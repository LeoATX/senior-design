import multiprocessing.synchronize
from transformers import logging
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer
from ultralytics import YOLO
import cv2
import flask
from flask import Flask, render_template, request
import librosa
import multiprocessing
import multiprocessing.sharedctypes
import numpy as np
import pyaudio
import tensorflow as tf
import time
import warnings
import whisper
import json
from google.oauth2 import service_account
from google.cloud import speech

# HARDCORE debug mode
DEBUG = False

client_file = '/Users/lauren/Desktop/senior-design-april/app/speech_api_keys.json'
credentials = service_account.Credentials.from_service_account_file(client_file)

RATE = 16000
CHUNK = int(RATE / 10)
audio_interface = pyaudio.PyAudio()

config = speech.RecognitionConfig(
    encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
    sample_rate_hertz=RATE,
    language_code='en-US'
)
streaming_config = speech.StreamingRecognitionConfig(config=config, interim_results=True)

LABELS = {0: 'cup', 1: 'bottle', 2: 'fork', 3: 'cell phone', 4: 'neutral'}
YOLO_CLASSES = {'cup': 41, 'bottle': 39, 'fork': 42, 'cell phone': 67}

def run_workflow(
    start_event: multiprocessing.synchronize.Event,
    status_message: multiprocessing.sharedctypes.Synchronized
):
    def generate_audio_chunks(stream):
        for _ in range(100):
            if not stream.is_active():
                break
            try:
                data = stream.read(CHUNK, exception_on_overflow=False)
                if not data:
                    break
                yield speech.StreamingRecognizeRequest(audio_content=data)
                time.sleep(0.1)
            except OSError:
                break
    # instantiate the models at startup
    status = 'Loading models, please wait...'
    status_message.value = status
    print(status)
    start_time = time.time()
    model_save_path = '../trained_v10'
    text_model = TFAutoModelForSequenceClassification.from_pretrained(
        model_save_path)
    tokenizer = AutoTokenizer.from_pretrained(model_save_path)
    yolo_model = YOLO('yolov8s.pt')
    end_time = time.time()
    status = f'Models loaded! Models loaded in {end_time - start_time} seconds'
    print(status)
    status_message.value = status

    while True:
        status_message.value = "Click the Start Workflow button to start."
        print("Click the \"Start Workflow\" button to start.")
        start_event.wait()  # blocks until start_event.set() is called

        # Clear event to allow it to be set again
        start_event.clear()

        try:
            sentence = ""
            try:
                stream = audio_interface.open(format=pyaudio.paInt16, channels=1, rate=RATE, input=True, frames_per_buffer=CHUNK, start=False)
                stream.start_stream()
                with speech.SpeechClient(credentials=credentials) as client:
                    status_message.value = 'Recording... Please speak into the microphone.'
                    print("Listening... Say the object to detect.")
                    responses = client.streaming_recognize(config=streaming_config, requests=generate_audio_chunks(stream))
                    for response in responses:
                        for result in response.results:
                            if result.is_final:
                                sentence = result.alternatives[0].transcript
                                status_message.value = f'Transcript: {sentence}'
                                print("Transcript:", sentence)
                                raise StopIteration
            except StopIteration:
                print("Processing the detected speech...")
            except KeyboardInterrupt:
                print("Stopped by user.")
            finally:
                if 'stream' in locals() and stream is not None:
                    if stream.is_active():
                        stream.stop_stream()
                    stream.close()
                #audio_interface.terminate()

            inputs = tokenizer(sentence, return_tensors='tf')
            outputs = text_model(**inputs)
            predicted_class_id = tf.argmax(outputs.logits, axis=-1).numpy()[0]
            detected_label = LABELS.get(predicted_class_id, 'neutral')
            status_message.value = f'Transcript: {sentence} Detected label: {detected_label}'
            print(f'Detected label: {detected_label}')
            time.sleep(1)
            if detected_label == 'neutral':
                status_message.value = 'Neutral detected. No object detection needed.'
                print('Neutral detected. No object detection needed.')
                time.sleep(1)
                continue

            target_class = YOLO_CLASSES.get(detected_label, -1)
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                status_message.value = 'Error: Could not open camera.'
                continue

            object_detected = False
            status_message.value = f"Looking for: {detected_label}"
            print(f"Looking for: {detected_label}")
            while not object_detected:
                ret, frame = cap.read()
                if not ret:
                    status_message.value = 'Error: Failed to capture frame.'
                    print("Error: Failed to capture frame.")
                    break
                results = yolo_model(frame, verbose=False)
                for result in results:
                    for box in result.boxes:
                        cls = int(box.cls.item())
                        if cls == target_class:
                            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                            center_x = (x1 + x2) // 2
                            center_y = (y1 + y2) // 2
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(frame, detected_label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                            print(f"{detected_label.capitalize()} detected at: ({x1}, {y1}) to ({x2}, {y2})")
                            status_message.value = f"Center of bounding box: ({center_x}, {center_y})"                    
                            print(f"Center of bounding box: ({center_x}, {center_y})")
                            time.sleep(2)
                            object_detected = True
                            break
                    if object_detected:
                        break
                cv2.imshow('YOLO Object Detection', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            cap.release()

        except Exception as e:
            print(f"Exception during workflow: {e}")
            status_message.value = f'Error: {str(e)}'

if __name__ == '__main__':

    # set some warning levels
    warnings.filterwarnings('ignore')
    logging.set_verbosity_error()

    start_event = multiprocessing.Event()
    status_message = multiprocessing.Manager().Value('s', 'Initializing workflow..')
    multiprocessing.Process(
        target=run_workflow, args=(start_event, status_message, )
    ).start()

    app = flask.Flask(__name__)

    @app.route('/')
    def index():
        return render_template('index.html')

    @app.route('/start_workflow', methods=['POST'])
    def start_workflow():
        start_event.set()
        return json.dumps({'message': 'Workflow started!', 'status': status_message.value})
    
    @app.route('/status')
    def get_status():
        return json.dumps({'status': status_message.value})

    app.logger.setLevel('ERROR')
    app.run(host='0.0.0.0', port=8080, debug=False)