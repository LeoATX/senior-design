from transformers import logging
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer
from ultralytics import YOLO
import cv2
import nemo.collections.asr as nemo_asr
import numpy as np
import pyaudio
import tensorflow as tf
import time
import warnings
import wave

# import os
# print(os.path.exists("text_classification/audio1.wav"))  # Should print True if the file exists

warnings.filterwarnings("ignore")
logging.set_verbosity_error()
LABELS = {0: "cup", 1: "bottle", 2: "fork", 3: "cell phone", 4: "neutral"}
YOLO_CLASSES = {"cup": 41, "bottle": 39, "fork": 42, "cell phone": 67}

RATE = 16000
CHUNK = int(RATE / 10)  # 100ms chunks
SILENCE_THRESHOLD = 1000  # Adjust as needed
MAX_SILENCE_CHUNKS = 50  # Number of silent chunks before stopping

# audio_interface = pyaudio.PyAudio()


def save_audio_to_wav(audio_samples, filename="debug_audio.wav"):
    """Saves recorded audio samples to a WAV file for debugging."""
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit PCM
        wf.setframerate(RATE)
        wf.writeframes(audio_samples.astype(np.int16).tobytes())


def generate_audio_chunks(stream):
    audio_data = []
    silent_chunks = 0

    print("Listening... Speak now.")

    while True:
        try:
            data = stream.read(CHUNK, exception_on_overflow=False)
            audio_array = np.frombuffer(data, dtype=np.int16)
            audio_data.append(audio_array)

            # Check if audio is silent
            if np.abs(audio_array).mean() < SILENCE_THRESHOLD:
                silent_chunks += 1
            else:
                silent_chunks = 0  # Reset counter if speech is detected

            # Stop recording after `MAX_SILENCE_CHUNKS` of silence
            if silent_chunks >= MAX_SILENCE_CHUNKS:
                print("Silence detected, stopping recording.")
                break

            time.sleep(0.1)

        except OSError:
            break

    return np.concatenate(audio_data, axis=0) if audio_data else None


sentence = ""
try:
    # stream = audio_interface.open(format=pyaudio.paInt16, channels=1, rate=RATE, input=True, frames_per_buffer=CHUNK, start=False)
    # stream.start_stream()
    # print("Listening... Say the object to detect.")

    # # Collect audio data
    # audio_samples = generate_audio_chunks(stream)
    # save_audio_to_wav(audio_samples)
    # if audio_samples is not None:
    #     audio_samples = audio_samples.astype(np.float32) / 32768.0
    #     transcript = asr_model.transcribe([audio_samples], source_lang='en', target_lang='en', pnc='nopnc', task='asr')  # NeMo transcribe returns a list
    #     sentence = " ".join(transcript)
    #     print("Transcript:", sentence)
    # transcript = asr_model.transcribe(["speech_recognition/audio1.wav"])
    pass
except KeyboardInterrupt:
    print("Stopped by user.")
finally:
    # if 'stream' in locals() and stream is not None:
    #     if stream.is_active():
    #         stream.stop_stream()
    #     stream.close()
    # audio_interface.terminate()
    pass

if __name__ == '__main__':

    asr_model = nemo_asr.models.ASRModel.from_pretrained("nvidia/canary-1b")
    transcript = asr_model.transcribe(
        ["text_classification/audio2.wav"], batch_size=1, source_lang="en", task="asr", pnc="pnc")
    print('\n\n\n')
    print('transcribing complete')
    print(transcript)
    sentence = ' '.join(transcript)

    model_save_path = 'text_classification/models/trained_v4'
    model = TFAutoModelForSequenceClassification.from_pretrained(model_save_path)
    tokenizer = AutoTokenizer.from_pretrained(model_save_path)

    inputs = tokenizer(sentence, return_tensors="tf")
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_id = tf.argmax(logits, axis=-1).numpy()[0]

    detected_label = LABELS.get(predicted_class_id, "neutral")
    print(f"Predicted label: {detected_label}")

    if detected_label == "neutral":
        print("Neutral detected. No object detection needed.")
        exit()

    yolo_model = YOLO('yolov8s.pt')

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        exit()

    object_detected = False
    target_class = YOLO_CLASSES.get(detected_label, -1)

    print(f"Looking for: {detected_label}")
    while not object_detected:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        results = yolo_model(frame)

        for result in results:
            for box in result.boxes:
                cls = int(box.cls.item())
                if cls == target_class:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, detected_label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                    print(
                        f"{detected_label.capitalize()} detected at: ({x1}, {y1}) to ({x2}, {y2})")
                    print(f"Center of bounding box: ({center_x}, {center_y})")
                    object_detected = True
                    break
            if object_detected:
                break

        cv2.imshow('YOLO Object Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.waitKey(1000)
    cv2.destroyAllWindows()
    cv2.waitKey(1)
