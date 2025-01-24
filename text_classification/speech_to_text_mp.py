import time
import pyaudio
from google.oauth2 import service_account
from google.cloud import speech
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer
import tensorflow as tf
import dotenv
import multiprocessing.connection


def speech_to_text(conn: multiprocessing.connection.Connection = None):
    """
    A wrapper for speech_to_text.py for multiprocessing
    """

    sentence = ''

    dotenv.load_dotenv('../.env')
    client_file = 'text_classification/speech_api_keys.json'
    credentials = service_account.Credentials.from_service_account_file(
        client_file)

    RATE = 16000
    CHUNK = int(RATE / 10)

    audio_interface = pyaudio.PyAudio()

    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=RATE,
        language_code='en-US'
    )
    streaming_config = speech.StreamingRecognitionConfig(
        config=config, interim_results=True)

    def generate_audio_chunks(stream):
        """Generator that yields audio chunks from the microphone."""
        for _ in range(100):
            data = stream.read(CHUNK, exception_on_overflow=False)
            if not data:
                break
            yield speech.StreamingRecognizeRequest(audio_content=data)
            time.sleep(0.1)

    try:
        stream = audio_interface.open(
            format=pyaudio.paInt16, channels=1, rate=RATE, input=True, frames_per_buffer=CHUNK)

        with speech.SpeechClient(credentials=credentials) as client:

            print("Listening... Press Ctrl+C to stop.")

            responses = client.streaming_recognize(
                config=streaming_config, requests=generate_audio_chunks(stream))

            for response in responses:
                for result in response.results:
                    if result.is_final:
                        print("Transcript:", result.alternatives[0].transcript)
                        sentence = result.alternatives[0].transcript
                        raise StopIteration

    except StopIteration:
        print("Stopping after first sentence.")
    except KeyboardInterrupt:
        print("Stopped by user.")
    finally:
        time.sleep(0.5)
        try:
            if stream.is_active():
                stream.stop_stream()
            stream.close()
        except OSError:
            print("Error stopping/closing the stream")
        audio_interface.terminate()

    model_save_path = 'text_classification/models/trained_v4'
    model = TFAutoModelForSequenceClassification.from_pretrained(
        model_save_path)
    tokenizer = AutoTokenizer.from_pretrained(model_save_path)

    # input_text = sentence
    input_text = 'Can you hand me the eraser?'

    inputs = tokenizer(input_text, return_tensors="tf")

    outputs = model(**inputs)
    logits = outputs.logits

    probabilities = tf.nn.softmax(logits, axis=-1)

    predicted_class_id = tf.argmax(logits, axis=-1).numpy()[0]

    print(f"Predicted class ID: {predicted_class_id}")
    print(f"Class probabilities: {probabilities.numpy()}")

    conn.send(predicted_class_id)


if __name__ == '__main__':
    speech_to_text()