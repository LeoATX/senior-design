import nemo.collections.asr as nemo_asr
import time
import whisper
import logging

if __name__ == '__main__':
    logging.getLogger('nemo_logger').setLevel(logging.ERROR)

    nvidia_asr = nemo_asr.models.ASRModel.from_pretrained('nvidia/canary-1b')
    openai_asr = whisper.load_model('turbo')

    print('\n\n')

    nvidia_start = time.time()
    transcript = nvidia_asr.transcribe(['speech_recognition/audio1.wav'])
    print(transcript[0])
    print('nvidia:', time.time() - nvidia_start)

    print('\n')

    openai_start = time.time()
    transcript = openai_asr.transcribe('speech_recognition/audio1.wav')
    print(transcript['text'])
    print('openai:', time.time() - nvidia_start)
