from object_detection import keras_cv_debug
from text_classification import speech_to_text_mp
import multiprocessing
import robot

if __name__ == '__main__':

    upstream, downstream = multiprocessing.Pipe()

    object_detection_p = multiprocessing.Process(target=keras_cv_debug.inference, args=(False, downstream))
    robot_p = multiprocessing.Process(target=speech_to_text_mp.speech_to_text, args=(downstream, ))

    object_detection_p.start()
    object_detection_p.join()
    robot_p.start()

    object_detection_p.join()
    robot_p.join()