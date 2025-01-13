from object_detection import keras_cv_debug
from object_detection import opencv_keras
import multiprocessing
import robot

if __name__ == '__main__':

    upstream, downstream = multiprocessing.Pipe()

    object_detection_p = multiprocessing.Process(target=keras_cv_debug.inference, args=(False, upstream))
    robot_p = multiprocessing.Process(target=robot.f, args=(downstream, ))

    object_detection_p.start()
    object_detection_p.join()
    robot_p.start()

    object_detection_p.join()
    robot_p.join()