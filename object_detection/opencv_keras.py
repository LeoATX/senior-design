import cv2
import keras
import keras_cv
import multiprocessing.connection

def inference(debug: bool = True, conn: multiprocessing.connection.Connection = None):
    debug = False

    # model = keras_cv.models.RetinaNet.from_preset(
    #     preset='retinanet_resnet50_pascalvoc',
    #     load_weights=True,
    #     num_classes=20,
    #     bounding_box_format='XYXY'
    # )

    # model = keras_cv.models.YOLOV8Detector(
    #     backbone=keras_cv.models.YOLOV8Backbone.from_preset(
    #         'yolo_v8_m_backbone_coco'),
    #     num_classes=80,
    #     bounding_box_format='XYXY'
    # )
    model = keras_cv.models.YOLOV8Detector.from_preset(
        preset='yolo_v8_m_pascalvoc',
        load_weights=True
    )
    video = cv2.VideoCapture(0)

    while True:
        _, frame = video.read()

        # Convert the captured frame into RGB
        # im = Image.fromarray(frame, 'RGB')
        frame = cv2.resize(src=frame, dsize=(640, 640))

        # Resizing into dimensions you used while training
        keras_frame = keras.utils.img_to_array(frame)

        # Expand dimensions to match the 4D Tensor shape.
        keras_frame = keras.ops.expand_dims(x=keras_frame, axis=0)
        print(keras_frame.shape) if debug else None

        # Calling the predict function using keras
        predictions = model.predict(keras_frame)
        conn.send(predictions['classes']) if conn != None else None

        for index in range(predictions['num_detections'][0]):
            box = predictions['boxes'][0][index]
            print(box) if debug else None
            x, y, w, h = [int(item) for item in box[0:4]]
            print(x, y, w, h) if debug else None
            box_conf = predictions['confidence'][0][index]
            box_class = predictions['classes'][0][index]

            # Draw bounding box and label on the frame
            label = f'{box_class}: {box_conf}'
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        cv2.imshow('YOLOv8', frame)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    inference()