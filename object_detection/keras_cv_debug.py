import keras
import keras_cv
import multiprocessing.connection
import os


def inference(debug: bool = True, conn: multiprocessing.connection.Connection = None):

    # model = keras_cv.models.RetinaNet.from_preset(
    #     preset='retinanet_resnet50_pascalvoc',
    #     load_weights=True,
    #     num_classes=20,
    #     bounding_box_format='XYWH'
    # )

    # model = keras_cv.models.YOLOV8Detector(
    #     backbone=keras_cv.models.YOLOV8Backbone.from_preset('yolo_v8_m_backbone_coco'),
    #     num_classes=80,
    #     bounding_box_format='XYWH'
    # )

    model = keras_cv.models.YOLOV8Detector.from_preset(
        preset='yolo_v8_m_pascalvoc', 
        load_weights=True
    )

    image1 = keras.utils.load_img(
        path=os.path.join(os.path.dirname(
            os.path.abspath(__file__)), 'test.jpg'),
        color_mode='rgb',
        target_size=None,
        interpolation='nearest',
        keep_aspect_ratio=False,
    )
    image1 = keras.utils.img_to_array(image1)
    image1 = keras.layers.Resizing(640, 640)(image1)
    image1 = keras.ops.reshape(x=image1, newshape=(1, *image1.shape))

    image2 = keras.utils.load_img(
        path=os.path.join(os.path.dirname(
            os.path.abspath(__file__)), 'people.jpg'),
        color_mode='rgb',
        target_size=None,
        interpolation='nearest',
        keep_aspect_ratio=False,
    )
    image2 = keras.utils.img_to_array(image2)
    image2 = keras.layers.Resizing(640, 640)(image2)
    image2 = keras.ops.reshape(x=image2, newshape=(1, *image2.shape))

    images = keras.ops.concatenate(xs=[image1, image2], axis=0)
    # images = keras.ops.reshape(x=images, newshape=(1, *images.shape))  # one image
    print(images.shape) if debug else None

    predictions: dict = model.predict(images)
    # dict_keys(['boxes', 'confidence', 'classes', 'num_detections'])
    if debug:
        print(predictions)
        for each in predictions['num_detections']:
            print(f'number of predictions: {each}')

    class_ids = [
        'Aeroplanes',
        'Bicycles',
        'Birds',
        'Boats',
        'Bottles',
        'Buses',
        'Cars',
        'Cats',
        'Chairs',
        'Cows',
        'Dining tables',
        'Dogs',
        'Horses',
        'Motorbikes',
        'People',
        'Potted plants',
        'Sheep',
        'Sofas',
        'Trains',
        'TV/Monitors'
    ]

    class_mapping = dict(zip(range(len(class_ids)), class_ids))

    # y_pred = keras_cv.bounding_box.to_ragged(y_pred)
    keras_cv.visualization.plot_bounding_box_gallery(
        images,
        value_range=(0, 255),
        bounding_box_format='XYWH',
        y_true=None,
        y_pred=predictions,
        scale=4,
        rows=1,
        cols=2,
        show=True,
        font_scale=0.5,
        class_mapping=class_mapping,
    ) if debug else None

    conn.send(predictions['classes']) if conn != None else None


if __name__ == '__main__':
    print(inference(debug=True))
