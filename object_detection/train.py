import keras
import keras_cv
import tensorflow as tf
import tensorflow_datasets as tfds

# Load the coco dataset
ds, ds_info = tfds.load(
    name='coco/2017',
    split='test[:5%]',  # +train[:5%]+validation[:5%]
    data_dir='~/.tensorflow_datasets/',
    with_info=True
)

# Optionally show examples
# tfds.show_examples(ds=ds, ds_info=ds_info)

# Function to preprocess the dataset
IMG_SIZE = 224  # Adjust size as needed


def preprocess(example):
    """Preprocess images and bounding boxes for DETR."""
    image = tf.image.resize(
        example['image'], (IMG_SIZE, IMG_SIZE)) / 255.0  # Normalize image

    # Extract bounding box and labels
    bbox = example['objects']['bbox']  # Normalized [ymin, xmin, ymax, xmax]
    label = example['objects']['label']  # Object classes

    return image, {'boxes': bbox, 'classes': label}


# Prepare dataset
train_ds = ds.map(preprocess).batch(32).prefetch(
    tf.data.AUTOTUNE)  # .shuffle(1000).batch(32)

for img, labels in train_ds.take(1):
    print("Image shape:", img.shape)  # Should be (batch_size, 512, 512, 3)
    print("Labels type:", type(labels))  # Should be dict
    print("Keys in labels:", labels.keys())  # Should contain "boxes" and "classes"
    print("BBox shape:", labels["boxes"].shape)  # Should match batch size
    print("Classes shape:", labels["classes"].shape)

model = keras_cv.models.YOLOV8Detector(
    backbone=keras_cv.models.YOLOV8Backbone.from_preset(
        preset='yolo_v8_s_backbone_coco',
        load_weights=True
    ),
    num_classes=80,
    bounding_box_format='XYWH'
)
model.compile(
    classification_loss='binary_crossentropy',
    box_loss='ciou',
    optimizer=keras.optimizers.SGD(global_clipnorm=10.0),
    jit_compile=False,
)
model.fit(train_ds, epochs=10)
