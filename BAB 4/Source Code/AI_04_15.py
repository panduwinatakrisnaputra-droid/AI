# Example 4.15
# https://pypi.org/project/image-classifiers/

# Install library terlebih dahulu jika menggunakan Google Colab/Jupyter
# ============================
# Import libraries
# ============================

import cv2
import numpy as np

from keras.applications.imagenet_utils import decode_predictions
from classification_models.keras import Classifiers


# ============================
# Set up model
# ============================

# VGG16
clf, preprocess_input = Classifiers.get('vgg16')

# ResNet50
# clf, preprocess_input = Classifiers.get('resnet50')

# MobileNetV2
# clf, preprocess_input = Classifiers.get('mobilenetv2')

# DenseNet201
# clf, preprocess_input = Classifiers.get('densenet201')

# InceptionV3
# clf, preprocess_input = Classifiers.get('inceptionv3')


# Ukuran input gambar
sz = 224

# Untuk InceptionV3 gunakan:
# sz = 299


# ============================
# Create model
# ============================

model = clf(
    input_shape=(sz, sz, 3),
    weights='imagenet',
    classes=1000
)

model.summary()


# ============================
# Initialize camera
# ============================

camera = cv2.VideoCapture(0)


# ============================
# Real-time classification
# ============================

while True:

    # Membaca frame kamera
    ret, cam_frame = camera.read()

    if not ret:
        break

    # Resize frame
    frame = cv2.resize(
        cam_frame,
        (sz, sz)
    )

    # Convert frame menjadi NumPy array
    image = np.asarray(frame)

    # Menambahkan batch dimension
    image = np.expand_dims(
        image,
        axis=0
    )

    # Preprocess image
    image = preprocess_input(image)


    # ============================
    # Prediction
    # ============================

    preds = model.predict(
        image,
        verbose=0
    )

    label = decode_predictions(
        preds,
        top=1
    )


    # ============================
    # Display prediction
    # ============================

    text = "{}, {:.1f}".format(
        label[0][0][1],
        label[0][0][2]
    )

    cv2.putText(
        cam_frame,
        text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2
    )


    # Menampilkan kamera
    cv2.imshow(
        "Classification",
        cam_frame
    )


    # ============================
    # Exit
    # ============================

    key = cv2.waitKey(30)

    if key == 27:  # Press ESC to quit
        break


# ============================
# Release resources
# ============================

camera.release()
cv2.destroyAllWindows()