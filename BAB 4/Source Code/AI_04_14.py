# Example 4.14

import cv2
import numpy as np

from tensorflow.keras.preprocessing.image import img_to_array
from keras.applications.imagenet_utils import decode_predictions
from tensorflow.keras.applications import (
    vgg16,
    resnet50,
    mobilenet,
    inception_v3
)


# ============================
# Initialize the model
# ============================

# model = vgg16.VGG16(weights='imagenet')
model = resnet50.ResNet50(weights='imagenet')
# model = mobilenet.MobileNet(weights='imagenet')
# model = inception_v3.InceptionV3(weights='imagenet')

print(model.summary())


# ============================
# Initialize camera
# ============================

camera = cv2.VideoCapture(0)

image_size = 224
# image_size = 299


# ============================
# Real-time prediction
# ============================

while camera.isOpened():

    # Read frame from camera
    ok, cam_frame = camera.read()

    if not ok:
        break

    # Resize frame
    frame = cv2.resize(
        cam_frame,
        (image_size, image_size)
    )

    # Convert image to array
    numpy_image = img_to_array(frame)

    # Add batch dimension
    image_batch = np.expand_dims(
        numpy_image,
        axis=0
    )


    # ============================
    # Preprocess image
    # ============================

    # processed_image = vgg16.preprocess_input(
    #     image_batch.copy()
    # )

    processed_image = resnet50.preprocess_input(
        image_batch.copy()
    )

    # processed_image = mobilenet.preprocess_input(
    #     image_batch.copy()
    # )

    # processed_image = inception_v3.preprocess_input(
    #     image_batch.copy()
    # )


    # ============================
    # Prediction
    # ============================

    predictions = model.predict(
        processed_image,
        verbose=0
    )

    label = decode_predictions(
        predictions,
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
        1,
        (255, 0, 0),
        2
    )

    # Display camera
    cv2.imshow(
        'video image',
        cam_frame
    )


    # ============================
    # Exit program
    # ============================

    key = cv2.waitKey(30)

    if key == 27:  # Press ESC to quit
        break


# ============================
# Release resources
# ============================

camera.release()
cv2.destroyAllWindows()