# Example 4.12

from tensorflow.keras.preprocessing import image
from keras.applications.imagenet_utils import decode_predictions
from tensorflow.keras.applications import (
    vgg16,
    resnet50,
    mobilenet,
    inception_v3
)
import numpy as np


# ============================
# Initialize the model
# ============================

# model = vgg16.VGG16(weights='imagenet')
# model = resnet50.ResNet50(weights='imagenet')
model = mobilenet.MobileNet(weights='imagenet')
# model = inception_v3.InceptionV3(weights='imagenet')

print(model.summary())


# ============================
# Load image
# ============================

img_path = 'Elephant.jpg'

img = image.load_img(
    img_path,
    target_size=(224, 224)
)

x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)


# ============================
# Preprocess image
# ============================

# processed_image = vgg16.preprocess_input(x)
# processed_image = resnet50.preprocess_input(x)
processed_image = mobilenet.preprocess_input(x)
# processed_image = inception_v3.preprocess_input(x)


# ============================
# Prediction
# ============================

predictions = model.predict(processed_image)

results = decode_predictions(predictions)

print(results)