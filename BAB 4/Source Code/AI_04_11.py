# Example 4.11

import numpy as np

from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import (
    VGG16,
    preprocess_input
)
from keras.applications.imagenet_utils import decode_predictions


# ============================
# Initialize the VGG16 model
# ============================

model = VGG16(weights='imagenet')


# ============================
# Load image
# ============================

img_path = 'AI BAB 4/Elephant.jpg'

img = image.load_img(
    img_path,
    target_size=(224, 224)
)


# ============================
# Preprocess image
# ============================

x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)


# ============================
# Prediction
# ============================

predictions = model.predict(x, verbose=0)

results = decode_predictions(
    predictions,
    top=5
)


# ============================
# Display results
# ============================

for result in results[0]:
    print(
        "Label: {} | Confidence: {:.2f}%".format(
            result[1],
            result[2] * 100
        )
    )