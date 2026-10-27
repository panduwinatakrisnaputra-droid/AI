
# Example 4.10 - VGG16

from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications.imagenet_utils import decode_predictions
import numpy as np

# Load VGG16 model with pretrained ImageNet weights
model = VGG16(weights='imagenet')

# Display model architecture
model.summary()
