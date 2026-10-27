# Example 4.17 CNN Visualize Filters.ipynb
#Modified from:

#https://machinelearningmastery.com/how-to-visualize-filters-and-feature-maps-in-convolutional-neural-networks/

# load vgg model
from keras.applications.vgg19 import VGG19
# load the model
model = VGG19()
# summarize the model
model.summary()
n=0
for layer in model.layers:
    print(n,layer.name)
    n+=1

n=0
for layer in model.layers:
    if 'conv' in layer.name:
        filters, biases = layer.get_weights()
        print(n,layer.name, filters.shape, biases.shape)
    n+=1