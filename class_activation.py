import os
import numpy as np
import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers.core import Dense, Lambda
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import array_to_img
from keras.applications.vgg16 import preprocess_input
from keras.optimizers import SGD
import cv2


def visualize_class_activation_map(model_path, img_paths, output_paths):
    # Load model
    vgg16_model = keras.applications.vgg16.VGG16()

    # Remove all after last convolutional layer
    for i in range(5):
        vgg16_model.layers.pop()

    model = Sequential()
    for layer in vgg16_model.layers:
        model.add(layer)

    # Load weights
    model.load_weights(model_filepath, by_name=True)

    # Add global average pooling layers
    model.add(Lambda(global_average_pooling,
              output_shape=global_average_pooling_shape))
    model.add(Dense(2, activation='softmax', init='uniform'))
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.5, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    del vgg16_model  # Clear memory

    print(model.summary())

    # Access each image
    for i in range(len(img_paths)):
        img_path = img_paths[i]
        output_path = output_paths[i]

        # load an image from file
        img = load_img(img_path, target_size=(224, 224))

        # convert the image pixels to a numpy array
        img = img_to_array(img)

        # reshape data for the model
        img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
        height, width = img.shape[1], img.shape[2]

        # prepare the image for the VGG model
        img = preprocess_input(img)

        # Get the 512 input weights to the softmax.
        class_weights = model.layers[16].get_weights()[1]
        final_conv_layer = get_output_layer(model, "block5_conv3")
        get_output = K.function([model.layers[0].get_input_at(0)], [final_conv_layer.get_output_at(-1), model.layers[16].get_output_at(-1)])
        [conv_outputs, predictions] = get_output([img])
        conv_outputs = conv_outputs[0, :, :, :]

        # Create the class activation map.
        cam = np.zeros(dtype=np.float32, shape=conv_outputs.shape[0:2])
        for i, w in enumerate(class_weights):
            conv_output_image = conv_outputs[:, :, i]
            cam += w * conv_output_image

        cam /= np.max(cam)
        cam = cv2.resize(cam, (height, width))
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap[np.where(cam < 0.2)] = 0
        img = heatmap * 0.5 + img

        # Convert back to image
        img = np.squeeze(img)
        img_pil = array_to_img(img)
        img_pil.show()
        img_pil.save(output_path)
        print("Saved image %s" % output_path)

    print("Success")


def get_output_layer(model, layer_name):
    # get the symbolic outputs of each "key" layer (we gave them unique names).
    layer_dict = dict([(layer.name, layer) for layer in model.layers])
    layer = layer_dict[layer_name]
    return layer


def global_average_pooling(x):
    return K.mean(x, axis=(2, 3))


def global_average_pooling_shape(input_shape):
    return input_shape[0:2]


model_filepath = './Models/Model_1.h5'
# image_folder = 'Q2/Test/G7/'
# output_folder = 'Q2/Plot/G7_sum/'
image_folder = 'Q2/Test/LE7/'
output_folder = 'Q2/Plot/LE7_sumV2/'
image_files = [f for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f))]
image_paths = [os.path.join(image_folder, f) for f in image_files]
output_image_paths = [os.path.join(output_folder, f) for f in image_files]
visualize_class_activation_map(model_filepath, image_paths, output_image_paths)
