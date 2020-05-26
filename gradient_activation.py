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
from tensorflow.python.framework import ops
import tensorflow as tf
import cv2


def target_category_loss(x, category_index, nb_classes):
    return tf.multiply(x, K.one_hot([category_index], nb_classes))


def target_category_loss_output_shape(input_shape):
    return input_shape


def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)


def register_gradient():
    if "GuidedBackProp" not in ops._gradient_registry._registry:
        @ops.RegisterGradient("GuidedBackProp")
        def _GuidedBackProp(op, grad):
            dtype = op.inputs[0].dtype
            return grad * tf.cast(grad > 0., dtype) * \
                tf.cast(op.inputs[0] > 0., dtype)


def modify_backprop(model, name):
    g = tf.compat.v1.get_default_graph()  # Updated for Tensorflow 2.0
    with g.gradient_override_map({'Relu': name}):

        # get layers that have an activation
        layer_dict = [layer for layer in model.layers[1:]
                      if hasattr(layer, 'activation')]

        # replace relu activation
        for layer in layer_dict:
            if layer.activation == keras.activations.relu:
                layer.activation = tf.nn.relu


def compile_saliency_function(model, activation_layer='block5_conv3'):
    input_img = model.input
    final_conv_layer = get_output_layer(model, "block5_conv3")
    layer_output = final_conv_layer.get_output_at(-1)
    max_output = K.max(layer_output, axis=3)
    saliency = K.gradients(K.sum(max_output), input_img)[0]
    return K.function([input_img, K.learning_phase()], [saliency])


def deprocess_image(x):
    """
    Same normalization as in:
    https://github.com/fchollet/keras/blob/master/examples/conv_filter_visualization.py
    """
    if np.ndim(x) > 3:
        x = np.squeeze(x)
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    if K.image_data_format() == 'channels_first':
        x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x


def grad_cam(input_model, image, category_index, layer_name):
    model = input_model
    loss = K.sum(model.layers[16].get_output_at(-1))
    final_conv_layer = get_output_layer(model, "block5_conv3")
    conv_output = final_conv_layer.get_output_at(-1)
    grads = normalize(K.gradients(loss, conv_output)[0])
    gradient_function = K.function([model.layers[0].get_input_at(0)], [conv_output, grads])

    output, grads_val = gradient_function([image])
    output, grads_val = output[0, :], grads_val[0, :, :, :]

    weights = np.mean(grads_val, axis = (0, 1))
    cam = np.ones(output.shape[0 : 2], dtype = np.float32)

    for i, w in enumerate(weights):
        cam += w * output[:, :, i]

    cam = cv2.resize(cam, (224, 224))
    cam = np.maximum(cam, 0)
    heatmap = cam / np.max(cam)

    # Return to BGR [0..255] from the preprocessed image
    image = image[0, :]
    image -= np.min(image)
    image = np.minimum(image, 255)

    cam = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)
    cam[np.where(cam < 0.2)] = 0
    cam = np.float32(cam) + np.float32(image)
    cam = 255 * cam / np.max(cam)
    return np.uint8(cam), heatmap


def visualize_class_activation_map(model_path, img_paths, output_paths1, output_paths2):
    # Load model
    vgg16_model = keras.applications.vgg16.VGG16()
    vgg16_model.layers.pop()
    model = Sequential()
    for layer in vgg16_model.layers:
        model.add(layer)

    model.add(Dense(2, activation='softmax'))
    del vgg16_model  # Clear memory

    model.load_weights(model_filepath)
    print(model.summary())

    # Access each image
    for i in range(len(img_paths)):
        img_path = img_paths[i]
        output_path1 = output_paths1[i]
        output_path2 = output_paths2[i]

        # load an image from file
        img = load_img(img_path, target_size=(224, 224))

        # convert the image pixels to a numpy array
        img = img_to_array(img)

        # reshape data for the model
        img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
        height, width = img.shape[1], img.shape[2]

        # prepare the image for the VGG model
        img = preprocess_input(img)

        # Predict
        predictions = model.predict(img)
        predicted_class = np.argmax(predictions)
        cam, heatmap = grad_cam(model, img, predicted_class, "block5_conv3")

        # Convert back to image
        cam = np.squeeze(cam)
        img_pil = array_to_img(cam)
        img_pil.show()
        img_pil.save(output_path1)
        print("Saved Grad-CAM:%s" % output_path1)

        register_gradient()
        modify_backprop(model, 'GuidedBackProp')
        saliency_fn = compile_saliency_function(model)
        saliency = saliency_fn([img, 0])
        gradcam = saliency[0] * heatmap[..., np.newaxis]

        # Convert back to image
        gradcam = deprocess_image(gradcam)
        img_pil2 = array_to_img(gradcam)
        img_pil2.show()
        img_pil2.save(output_path2)
        print("Saved guided Grad-CAM:%s" % output_path2)

    print("Success")


def get_output_layer(model, layer_name):
    # get the symbolic outputs of each "key" layer (we gave them unique names).
    layer_dict = dict([(layer.name, layer) for layer in model.layers])
    layer = layer_dict[layer_name]
    return layer


model_filepath = './Models/Model_1.h5'

image_folder = 'Q2/Test/G7/'
output_folder1 = 'Q2/Plot/G7_gradient1/'
output_folder2 = 'Q2/Plot/G7_gradient2/'

# image_folder = 'Q2/Test/LE7/'
# output_folder1 = 'Q2/Plot/LE7_gradient1/'
# output_folder2 = 'Q2/Plot/LE7_gradient2/'

image_files = [f for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f))]
image_paths = [os.path.join(image_folder, f) for f in image_files]
output_image_paths1 = [os.path.join(output_folder1, f) for f in image_files]
output_image_paths2 = [os.path.join(output_folder2, f) for f in image_files]

visualize_class_activation_map(model_filepath, image_paths, output_image_paths1, output_image_paths2)
