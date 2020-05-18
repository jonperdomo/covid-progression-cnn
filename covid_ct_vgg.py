import os
import sys
import time
import numpy as np
import tensorflow as tf
import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense, Flatten
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import *
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools


# plots images with labels within jupyter notebook
def plots(ims, figsize=(12,6), rows=1, interp=False, titles=None):
    if type(ims[0]) is np.ndarray:
        ims = np.array(ims).astype(np.uint8)
        if (ims.shape[-1] != 3):
            ims = ims.transpose((0,2,3,1))
    f = plt.figure(figsize=figsize)
    cols = len(ims)//rows if len(ims) % 2 == 0 else len(ims)//rows + 1
    for i in range(len(ims)):
        sp = f.add_subplot(rows, cols, i+1)
        sp.axis('Off')
        if titles is not None:
            title = str(titles[i])
            sp.set_title(title, fontsize=16)
        plt.imshow(ims[i], interpolation=None if interp else 'none')
    plt.show()


# Plot
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting 'normalize=True'.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print("Confusion matrix, without normalization")

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment='center',
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


def train(number_of_epochs=5):
    # Don't pre-allocate GPU memory (Tensorflow 2)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)

    output_path = 'Models/'
    train_path = 'Q2/Train/'
    valid_path = 'Q2/Valid/'
    test_path = 'Q2/Test'
    labels = ['G7', 'LE7']

    # 112 training images
    train_batches = ImageDataGenerator().flow_from_directory(train_path, target_size=(224, 224), classes=labels, batch_size=8)

    # 16 validation images
    valid_batches = ImageDataGenerator().flow_from_directory(valid_path, target_size=(224, 224), classes=labels, batch_size=8)

    # 14 test images
    test_batches = ImageDataGenerator().flow_from_directory(test_path, target_size=(224, 224), classes=labels, batch_size=7)

    # Build fine-tuned VGG-16 model
    vgg16_model = keras.applications.vgg16.VGG16()

    # Pop the last output layer
    vgg16_model.layers.pop()

    # Transform VGG-16 from type Model to Sequential
    model = Sequential()
    for layer in vgg16_model.layers:
        model.add(layer)

    # Freeze layers from future training so weights are not updated
    for layer in model.layers:
        layer.trainable = False

    # Add an updated dense layer for the 2 categories
    model.add(Dense(2, activation='softmax'))
    # model.summary()
    del vgg16_model  # Clear memory

    # Compile
    model.compile(Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

    # Train
    epochs = 100
    start = time.time()
    model.fit_generator(train_batches, steps_per_epoch=14, validation_data=valid_batches, validation_steps=2, epochs=epochs, verbose=2)
    end = time.time()
    time_elapsed = end - start
    print("Time (s): %.3f" % time_elapsed)

    # Save the model
    model_version = 0
    filename = "Model_%d.h5" % 0
    output_filepath = os.path.join(output_path, filename)
    while os.path.exists(output_filepath):
        model_version += 1
        output_filepath = os.path.join(output_path, filename)

    model.save(output_filepath)

    print('Saved model %s' % output_filepath)


# Set number of epochs in the command line
if __name__ == "__main__":
    if len(sys.argv) == 2:
        epochs = sys.argv[1]
        print(f"Number of epochs: {epochs}")
        train(epochs)
    else:
        print("Missing number of epochs as input argument")

