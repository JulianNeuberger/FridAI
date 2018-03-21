from datetime import datetime
from os import path
from numpy import random

# for reproducible experiments, you should seed your random generator
# training of ANNs, as well as their testing is heavily dependent on order
# therefore it is dependent on chance (==> shuffling of samples!)
random.seed(1337)

import numpy
from PIL import Image
from keras import callbacks
from sklearn import datasets

from util import get_model_for_name, AVAILABLE_SOLUTION_ARCHS, image_to_ndarray
from config import NUM_DIGITS, MAX_FEATURE


arch = input('How do you want to solve the problem? Choose one of {}'.format(AVAILABLE_SOLUTION_ARCHS.keys()))

# load prepared data set containing 1797 digits as 8x8 images
digit_features, digit_classes = datasets.load_digits(n_class=NUM_DIGITS, return_X_y=True)
num_samples = digit_classes.shape[0]

# normalize features, see documentation of sklearn.datasets.load_digits!
# neural networks work best with normalized data
digit_features /= MAX_FEATURE

# we need so called "one-hot" vectors
# one-hots are vectors, where all entries are 0 except the target class, which is 1
digit_labels = numpy.zeros(shape=(num_samples, NUM_DIGITS))
for index, digit_class in enumerate(digit_classes):
    digit_labels[index][digit_class] = 1.

# get a neural net, that can fit our problem
model = get_model_for_name(arch)

# prints a human readable summary of the model to the out-stream
model.summary()

# a callback to log loss/accuracy etc. for tensorboard to visualize
run_name = 'digits-{}-{:%d-%b_%H-%M-%S}'.format(arch, datetime.now())
tb_callback = callbacks.TensorBoard(log_dir=path.join('digits', 'logs', run_name))

# training the model
model.fit(digit_features, digit_labels, batch_size=32, epochs=30, validation_split=.2, callbacks=[tb_callback])

# finally save the model's weights
# for compatibility reasons we don't save the entire model
# instead save only the weights, on load build the same model and load them
model.save_weights(path.join('digits', 'weights', run_name))

# load our test image and show it
image = Image.open(path.join('digits', 'img', '7.bmp'))
image.show()

# convert the image to a numpy array
pixels = image_to_ndarray(image)
# add a dimension, which is the batch size dimension (therefore it is 1 for our single sample)
pixels.shape = (1,) + pixels.shape

# predict the class (which digit?) for this sample
# model.predict returns a prediction for each input sample, as list
# the first one corresponds to the first sample
# here: each prediction is a vector, with [number of digits: 0-9 = 10] entries
# each entry is a probability of 0 (0%) - 1 (100%), how likely the input sample is this digit
prediction = model.predict(pixels)
print('Raw prediction is {}'.format(prediction[0]))
print('This has to be a "{}"'.format(prediction[0].argmax()))
