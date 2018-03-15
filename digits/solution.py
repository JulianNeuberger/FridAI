from datetime import datetime
from os import path

import numpy
from PIL import Image
from keras import callbacks
from sklearn import datasets

from digits import util, config
from digits.util import get_model_for_name, AVAILABLE_SOLUTION_ARCHS

arch = input('How do you want to solve the problem? Choose one of {}'.format(AVAILABLE_SOLUTION_ARCHS.keys()))

# load prepared data set containing 1797 digits as 8x8 images
digit_features, digit_classes = datasets.load_digits(n_class=config.NUM_DIGITS, return_X_y=True)
num_samples = digit_classes.shape[0]

# normalize features, see documentation of sklearn.datasets.load_digits!
# neural networks work best with normalized data
digit_features /= config.MAX_FEATURE

# we need so called "one-hot" vectors
# one-hots are vectors, where all entries are 0 except the target class, which is 1
digit_labels = numpy.zeros(shape=(num_samples, config.NUM_DIGITS))
for index, digit_class in enumerate(digit_classes):
    digit_labels[index][digit_class] = 1.

# get a neural net, that can fit our problem
model = get_model_for_name(arch)

# prints a human readable summary of the model to the out-stream
model.summary()

# a callback to log loss/accuracy etc. for tensorboard to visualize
run_name = 'digits-{}-{:%d-%b_%H-%M-%S}'.format(arch, datetime.now())
tb_callback = callbacks.TensorBoard(log_dir=path.join('logs', run_name))

# training the model
model.fit(digit_features, digit_labels, batch_size=32, epochs=30, validation_split=.2, callbacks=[tb_callback])

# finally save the model's weights
# for compatibility reasons we don't save the entire model
# instead save only the weights, on load build the same model and load them
model.save_weights(path.join('weights', run_name))

image = Image.open(path.join('img', '0.bmp'))
image.show()
pixels = util.image_to_ndarray(image)
pixels.shape = (1,) + pixels.shape
prediction = model.predict(pixels)
print('Raw prediction is {}'.format(prediction[0]))
print('This has to be a "{}"'.format(prediction[0].argmax()))
