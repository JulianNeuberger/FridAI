from datetime import datetime
from os import path
from numpy import random

# for reproducible experiments, you should seed your random generator
# training of ANNs, as well as their testing is heavily dependent on order
# therefore it is dependent on chance (==> shuffling of samples!)
random.seed(1337)

from config import NUM_CLASSES
from keras import utils, callbacks
from models import get_model
from sklearn import datasets

features, labels = datasets.load_breast_cancer(return_X_y=True)
labels = utils.to_categorical(labels, NUM_CLASSES)

# this is your task, implement the method get_model() in the models.py file
model = get_model()

run_name = 'cancer-{:%d-%b_%H-%M-%S}'.format(datetime.now())
log_dir = path.join('cancer', 'logs', run_name)
print('logging to "{}"'.format(log_dir))
tb_callback = callbacks.TensorBoard(log_dir=log_dir)
model.fit(features, labels, batch_size=64, epochs=100, validation_split=.2, callbacks=[tb_callback])
