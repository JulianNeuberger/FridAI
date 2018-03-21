from datetime import datetime

from boston.config import MAX_TARGET
from keras import callbacks
from boston.models import get_model
from os import path
from sklearn import datasets

features, labels = datasets.load_boston(return_X_y=True)
# normalize targets (so they are in range [0;1]
labels /= MAX_TARGET

# this is your task, implement the method get_model() in the models.py file
model = get_model()

run_name = 'boston-{:%d-%b_%H-%M-%S}'.format(datetime.now())
dir_path = path.dirname(path.realpath(__file__))
log_dir = path.join(dir_path, 'logs', run_name)
print('logging to "{}"'.format(log_dir))
tb_callback = callbacks.TensorBoard(log_dir=log_dir)
model.fit(features, labels, batch_size=32, epochs=100, validation_split=.2, callbacks=[tb_callback])
