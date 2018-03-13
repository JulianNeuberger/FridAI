from sklearn import datasets
from keras import layers, engine, callbacks, utils
from datetime import datetime
from os import path

# there are 10 digits
num_digits = 10
num_features = 64
max_feature = 16

# load prepared data set containing 1797 digits as 8x8 images
digit_features, digit_classes = datasets.load_digits(n_class=num_digits, return_X_y=True)

# normalize features, see documentation of sklearn.datasets.load_digits!
digit_features /= max_feature

# we need so called "one-hot" vectors
digit_classes = utils.to_categorical(digit_classes, num_classes=num_digits)

# we need a layer that acts as input.
# shape of that input has to be known and depends on data.
input_layer = layers.Input(shape=(num_features,))

# hidden layers are the model's power to fit data.
# number of neurons and type of layers are crucial.
# idea behind decreasing number of units per layer:
# increase the "abstraction" in each layer...
hidden_layer = layers.Dense(units=256)(input_layer)
hidden_layer = layers.Dense(units=512)(hidden_layer)
hidden_layer = layers.Dense(units=512)(hidden_layer)
hidden_layer = layers.Dense(units=512)(hidden_layer)
hidden_layer = layers.Dense(units=512)(hidden_layer)

# last layer represents output.
# activation of each neuron corresponds to the models decision of
# choosing that class.
# softmax ensures that all activations summed up are equal to 1.
# this lets one interpret that output as a probability
output_layer = layers.Dense(units=num_digits, activation='softmax')(hidden_layer)

# a callback to log loss/accuracy etc. for tensorboard to visualize
run_name = 'run-{:%d-%b_%H-%M-%S}'.format(datetime.now())
tb_callback = callbacks.TensorBoard(log_dir=path.join('logs', run_name))

# actual creation of the model with in- and output layers
model = engine.Model(inputs=[input_layer], outputs=[output_layer])
# transform into a trainable model by specifying the optimizing function
# (here stochastic gradient descent),
# as well as the loss (eg. how big of an error is produced by the model)
# track the model's accuracy as an additional metric (only possible for classification)
model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

# prints a human readable summary of the model to the outstream
model.summary()

# training the model
model.fit(digit_features, digit_classes, batch_size=32, epochs=100, validation_split=.2, callbacks=[tb_callback])
