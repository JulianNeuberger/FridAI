from boston.config import FEATURE_DIMENSIONALITY
from keras.engine import Model
from keras.layers import Input, Dense
from keras.optimizers import Adam


def get_model() -> Model:
    # here starts your task!
    # implement an ANN that solves the boston house prices task.
    # solving means we want to approximate an unknown function which,
    # given some features like house size, location etc. predicts it's price

    # there are some useful imports already, check the imports from config and the
    # different layers from keras.layers!

    # be sure to delete this line after you implemented your ANN and want to run it
    raise NotImplementedError('You have to implement {}.get_model'.format(__file__))

    # create a new model by specifying input/output layer(s)
    model = Model(inputs=[], outputs=[])
    # I already chose optimizer and loss function, you won't need to teak them (but you can of course!)
    optimizer = Adam()
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model
