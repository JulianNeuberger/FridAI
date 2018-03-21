from config import NUM_CLASSES, FEATURE_DIMENSIONALITY
from keras.engine import Model
from keras.layers import Input, Dense
from keras.optimizers import Adam


def get_model() -> Model:
    # here starts your task!
    # implement an ANN that solves the iris task
    # (solving means it is better than guessing, which is correct 33% of the time)
    raise NotImplementedError('You have to implement {}.get_model'.format(__file__))

    # create a new model by specifying input/output layer(s)
    model = Model(inputs=[], outputs=[])
    # I already chose optimizer and loss function, you won't need to teak them (but you can of course!)
    optimizer = Adam()
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model
