import os
from config import NUM_CLASSES, FEATURE_DIMENSIONALITY
from keras.engine import Model
from keras.layers import Input, Dense
from keras.optimizers import Adam, SGD


def get_model() -> Model:
    # here starts your task!
    # implement an ANN that solves the cancer task
    # (solving means it is better than guessing, which is correct 50% of the time)
    # delete the following (single) line and replace it with your model
    #raise NotImplementedError('You have to implement {}.get_model'.format(os.path.basename(__file__)))

    input_layer = Input(shape=(FEATURE_DIMENSIONALITY,))
    hidden_layer = Dense(units=100, activation='relu')(input_layer)
    hidden_layer = Dense(units=100, activation='relu')(hidden_layer)
    hidden_layer = Dense(units=100, activation='relu')(hidden_layer)
    hidden_layer = Dense(units=100, activation='relu')(hidden_layer)
    hidden_layer = Dense(units=50, activation='relu')(hidden_layer)
    hidden_layer = Dense(units=50, activation='relu')(hidden_layer)
    hidden_layer = Dense(units=50, activation='relu')(hidden_layer)
    hidden_layer = Dense(units=50, activation='relu')(hidden_layer)
    output_layer = Dense(units=NUM_CLASSES, activation='softmax')(hidden_layer)

    # create a new model by specifying input/output layer(s)
    model = Model(inputs=[input_layer], outputs=[output_layer])
    # I already chose optimizer and loss function, you won't need to tweak them (but you can of course!)
    optimizer = Adam()
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model
