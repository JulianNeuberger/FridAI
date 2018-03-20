from config import NUM_CLASSES, FEATURE_DIMENSIONALITY
from keras.engine import Model
from keras.layers import Input, Dense
from keras.optimizers import Adam


def get_model() -> Model:
    # here starts your task!
    # implement an ANN that solves the iris task
    # (solving means it is better than guessing, which is correct 33% of the time)
    input_layer = Input(shape=(FEATURE_DIMENSIONALITY,))
    hidden_layer = Dense(units=256, activation='relu', use_bias=True)(input_layer)
    hidden_layer = Dense(units=128, activation='relu', use_bias=True)(hidden_layer)
    hidden_layer = Dense(units=64, activation='relu', use_bias=True)(hidden_layer)
    output_layer = Dense(units=NUM_CLASSES, activation='softmax')(hidden_layer)

    # create a new model by specifying input/output layer(s)
    model = Model(inputs=[input_layer], outputs=[output_layer])
    # I already chose optimizer and loss function, you won't need to teak them (but you can of course!)
    optimizer = Adam()
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model
