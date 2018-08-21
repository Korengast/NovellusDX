from models.KerasModel import KerasModel
from keras.layers import Input, Flatten, Dense, Dropout
from keras.models import Model


class FCModel(KerasModel):
    def __init__(self, input_shape):
        super(FCModel, self).__init__(input_shape)

    def build(self, input_shape):
        X_input = Input(input_shape)
        X = Flatten()(X_input)
        X = Dense(128, activation='sigmoid')(X)
        X = Dropout(0.3)(X)
        X = Dense(64, activation='sigmoid')(X)
        X = Dropout(0.3)(X)
        X = Dense(1, activation='sigmoid')(X)
        model = Model(inputs=X_input, outputs=X, name='FcModel')
        return model


