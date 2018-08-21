from models.resnet50 import ResNet50
from keras.layers import Dense, Flatten, GlobalAveragePooling2D
from keras.layers import Input
from keras.models import Model
from models.KerasModel import KerasModel


class TlResnet50(KerasModel):
    def __init__(self, input_shape):
        super(TlResnet50, self).__init__(input_shape)

    def build(self, input_shape):
        tl_model = ResNet50(weights='imagenet',include_top=False, input_shape=input_shape)
        last_layer = tl_model.output
        X = GlobalAveragePooling2D()(last_layer)
        X = Dense(1, activation='sigmoid')(X)

        model = Model(inputs=tl_model.input, outputs=X)
        for layer in model.layers[:-1]:
            layer.trainable = False
        return model
