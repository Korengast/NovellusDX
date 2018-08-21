from keras import optimizers
class KerasModel(object):

    def __init__(self, input_shape):
        self.model = self.build(input_shape)

    def build(self, input_shape):
        pass

    def compile(self):
        adam = optimizers.adam()
        self.model.compile(optimizer=adam, loss="binary_crossentropy", metrics=["accuracy"])

    def fit(self, x, y, epochs=10, batch_size=15):
        self.model.fit(x=x, y=y, epochs=epochs, batch_size=batch_size)

    def evaluate(self, x, y):
        loss_acc = self.model.evaluate(x=x, y=y)
        return loss_acc

    def predict(self, x):
        preds = self.model.predict(x=x)
        return preds

    def save_model_weights(self, name):
        self.model.save_weights('Weights/'+name)

    def load_model_weights(self, name):
        self.model.load_weights('Weights/'+name, by_name=False)

