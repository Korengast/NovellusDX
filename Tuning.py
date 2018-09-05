import numpy as np
from models.BayramogluetAl2015Model import BayramogluetAl2015Model
from matplotlib import pyplot as plt
plt.style.use('ggplot')

EPOCHS = 100
BATCH_SIZE = 30

RES_AUG = '192_rotate'  # '1920_no', '384_flip', '192_rotate'
X_train = np.load('array_data/' + RES_AUG + '_X_train.npy')
X_valid = np.load('array_data/' + RES_AUG + '_X_valid.npy')
Y_train = np.load('array_data/' + RES_AUG + '_Y_train.npy')
Y_valid = np.load('array_data/' + RES_AUG + '_Y_valid.npy')

model = BayramogluetAl2015Model(X_train[0].shape)


model.compile()
model.model.summary()
fit_hist = model.fit(X_train, Y_train, EPOCHS, BATCH_SIZE, validation_data=(X_valid, Y_valid))


plt.plot(fit_hist.history['acc'])
plt.plot(fit_hist.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

