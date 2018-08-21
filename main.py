import numpy as np
from models.GaoetAl2017Model import GaoetAl2017Model
from models.BayramogluetAl2017Model import BayramogluetAl2017Model
from models.FCModel import FCModel
from models.transfer_learning_resnet50 import TlResnet50

EPOCHS = 100
BATCH_SIZE = 15

RES_AUG = '384_flip'  # '1920_no', '384_flip', '192_rotate'
X_train = np.load('array_data/' + RES_AUG + '_X_train.npy')
X_valid = np.load('array_data/' + RES_AUG + '_X_valid.npy')
Y_train = np.load('array_data/' + RES_AUG + '_Y_train.npy')
Y_valid = np.load('array_data/' + RES_AUG + '_Y_valid.npy')

RES = '384'
X_test = np.load('array_data/' + RES + '_X_test.npy')
Y_test = np.load('array_data/' + RES + '_Y_test.npy')

model = FCModel(X_train[0].shape)
# model = TlResnet50(X_train[0].shape)
# model = GaoetAl2017Model(X_train[0].shape)
# model = BayramogluetAl2017Model(X_train[0].shape)

model.compile()
model.model.summary()
model.fit(X_train, Y_train, EPOCHS, BATCH_SIZE)
model.save_model_weights('weigths')
model.load_model_weights('weigths')
acc_train = model.evaluate(X_train, Y_train)[1]
acc_valid = model.evaluate(X_valid, Y_valid)[1]

print()
print("Train Accuracy = " + str(acc_train))
print("Valid Accuracy = " + str(acc_valid))
