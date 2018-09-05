import numpy as np
from models.FCModel import FCModel
from models.GaoetAl2017Model import GaoetAl2017Model
from models.BayramogluetAl2015Model import BayramogluetAl2015Model
from models.transfer_learning_resnet50 import TlResnet50
from models.transfer_learning_vgg16 import TlVGG16

EPOCHS = 50
BATCH_SIZE = 30

RES_AUG = '192_rotate'  # '1920_no', '384_flip', '192_rotate'
X_train = np.load('array_data/' + RES_AUG + '_X_train.npy')
X_valid = np.load('array_data/' + RES_AUG + '_X_valid.npy')
Y_train = np.load('array_data/' + RES_AUG + '_Y_train.npy')
Y_valid = np.load('array_data/' + RES_AUG + '_Y_valid.npy')

# RES = '1920'
# X_test = np.load('array_data/' + RES + '_X_test.npy')
# Y_test = np.load('array_data/' + RES + '_Y_test.npy')

# model = FCModel(X_train[0].shape, layers_num=3)
model = BayramogluetAl2015Model(X_train[0].shape)
# model = GaoetAl2017Model(X_train[0].shape)
# model = TlResnet50(X_train[0].shape)
# model = TlVGG16(X_train[0].shape)


model.compile()
model.model.summary()
# model.fit(X_train, Y_train, EPOCHS, BATCH_SIZE)
# model.save_model_weights('weights')
# model.load_model_weights('weights')
acc_train = model.evaluate(X_train, Y_train)[1]
print()
print("Train Accuracy = " + str(acc_train))
acc_valid = model.evaluate(X_valid, Y_valid)[1]
print("Valid Accuracy = " + str(acc_valid))
