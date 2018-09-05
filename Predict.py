import numpy as np
from models.BayramogluetAl2015Model import BayramogluetAl2015Model
import shutil

EPOCHS = 50
BATCH_SIZE = 30

RES_AUG = '192_rotate'
X_train = np.load('array_data/' + RES_AUG + '_X_train.npy')
X_valid = np.load('array_data/' + RES_AUG + '_X_valid.npy')
Y_train = np.load('array_data/' + RES_AUG + '_Y_train.npy')
Y_valid = np.load('array_data/' + RES_AUG + '_Y_valid.npy')

X_train = np.concatenate((X_train, X_valid), axis=0)
Y_train = np.concatenate((Y_train, Y_valid), axis=0)
model = BayramogluetAl2015Model(X_train[0].shape)

model.compile()
model.model.summary()
model.fit(X_train, Y_train, EPOCHS, BATCH_SIZE)

RES = '192'
X_test = np.load('array_data/' + RES + '_X_test.npy')
test_file_names = np.load('array_data/' + RES + 'test_file_names.npy')

predictions = model.predict(X_test)

for tfn, pred in zip(test_file_names, predictions):
    if pred == 1:
        shutil.copy2('DL_task/test/' + tfn, 'Predictions/MT')
    else:
        shutil.copy2('DL_task/test/' + tfn, 'Predictions/WT')


