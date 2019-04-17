# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 13:50:22 2019

A simple Neural Net work trained on 1D Mie scattering simulations


Editor:
    Shihao Ran
    STIM Laboratory
"""


import numpy as np
from matplotlib import pyplot as plt

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.callbacks import EarlyStopping
from keras.layers import Dropout
from keras.callbacks import ModelCheckpoint
from keras.models import load_model

res = 128
padding=2
simRes = res*(padding*2+1)
line_length = int(simRes/2)
num = 30
num_total_sample = num**3
channel = 1

regressor = Sequential()

regressor.add(Dense(640, input_shape=(line_length,channel,), activation='relu'))

regressor.add(Dropout(0.1))

regressor.add(Dense(320, activation='relu'))

regressor.add(Flatten())

regressor.add(Dense(160, activation='relu'))

regressor.add(Dense(3))

regressor.compile('adam', loss = 'mean_squared_error')

X_train = np.load(r'D:\irimages\irholography\CNN\data_v12_far_line\split_data\X_train_intensity.npy')
X_test = np.load(r'D:\irimages\irholography\CNN\data_v12_far_line\split_data\X_test_intensity.npy')
y_train = np.load(r'D:\irimages\irholography\CNN\data_v12_far_line\split_data\y_train.npy')
y_test = np.load(r'D:\irimages\irholography\CNN\data_v12_far_line\split_data\y_test.npy')

y_train[:,1] *= 100

model_path = r'D:\irimages\irholography\CNN\ANN_v2_far_line\intensity_model_dropout'
#early_stopping_monitor = EarlyStopping(patience=4)
model_check = ModelCheckpoint(model_path, save_best_only=True)
#%%
history = regressor.fit(x = X_train, y = y_train, batch_size = 100,
                        epochs = 35,
                        validation_split = 0.2,
                        callbacks=[model_check])

#%%
# get the prediction from the network
regressor = load_model(r'D:\irimages\irholography\CNN\ANN_v2_far_line\intensity_model_dropout')

y_pred = regressor.predict(X_test)

# down scale it
y_pred[:, 1] /= 100

y_off = np.abs(y_pred - y_test)
y_off_perc = np.mean(y_off/[2.0, 0.05, 2.0], axis = 0) * 100

print('Current Model:')
print('Refractive Index (Real) Error: ' + str(y_off_perc[0]) + ' %')
print('Refractive Index (Imaginary) Error: ' + str(y_off_perc[1]) + ' %')
print('Redius of the Sphere Error: ' + str(y_off_perc[2]) + ' %')

# plot the prediction and ground truth
plt.figure()
plt.subplot(3,1,1)
plt.plot(y_test[::32,0], label = 'Ground Truth')
plt.plot(y_pred[::32,0], linestyle='dashed', label = 'Prediction')
plt.legend()
plt.title('Real Part')

plt.subplot(3,1,2)
plt.plot(y_test[::32,1], label = 'Ground Truth')
plt.plot(y_pred[::32,1], linestyle='dashed', label = 'Prediction')
plt.legend()
plt.title('Imaginary Part')

plt.subplot(3,1,3)
plt.plot(y_test[::32,2], label = 'Ground Truth')
plt.plot(y_pred[::32,2], linestyle='dashed', label = 'Prediction')
plt.legend()
plt.title('Radius')

# plot training history
plt.figure()
#plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
#plt.legend(['train', 'test'], loc='upper left')
plt.show()

