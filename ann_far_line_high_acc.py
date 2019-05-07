# -*- coding: utf-8 -*-
"""
Created on Sun May  5 18:47:55 2019

Editor:
    Shihao Ran
    STIM Laboratory
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 13:50:22 2019

A simple Neural Net work trained on 1D Mie scattering simulations


Editor:
    Shihao Ran
    STIM Laboratory
"""

# import necessary packages
import numpy as np
from matplotlib import pyplot as plt

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.callbacks import EarlyStopping
from keras.layers import Dropout
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras.initializers import glorot_normal

#%%
# specify the related training parameters

# resolution of the simulation BEFORE padding
res = 128
# padding of the simulation
padding=2
# simulated resolution
simRes = res*(padding*2+1)
# length of the 1-D simulation
line_length = int(simRes/2)
# feature size for each feature
num = 30
num_bp = 100
num_nodes = 5
# total number of samples
num_total_sample = num**3
# number of channel of the input image
# if the input is complex images
# 	channel = 2 for real and imaginary part
# if the input is intensity images
# 	channel = 1 for real part only
# please specify channel and data set MANUALLY
channel = 2

#%%
# initialize regressor
regressor = Sequential()
# add the first hidden layer
regressor.add(Dense(640, input_shape=(line_length,channel,), activation='relu'))
regressor.add(Dense(320))
# add a Flatten layer to reduce the dimention of the feature map to 1-D
regressor.add(Flatten())
# add the output layer with 3 units
regressor.add(Dense(160))
# add the output layer with 3 units
regressor.add(Dense(3))
# compile the regressor with 'adam' optimizer
regressor.compile('adam', loss = 'mean_squared_error')
print(regressor.summary())
#%%

# preload the training and testing set and the label set
X_train = np.load(r'D:\irimages\irholography\CNN\ANN_simple_test\data\X_train.npy')
X_test = np.load(r'D:\irimages\irholography\CNN\ANN_simple_test\data\X_test.npy')
y_train = np.load(r'D:\irimages\irholography\CNN\ANN_simple_test\data\y_train.npy')
y_test = np.load(r'D:\irimages\irholography\CNN\ANN_simple_test\data\y_test.npy')

# up scale the training labels
y_train[:,1] *= 100

# specify the path of the trained model
model_path = r'D:\irimages\irholography\CNN\ANN_simple_test\model\beta_complex_model_'+str(num_nodes)+'nodes'
# early_stopping_monitor = EarlyStopping(patience=4)
model_check = ModelCheckpoint(model_path, save_best_only=True)
#%%
# fit the model with training and testing set
history = regressor.fit(x = X_train, y = y_train, batch_size = 200,
                        epochs = 200,
                        validation_split = 0.2,
                        callbacks=[model_check])

#%%
# get the prediction
y_pred = regressor.predict(X_test)

# down scale it
y_pred[:, 1] /= 100

# calculate the relative RMSE
y_off = np.abs(y_pred - y_test)
y_off_perc = np.mean(y_off/[2.0, 0.05, 2.0], axis = 0) * 100


# print the calculated relative RMSE
print('Current Model:')
print('Refractive Index (Real) Error: ' + str(y_off_perc[0]) + ' %')
print('Refractive Index (Imaginary) Error: ' + str(y_off_perc[1]) + ' %')
print('Redius of the Sphere Error: ' + str(y_off_perc[2]) + ' %')


#%%
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

#%%
print(regressor.summary())