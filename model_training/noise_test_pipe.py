# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 16:19:51 2019

This script runs a pipeline including:
    training the model
    test the model
    plot the performance
    test the model on synthetic noisy data
    plot the noise sensitivity

Editor:
    Shihao Ran
    STIM Laboratory
"""


# import necessary packages
import numpy as np
from matplotlib import pyplot as plt
import sys

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.callbacks import ModelCheckpoint
from keras.initializers import glorot_normal

def calculate_error(imdata, option = 'complex'):
    """
    make a prediction based on the input data set
    calculate the relative error between the prediction and the testing ground truth
    if the input data is intensity images, set the channel number to 1
    otherwise it is complex images, set the channel number to 2
    
    params:
        imdata: the testing set
        option: the type of the testing
    
    return:
        y_off_perc: relative RMSE
    
    """
    # use different models to test depend on the data set type
    if option == 'intensity':
        y_pred = intensity_ANN.predict(imdata)
    else:
        y_pred = complex_ANN.predict(imdata)
    
    y_pred[:, 1] /= 100
    
    # calculate the relative error of the sum of the B vector
    y_off = np.abs(y_test - y_pred)
    
    y_off_perc = np.average(y_off / [2.0, 0.05, 2.0], axis = 0) * 100
    
    return y_off_perc


def band_noise(S, low, high):
    # S is the input signal
    # amp is the amplitude of the noise or the standard deviation of the Gaussian noise
    
    S_noise = np.zeros(S.shape)
    # for each image in the data set
    for i in range(S.shape[0]):
        # for each channel in the image
        # real and imaginary part
        for j in range(S.shape[-1]):
            amp = np.std(S[i, :, j])*2
            # Gaussian noise in frequency domain
            eta_f = np.random.normal(0, amp, S.shape[1])
            
            # calculate the number of values in the signal
            N = eta_f.shape[0]
            
            # calculate the factor indices
            lfi = N * low / 2
            hfi = N * high / 2
            
            lf0 = int(lfi)
            lf1 = N-int(lfi)
            
            hf0 = int(hfi)
            hf1 = N - int(hfi)
            
            # apply the bandpass filter
            eta_f[0:lf0] = 0
            eta_f[hf0:hf1] = 0
            eta_f[lf1:N] = 0
            
            # inverse FFT to generate noise in the real domain
            eta_if = np.fft.ifft(eta_f)
            S_noise[i, :, j] = S[i, :, j] + np.real(eta_if)
   
    return S_noise

def perc_noise(S, perc):
    # S is the input signal
    # amp is the amplitude of the noise or the standard deviation of the Gaussian noise
    
    S_noise = np.zeros(S.shape)
    # for each image in the data set
    for i in range(S.shape[0]):
        # for each channel in the image
        # real and imaginary part
        for j in range(S.shape[-1]):
            amp = perc*0.25
            # Gaussian noise in frequency domain
            eta_f = np.random.normal(0, amp, S.shape[1])
            
            S_noise[i, :, j] = S[i, :, j] + eta_f
   
    return S_noise

def square_noise(S, perc):
    
    # S is the input signal
    # amp is the amplitude of the noise or the standard deviation of the Gaussian noise
    
    S_noise = np.zeros(S.shape)
    # for each image in the data set
    for i in range(S.shape[0]):
        # for each channel in the image
        # real and imaginary part
        for j in range(S.shape[-1]):
            amp = perc*0.25
            # Gaussian noise in frequency domain
            eta_f = np.random.normal(0, amp, S.shape[1])
            
            S_noise[i, :, j] = S[i, :, j] + eta_f**2 * 2
   
    return S_noise

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
# total number of samples
num_total_sample = num**3
# number of nodes in the hidden layer
num_nodes = 10      
# number of channel of the input image
# if the input is complex images
# 	channel = 2 for real and imaginary part
# if the input is intensity images
# 	channel = 1 for real part only
# preload the training and testing set and the label set
X_train_intensity = np.load(r'D:\irimages\irholography\CNN\ANN_simple-real\data\X_train_intensity.npy')
X_test_intensity = np.load(r'D:\irimages\irholography\CNN\ANN_simple-real\data\X_test_intensity.npy')

X_train = np.load(r'D:\irimages\irholography\CNN\ANN_simple-real\data\X_train.npy')
X_test = np.load(r'D:\irimages\irholography\CNN\ANN_simple-real\data\X_test.npy')

y_train = np.load(r'D:\irimages\irholography\CNN\ANN_simple-real\data\y_train.npy')
y_test = np.load(r'D:\irimages\irholography\CNN\ANN_simple-real\data\y_test.npy')

#%%
# up scale the training labels
y_train[:,1] *= 100

#%%
# initialize regressor
real_model = Sequential()
# add the first hidden layer
real_model.add(Dense(num_nodes, input_shape=(line_length,1,),
                     kernel_initializer=glorot_normal(seed=25),
                     activation='relu'))

real_model.add(Dense(num_nodes, kernel_initializer=glorot_normal(seed=25)))
real_model.add(Dense(num_nodes, kernel_initializer=glorot_normal(seed=25)))
# add a Flatten layer to reduce the dimention of the feature map to 1-D
real_model.add(Flatten())
# add the output layer with 3 units
real_model.add(Dense(3, kernel_initializer=glorot_normal(seed=25)))
# compile the regressor with 'adam' optimizer
real_model.compile('adam', loss = 'mean_squared_error')
# specify the path of the trained model
model_path_real = r'D:\irimages\irholography\CNN\ANN_simple-real\model\intensity_model_'+str(num_nodes)+'nodes_sqrt_3layers'

# early_stopping_monitor = EarlyStopping(patience=4)
model_check1 = ModelCheckpoint(model_path_real, save_best_only=True)

# fit the model with training and testing set
history1 = real_model.fit(x = X_train_intensity, y = y_train, batch_size = 135,
                        epochs = 400,
                        validation_split = 0.2,
                        callbacks=[model_check1])

#%%
# initialize regressor
complex_model = Sequential()
# add the first hidden layer
complex_model.add(Dense(num_nodes, input_shape=(line_length,2,),
                        kernel_initializer=glorot_normal(seed=25),
                        activation='relu'))
complex_model.add(Dense(num_nodes, kernel_initializer=glorot_normal(seed=25)))
complex_model.add(Dense(num_nodes, kernel_initializer=glorot_normal(seed=25)))
# add a Flatten layer to reduce the dimention of the feature map to 1-D
complex_model.add(Flatten())
# add the output layer with 3 units
complex_model.add(Dense(3, kernel_initializer=glorot_normal(seed=25)))
# compile the regressor with 'adam' optimizer
complex_model.compile('adam', loss = 'mean_squared_error')
# specify the path of the trained model
model_path_complex = r'D:\irimages\irholography\CNN\ANN_simple-real\model\complex_model_'+str(num_nodes)+'nodes_sqrt_3layers'

# early_stopping_monitor = EarlyStopping(patience=4)
model_check2 = ModelCheckpoint(model_path_complex, save_best_only=True)

# fit the model with training and testing set
history2 = complex_model.fit(x = X_train, y = y_train, batch_size = 135,
                        epochs = 400,
                        validation_split = 0.2,
                        callbacks=[model_check2])

#%%
# get the prediction from the network

# get the prediction
y_pred = real_model.predict(X_test_intensity)

# down scale it
y_pred[:, 1] /= 100

# calculate the relative RMSE
y_off = np.abs(y_pred - y_test)
y_off_perc = np.mean(y_off/[2.0, 0.05, 2.0], axis = 0) * 100


# print the calculated relative RMSE
print('Absolute Model:')
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
plt.suptitle('Absolute ' +str(num_nodes)+' nodes')

# plot training history
plt.figure()
#plt.plot(history.history['loss'])
plt.plot(history1.history['val_loss'])
plt.title('Absolute Model Loss '+str(num_nodes)+' nodes')
plt.ylabel('loss')
plt.xlabel('epoch')
#plt.legend(['train', 'test'], loc='upper left')
plt.show()

#%%
# get the prediction from the network
y_pred = complex_model.predict(X_test)

# down scale it
y_pred[:, 1] /= 100

# calculate the relative RMSE
y_off = np.abs(y_pred - y_test)
y_off_perc = np.mean(y_off/[2.0, 0.05, 2.0], axis = 0) * 100


# print the calculated relative RMSE
print('Complex Model:')
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
plt.suptitle('Complex '+str(num_nodes)+' nodes')

# plot training history
plt.figure()
#plt.plot(history.history['loss'])
plt.plot(history2.history['val_loss'])
plt.title('Complex Model Loss '+str(num_nodes)+' nodes')
plt.ylabel('loss')
plt.xlabel('epoch')
#plt.legend(['train', 'test'], loc='upper left')
plt.show()


#%%
low = 0.0
perc_low = 0
perc_high = 1

# number of different numerical apertures to be tested
nb_noise = 30

# allocate a list of the NA
perc_list = np.linspace(perc_low, perc_high, nb_noise)

# pre load y train and y test
X_test = np.load(r'D:\irimages\irholography\CNN\ANN_simple-real\data\X_test.npy')
X_test_intensity = np.load(r'D:\irimages\irholography\CNN\ANN_simple-real\data\X_test_intensity.npy')
y_test = np.load(r'D:\irimages\irholography\CNN\ANN_simple-real\data\y_test.npy')

# pre load intensity and complex CNNs
complex_ANN = complex_model
intensity_ANN = real_model

# parent directory of the data set
#data_dir = r'D:\irimages\irholography\CNN\ANN_v2_far_line'

# allocate space for complex and intensity accuracy
complex_error = np.zeros((nb_noise, 3), dtype = np.float64)
intensity_error = np.zeros((nb_noise, 3), dtype = np.float64)

cnt = 0
for perc_idx in range(nb_noise):
    
    perc = perc_list[perc_idx]
    
    X_test_noise = perc_noise(X_test, perc)
    X_test_intensity_noise = perc_noise(X_test_intensity, np.sqrt(2)*perc)
    
    # handle complex model first
    complex_error[perc_idx, :] = calculate_error(X_test_noise, option = 'complex')
    
    # handle intensity model second
    intensity_error[perc_idx, :] = calculate_error(X_test_intensity_noise, option = 'intensity')
    
    # print progress
    cnt += 1
    sys.stdout.write('\r' + str(round(cnt / nb_noise * 100, 2))  + ' %')
    sys.stdout.flush() # important
    
# save the error file
np.save(r'D:\irimages\irholography\CNN\ANN_simple-real\results\complex_error_noise_'+str(num_nodes)+'nodes_sqrt_3layers', complex_error)
np.save(r'D:\irimages\irholography\CNN\ANN_simple-real\results\intensity_error_noise_'+str(num_nodes)+'nodes_sqrt_3layers', intensity_error)

#%%
# plot out the error
plt.figure()
plt.subplot(311)
plt.plot(perc_list, complex_error[:, 0], label = 'Complex ANN')
plt.plot(perc_list, intensity_error[:, 0], label = 'Absolute ANN')
plt.xlabel('Noise Percentage')
plt.ylabel('Relative Error (Refractive Index)')
plt.legend()

plt.subplot(312)
plt.plot(perc_list, complex_error[:, 1], label = 'Complex ANN')
plt.plot(perc_list, intensity_error[:, 1], label = 'Absolute ANN')
plt.xlabel('Noise Percentage')
plt.ylabel('Relative Error (Attenuation Coefficient)')
plt.legend()

plt.subplot(313)
plt.plot(perc_list, complex_error[:, 2], label = 'Complex ANN')
plt.plot(perc_list, intensity_error[:, 2], label = 'Absolute ANN')
plt.xlabel('Noise Percentage')
plt.ylabel('Relative Error (Sphere Radius)')
plt.legend()
plt.suptitle('Noise Test with '+str(num_nodes)+' nodes')
