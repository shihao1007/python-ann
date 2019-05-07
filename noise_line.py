# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 15:21:52 2019

Add noise to the far field LINE simulation

Editor:
    Shihao Ran
    STIM Laboratory
"""

import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
import scipy as sp
import scipy.special
import math
import sys


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
    
    amp = perc*0.25
    # Gaussian noise in frequency domain
    eta_r = np.random.normal(0, amp, S.shape)
    eta_i = np.random.normal(0, amp, S.shape)
    
    S_noise = S + eta_r + 1j*eta_i
   
    return S_noise
#%%         

# center obscuration of the objective when calculating bandpass filter
low = 0.0
perc_low = 0
# numerical aperture of the objective
perc_high = 1

# number of different numerical apertures to be tested
nb_noise = 40

# allocate a list of the NA
perc_list = np.linspace(perc_low, perc_high, nb_noise)

# pre load y train and y test
X_test = np.load(r'D:\irimages\irholography\CNN\ANN_noisy_data\data\X_test.npy')
X_test_absolute = np.load(r'D:\irimages\irholography\CNN\ANN_noisy_data\data\X_test_absolute.npy')
y_test = np.load(r'D:\irimages\irholography\CNN\ANN_noisy_data\data\y_test.npy')

# pre load intensity and complex CNNs
complex_ANN = load_model(r'D:\irimages\irholography\CNN\ANN_noisy_data\model\complex_model_30nodes')
intensity_ANN = load_model(r'D:\irimages\irholography\CNN\ANN_noisy_data\model\absolute_model_30nodes')

# parent directory of the data set
#data_dir = r'D:\irimages\irholography\CNN\ANN_v2_far_line'

# allocate space for complex and intensity accuracy
complex_error = np.zeros((nb_noise, 3), dtype = np.float64)
intensity_error = np.zeros((nb_noise, 3), dtype = np.float64)

cnt = 0
for perc_idx in range(nb_noise):
    
    perc = perc_list[perc_idx]
    
    X_test_noise = perc_noise(X_test, perc)
    X_test_intensity_noise = np.abs(X_test_noise[...,0] + 1j*X_test_noise[...,1])
    X_test_intensity_noise = X_test_intensity_noise[...,None]
    # handle complex model first
    complex_error[perc_idx, :] = calculate_error(X_test_noise, option = 'complex')
    
    # handle intensity model second
    intensity_error[perc_idx, :] = calculate_error(X_test_intensity_noise, option = 'intensity')
    
    # print progress
    cnt += 1
    sys.stdout.write('\r' + str(round(cnt / nb_noise * 100, 2))  + ' %')
    sys.stdout.flush() # important
    
# save the error file
np.save(r'D:\irimages\irholography\CNN\ANN_noisy_data\result\complex_error_noise', complex_error)
np.save(r'D:\irimages\irholography\CNN\ANN_noisy_data\result\absolute_error_noise', intensity_error)

#%%

complex_error = np.load(r'D:\irimages\irholography\CNN\ANN_noisy_data\result\complex_error_noise.npy')
intensity_error = np.load(r'D:\irimages\irholography\CNN\ANN_noisy_data\result\absolute_error_noise.npy')

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

##%%
#plt.figure()
#plt.subplot(311)
#plt.plot(NA_list[2:], complex_error[2:, 0], label = 'Complex ANN')
#plt.plot(NA_list[2:], intensity_error[2:, 0], label = 'Intensity ANN')
#plt.xlabel('NA')
#plt.ylabel('Relative Error (Refractive Index)')
#plt.legend()
#
#plt.subplot(312)
#plt.plot(NA_list[2:], complex_error[2:, 1], label = 'Complex ANN')
#plt.plot(NA_list[2:], intensity_error[2:, 1], label = 'Intensity ANN')
#plt.xlabel('NA')
#plt.ylabel('Relative Error (Attenuation Coefficient)')
#plt.legend()
#
#plt.subplot(313)
#plt.plot(NA_list[2:], complex_error[2:, 2], label = 'Complex ANN')
#plt.plot(NA_list[2:], intensity_error[2:, 2], label = 'Intensity ANN')
#plt.xlabel('NA')
#plt.ylabel('Relative Error (Sphere Radius)')
#plt.legend()

