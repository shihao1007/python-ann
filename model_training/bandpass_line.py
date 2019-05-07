# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 09:47:43 2019

This script loads in the data already been bandpassed
And evaluate the models to the new test set
then plot the result

Editor:
    Shihao Ran
    STIM Laboratory
"""

import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
import sys
#%%
def cal_feature_space(a_min, a_max,
                      nr_min, nr_max,
                      ni_min, ni_max, num):
    """
    get the feature space of the bandpass sensitivity test by setting
    the range of the features
    
    params:
        a_max: the maximal of the radius of the spheres
        a_min: the minimal of the radius of the spheres
        nr_max: the maximal of the refractive indices
        nr_min: the minimal of the refractive indices
        ni_max: the maximal of the attenuation coefficients
        ni_min: the minimal of the attenuation coefficients
        num: dimention of each feature
        
    return:
        a: list containing the radius
        nr: list containing the refractive index
        ni: list containing the attenuation coefficient
    """

    a = np.linspace(a_min, a_max, num)
    nr = np.linspace(nr_min, nr_max, num)
    ni = np.linspace(ni_min, ni_max, num)
    
    return a, nr, ni

def calculate_error(imdata, y_test, option = 'complex'):
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
#%%

# dimention of the data set
num = 10
num_bp = 100

# center obscuration of the objective when calculating bandpass filter
NA_in = 0.0
# numerical aperture of the objective
NA_out = 1.01

# number of different numerical apertures to be tested
nb_NA = 100

# allocate a list of the NA
NA_list = np.linspace(0.02, NA_out, nb_NA)


# total number of images in the data set
num_samples = num ** 3 * num_bp

test_size = 0.1
num_test = int(num_samples * test_size)
num_test_in_group = int(num_test / num)

num_nodes = 5

# pre load y train and y test
#y_train = np.load(r'D:\irimages\irholography\CNN\data_v9_far_field\split_data\train\y_train.npy')
y_test_raw = np.load(r'D:\irimages\irholography\CNN\\ANN_more_bandpass_HD\data\y_test.npy')

# down sample y_test here also
ds_ratio = 50
y_test_ds = y_test_raw[::ds_ratio]
num_test_ds = int(num_test/ds_ratio)

# pre load intensity and complex CNNs
complex_ANN = load_model(r'D:\irimages\irholography\CNN\\ANN_more_bandpass_HD\model\complex_model_'+str(num_nodes)+'nodes')
intensity_ANN = load_model(r'D:\irimages\irholography\CNN\\ANN_more_bandpass_HD\model\absolute_model_'+str(num_nodes)+'nodes')

# parent directory of the data set
data_dir = r'D:\irimages\irholography\CNN\\ANN_more_bandpass_HD\bandpass_test'

# allocate space for complex and intensity accuracy
complex_error = np.zeros((nb_NA, 3), dtype = np.float64)
absolute_error = np.zeros((nb_NA, 3), dtype = np.float64)

cnt = 0
for NA_idx in range(nb_NA):
    
    X_test = np.load(data_dir + '\\' + 'X_test_'+str(NA_idx)+'.npy')
    X_test_absolute = np.load(data_dir + '\\' + 'X_test_abs_'+str(NA_idx)+'.npy')
    
    # handle complex model first
    complex_error[NA_idx, :] = calculate_error(X_test, y_test_ds, option = 'complex')
    
    # handle intensity model second
    absolute_error[NA_idx, :] = calculate_error(X_test_absolute, y_test_ds, option = 'intensity')
    # print progress
    cnt += 1
    sys.stdout.write('\r' + str(round(cnt / nb_NA * 100, 2))  + ' %')
    sys.stdout.flush() # important
    
# save the error file
np.save(r'D:\irimages\irholography\CNN\ANN_more_bandpass_HD\result\complex_error_bp'+str(num_nodes)+'nodes', complex_error)
np.save(r'D:\irimages\irholography\CNN\ANN_more_bandpass_HD\result\absolute_error_bp'+str(num_nodes)+'nodes', absolute_error)

#%%
# plot out the error
plt.figure()
plt.subplot(131)
plt.plot(NA_list, complex_error[:, 0], label = 'Complex ANN')
plt.plot(NA_list, absolute_error[:, 0], label = 'Absolute ANN')
plt.xlabel('NA')
plt.ylabel('Relative Error (Refractive Index)')
plt.legend()

plt.subplot(132)
plt.plot(NA_list, complex_error[:, 1], label = 'Complex ANN')
plt.plot(NA_list, absolute_error[:, 1], label = 'Absolute ANN')
plt.xlabel('NA')
plt.ylabel('Relative Error (Attenuation Coefficient)')
plt.legend()

plt.subplot(133)
plt.plot(NA_list, complex_error[:, 2], label = 'Complex ANN')
plt.plot(NA_list, absolute_error[:, 2], label = 'Absolute ANN')
plt.xlabel('NA')
plt.ylabel('Relative Error (Sphere Radius)')
plt.legend()

plt.suptitle('Bandpass Error '+str(num_nodes)+' nodes')

#%%
#plt.figure()
#plt.subplot(311)
#plt.plot(NA_list[10:], complex_error[10:, 0], label = 'Complex ANN')
#plt.plot(NA_list[10:], absolute_error[10:, 0], label = 'Absolute ANN')
#plt.xlabel('NA')
#plt.ylabel('Relative Error (Refractive Index)')
#plt.legend()
#
#plt.subplot(312)
#plt.plot(NA_list[10:], complex_error[10:, 1], label = 'Complex ANN')
#plt.plot(NA_list[10:], absolute_error[10:, 1], label = 'Absolute ANN')
#plt.xlabel('NA')
#plt.ylabel('Relative Error (Attenuation Coefficient)')
#plt.legend()
#
#plt.subplot(313)
#plt.plot(NA_list[10:], complex_error[10:, 2], label = 'Complex ANN')
#plt.plot(NA_list[10:], absolute_error[10:, 2], label = 'Absolute ANN')
#plt.xlabel('NA')
#plt.ylabel('Relative Error (Sphere Radius)')
#plt.legend()
#
#plt.suptitle('Bandpass Error '+str(num_nodes)+' nodes (Partial)')