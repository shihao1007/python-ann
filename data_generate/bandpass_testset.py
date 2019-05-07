# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 10:54:20 2019

This script generates the new testing data for bandpass filtering

Editor:
    Shihao Ran
    STIM Laboratory
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import scipy.special
import math
import sys
import chis.MieScattering as ms


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


#%%
# set the size and resolution of both planes
fov = 32                    # field of view
res = 256                   # resolution

lambDa = 1                  # wavelength

k = 2 * math.pi / lambDa    # wavenumber
padding = 2                 # padding

simRes, simFov = ms.pad(res, fov, padding)
working_dis = ms.get_working_dis(padding)
scale_factor = ms.get_scale_factor(res, fov, working_dis)

line_size = int(simRes/2)

ps = [0, 0, 0]              # position of the sphere
k_dir = [0, 0, -1]          # propagation direction of the plane wave
E = [1, 0, 0]               # electric field vector


# features
a_min = 1.0
a_max = 2.0

nr_min = 1.1
nr_max = 2.0

ni_min = 0.01
ni_max = 0.05

# dimention of the data set
num = 10
num_bp = 100

a, nr, ni = cal_feature_space(a_min, a_max,
                              nr_min, nr_max,
                              ni_min, ni_max, num)

#get the maximal order of the integration
l = ms.get_order(a_max, lambDa)

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

# pre load y train and y test
#y_train = np.load(r'D:\irimages\irholography\CNN\data_v9_far_field\split_data\train\y_train.npy')
y_test = np.load(r'D:\irimages\irholography\CNN\\ANN_more_bandpass_HD\data\y_test.npy')

# down sample the testing set
ds_ratio = 50
y_test_ds = y_test[::ds_ratio]
num_test_ds = int(num_test/ds_ratio)

# parent directory of the data set
data_dir = r'D:\irimages\irholography\CNN\\ANN_more_bandpass_HD\bandpass_test'

scatter = ms.scatter_matrix(simRes, simFov, working_dis,
                            a_max, lambDa, k_dir, 1, 'far')

for NA_idx in range(nb_NA):
    
    NA = NA_list[NA_idx]
    
    X_test = np.zeros((num_test_ds, line_size, 2))
    X_test_absolute = np.zeros((num_test_ds, line_size, 1))
    
    print('Banbpassing the ' + str(NA_idx + 1) + 'th filter \n')
    cnt = 0
    for idx in range(num_test_ds):
        
        n = y_test_ds[idx, 0] + 1j*y_test_ds[idx, 1]
        a = y_test_ds[idx, 2]
    
        B = ms.coeff_b(l, k, n, a)
        
        # integrate through all the orders to get the farfield in the Fourier Domain
        E_scatter_fft = np.sum(scatter * B, axis = -1) * scale_factor
        
        bpf_line = ms.bandpass_filter(simRes, simFov, 0, NA, dimension=1)

        # convert back to spatial domain
        E_near_line, E_near_x = ms.idhf(simRes, simFov, E_scatter_fft*bpf_line)
        
        E_near_line = (E_near_line-np.mean(E_near_line))/np.std(E_near_line)
        
        # shift the scattering field in the spacial domain for visualization
        Et = E_near_line + 1
        
        X_test[idx,:,0] = np.real(Et)
        X_test[idx,:,1] = np.imag(Et)
        
        X_test_absolute[idx,:,0] = np.abs(Et)
        
        # print progress
        cnt += 1
        sys.stdout.write('\r' + str(round(cnt / num_test_ds * 100, 2))  + ' %')
        sys.stdout.flush() # important
    

    # save the bandpassed
    np.save(data_dir + '\\' + 'X_test_'+str(NA_idx), X_test)
    np.save(data_dir + '\\' + 'X_test_abs_'+str(NA_idx), X_test_absolute)

