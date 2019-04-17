# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 09:47:43 2019

Bandpass filtering of the far field LINE simulation

Load in the labels and generate the field at the far plane

Limit the aperture in the fourier domain to implement bandpass filtering

Then use discrete hankel transform to get near field line

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
#%%
def coeff_b(l, k, n, a):
    
    """
    Calculate the B vector with respect to the sphere properties
    Note that B vector is independent to scatter matrix and only
    relys on n and a
    params:
        l: order of the field
        k: wavenumber of the incident field
        n: refractive index of the sphere
        a: radius of the sphere
    return:
        B: B vector
    """
    
    jka = sp.special.spherical_jn(l, k * a)
    jka_p = sp.special.spherical_jn(l, k * a, derivative=True)
    jkna = sp.special.spherical_jn(l, k * n * a)
    jkna_p = sp.special.spherical_jn(l, k * n * a, derivative=True)

    yka = sp.special.spherical_yn(l, k * a)
    yka_p = sp.special.spherical_yn(l, k * a, derivative=True)

    hka = jka + yka * 1j
    hka_p = jka_p + yka_p * 1j

    bi = jka * jkna_p * n
    ci = jkna * jka_p
    di = jkna * hka_p
    ei = hka * jkna_p * n

    # return ai * (bi - ci) / (di - ei)
    B = (bi - ci) / (di - ei)
    return B

def idhf(simFov, simRes, y):
    """
    Inverse Discrete Hankel Transform of an 1D array
    param:
        simFov: simulated field of view
        simRes: simulated resolution
        y: the 1-D matrix to be transformed
    return:
        F: the 1-D matrix after inverse Hankel Transform
        F_x: the sample index in the transformed space
    """
    
    # the range of the sample index in real space
    X = int(simFov/2)
    # number of samples
    n_s = int(simRes/2)
    
    # order of the bessel function
    order = 0
    # roots of the bessel function
    jv_root = sp.special.jn_zeros(order, n_s)
    # the max root
    jv_M = jv_root[-1]
    # the rest of the roots
    jv_m = jv_root[:-1]
    #jv_mX = jv_m/X
    
    # the sequence is "supposed" to be scaled by the roots
    # uncomment the following line to do this
    #F_term = np.interp(jv_mX, x, y)
    
    # just use the original sequence to do the transformation
    F_term = y[1:]
    # inverse DHT
    F = np.zeros(n_s, dtype=np.complex128)
    
    # the calculation is vectorized
    jv_k = jv_root[None,...]
    prefix = 2/(X**2)

    Jjj = jv_m[...,None]*jv_k/jv_M
    numerator = sp.special.jv(order, Jjj)
    denominator = sp.special.jv(order+1, jv_m[...,None])**2
    
    summation = np.sum(numerator / denominator * F_term[:-1][...,None], axis=0)
        
    F = prefix * summation
    
    F_x = jv_root*X/jv_M
    
    return F, F_x
    
#%%
def cal_line_scatter_matrix(l, k, k_dir, res, fov, working_dis, scale_factor):
    """
    Calculate the scatter matrix
    Note that the scatter matrix only depends on k, r, and fov
    Therefore it is pre-calculated to speed up the data generation
    
    params:
        l: order of the field
        k: wavenumber
        k_dir: propagation direction of the incident field
        res: resolution BEFORE padding (padding is a global variabel)
        fov: field of view BEFORE padding
        working_dis: working distance of the objective
        scale_factor: scale factor to scale up the intensity
    
    return:
        scatter_matrix
    """    
    # construct the evaluate plane    
    # simulation resolution
    # in order to do fft and ifft, expand the image use padding
    simRes = res*(2*padding + 1)
    simFov = fov*(2*padding + 1)
    center = int(simRes/2)
    # halfgrid is the size of a half grid
    halfgrid = np.ceil(simFov/2)
    # range of x, y
    gx = np.linspace(-halfgrid, +halfgrid, simRes)[:center+1]
    gy = gx[0]     
    # make it a plane at z = 0 (plus the working distance) on the Z axis
    z = working_dis
    
    # calculate the distance matrix
    rMag = np.sqrt(gx**2+gy**2+z**2)
    kMag = 2 * np.pi / lambDa
    # calculate k dot r
    kr = kMag * rMag
    
    # calculate the asymptotic form of hankel funtions
    hlkr_asym = np.zeros((kr.shape[0], l.shape[0]), dtype = np.complex128)
    for i in l:
        hlkr_asym[..., i] = np.exp(1j*(kr-i*math.pi/2))/(1j * kr)
    
    # calculate the legendre polynomial
    # get the frequency components
    fx = np.fft.fftfreq(simRes, simFov/simRes)[:center+1]
    fy = fx[0]
    
    # calculate the sum of kx ky components so we can calculate 
    # cos_theta in the Fourier Domain later
    kxky = fx ** 2 + fy ** 2
    # create a mask where the sum of kx^2 + ky^2 is 
    # bigger than 1 (where kz is not defined)
    mask = kxky > 1
    # mask out the sum
    kxky[mask] = 0
    # calculate cos theta in Fourier domain
    cos_theta = np.sqrt(1 - kxky)
    cos_theta[mask] = 0
    # calculate the Legendre Polynomial term
    pl_cos_theta = sp.special.eval_legendre(l, cos_theta[..., None])
    # mask out the light that is propagating outside of the objective
    pl_cos_theta[mask] = 0
    
    # calculate the prefix alpha term
    alpha =(2*l + 1) * 1j ** l
    
    # calculate the matrix besides B vector
    scatter_matrix = hlkr_asym * pl_cos_theta * alpha
    
    return scatter_matrix


#%%
def get_order(a, lambDa):
    """
    calculate the order of the integration based on size of the sphere 
    and the wavelength
    
    params:
        a: radius of the sphere
        lambDa: wavelength of the incident field
    
    return:
        l: order of the field as a 1-D matrix
    """

    l_max = math.ceil(2*np.pi * a / lambDa + 4 * (2 * np.pi * a / lambDa) ** (1/3) + 2)
    l = np.arange(0, l_max+1, 1)
    
    return l

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

def new_bpf(simFov, simRes, NA_in, NA_out):
    """
    get the bandpass filter based on the in and out Numerical Aperture
    basically, a bandpass filter is just a circular mask
    with inner and outer diamater specified by the in and out NA
    
    params:
        simFov: simulated field of view
        simRes: simulated resolution
        NA_in: center obscuration of the objective
        NA_out: outer back aperture of the objective
    
    return:
        bpf_test: the bandpass filter
    """
    
    # get the axis in the fourier domain
    f_x = np.fft.fftfreq(simRes, simFov/simRes)
    
    # create a meshgrid
    fx, fy = np.meshgrid(f_x, f_x)
    
    # compute the map
    fxfy = np.sqrt(fx ** 2 + fy ** 2)
    
    # initialize the filter
    bpf_test = np.zeros((simRes, simRes))
    
    # draw the filter
    mask_out = fxfy <= NA_out
    mask_in = fxfy >= NA_in
    
    mask = np.logical_and(mask_out, mask_in)
    
    bpf_test[mask] = 1
    
    return bpf_test
#%%
# set the size and resolution of both planes
fov = 16                    # field of view
res = 128                   # resolution

lambDa = 1                  # wavelength

k = 2 * math.pi / lambDa    # wavenumber
padding = 2                 # padding
working_dis = 10000 * (2 * padding + 1)          # working distance

# scale factor of the intensity
scale_factor = working_dis * 2 * math.pi * res/fov           

# simulation resolution
# in order to do fft and ifft, expand the image use padding
simRes = res*(2*padding + 1)
simFov = fov*(2*padding + 1)

line_size = int(simRes/2)

ps = [0, 0, 0]              # position of the sphere
k_dir = [0, 0, -1]          # propagation direction of the plane wave
E = [1, 0, 0]               # electric field vector
# half of the grid size
halfgrid = np.ceil(fov / 2) * (padding * 2 + 1)

# features
a_min = 1.0
a_max = 2.0

nr_min = 1.1
nr_max = 2.0

ni_min = 0.01
ni_max = 0.05

# dimention of the data set
num = 30

a, nr, ni = cal_feature_space(a_min, a_max,
                              nr_min, nr_max,
                              ni_min, ni_max, num)

#get the maximal order of the integration
l = get_order(a_max, lambDa)

# center obscuration of the objective when calculating bandpass filter
NA_in = 0.0
# numerical aperture of the objective
NA_out = 0.3

# number of different numerical apertures to be tested
nb_NA = 8

# allocate a list of the NA
NA_list = np.linspace(0.05, NA_out, nb_NA)


# total number of images in the data set
num_samples = num ** 3

test_size = 0.2
num_test = int(num_samples * test_size)
num_test_in_group = int(num_test / num)

# pre load y train and y test
#y_train = np.load(r'D:\irimages\irholography\CNN\data_v9_far_field\split_data\train\y_train.npy')
y_test = np.load(r'D:\irimages\irholography\CNN\data_v12_far_line\split_data\y_test.npy')

# pre load intensity and complex CNNs
complex_ANN = load_model(r'D:\irimages\irholography\CNN\ANN_v2_far_line\complex_model_dropout')
intensity_ANN = load_model(r'D:\irimages\irholography\CNN\ANN_v2_far_line\intensity_model_dropout')

# parent directory of the data set
data_dir = r'D:\irimages\irholography\CNN\ANN_v2_far_line'

# allocate space for complex and intensity accuracy
complex_error = np.zeros((nb_NA, 3), dtype = np.float64)
intensity_error = np.zeros((nb_NA, 3), dtype = np.float64)

for NA_idx in range(nb_NA):
    
    NA = NA_list[NA_idx]
    scatter_matrix = cal_line_scatter_matrix(l, k, k_dir, res, fov,
                                             working_dis, scale_factor)
    
    X_test = np.zeros((num_test, line_size, 2))
    X_test_intensity = np.zeros((num_test, line_size, 1))
    
    print('Banbpassing the ' + str(NA_idx + 1) + 'th filter \n')
    cnt = 0
    for idx in range(num_test):
        
        n = y_test[idx, 0] + 1j*y_test[idx, 1]
        a = y_test[idx, 2]
    
        B = coeff_b(l, k, n, a)
        
        # integrate through all the orders to get the farfield in the Fourier Domain
        E_scatter_fft = np.sum(scatter_matrix * B, axis = -1) * scale_factor
        
        bpf = new_bpf(simFov, simRes, NA_in, NA)
        bpf_line = bpf[0, :int(simRes/2)+1]

        # convert back to spatial domain
        E_near_line, E_near_x = idhf(simFov, simRes, E_scatter_fft*bpf_line)
        
        E_near_line = (E_near_line-np.mean(E_near_line))/np.std(E_near_line)
        
        # shift the scattering field in the spacial domain for visualization
        Et = E_near_line + 1
        
        X_test[idx,:,0] = np.real(Et)
        X_test[idx,:,1] = np.imag(Et)
        
        X_test_intensity[idx,:,0] = np.abs(Et)**2
        
        # print progress
        cnt += 1
        sys.stdout.write('\r' + str(round(cnt / num_test * 100, 2))  + ' %')
        sys.stdout.flush() # important
    
    print()
    print('Evaluating complex model \n')
    # handle complex model first
    complex_error[NA_idx, :] = calculate_error(X_test, option = 'complex')
    
    print('Evaluating intensity model \n')
    # handle intensity model second
    intensity_error[NA_idx, :] = calculate_error(X_test_intensity, option = 'intensity')
    
# save the error file
np.save(r'D:\irimages\irholography\CNN\ANN_v2_far_line\complex_error_to0.3', complex_error)
np.save(r'D:\irimages\irholography\CNN\ANN_v2_far_line\intensity_error_to0.3', intensity_error)

#%%
# plot out the error
plt.figure()
plt.subplot(311)
plt.plot(NA_list, complex_error[:, 0], label = 'Complex CNN')
plt.plot(NA_list, intensity_error[:, 0], label = 'Intensity CNN')
plt.xlabel('NA')
plt.ylabel('Relative Error (Refractive Index)')
plt.legend()

plt.subplot(312)
plt.plot(NA_list, complex_error[:, 1], label = 'Complex CNN')
plt.plot(NA_list, intensity_error[:, 1], label = 'Intensity CNN')
plt.xlabel('NA')
plt.ylabel('Relative Error (Attenuation Coefficient)')
plt.legend()

plt.subplot(313)
plt.plot(NA_list, complex_error[:, 2], label = 'Complex CNN')
plt.plot(NA_list, intensity_error[:, 2], label = 'Intensity CNN')
plt.xlabel('NA')
plt.ylabel('Relative Error (Sphere Radius)')
plt.legend()