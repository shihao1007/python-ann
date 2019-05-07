# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 15:40:48 2019

this script generates dataset with bandpass filtering
unlike data_generate_bandpass.py which generates bandpassed testing set only
this script regenerates the whole dataset with bandpass as the fourth feature
similar to data_generate_noisy_line.py


Editor:
    Shihao Ran
    STIM Laboratory
"""

# import necessary packages
import numpy as np
import scipy as sp
import scipy.special
import math
import matplotlib.pyplot as plt
import sys
import chis.MieScattering as ms
#%%
def cal_feature_space(a_min, a_max,
                      nr_min, nr_max,
                      ni_min, ni_max,
                      noise_pc_min, noise_pc_max,
                      num, num_bp):
# set the range of the features:
    # a_max: the maximal of the radius of the spheres
    # a_min: the minimal of the radius of the spheres
    # nr_max: the maximal of the refractive indices
    # nr_min: the minimal of the refractive indices
    # ni_max: the maximal of the attenuation coefficients
    # ni_min: the minimal of the attenuation coefficients
    # num: dimention of each feature
    a = np.linspace(a_min, a_max, num)
    nr = np.linspace(nr_min, nr_max, num)
    ni = np.linspace(ni_min, ni_max, num)
    noise_pc = np.linspace(noise_pc_min, noise_pc_max, num_bp)

    return a, nr, ni, noise_pc


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

# define feature space

num = 10
num_bp = 100
num_samples = num ** 3 * num_bp

a_min = 1.1
a_max = 2.0

nr_min = 1.1
nr_max = 2.0

ni_min = 0.01
ni_max = 0.05

bp_min = 0.02
bp_max = 1.01

a, nr, ni, bp = cal_feature_space(a_min, a_max,
                                  nr_min, nr_max,
                                  ni_min, ni_max,
                                  bp_min, bp_max,
                                  num, num_bp)

#get the maximal order of the integration
l = ms.get_order(a_max, lambDa)

# pre-calculate the scatter matrix and the incident field
scatter = ms.scatter_matrix(simRes, simFov, working_dis, a_max,
                            lambDa, k_dir, 1, 'far')

# allocate space for data set
sphere_data = np.zeros((3, num, num, num, num_bp))
im_data = np.zeros((line_size, 2, num, num, num, num_bp))
im_dir = r'D:\irimages\irholography\CNN\data_v12_far_line\more_bandpass_HD\im_data'

# get all the bandpass filters
bpfs = []
for b in range(num_bp):
                
    bp0 = bp[b]
    bpf = ms.bandpass_filter(simRes, simFov, 0, bp0, dimension=1)
    bpfs.append(bpf)

cnt = 0
for h in range(num):
    for i in range(num):
        for j in range(num):
            
            a0 = a[h]
            n0 = nr[i] + 1j*ni[j]
            
            B = ms.coeff_b(l, k, n0, a0)

            # integrate through all the orders to get the farfield in the Fourier Domain
            E_scatter_fft = np.sum(scatter * B, axis = -1) * scale_factor
            
            for b in range(num_bp):
        
                # convert back to spatial domain
                E_near_line, E_near_x = ms.idhf(simRes, simFov, E_scatter_fft*bpfs[b])
    
                E_near_line = (E_near_line-np.mean(E_near_line))/np.std(E_near_line)
    
                # shift the scattering field in the spacial domain for visualization
                Et = E_near_line + 1
    
                im_data[:, 0, i, j, h, b] = np.real(Et)
                im_data[:, 1, i, j, h, b] = np.imag(Et)
    
                sphere_data[:, i, j, h, b] = [nr[i], ni[j], a[h]]
    
                            # print progress
                cnt += 1
                sys.stdout.write('\r' + str(round(cnt / (num_samples) * 100, 2))  + ' %')
                sys.stdout.flush() # important

# save the data
np.save(im_dir, im_data)
np.save(r'D:\irimages\irholography\CNN\data_v12_far_line\more_bandpass_HD\label_data', sphere_data)