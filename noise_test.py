# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 15:27:56 2019

Editor:
    Shihao Ran
    STIM Laboratory
"""
import matplotlib.pyplot as plt
import numpy as np

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
imdata = np.load(r'D:\irimages\irholography\CNN\data_v12_far_line\bandpass_data\im_data.npy')

#%%
sample1 = imdata[:, 0, 5, 5, 5, 18]
sample2 = imdata[:, 1, 5, 5, 5, 18]

plt.figure()
plt.plot(np.real(sample1), label='Real')
plt.plot((sample2), label='Imaginary')
plt.xlabel('Pixel Index')
plt.ylabel('Pixel Intensity')
plt.title('Training Sample # 1252 NA 0.9')
#plt.plot(eta_if, label='Noise')
#plt.plot(sample1_w_noise, label='WithNoise')
#plt.plot(sample1[:,1], label='Imaginary')
plt.legend()
