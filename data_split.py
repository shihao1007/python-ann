# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 13:56:12 2019

data split for 1D data
load the raw data then split it and calculate the intensity version too

Editor:
    Shihao Ran
    STIM Laboratory
"""

# import packages
import numpy as np
from sklearn.model_selection import train_test_split

#%%
# specify the parameters
res = 128
padding=2
simRes = res*(padding*2+1)
line_length = int(simRes/2)
num = 30
num_bp = 100
num_total_sample = num**3
channel = 2

# load the raw data
X = np.load(r'D:\irimages\irholography\CNN\data_v12_far_line\raw_data\im_data.npy')
y = np.load(r'D:\irimages\irholography\CNN\data_v12_far_line\raw_data\label_data.npy')

# reshape and swap the axis so that the number of samples is at the first dimention
X = np.reshape(X, (line_length, channel, num_total_sample))
X = np.swapaxes(X, 0, -1)
X = np.swapaxes(X, -2, -1)

y = np.reshape(y, (3, num_total_sample))
y = np.swapaxes(y, 0, 1)

# split the data set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .1,
                                                    random_state = 25)

# save the splitted data sets
np.save(r'D:\irimages\irholography\CNN\ANN_simple_test\data\X_train', X_train)
np.save(r'D:\irimages\irholography\CNN\ANN_simple_test\data\y_train', y_train)
np.save(r'D:\irimages\irholography\CNN\ANN_simple_test\data\X_test', X_test)
np.save(r'D:\irimages\irholography\CNN\ANN_simple_test\data\y_test', y_test)

#%%
# calculate the intensity
X_train_absolute = np.abs(X_train[:,:,0] + 1j*X_train[:,:,1])
X_test_absolute = np.abs(X_test[:,:,0] + 1j*X_test[:,:,1])

# reshape the intensity set since neural network only accepts (..., 1) data
X_train_absolute = X_train_absolute[...,None]
X_test_absolute = X_test_absolute[...,None]
np.save(r'D:\irimages\irholography\CNN\ANN_simple_test\data\X_train_absolute', X_train_absolute)
np.save(r'D:\irimages\irholography\CNN\ANN_simple_test\data\X_test_absolute', X_test_absolute)


#%%
# calculate the intensity
X_train_intensity = np.abs(X_train[:,:,0] + 1j*X_train[:,:,1])**2
X_test_intensity = np.abs(X_test[:,:,0] + 1j*X_test[:,:,1])**2

# reshape the intensity set since neural network only accepts (..., 1) data
X_train_intensity = X_train_intensity[...,None]
X_test_intensity = X_test_intensity[...,None]

np.save(r'D:\irimages\irholography\CNN\ANN_simple_test\data\X_train_intensity', X_train_intensity)
np.save(r'D:\irimages\irholography\CNN\ANN_simple_test\data\X_test_intensity', X_test_intensity)

#%%
X_train_inten_beta = np.zeros((21600, 320, 2))
X_test_inten_beta = np.zeros((5400, 320, 2))

X_train_inten_beta[...,0] = X_train_intensity
X_train_inten_beta[...,1] = X_train_intensity
#X_train_inten_beta[...,2] = X_train_inten

X_test_inten_beta[...,0] = X_train_intensity
X_test_inten_beta[...,1] = X_train_intensity
#X_test_inten_beta[...,2] = X_test_inten

np.save(r'D:\irimages\irholography\CNN\data_v12_far_line\split_data\X_train_inten_2channels', X_train_inten_beta)
np.save(r'D:\irimages\irholography\CNN\data_v12_far_line\split_data\X_test_inten_2channels', X_test_inten_beta)

#%%
X_train_real = np.zeros((21600, 320, 1))
X_test_real = np.zeros((5400, 320, 1))

X_train_real[...,0] = X_train[...,0]
#X_train_real[...,1] = X_train[...,0]

X_test_real[...,0] = X_test[...,0]
#X_test_real[...,1] = X_test[...,0]

np.save(r'D:\irimages\irholography\CNN\ANN_simple-real\data\X_train_real', X_train_real)
np.save(r'D:\irimages\irholography\CNN\ANN_simple-real\data\X_test_real', X_test_real)

#%%
X_train_real_2ch = np.zeros((21600, 320, 2))
X_test_real_2ch = np.zeros((5400, 320, 2))

X_train_real_2ch[...,0] = X_train[...,0]
X_train_real_2ch[...,1] = X_train[...,0]

X_test_real_2ch[...,0] = X_test[...,0]
X_test_real_2ch[...,1] = X_test[...,0]

np.save(r'D:\irimages\irholography\CNN\ANN_simple-real\data\X_train_real_2ch', X_train_real_2ch)
np.save(r'D:\irimages\irholography\CNN\ANN_simple-real\data\X_test_real_2ch', X_test_real_2ch)