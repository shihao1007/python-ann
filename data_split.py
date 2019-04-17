# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 13:56:12 2019

data split for 1D data

Editor:
    Shihao Ran
    STIM Laboratory
"""

import numpy as np
from sklearn.model_selection import train_test_split

#%%

res = 128
padding=2
simRes = res*(padding*2+1)
line_length = int(simRes/2)
num = 30
num_total_sample = num**3
channel = 2

X = np.load(r'D:\irimages\irholography\CNN\data_v12_far_line\raw_data\im_data.npy')
y = np.load(r'D:\irimages\irholography\CNN\data_v12_far_line\raw_data\label_data.npy')

X = np.reshape(X, (line_length, channel, num_total_sample))
X = np.swapaxes(X, 0, -1)
X = np.swapaxes(X, -2, -1)

y = np.reshape(y, (3, num_total_sample))
y = np.swapaxes(y, 0, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .2,
                                                    random_state = 5)

np.save(r'D:\irimages\irholography\CNN\data_v12_far_line\split_data\X_train', X_train)
np.save(r'D:\irimages\irholography\CNN\data_v12_far_line\split_data\y_train', y_train)
np.save(r'D:\irimages\irholography\CNN\data_v12_far_line\split_data\X_test', X_test)
np.save(r'D:\irimages\irholography\CNN\data_v12_far_line\split_data\y_test', y_test)

#%%
X_train_inten = np.abs(X_train[:,:,0] + 1j*X_train[:,:,1])**2
X_test_inten = np.abs(X_test[:,:,0] + 1j*X_test[:,:,1])**2

X_train_inten = X_train_inten[...,None]
X_test_inten = X_test_inten[...,None]
np.save(r'D:\irimages\irholography\CNN\data_v12_far_line\split_data\X_train_intensity', X_train_inten)
np.save(r'D:\irimages\irholography\CNN\data_v12_far_line\split_data\X_test_intensity', X_test_inten)