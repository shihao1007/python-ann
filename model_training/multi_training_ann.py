# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 13:32:48 2019

this is a program runs the training process of the ANN and measure the
average performance

Editor:
    Shihao Ran
    STIM Laboratory
"""


#%%
# import necessary packages
import numpy as np

from keras.models import Sequential
from keras.layers import Flatten
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint
from keras.models import load_model

#%%
# define data size
res = 128
padding=2
simRes = res*(padding*2+1)
line_length = int(simRes/2)
num = 30
num_total_sample = num**3
test_size = .2
num_test_samples = int(num_total_sample * test_size)
num_training = 30
#%%
# define ANN structure

def create_ann(option):
    """
    create a 5-layer CNN based on the type
    parameter:
        option: 'complex' or 'intensity'
            if it is a complex CNN, the input images will have two channels
            otherwise just one channel for intensity images
    return:
        regressor: a sequential CNN model
    """
    
    if option == 'complex':
        channel = 2
    else:
        channel = 1
    
    regressor = Sequential()
    
    regressor.add(Dense(640, input_shape=(line_length,channel,), activation='relu'))
    
    regressor.add(Dense(320, activation='relu'))
    
    regressor.add(Flatten())
    
    regressor.add(Dense(160, activation='relu'))
    
    regressor.add(Dense(3))
    
    regressor.compile('adam', loss = 'mean_squared_error')
    
    return regressor
#%%
# load data set

X_train_complex = np.load(r'D:\irimages\irholography\CNN\data_v12_far_line\split_data\X_train.npy')
X_test_complex = np.load(r'D:\irimages\irholography\CNN\data_v12_far_line\split_data\X_test.npy')

X_train_intensity = np.load(r'D:\irimages\irholography\CNN\data_v12_far_line\split_data\X_train_intensity.npy')
X_test_intensity = np.load(r'D:\irimages\irholography\CNN\data_v12_far_line\split_data\X_test_intensity.npy')

y_train = np.load(r'D:\irimages\irholography\CNN\data_v12_far_line\split_data\y_train.npy')
y_test = np.load(r'D:\irimages\irholography\CNN\data_v12_far_line\split_data\y_test.npy')

model_save_dir = r'D:\irimages\irholography\CNN\ANN_v2_far_line\multi_training'

#%%
# scale the attenuation coefficient feature

y_train[:, 1] *= 100

#%%
def multi_training(num_training, option):

    # initialize error list
    rmse = np.zeros((num_training, num_test_samples, 3))
    
    r_rmse = np.zeros((num_training, 3))

    # train the network
    for i in range(num_training):
        
        model_path = r'D:\irimages\irholography\CNN\ANN_v2_far_line\complex_model'+str(i)
        
        model_check = ModelCheckpoint(model_path, save_best_only=True)
        
        # for each loop
        
        # initialize CNN model
        ann = create_ann(option)
        
        # load training and testing data
        if option == 'complex':
            X_train = X_train_complex
            X_test = X_test_complex
        else:
            X_train = X_train_intensity
            X_test = X_test_intensity
            
        # train the complex cnn
        print('Training the ' + str(i+1) + 'th '+option+' model!')
        
        ann.fit(x=X_train, y=y_train, batch_size=50,
                epochs=35, validation_split=0.2,
                callbacks=[model_check])
        
        ann = load_model(model_path)
        # get the predictions
        y_pred = ann.predict(X_test)
        
        # down scale the predictions
        y_pred[:, 1] /= 100
        
        #evaluation for a and n
        y_off = np.abs(y_pred - y_test)
        y_r_off = np.mean(y_off / [2, 0.05, 2], axis=0) * 100

        rmse[i, ...] = y_off
        r_rmse[i, :] = y_r_off
        
    return rmse, r_rmse

#%%
complex_rmse, complex_r_rmse = multi_training(num_training, 'complex')
intensity_rmse, intensity_r_rmse = multi_training(num_training, 'intensity')

np.save(model_save_dir + '\\complex_multi_rmse1', complex_rmse)
np.save(model_save_dir + '\\intensity_multi_rmse1', intensity_rmse)
np.save(model_save_dir + '\\complex_multi_r_rmse1', complex_r_rmse)
np.save(model_save_dir + '\\intensity_multi_r_rmse1', intensity_r_rmse)
