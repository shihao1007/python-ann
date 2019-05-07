# python-ann
Python implementation of pipelines for ANNs to solve the inverse Mie problem

1 ./data_generate contains scripts for generating 1-D far field raw data, bandpassed data, noisy data, and new bandpassed test set

2 ./data_loader contains a single script that takes care of load the raw data, splitting them plus other preprocessings such as calculating the intensity, absolute value, of the raw data.

3 ./model_training contains scripts for building different ANN models for noisy data, bandpassed data with simple ANN structure or deep ANN structure.

Please pull "chis" repository and include it into Python path before running any script.
