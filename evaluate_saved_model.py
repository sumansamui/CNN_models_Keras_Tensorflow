import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import os
import numpy as np

import config
from utility_functions import load_dataset, create_model



print('Loading data from disk...')
X_train, y_train, X_valid, y_valid, X_test, y_test = load_dataset()


# Create a basic model instance 

new_model = create_model()

saved_model_name = 'dnn_model.h5'

# Loads the weights (restore the trained model)
new_model.load_weights(os.path.join(config.path_to_checkpoint, saved_model_name))


# Performance of the restored model
loss, acc = new_model.evaluate(X_test,  y_test, verbose=2)
print('Restored model, accuracy: {:5.2f}%'.format(100*acc))