"""
Author: Suman Samui
Email: samuisuman@gmail.com

"""





import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import os

import config

def download_mnist_fashion():

	fashion_mnist = keras.datasets.fashion_mnist

	(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()



	X_valid, X_train = X_train_full[:5000], X_train_full[5000:]

	y_valid, y_train = y_train_full[:5000], y_train_full[5000:]



	os.makedirs(config.path_to_data,exist_ok=True)
	np.save(os.path.join(config.path_to_data,'X_train.npy'),X_train)
	np.save(os.path.join(config.path_to_data,'y_train.npy'),y_train)
	np.save(os.path.join(config.path_to_data,'X_valid.npy'),X_valid)
	np.save(os.path.join(config.path_to_data,'y_valid.npy'),y_valid)
	np.save(os.path.join(config.path_to_data,'X_test.npy'),X_test)
	np.save(os.path.join(config.path_to_data,'y_test.npy'),y_test)

	print('saving data....complete!')


def load_dataset():

	X_train = np.load(os.path.join(config.path_to_data,'X_train.npy')) 
	y_train = np.load(os.path.join(config.path_to_data,'y_train.npy'))

	X_valid = np.load(os.path.join(config.path_to_data,'X_valid.npy'))
	y_valid = np.load(os.path.join(config.path_to_data,'y_valid.npy'))
	
	X_test = np.load(os.path.join(config.path_to_data,'X_test.npy'))
	y_test = np.load(os.path.join(config.path_to_data,'y_test.npy'))

	#scaling
	X_mean = X_train.mean(axis=0, keepdims=True)
	X_std = X_train.std(axis=0, keepdims=True) + 1e-7

	X_train = (X_train - X_mean) / X_std
	X_valid = (X_valid - X_mean) / X_std
	X_test = (X_test - X_mean) / X_std

	X_train = X_train[..., np.newaxis]
	X_valid = X_valid[..., np.newaxis]
	X_test = X_test[..., np.newaxis]

	return X_train, y_train, X_valid, y_valid, X_test, y_test


# Define a simple sequential model
def create_cnn_model():
	model = keras.models.Sequential([
		keras.layers.Conv2D(64,7,activation='relu',padding='same',input_shape=[28,28,1]),
		keras.layers.BatchNormalization(),
		keras.layers.MaxPooling2D(2),
		keras.layers.Conv2D(128,3,activation='relu',padding='same'),
		keras.layers.BatchNormalization(),
		keras.layers.Conv2D(128,3,activation='relu',padding='same'),
		keras.layers.BatchNormalization(),
		keras.layers.MaxPooling2D(2),
		keras.layers.Conv2D(256,3,activation='relu',padding='same'),
		keras.layers.BatchNormalization(),
		keras.layers.Conv2D(256,3,activation='relu',padding='same'),
		keras.layers.BatchNormalization(),
		keras.layers.MaxPooling2D(2),
		keras.layers.Flatten(),
		keras.layers.BatchNormalization(),
		keras.layers.Dense(128,activation='relu'),
		keras.layers.BatchNormalization(),
		keras.layers.Dropout(0.5),
		keras.layers.Dense(64,activation='relu'),
		keras.layers.BatchNormalization(),
		keras.layers.Dropout(0.5),
		keras.layers.Dense(10,activation='softmax')
		])
	model.compile(optimizer='nadam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
	return model


