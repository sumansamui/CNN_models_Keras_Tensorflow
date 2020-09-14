import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import os
import numpy as np
import sys



path_to_checkpoint = './ckpnt/'

path_to_data = './mnist_fashion_data/'

path_to_results = './results/'


fashion_mnist = keras.datasets.fashion_mnist

(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()



X_valid, X_train = X_train_full[:5000], X_train_full[5000:]

y_valid, y_train = y_train_full[:5000], y_train_full[5000:]



#scaling
X_mean = X_train.mean(axis=0, keepdims=True)
X_std = X_train.std(axis=0, keepdims=True) + 1e-7

X_train = (X_train - X_mean) / X_std
X_valid = (X_valid - X_mean) / X_std
X_test = (X_test - X_mean) / X_std

X_train = X_train[..., np.newaxis]
X_valid = X_valid[..., np.newaxis]
X_test = X_test[..., np.newaxis]





print('*'*50)
print('TF version:' + tf.__version__)
print('keras version:'+ keras.__version__)
print('*'*50)

print('Loading data from disk...')


print('*'*50)
print('X_train shape:' + str(X_train.shape))
print('y_train shape:'+ str(y_train.shape))


print('X_valid shape:' + str(X_valid.shape))
print('y_valid shape:'+ str(y_valid.shape))

print('X_test shape:' + str(X_test.shape))
print('y_test shape:'+ str(y_test.shape))


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



print('creating Sequential Keras Model...')
# Create a basic model instance
model = create_cnn_model()

# Evaluate the untrained model
loss, acc = model.evaluate(X_test,  y_test, verbose=2)
print("Untrained model, accuracy: {:5.2f}%".format(100*acc))


# Display the model's architecture
model.summary()



print('starting model training ...')

# Training params

batch_size = 64
epochs=100

# Model name for saving
model_name='cnn_model.h5'

# create checkpoint folder and removes the contents if already exits
os.makedirs(path_to_checkpoint,exist_ok=True)




checkpoint_cb = keras.callbacks.ModelCheckpoint(os.path.join(path_to_checkpoint, model_name),save_best_only=True)

early_stopping_cb = keras.callbacks.EarlyStopping(patience=20,restore_best_weights=True)


# Plot training history

history = model.fit(X_train, 
					y_train, 
					epochs = epochs,
					batch_size= batch_size,
					validation_data=(X_valid, y_valid),
					callbacks=[checkpoint_cb,early_stopping_cb])


# Plot training history

pd.DataFrame(history.history).plot(figsize=(8,5))
plt.grid(True)
plt.gca().set_ylim(0,1)
os.makedirs(path_to_results,exist_ok=True)
plt.savefig(os.path.join(path_to_results,'training_history.png'))
plt.show()

# Evaluate the model
loss, acc = model.evaluate(X_test,  y_test, verbose=2)
print("Trained model, accuracy: {:5.2f}%".format(100*acc))

# Save the result
sys.stdout=open(os.path.join(path_to_results,'result.txt'),"w")
print("Trained model, accuracy: {:5.2f}%".format(100*acc))
sys.stdout.close()