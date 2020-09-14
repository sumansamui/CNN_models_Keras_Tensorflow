import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import os
import numpy as np
import sys

import config
from utility_functions import load_dataset, create_cnn_model,download_mnist_fashion


flag_download=0 # it should be set to 1 for the first time

if flag_download==1:
	download_mnist_fashion()


print('*'*50)
print('TF version:' + tf.__version__)
print('keras version:'+ keras.__version__)
print('*'*50)

print('Loading data from disk...')
X_train, y_train, X_valid, y_valid, X_test, y_test = load_dataset()

print('*'*50)
print('X_train shape:' + str(X_train.shape))
print('y_train shape:'+ str(y_train.shape))


print('X_valid shape:' + str(X_valid.shape))
print('y_valid shape:'+ str(y_valid.shape))

print('X_test shape:' + str(X_test.shape))
print('y_test shape:'+ str(y_test.shape))



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

batch_size = 32
epochs=100

# Model name for saving
model_name='cnn_model.h5'

# create checkpoint folder and removes the contents if already exits
os.makedirs(config.path_to_checkpoint,exist_ok=True)




checkpoint_cb = keras.callbacks.ModelCheckpoint(os.path.join(config.path_to_checkpoint, model_name),save_best_only=True)

early_stopping_cb = keras.callbacks.EarlyStopping(patience=10,restore_best_weights=True)


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
os.makedirs(config.path_to_results,exist_ok=True)
plt.savefig(os.path.join(config.path_to_results,'training_history.png'))
plt.show()

# Evaluate the model
loss, acc = model.evaluate(X_test,  y_test, verbose=2)
print("Trained model, accuracy: {:5.2f}%".format(100*acc))

# Save the result
sys.stdout=open(os.path.join(config.path_to_results,'result.txt'),"w")
print("Trained model, accuracy: {:5.2f}%".format(100*acc))
sys.stdout.close()