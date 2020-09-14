

## A basic tutorial: Building an image classifier using CNN (Keras Tensorflow backend)




## General steps to be followed:

* First, we need to load a dataset (Fashion MNIST)

* Build a keras model using deep CNN
   
* Train and save the model

* Evaluate the trained model on test data



## Fashion MNIST Dataset:

* Fashion MNIST, which is a drop-in replacement of MNIST (hadwritten digits). It has the exact same format as MNIST (70,000 grayscale images of 28×28 pixels each, with 10 classes), but the images represent fashion items rather than handwritten digits, so each class is more diverse and the problem turns out to be significantly more challenging than MNIST.

class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
 

<img src="/images/fashion_mnist.png" width="800" />
 
 
* The dataset is already split into a training set, a test set, and validation (dev) set. Since we are going to train the neural network using Gradient Descent types algorithm, we must scale the input features. 

* Train, test and validation set will be stored as numpy arrays in ./data folder after scaling.

* Please run set the flag_download=1 if you run the model for the first time


## Description of scripts in the repo:


* config.py -- it contains description of all the necessary paths (data path, checkpoint path, and result saving path)

* utility_functions.py -- it contains functions like loading dataset, creating Keras model, etc.

* main_train_evaluate.py -- It is the main top-level scripts to create, train and evaluate a model. It will also the latest trained model into ./ckpnt folder.

* evaluate_saved_model.py -- You can use this script to restore a saved model in ./ckpnt folder and reproduce the result


## Next Task to do:

This model is not fine-tuned. 

Current accuracy on test data ~ 92 % (approx)

You should fine-tune the hyperparameters (learning rate, the number of layers, the number of nodes in each layer, activation functions, and optimization algorithms) and settle down to a best possible DNN model for this classification task.




### For queries and further information:

please contact: samuisuman@gmail.com
 
### © Suman Samui All Rights Reserved 
