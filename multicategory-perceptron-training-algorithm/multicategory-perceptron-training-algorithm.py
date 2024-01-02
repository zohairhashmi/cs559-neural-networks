# CS 559: Neural Networks
# Assignment 3
# Authors: Zohair Hashmi 
# UIN: 668913771

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gzip

image_size = 28
n_test = 10000

## Defining Functions

def trainModel(epsilon, learning_rate, randomSeed):
    np.random.seed(randomSeed)
    y_train_one_hot = np.zeros((n, 10))
    y_train_one_hot[np.arange(n), y_train] = 1
    epochs = 0
    W = np.random.randn(10,784) #initializing weights
    epoch_errors = [100000]
    while (epochs < 1000 and epoch_errors[epochs]/ n > epsilon):
        print('Training epoch: ', epochs)    
        errors = 0
        y_pred = np.zeros((n,10))
        for i in range(n):
            v = np.dot(W,X_train[i].reshape(784,1)) #calculating v    
            max_v = np.argmax(v) #finding the index of the maximum value in v
            y_pred[i][max_v] = 1
            
            if max_v != y_train[i]:
                errors += 1      
        
        W = W + learning_rate*(y_train_one_hot-y_pred).T@X_train.reshape(n, 784)
            
        epoch_errors.append(errors)
        epochs += 1
        print('Number of errors: %d \t Error rate: %.2f' %(errors, errors/n))
        print('--------------------------------------------------')
    return epochs, epoch_errors, W

def testModel(W):
    errors = 0
    y_test_pred = np.zeros((n_test,10))

    for i in range(n_test):
        v_test = np.dot(W,X_test[i].reshape(784,1)) #calculating v    
        max_v_test = np.argmax(v_test) #finding the index of the maximum value in v
        y_test_pred[i][max_v_test] = 1

        if max_v_test != y_test[i]:
            errors += 1

    print('Number of errors: ', errors)
    print('Accuracy: ', (n_test-errors)/n_test*100, '%')


## Reading Test Data
# Reading test images to numpy array
f = gzip.open('dataset/t10k-images-idx3-ubyte.gz','r')

f.read(16)
buf = f.read(image_size * image_size * n_test)
X_test = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
X_test = X_test.reshape(n_test, image_size, image_size, 1)

# Reading test labels to numpy array
f = gzip.open('dataset/t10k-labels-idx1-ubyte.gz','r')
f.read(8)
buf = f.read(n_test)
y_test = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)


# Ask user to input number of training samples
n = int(input('Enter number of training samples: '))
print('--------------------------------------------------')
# Ask user to input epsilon
epsilon = float(input('Enter epsilon: '))
print('--------------------------------------------------')
# Ask user to input learning rate
learning_rate = float(input('Enter learning rate: '))
print('--------------------------------------------------')
# Ask user to input random seed
randomSeed = int(input('Enter random seed: '))
print('--------------------------------------------------')


# Reading training images to numpy array
f = gzip.open('dataset/train-images-idx3-ubyte.gz','r')

f.read(16)
buf = f.read(image_size * image_size * n)
X_train = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
X_train = X_train.reshape(n, image_size, image_size, 1)

# Reading training labels to numpy array
f = gzip.open('dataset/train-labels-idx1-ubyte.gz','r')
f.read(8)
buf = f.read(n)
y_train = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)

# Data Training Phase
epochs, epoch_errors, W = trainModel(epsilon, learning_rate, randomSeed)

# Testing the model
testModel(W)

# Plotting the error rate vs epochs
plt.plot(range(1,epochs+1), epoch_errors[1:])
plt.xlabel('Epochs')
plt.ylabel('Error Rate')
plt.title('Error Rate vs Epochs')
plt.show()

exit()