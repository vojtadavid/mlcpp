# import mnist_loader
import network
# import tensorflow as tf
# import keras
import numpy as np
import pickle


b = None
with open('mnist.data','rb') as input:
    b = pickle.load(input)
(x_train, y_train), (x_test, y_test) = b


# _training_data, _validation_data, _test_data = mnist_loader.load_data_wrapper()


train_data=[]
for i in range(len(x_train)):
    y = np.zeros((10,1), dtype=int)
    y[y_train[i]]=1
    train_data.append((x_train[i].reshape(784,1)/255,y))

test_data=[]
for i in range(len(x_test)):
    y = np.zeros((10,1), dtype=int)
    y[y_test[i]]=1
    test_data.append((x_test[i].reshape(784,1)/255,y_test[i]))

net = network.Network([784, 30, 10])
net.load()
net.SGD(train_data, 30, 10, 3.0, test_data=test_data)

