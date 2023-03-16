
import tensorflow as tf

print("using tf verion : ", tf.__version__)


from tensorflow.keras.datasets import mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data() # this is a helper function that returns train and text data
'''
train set= data to train the nn

test set = to validate the perfomace of the nn
'''

print("x_train shape: {} \ny_train shape {}\nx_text shape: {}\ny_train shape: {}".format(
    x_train.shape, y_train.shape, x_test.shape, y_test.shape))

'''

x_train shape: (60000, 28, 28) - 60000 training examples 28px-w 28px-h (r,w)
y_train shape (60000,)
x_text shape: (10000, 28, 28)
y_train shape: (10000,)

'''

from matplotlib import pyplot as plt
plt.imshow(x_train[0], cmap='binary')
plt.show()

from tensorflow.keras.utils import to_categorical

y_train_encoded = to_categorical(y_train)
y_test_encoded = to_categorical(y_test)


'''
to make sure the encoding worksits now a 10 dimensional vector
'''
print("y_train_encoded shape : {}\ny_test_encoded shape: {} ".format(
y_train_encoded.shape, y_test_encoded.shape))

'''
y_train_encoded shape : (60000, 10)
y_test_encoded shape: (10000, 10) 
now each eaample is s 10-d vector
think if it like a switch.. it knows which one is on / off 
'''


'''
now each examples is 28 by 28 - we want to change that to 178 by 1


'''
import numpy as np

x_train_reshaped = np.reshape(x_train, (60000,784))
x_test_reshaped = np.reshape(x_test, (10000, 784))

print('x_train_reshaped shape: {}\nx_test_reshaped shape: {}'.format(
x_train_reshaped.shape, x_test_reshaped.shape))




x_mean = np.mean(x_train_reshaped)
x_std = np.std(x_train_reshaped) # standard deviation

epsilon = 1e-10  # a very small value constant

x_train_norm = (x_train_reshaped - x_mean) / (x_std + epsilon)
x_test_norm = (x_test_reshaped - x_mean) / (x_std + epsilon)
