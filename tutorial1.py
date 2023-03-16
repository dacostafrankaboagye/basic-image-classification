
import tensorflow as tf

print("using tf verion : ", tf.__version__)




#============== The Dataset Import MNIST
from tensorflow.keras.datasets import mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data() # this is a helper function that returns train and text data
'''
train set= data to train the nn

test set = to validate the perfomace of the nn
'''



#============== Shapes of Imported Arrays
print("x_train shape: {} \ny_train shape {}\nx_text shape: {}\ny_train shape: {}".format(
    x_train.shape, y_train.shape, x_test.shape, y_test.shape))

'''

x_train shape: (60000, 28, 28) - 60000 training examples 28px-w 28px-h (r,w)
y_train shape (60000,)
x_text shape: (10000, 28, 28)
y_train shape: (10000,)

'''


#============= Plot an Image Example

from matplotlib import pyplot as plt
#%matplotlib inline
plt.imshow(x_train[0], cmap='binary')
plt.show()


#========= Display Labels
y_train[0]

'''
the unique values 
'''
print(set(y_train))




#=============One Hot Encoding
'''
fter this encoding, every label 
will be converted to a list with 10 elements and the 
element at index to the corresponding class will be set to 1, rest will be set to 0
'''


#============Encoding Labels

from tensorflow.keras.utils import to_categorical

y_train_encoded = to_categorical(y_train)
y_test_encoded = to_categorical(y_test)






#=============== Validated Shapes
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

#========== Display Encoded Labels
y_train_encoded[0]





#============Neural Networks



#=============== Preprocessing the Examples Unrolling N-dimensional Arrays to Vectors


'''
now each examples is 28 by 28 - we want to change that to 178 by 1


'''
import numpy as np

x_train_reshaped = np.reshape(x_train, (60000,784))
x_test_reshaped = np.reshape(x_test, (10000, 784))

print('x_train_reshaped shape: {}\nx_test_reshaped shape: {}'.format(
x_train_reshaped.shape, x_test_reshaped.shape))




#=======Display Pixel Values

#x_train_reshaped[0]
print(set(x_train_reshaped[0])) # printing the unique values

'''
that is how pixel values are 0 - 255
'''


#========= Data Normalisation


x_mean = np.mean(x_train_reshaped)
x_std = np.std(x_train_reshaped) # standard deviation

epsilon = 1e-10  # a very small value constant

x_train_norm = (x_train_reshaped - x_mean) / (x_std + epsilon)
x_test_norm = (x_test_reshaped - x_mean) / (x_std + epsilon)




#============Display Normalized Pixel Values

print(set(x_train_norm[0]))
'''
they are small values now
'''




#======================Creating a Model Creating the Model

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


'''
using the sequential class to create the model
- we pass in a list of layers

with sequential class, your input layer
is your input examples - you donot define it separately
but let it correspond to the input shape
- the output of one layer is the input to the next layer
- you can change the number of nodes
 - or add more layers
 
 think of computaion power
 
define which algorithm it should use to optimise the w and b


'''

model = Sequential([
    Dense(128, activation='relu', input_shape=(784,)),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
    
])


#Activation fnx = give model flexibility = helps you to find non linear patterns in the data

#=============== Compiling the model


'''

optimizer fnx, loss fnx, 
sgd=stochastic gradient descent
'categorical_crossentropy' = like the difference btn actual ouput and predicted output
it needs to be minimised

'''

model.compile(
    optimizer='sgd',
    loss = 'categorical_crossentropy',
    metrics = ['accuracy']
)

model.summary() # will display the architecture of the model


#=============== training the model

'''

'''

model.fit(x_train_norm, y_train_encoded, epochs=3)



#========================== evaluate the model
'''
we should know if it has understood the nderlying fnx

'''

loss, accuracy = model.evaluate(x_test_norm, y_test_encoded)

'''
it uses the model state as it is
does a forward pass to understand the predictions
the accuracy should be higher for us in this case than the last epoch computed accuracy

'''

print("test set accuracy = ", accuracy*100)
'''
test set accuracy =  96.05000019073486
it is not significantly lower = successful
'''



#====================== prediction on test set

preds = model.predict(x_test_norm)

print('shape of preds: ', preds.shape)



#=============== plotting the result
'''
only 25 of them
'''


plt.figure(figsize=(12, 12))

start_index = 0

for i in range(25):
    plt.subplot(5,5, i+1)
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    
    pred = np.argmax(preds[start_index+i])
    gt = y_test[start_index+i]
    
    col = 'g'
    if pred!= gt:
        col = 'r'
        
    plt.xlabel('i={}, pred={}, gt={}'.format(
        start_index+i, pred, gt
    ), color=col)
    
    plt.imshow(x_test[start_index+i], cmap='binary')
plt.show()





#============== plotting the inaccuracte predicted value

'''
take a look at the prediction that wasn't accurate
for me it was index 8
'''

plt.plot(preds[8])
plt.show()

'''
you will see the softmax probability output
'''

