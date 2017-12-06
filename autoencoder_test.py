from numpy.random import seed
#from sklearn.preprocessing import minmax_scale
#from sklearn.model_selection import train_test_split
from keras.layers import Input, Dense
from keras.models import Model
import glob
import load_data
import matplotlib.pyplot as plt
import numpy as np 

#(X_train, Y_train), (X_test, Y_test) = load_data.load_data('datasets',0.5)### AN EXAMPLE OF SIMPLE AUTOENCODER ###
data = glob.glob('datasets/test_set/positive_set/*.png')

# InputLayer (None, 10)
#      Dense (None, 5
#      Dense (None, 10)
ncol = len(data)
#print(X_train.shape)
#print(X_test.shape)
input_dim = Input(shape = (ncol, ))

# DEFINE THE DIMENSION OF ENCODER ASSUMED 3
#encoding_dim = 2

## DEFINE THE ENCODER LAYER
#encoded = Dense(encoding_dim, activation = 'relu')(input_dim)

## DEFINE THE DECODER LAYER
#decoded = Dense(ncol, activation = 'sigmoid')(encoded)

## COMBINE ENCODER AND DECODER INTO AN AUTOENCODER MODEL
#autoencoder = Model(input = input_dim, output = decoded)

## CONFIGURE AND TRAIN THE AUTOENCODER
#autoencoder.compile(optimizer = 'adadelta', loss = 'binary_crossentropy')
#autoencoder.fit(X_train, X_train, nb_epoch = 50, batch_size = 100, shuffle = True, validation_data = (X_test, X_test))

## THE ENCODER TO EXTRACT THE REDUCED DIMENSION FROM THE ABOVE AUTOENCODER
#encoder = Model(input = input_dim, output = encoded)
#encoded_input = Input(shape = (encoding_dim, ))
#encoded_out = encoder.predict(X_test)
##encoded_out[0:2]
#print(encoded_out)
plt.figure(figsize=(6, 6))
print(np.shape(data))
plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=y_test)
plt.show()
