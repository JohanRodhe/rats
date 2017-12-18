from numpy.random import seed
from sklearn.preprocessing import minmax_scale
#from sklearn.model_selection import train_test_split
from keras.layers import Input, Dense
from keras.models import Model
import glob
import load_data
import matplotlib.pyplot as plt
import numpy as np 
# this is the size of our encoded representations
#encoding_dim = 32  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats
#batch_size = 128
#original_dim = 5043
#latent_dim = 2
#intermediate_dim = 128
#
## this is our input placeholder
#input_img = Input(shape=(784,))
## "encoded" is the encoded representation of the input
#encoded = Dense(encoding_dim, activation='relu')(input_img)
## "decoded" is the lossy reconstruction of the input
#decoded = Dense(784, activation='sigmoid')(encoded)
#
## this model maps an input to its reconstruction
#autoencoder = Model(input_img, decoded)
#encoder = Model(input_img, encoded)
#encoded_input = Input(shape=(encoding_dim,))
#autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
#(x_train, y_train), (x_test, y_test) = load_data.load_data('datasets',0.8)### AN EXAMPLE OF SIMPLE AUTOENCODER ###
#x_train = x_train.astype('float32') / 255.
#x_test = x_test.astype('float32') / 255.
#x_train = np.reshape(x_train, (len(x_train), 123, 41, 1))  # adapt this if using `channels_first` image data format
#x_test = np.reshape(x_test, (len(x_test), 123, 41, 1))  # adapt this if using `channels_first` image data format
##data = glob.glob('datasets/test_set/positive_set/*.png')
##x_train = x_train.astype('float32') / 255.
##x_test = x_test.astype('float32') / 255.
##x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
##x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
#x = Input(batch_shape=(batch_size, original_dim))
#h = Dense(intermediate_dim, activation='relu')(x)
#z_mean = Dense(latent_dim)(h)
##autoencoder.fit(x_train, x_train, epochs= 50, batch_size=128, shuffle=True,
##                validation_data=(x_test,x_test))
## InputLayer (None, 10)
##      Dense (None, 5
##      Dense (None, 10)
##ncol = len(data)
##print(X_train.shape)
##print(X_test.shape)
##input_dim = Input(shape = (ncol, ))
## DEFINE THE DIMENSION OF ENCODER ASSUMED 3
##encoding_dim = 2
#
### DEFINE THE ENCODER LAYER
##encoded = Dense(encoding_dim, activation = 'relu')(input_dim)
#
### DEFINE THE DECODER LAYER
##decoded = Dense(ncol, activation = 'sigmoid')(encoded)
##encoder = Model(x, z_mean)
#x_test_encoded = encoder.predict(x_test)
### COMBINE ENCODER AND DECODER INTO AN AUTOENCODER MODEL
##autoencoder = Model(input = input_dim, output = decoded)
#
### CONFIGURE AND TRAIN THE AUTOENCODER
##autoencoder.compile(optimizer = 'adadelta', loss = 'binary_crossentropy')
##autoencoder.fit(X_train, X_train, nb_epoch = 50, batch_size = 100, shuffle = True, validation_data = (X_test, X_test))
#
### THE ENCODER TO EXTRACT THE REDUCED DIMENSION FROM THE ABOVE AUTOENCODER
##encoder = Model(input = input_dim, output = encoded)
##encoded_input = Input(shape = (encoding_dim, ))
##encoded_out = encoder.predict(X_test)
###encoded_out[0:2]
##print(encoded_out)
##print(np.shape(data))
##x_test_encoded = encoder.predict(x_test, batch_size=batch_size)
#print(np.shape(x_test_encoded))
##plt.figure(figsize=(6, 6))
##plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=y_test)
##plt.colorbar()
##plt.show()



### AN EXAMPLE OF DEEP AUTOENCODER WITH MULTIPLE LAYERS
# InputLayer (None, 10)
#      Dense (None, 20)
#      Dense (None, 10)
#      Dense (None, 5)
#      Dense (None, 3)
#      Dense (None, 5)
#      Dense (None, 10)
#      Dense (None, 20)
#      Dense (None, 10)
 

#(x_test, y_test) = load_data.load_data('datasets/green', 0.8)
(x_train, y_train), (x_test, y_test) = load_data.load_data('datasets/green', 0.8)
### AN EXAMPLE OF SIMPLE AUTOENCODER ###
(x_test2, y_test2) = load_data.load_data('datasets/red')
print (np.shape(x_train))
print (np.shape(x_train))
sX = minmax_scale(x_train, axis = 0)
ncol = sX.shape[1]
ncol2 = x_train.shape[1]
input_dim = Input(shape = (ncol, ))
# DEFINE THE DIMENSION OF ENCODER ASSUMED 3
encoding_dim = 2
# DEFINE THE ENCODER LAYER
encoded = Dense(encoding_dim, activation = 'relu')(input_dim)
# DEFINE THE DECODER LAYER
decoded = Dense(ncol, activation = 'sigmoid')(encoded)
# COMBINE ENCODER AND DECODER INTO AN AUTOENCODER MODEL
autoencoder = Model(input = input_dim, output = decoded)
# CONFIGURE AND TRAIN THE AUTOENCODER
autoencoder.compile(optimizer = 'adadelta', loss = 'binary_crossentropy')
autoencoder.fit(x_train, x_train, nb_epoch = 200, batch_size = 100, shuffle = True, validation_data = (x_test, x_test))
# DEFINE THE ENCODER LAYERS
# CONFIGURE AND TRAIN THE AUTOENCODER
autoencoder.compile(optimizer = 'adadelta', loss = 'binary_crossentropy')
#autoencoder.fit(x_train, x_train, nb_epoch = 400, batch_size = 128, shuffle = True, validation_data = (x_test, x_test))
# THE ENCODER TO EXTRACT THE REDUCED DIMENSION FROM THE ABOVE AUTOENCODER
encoder = Model(input = input_dim, output = encoded)
encoded_input = Input(shape = (encoding_dim, ))
x_test_encoded = encoder.predict(x_test2, batch_size=100)
print(x_test_encoded.shape)
plt.figure(figsize=(6, 6))
plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=y_test2)
plt.colorbar()
print(ncol)
print(ncol2)
plt.show()

