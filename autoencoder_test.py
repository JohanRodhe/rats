from numpy.random import seed
#from sklearn.preprocessing import minmax_scale
#from sklearn.model_selection import train_test_split
from keras.layers import Input, Dense
from keras.models import Model
import load_data
 
#df = read_csv("credit_count.txt")
#Y = df[df.CARDHLDR == 1].DEFAULTS
#X = df[df.CARDHLDR == 1].ix[:, 2:12]
## SCALE EACH FEATURE INTO [0, 1] RANGE
#sX = minmax_scale(X, axis = 0)
#ncol = sX.shape[1]
#X_train, X_test, Y_train, Y_test = train_test_split(sX, Y, train_size = 0.5, random_state = seed(2017))
(X_train, Y_train), (X_test, Y_test) = load_data.load_data('datasets',0.5)### AN EXAMPLE OF SIMPLE AUTOENCODER ###
# InputLayer (None, 10)
#      Dense (None, 5
#      Dense (None, 10)
ncol = X_train.shape[1]
print(X_train.shape)
print(X_test.shape)
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
autoencoder.fit(X_train, X_train, nb_epoch = 50, batch_size = 100, shuffle = True, validation_data = (X_test, X_test))
# THE ENCODER TO EXTRACT THE REDUCED DIMENSION FROM THE ABOVE AUTOENCODER
encoder = Model(input = input_dim, output = encoded)
encoded_input = Input(shape = (encoding_dim, ))
encoded_out = encoder.predict(X_test)
#encoded_out[0:2]
print(encoded_out)
