import keras
from keras.models import load_model
import matplotlib.image as img
import numpy as np
import glob
import load_data
from sklearn.metrics import confusion_matrix

img_rows, img_cols = 123,41
n = img_rows * img_cols
model = load_model('rats_cnn.h5')
num_classes = 2
(x_train, y_train), (x_test, y_test) = load_data.load_data('datasets/test_set', 0.0)
y_test_cat = keras.utils.to_categorical(y_test, num_classes) 
y_pred = []
x_test_re = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
probs = model.predict_on_batch(x_test_re)

for i in range(0,probs.shape[0]):
    y_pred.append(np.argmax(probs[i]))

preds = keras.utils.to_categorical(y_pred, num_classes)
conf = confusion_matrix(y_test, y_pred)
print (conf)
rate = np.count_nonzero(y_test - y_pred, 0)
acc = (1 - (float(rate)/float(len(y_test)))) * 100
print ("accuracy: %s" % acc)
