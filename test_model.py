import keras
import itertools
from keras.models import load_model
from keras.utils import plot_model
import matplotlib.image as img
import matplotlib.pyplot as plt
import numpy as np
import glob
import load_data
from sklearn.metrics import confusion_matrix

img_rows, img_cols = 123,41
n = img_rows * img_cols
model = load_model('rats_cnn.h5')
num_class_names = 2
(x_train, y_train), (x_test, y_test) = load_data.load_data('datasets/test_set', 0.0)
y_test_cat = keras.utils.to_categorical(y_test, num_class_names) 
y_pred = []
x_test_re = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
probs = model.predict_on_batch(x_test_re)

for i in range(0,probs.shape[0]):
    y_pred.append(np.argmax(probs[i]))

preds = keras.utils.to_categorical(y_pred, num_class_names)
cm = confusion_matrix(y_test, y_pred)
#print (cm)
rate = np.count_nonzero(y_test - y_pred, 0)
acc = (1 - (float(rate)/float(len(y_test)))) * 100

#plot_model(model,show_shapes=True, to_file='model.png')

class_names = ['Sound','Not sound']
title = 'Confusion matrix'
cmap = plt.cm.Blues
plt.figure()
plt.imshow(cm, interpolation='nearest', cmap=cmap)
plt.title(title)
plt.colorbar()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names, rotation=45)
plt.yticks(tick_marks, class_names)
normalize = False
fmt = '.2f' if normalize else 'd'
thresh = cm.max() / 2.
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, format(cm[i, j], fmt),
        horizontalalignment="center",
        color="white" if cm[i, j] > thresh else "black")

plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
#print("HEJ")
plt.show()
