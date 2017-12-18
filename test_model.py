import keras
from keras.models import load_model
from keras.utils import plot_model
import matplotlib.image as img
import matplotlib.pyplot as plt
import numpy as np
import glob
import load_data
from sklearn.metrics import confusion_matrix
import itertools

img_rows, img_cols = 123,41
n = img_rows * img_cols
model = load_model('rats_cnn.h5')
num_classes = 2
(x_test, y_test) = load_data.load_data('datasets/green')
y_pred = []
x_test_re = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
probs = model.predict_on_batch(x_test_re)
for i in range(0,probs.shape[0]):
    y_pred.append(np.argmax(probs[i]))
# Show wrong examples:
#    if y_pred[i] != y_test[i]:
#        plt.figure()
#        plt.imshow(x_test_re[i,:,:,0], cmap='gray')
#        print ('true label: %s ' % y_test[i], ' pred label: %s ' % y_pred[i])
#        plt.show()

cm = confusion_matrix(y_test, y_pred)
rate = np.count_nonzero(y_test - y_pred, 0)
acc = (1 - (float(rate)/float(len(y_test)))) * 100
print (acc)
#plot_model(model,show_shapes=True, to_file='model.png')

class_names = ['Not sound','Sound']
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
plt.show()
