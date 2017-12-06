import numpy as np
from sklearn import svm
from load_data import load_data
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt

#load the data
(x_train, y_train) = load_data('datasets/red')
(x_test, y_test) = load_data('datasets/green')
clf = svm.NuSVC()
clf.fit(x_train, y_train)
accuracy = clf.score(x_test, y_test)
print "accuracy=%f" % accuracy

# for conf matrix
y_pred = clf.predict(x_test)
print (y_pred)
cm = confusion_matrix(y_test, y_pred)
#print (cm)
rate = np.count_nonzero(y_test - y_pred, 0)
acc = (1 - (float(rate)/float(len(y_test)))) * 100

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
plt.show()
