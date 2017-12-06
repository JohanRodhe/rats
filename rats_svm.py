import numpy as np
from sklearn import svm
from load_data import load_data

#load the data
(x_train, y_train) = load_data('datasets/red')
(x_test, y_test) = load_data('datasets/green')
clf = svm.NuSVC()
clf.fit(x_train, y_train)
accuracy = clf.score(x_test, y_test)
print "accuracy=%f" % accuracy
