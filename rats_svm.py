import numpy as np
from sklearn.svm import SVC
from load_data import load_data

#load the data
(x_train, y_train) = load_data('datasets/red', 1.0)
(x_test, y_test) = load_data('datasets/green', 1.0)
clf = SVC()
clf.fit(x_train, y_train)
accuracy = clf.score(x_test, y_test)
print "accuracy=%f" % accuracy
