# Rats
In this project the goal is to study sonograms of rat speech in order to separate communication between rats from background noise. Since rats speak in an ultrasound range we can discard any frequencies below $20kHz$.
Once such communication can be successfully discerned, there is an interest in clustering the communication samples to try and glean some information on how rats might communicate. Examples of such communication might be different "words" for different situations. For the detection part, we implement a Support Vector Machine (SVM) based classifier as well as one based on a Convolutional Neural Network (CNN) and compared the two.

# CNN
Accuracy: 90.2%
![](https://github.com/JohanRodhe/rats/blob/master/plots/confmat_cnn.png)
# SVM
Accuracy: 85.0%
![](https://github.com/JohanRodhe/rats/blob/master/plots/confmat_svm.png)
