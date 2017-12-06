import numpy as np
import matplotlib.image as img
import matplotlib.pyplot as plt

img_rows,img_cols = 123,41
with open('sounds.txt','r') as fp:
    sounds = [sound.strip() for sound in fp.readlines()]
y =  np.zeros((img_rows*img_cols, len(sounds)))
for k in range(len(sounds)):
    y[:,k] = np.reshape(img.imread(sounds[k]), (img_rows * img_cols,))
y = ((y - np.mean(y)) / np.sqrt(y.shape[0])).T
U,s,V = np.linalg.svd(y)
s = np.square(s)
#sigma = V * np.diag(np.square(s)) * V.T
W = np.concatenate((V[0].reshape(V.shape[0],1),V[1].reshape(V.shape[0],1)), axis=1)
T = np.dot(y,W).T
print T.shape
plt.plot(T[0,:], T[1,:], 'ro')
plt.axis([-0.2, 0.2, -0.2, 0.2])
plt.show()
