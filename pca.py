import numpy as np
import matplotlib.image as img
import matplotlib.pyplot as plt

img_rows,img_cols = 123,41
with open('sounds.txt','r') as fp:
    sounds = [sound.strip() for sound in fp.readlines()]
y =  np.zeros((img_rows*len(sounds), img_cols))
for k in range(len(sounds)):
    y[k*img_rows:k*img_rows+img_rows,:] = img.imread(sounds[k])
y = ((y - np.mean(y)) / np.sqrt(y.shape[0]))
U,s,V = np.linalg.svd(y)
#s = np.square(s)
##sigma = V * np.diag(np.square(s)) * V.T
W = np.concatenate((V[0].reshape(V.shape[0],1),V[1].reshape(V.shape[0],1), V[2].reshape(V.shape[0],1)), axis=1)
print W.shape
T = np.dot(y,W)
plt.plot(T[:,0], T[:,1],T[:,2], 'ro')
plt.axis([-0.2, 0.2, -0.2, 0.2,-0.2,0.2])
plt.show()
