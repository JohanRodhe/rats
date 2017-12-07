import matplotlib.pyplot as plt
import numpy as np
import glob
import matplotlib.image as mpimg

images = glob.glob('datasets/green/positive_set/*.png')
i = 0
for im in images:
    img = mpimg.imread(im)
    fig = plt.figure()
    plt.ion()
    plt.imshow(img)
    plt.show()
    x = plt.ginput(1)
    plt.close(fig)
    i += 1
    print ("clicked", x)
    print (i, len(images))
    if x[0][1] > 50:
        print("short")
        mpimg.imsave('datasets/green/short/short_%s.png' % i, img)
    else:
        print("long")
        mpimg.imsave('datasets/green/long/long_%s.png' % i, img)
