from keras.models import load_model
import matplotlib.image as img
import numpy as np
import glob

img_rows, img_cols = 123,41
n = img_rows * img_cols
model = load_model('rats_cnn.h5')
data = glob.glob('datasets/untagged_set/*.png')
sounds = []
probs  = []
for image in data:
    current_image = img.imread(image)
    current_image = current_image.reshape(1, img_rows, img_cols, 1)
    prob = model.predict(current_image)
    if np.argmax(prob) == 0 and np.max(prob) >= 0.75:
        sounds.append(image)
with open('sounds.txt', 'w') as fp:
    for sound in sounds:
        print>>fp, sound
