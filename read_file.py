import matplotlib.pyplot as plt
import wave
import numpy as np
from scipy.fftpack import fft
from scipy import signal



fp = wave.open('../rattljud_kandidat_2015/data/red.wav', 'r')
sig = fp.readframes(-1)
data = np.fromstring(sig, dtype=np.int16)
fs = fp.getframerate()
data = data.astype(np.float64)
data = [(d+2**15)/(2**16-1) for d in data]

#NN = 2**9
#sigN = (len(data)/512)
#arr_data = np.array(data[0:sigN * 512]) #NN             (82329600) 
#data = np.reshape(arr_data,(NN,sigN))

#fy = fft(data)

f,t,Zxx = signal.stft(data,fs,nperseg=256)
b, a = signal.butter(4, 20000, 'high', analog=False)
w, h = signal.freqs(b, a)

