import matplotlib.pyplot as plt
import wave
import numpy as np
from scipy.fftpack import fft
from scipy import signal


b, a = signal.butter(4, 100, btype='low', analog=True)
w, h = signal.freqs(b, a) 


