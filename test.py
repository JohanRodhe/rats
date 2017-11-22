import wave
import numpy as np
from scipy.fftpack import fft
from scipy.ndimage import filters
import matplotlib.pyplot as plt
from scipy import signal
#import peakutils as pu
from detect_peaks import detect_peaks

fp =  wave.open('data/red.wav', 'r')
data_str = fp.readframes(-1)
fs = fp.getframerate()
# the sound data is in 16bit PCM, hence the np.int16
data = np.fromstring(data_str, dtype=np.int16)
data = data.astype(np.float64)
data = [d/(2**15) for d in data]
N = fp.getnframes()
numsamples = 2**20
# see if we need to zero pad
if N / numsamples != 0:
    pad = np.ceil(float(N) / numsamples) * numsamples - N
    padding = np.zeros((int((pad)),))
    data = np.concatenate((data,padding))
data_matrix = np.reshape(data, (numsamples, int((N+pad) / numsamples)))
nyq = float(0.5 * fs)
print "nyq=%f" % nyq
stop_low = 20000.0 / nyq
print "stop_low=%f" % stop_low
low = 25000.0 / nyq
print "low=%f" % low
#high = 100000.0 / nyq
#print "high=%f" % high
#stop_high = 115000 / nyq
#print "stop_high=%f" % stop_high
#N_filt,wn = signal.buttord(wp=[low, high], ws=[stop_low, stop_high], gpass=0.1, gstop=30.0)
N_filt,wn = signal.buttord(wp=low, ws=stop_low, gpass=0.1, gstop=60.0)
b,a = signal.butter(N_filt, wn, 'highpass')
filt_sig = np.zeros(np.shape(data_matrix))
peaks = []
for i in range(data_matrix.shape[1]):
    filt_sig[:,i] = signal.lfilter(b,a, data_matrix[:,i])
    filt_sig[:,i] = np.square(filt_sig[:,i])
    filt_sig[:,i] = filters.gaussian_filter1d(filt_sig[:,i],3)
    #peakind = pu.indexes(filt_sig[:,i], thres=0.0002, min_dist=8000)
    peakind = detect_peaks(filt_sig[:,i], mph=0.0002, mpd=8000)
    [peaks.append(idx + i*numsamples) for idx in peakind]
# look at spectrogram for the peaks
print "#peaks: %d" % len(peaks)
zoomWidth = 2**15
for i in range(len(peaks)):
    cursor = int(peaks[i] - round(zoomWidth / 2))
    if cursor < zoomWidth / 2 or cursor > N:
        continue
    cur_w = data[cursor:cursor+zoomWidth]
    print np.shape(cur_w)
    f,t,Sxx = signal.spectrogram(cur_w, fs, window=('hamming'), noverlap=2**15/256)
    plt.pcolormesh(t,f,Sxx)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()
#b,a = signal.butter(4, 0.16, btype='high', analog=False)
#w,h = signal.freqz(b,a, worN=2000)
#plt.plot(w / np.pi, 20*np.log10(abs(h)))
#plt.semilogx(w, 20 * np.log10(abs(h)))
#plt.xlabel('Frequency')
#plt.ylabel('Gain')
#plt.ylabel('Amplitude response [dB]')
#plt.grid()
#plt.show()
#plt.pcolormesh(t,f,np.log(np.abs(Zxx)))
#plt.title('STFT Magnitude')
#plt.ylabel('Frequency [Hz]')
#plt.xlabel('Time [sec]')
#plt.show()
#fft_data = fft(buffer(data, 2**9))
#fft_len = len(fft_data)/2
#plt.plot(np.log(abs(fft_data[:(fft_len-1)])),'r')
#plt.show()
