import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wavfile

# Read audio file
fs, data = wavfile.read('original.wav')

# Time axis
t = np.linspace(0, len(data)/fs, len(data))

# Time domain plot (original)
plt.figure(1)
plt.plot(t, data)
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.title('Time Domain')

# FFT
dataf = np.fft.fft(data)
fd = abs(dataf)

# Half spectrum
x = fd[0:int(len(fd)/2)-1]
faxis = np.linspace(0, fs/2, len(x))

# Frequency domain plot (original)
plt.figure(2)
plt.xlabel('Frequency')
plt.ylabel('Amplitude in dB')
plt.title('Frequency Domain')
plt.xscale('log')
plt.plot(faxis, 20*np.log10(x/len(data)))

# Frequency indices
k1 = int(len(dataf)/fs * 1000)
k2 = int(len(dataf)/fs * 10000)

n1 = int(len(dataf)/fs * 0.1)
n2 = int(len(dataf)/fs * 90)
n3 = int(len(dataf)/fs * 10001)
n4 = int(len(dataf)/fs * 25000)

# Noise reduction
dataf[n3:n4] /= 60
dataf[len(dataf)-n4:len(dataf)-n3] /= 60
dataf[n1:n2] /= 60
dataf[len(dataf)-n2:len(dataf)-n1] /= 60

# Harmonic amplification
dataf[k1:k2] *= 10
dataf[len(dataf)-k2:len(dataf)-k1] *= 10

# Improved frequency domain
plt.figure(3)
plt.plot(faxis, 20*np.log10(dataf[0:int(len(dataf)/2)-1] / len(dataf)))
plt.xlabel('Frequency')
plt.ylabel('Amplitude in dB')
plt.title('Improved Frequency Domain')
plt.xscale('log')

# IFFT (back to time domain)
enhanced = np.fft.ifft(dataf)
clr = np.real(enhanced)

# Convert to audio format
audio = clr.astype(np.int16)

# Improved time domain
plt.figure(4)
plt.plot(t, clr)
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.title('Improved Time Domain')

plt.show()

# Save enhanced audio
wavfile.write('improved.wav', fs, audio)