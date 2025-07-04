# import wave
# import numpy as np
# import pylab as pl

# fw = wave.open('Snare.wav','rb')

# params = fw.getparams()
# nchannels, sampwidth, framerate, nframes = params[:4]
# strData = fw.readframes(nframes)
# waveData = np.frombuffer(strData, dtype=np.int16)
# waveData = waveData * 1.0 / max(abs(waveData))
# fw.close()

# Data = waveData.tolist()

# pl.subplot(111)
# pl.plot(np.arange(len(Data)), Data, 'g')
# pl.grid()
# pl.show()

# from scipy.io.wavfile import write

# Data = Data[:20000]

# waveform = np.array(Data)
# scaled = np.int16(waveform / np.max(np.abs(waveform)) * 32767) 
# write('Cropped1.wav', 44100, scaled)

# import pyautogui
# import random

# pyautogui.hotkey('ctrl', 'win', 'd')

# while True: 

#     try: 
#         pyautogui.moveTo(random.random() * 1000 + 500, random.random() * 1000 + 500)
        
#         pyautogui.dragTo(random.random() * 1000 + 500, random.random() * 1000 + 500, duration=0.0001, button='left')

#     except Exception:

#         pyautogui.moveTo(random.random() * 1000 + 500, random.random() * 1000 + 500)
#         # pyautogui.dragTo(random.random() * 1000 + 500, random.random() * 1000 + 500, duration=0.0001, button='left')

# import numpy as np
# SoundArray = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
# Wave = [3, 3]
# SoundArray[2:4] += Wave

# a = np.array([1,2,3])
# a += [1] * 4; print(a)

# print(SoundArray)


# if (end < len(SoundArray)):

#             SoundArray[start:end] += Wave
#             print('1')

#         elif (start < len(SoundArray)):

#             SoundArray[start:] += Wave[:len(SoundArray) - start]
#             SoundArray.extend(Wave[len(SoundArray) - start:])
#             print('2')

#         else:

#             SoundArray.extend([0] * (start - len(SoundArray)))
#             SoundArray.extend(Wave)
#             print('3')

# import numpy as np
# import librosa
# import matplotlib.pyplot as plt

# # Load audio file
# y, sr = librosa.load('FluteSolo.wav')

# # Compute the Short-Time Fourier Transform (STFT)
# D = librosa.stft(y)

# # Convert amplitude to decibels
# S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

# # Plot the spectrogram
# plt.figure(figsize=(10, 4))
# librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='log')
# plt.colorbar()
# plt.title('Spectrogram')
# plt.tight_layout()
# plt.show()

# # Extract acoustic descriptors
# spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
# print(type(spectral_centroid))
# print(f'{spectral_centroid}')

# import numpy as np
# import librosa
# import matplotlib.pyplot as plt
# from scipy.signal import find_peaks

# # Load the audio file
# y, sr = librosa.load('FluteSolo.wav')

# # Compute the Short-Time Fourier Transform (STFT)
# D = librosa.stft(y)
# S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

# # Compute the mean spectrum
# mean_spectrum = np.mean(np.abs(D), axis=1)

# # Find the peaks in the spectrum
# peaks, _ = find_peaks(mean_spectrum, height=np.max(mean_spectrum) * 0.1)

# # Extract the frequencies and their amplitudes
# frequencies = librosa.fft_frequencies(sr=sr, n_fft=D.shape[0])
# peak_frequencies = frequencies[peaks]
# peak_amplitudes = mean_spectrum[peaks]

# # Calculate the ratios of harmonics
# fundamental_frequency = peak_frequencies[0]
# harmonic_ratios = peak_amplitudes / peak_amplitudes[0]

# # Print the results
# print("Fundamental Frequency:", fundamental_frequency)
# print("Harmonic Ratios:", harmonic_ratios)

# # Plot the spectrum with peaks highlighted
# plt.figure(figsize=(10, 6))
# plt.plot(frequencies, mean_spectrum)
# plt.plot(peak_frequencies, peak_amplitudes, 'ro')
# plt.title('Spectrum with Harmonic Peaks')
# plt.xlabel('Frequency (Hz)')
# plt.ylabel('Amplitude')
# plt.grid(True)
# plt.show()

# import numpy as np

# v = [1, 0.85805565, 0.25376666, 1.046945, 0.95703995, 0.1081515,
#  0.11217093, 0.10888793, 0.1345883, 0.12712286, 0.12303026, 0.19973917,
#  0.22603528, 0.48927325, 0.4070098, 0.4201747, 0.12063531, 0.11017737,
#  0.13960934, 0.11921344]

# vv = np.sum(v)
# for vi in range(0, len(v)): v[vi] /= vv
# print(v)

# import numpy as np
# import librosa
# import soundfile as sf
# from scipy.signal import find_peaks

# # Load the audio file
# y, sr = librosa.load('FluteSolo.wav')

# # Perform harmonic-percussive source separation
# y_harmonic, _ = librosa.effects.hpss(y)

# # Perform a high-resolution STFT
# D = librosa.stft(y_harmonic, n_fft=4096, hop_length=512)

# # Extract magnitude and phase
# magnitude, phase = np.abs(D), np.angle(D)

# # Identify harmonic peaks
# frequencies = librosa.fft_frequencies(sr=sr, n_fft=4096)
# mean_spectrum = np.mean(magnitude, axis=1)
# peaks, _ = find_peaks(mean_spectrum, height=np.max(mean_spectrum) * 0.1)
# peak_frequencies = frequencies[peaks]
# peak_amplitudes = mean_spectrum[peaks]

# # Resynthesize the sound using the harmonic components
# y_resynth = np.zeros_like(y_harmonic)
# for i, peak in enumerate(peaks):
#     resynth_component = peak_amplitudes[i] * np.cos(2 * np.pi * peak_frequencies[i] * np.arange(len(y_harmonic)) / sr)
#     y_resynth += resynth_component

# # Save the resynthesized sound
# sf.write('resynthesized_sound.wav', y_resynth, sr)

# print("Resynthesized sound saved to 'resynthesized_sound.wav'")

# import numpy as np
# from wave import open

# path = r'C:\Users\studi\OneDrive\桌面\PyStuff\output.wav'

# fw = open(path, 'rb')
# params = fw.getparams()
# nchannels, sampwidth, framerate, nframes = params[:4]
# strData = fw.readframes(nframes)
# waveData = np.frombuffer(strData, dtype=np.int16)
# waveData = waveData * 1.0 / max(abs(waveData))
# fw.close()

# import random

# waveData = waveData.tolist()
# for idx in range(0, len(waveData)): waveData[idx] += (random.random() / 10)

# from scipy.io.wavfile import write

# filename = "Noisy.wav"
# waveform = np.array(waveData)
# scaled = np.int16(waveform / np.max(np.abs(waveform)) * 32767)
# write(filename, framerate, scaled)
# print(f"Audio saved to {filename}")

# import pyaudio
# import time

# while True:

#     FORMAT = pyaudio.paInt16
#     CHANNELS = 1
#     RATE = 44100
#     CHUNK = 1024

#     timeStart = time.time()
#     print(timeStart, end = " ---> ")

#     audio = pyaudio.PyAudio()
#     stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

#     timeEnd = time.time()
#     print(timeEnd)
#     print(f"\n\n --- Open warmup: {timeEnd - timeStart}\n\n")

#     stream.stop_stream()
#     stream.close()
#     audio.terminate()

#     time.sleep(1)

# import random
# import numpy as np

# tt = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# print(np.nan + 4)

# while True:

#     AMP = [random.random(), random.random(), random.random(), random.random(), random.random(), random.random(), random.random(), random.random(), random.random(), random.random(), random.random(), random.random(), random.random(), random.random(), random.random(), random.random(), random.random(), random.random(), random.random(), random.random(), ]        
    
#     for idx in range(0, 20): 
        
#         tt[idx] += AMP[idx]
#         print(tt[idx], end = " & ")
#         print(AMP[idx])

# import numpy as np
# import time

# size = 100000000

# s = np.array([])
# t = np.linspace(1, size, size)

# t_start = time.time()
# s = s.tolist()
# for idx in range(0, size, size // 1000): s.extend(t[idx:idx+100000])
# s = np.array(s)
# t_end = time.time()

# print(s)
# print(t_end)
# print(t_start)
# print(t_end - t_start)

# s = np.array([])
# t = np.linspace(1, size, size)

# t_start = time.time()
# for idx in range(0, size, size // 1000): s = np.append(s, t[idx:idx+100000])
# t_end = time.time()

# print(s)
# print(t_end)
# print(t_end)
# print(t_start)
# print(t_end - t_start)

# import numpy as np
# import pylab as pl
# import random as rd
# from scipy.signal import find_peaks
# from scipy.fft import fft, fftfreq
# from scipy.signal.windows import hann, hamming, blackman, flattop

# multi_Size = 1 / 4
# multi_Zero = 0
# frameSize = int(2048 * multi_Size)
# zerosFrame = int(2048 * multi_Zero)
# waveData0 = np.zeros(frameSize)
# frequency = 163.33
# framerate = 44100
# AMP = [0.2, 0.1, 0.1, 0, 0.1, 0, 0.1, 0, 0, 0, 0.1, 0, 0.1, 0, 0, 0, 0.1, 0, 0.1, 0]

# for idx in range(0, 20): 
    
#     sinX = np.linspace(0, 2 * np.pi * frequency * (idx + 1) / (framerate / frameSize), frameSize)
#     waveData0 += np.sin(sinX) * AMP[idx] 

# temp = waveData0[:]

# for idx in range(0, frameSize - 1):

#     value = rd.random() * 5

#     if value > 2.2515 and value < 2.7038: 
        
#         value *= 2
#         waveData0[idx] += value; waveData0[idx + 1] -= value
    
    
# waveData0 = np.append(waveData0, [0] * zerosFrame)

# WindowFilter1 = hann(frameSize + zerosFrame)
# WindowFilter2 = hamming(frameSize + zerosFrame)
# WindowFilter3 = blackman(frameSize + zerosFrame)
# # WindowFilter4 = flattop(frameSize + zerosFrame)

# pl.subplot(221)

# pl.plot(np.arange(len(WindowFilter1)), WindowFilter1, label = "Hann", color = "r")
# pl.plot(np.arange(len(WindowFilter2)), WindowFilter2, label = "Hamming", color = "c")
# pl.plot(np.arange(len(WindowFilter3)), WindowFilter3, label = "Blackman", color = "b")
# # pl.plot(np.arange(len(WindowFilter4)), WindowFilter4, label = "Flat-Top", color = "g")

# pl.legend()
# pl.grid()

# waveData1 = waveData0 * WindowFilter1
# waveData2 = waveData0 * WindowFilter2
# waveData3 = waveData0 * WindowFilter3
# # waveData4 = waveData0 * WindowFilter4

# pl.subplot(222)

# pl.plot(np.arange(len(waveData0)), waveData0, label = "NoFilter", color = "black")
# pl.plot(np.arange(len(waveData1)), waveData1, label = "Hann", color = "r")
# pl.plot(np.arange(len(waveData2)), waveData2, label = "Hamming", color = "c")
# pl.plot(np.arange(len(waveData3)), waveData3, label = "Blackman", color = "b")
# # pl.plot(np.arange(len(waveData4)), waveData4, label = "Flat-Top", color = "g")

# pl.legend()
# pl.grid()

# x = fftfreq(frameSize, 1 / framerate)[:frameSize//2]

# fft0 = fft(waveData0)
# fft1 = fft(waveData1)
# fft2 = fft(waveData2)
# fft3 = fft(waveData3)
# # fft4 = fft(waveData4)

# y0 = 2.0 / frameSize * np.abs(fft0)[:frameSize//2]
# y1 = 2.0 / frameSize * np.abs(fft1)[:frameSize//2]
# y2 = 2.0 / frameSize * np.abs(fft2)[:frameSize//2]
# y3 = 2.0 / frameSize * np.abs(fft3)[:frameSize//2]
# # y4 = 2.0 / frameSize * np.abs(fft4)[:frameSize//2]

# x0, _ = find_peaks(y0, height = 0.0100)
# x1, _ = find_peaks(y1, height = 0.0100)
# x2, _ = find_peaks(y2, height = 0.0100)
# x3, _ = find_peaks(y3, height = 0.0100)
# # x4, _ = find_peaks(y4, height = 0.0100)

# pl.subplot(223)

# pl.plot(x, y0, label = "NoFilter", color = "black")
# pl.plot(x, y1, label = "Hann", color = "r")
# pl.plot(x, y2, label = "Hamming", color = "c")
# pl.plot(x, y3, label = "Blackman", color = "b")
# # pl.plot(x, y4, label = "Flat-Top", color = "g")

# pl.legend()
# pl.grid()

# pl.subplot(224)

# reAMP = [0.270, 0.38, 0.05, 0.20, 0.2, 0.3, 0.4, 0, 0.1, 0,0,0,0,0,0,0,0,0,0,0]

# base0 = x0[0] * framerate / frameSize
# base1 = x1[0] * framerate / frameSize
# base2 = x2[0] * framerate / frameSize
# base3 = x3[0] * framerate / frameSize

# print(base0)
# print(base1)
# print(base2)
# print(base3)

# wave = np.zeros(frameSize)

# for idx in range(0, 20): 
    
#     sinX = np.linspace(0, 2 * np.pi * frequency * (idx + 1) / (framerate / frameSize), frameSize)
#     wave += np.sin(sinX) * reAMP[idx] 

# wave0 = np.zeros(frameSize)

# for idx in range(0, 20): 
    
#     sinX = np.linspace(0, 2 * np.pi * base0 * (idx + 1) / (framerate / frameSize), frameSize)
#     wave0 += np.sin(sinX) * reAMP[idx] 

# wave1 = np.zeros(frameSize)

# for idx in range(0, 20): 
    
#     sinX = np.linspace(0, 2 * np.pi * base1 * (idx + 1) / (framerate / frameSize), frameSize)
#     wave1 += np.sin(sinX) * reAMP[idx] 

# wave2 = np.zeros(frameSize)

# for idx in range(0, 20): 
    
#     sinX = np.linspace(0, 2 * np.pi * base2 * (idx + 1) / (framerate / frameSize), frameSize)
#     wave2 += np.sin(sinX) * reAMP[idx] 

# wave3 = np.zeros(frameSize)

# for idx in range(0, 20): 
    
#     sinX = np.linspace(0, 2 * np.pi * base3 * (idx + 1) / (framerate / frameSize), frameSize)
#     wave3 += np.sin(sinX) * reAMP[idx] 

# pl.plot(np.arange(len(wave)), wave, label = "Original", color = "purple")
# pl.plot(np.arange(len(wave0)), wave0, label = "NoFilter", color = "black")
# pl.plot(np.arange(len(wave1)), wave1, label = "Hann", color = "r")
# pl.plot(np.arange(len(wave2)), wave2, label = "Hamming", color = "c")
# pl.plot(np.arange(len(wave3)), wave3, label = "Blackman", color = "b")

# pl.subplot(224)

# base0 = x0[0] * framerate / frameSize
# error = framerate / (frameSize + zerosFrame)
# amp0 = np.array([])

# for idx in range(0, 20): 
    
#     overtone = base0 * (idx + 1)
#     interval = [overtone - error, overtone + error]
#     indices = np.where((x >= interval[0]) & (x <= interval[1]))
#     summation = np.sum(y0[indices])
#     amp0 = np.append(amp0, summation)

# amp0 /= np.sum(amp0); print(amp0, end = "\n")
# amp0 -= AMP

# base1 = x1[0] * framerate / frameSize
# error = framerate / (frameSize + zerosFrame)
# amp1 = np.array([])

# for idx in range(0, 20): 
    
#     overtone = base1 * (idx + 1)
#     interval = [overtone - error, overtone + error]
#     indices = np.where((x >= interval[0]) & (x <= interval[1]))
#     summation = np.sum(y1[indices])
#     amp1 = np.append(amp1, summation)

# amp1 /= np.sum(amp1); print(amp1, end = "\n")
# amp1 -= AMP

# base2 = x2[0] * framerate / frameSize
# error = framerate / (frameSize + zerosFrame)
# amp2 = np.array([])

# for idx in range(0, 20): 
    
#     overtone = base2 * (idx + 1)
#     interval = [overtone - error, overtone + error]
#     indices = np.where((x >= interval[0]) & (x <= interval[1]))
#     summation = np.sum(y2[indices])
#     amp2 = np.append(amp2, summation)

# amp2 /= np.sum(amp2); print(amp2, end = "\n")
# amp2 -= AMP

# base3 = x3[0] * framerate / frameSize
# error = framerate / (frameSize + zerosFrame)
# amp3 = np.array([])

# for idx in range(0, 20): 
    
#     overtone = base3 * (idx + 1)
#     interval = [overtone - error, overtone + error]
#     indices = np.where((x >= interval[0]) & (x <= interval[1]))
#     summation = np.sum(y3[indices]) 
#     amp3 = np.append(amp3, summation)

# amp3 /= np.sum(amp3); print(amp3, end = "\n")
# amp3 -= AMP

# base4 = x4[0] * framerate / frameSize
# error = framerate / (frameSize + zerosFrame)
# amp4 = np.array([])

# for idx in range(0, 20): 
    
#     overtone = base4 * (idx + 1)
#     interval = [overtone - error, overtone + error]
#     indices = np.where((x >= interval[0]) & (x <= interval[1]))
#     summation = np.sum(y4[indices])
#     amp4 = np.append(amp4, summation)

# amp4 /= np.sum(amp4); print(amp4, end = "\n")
# amp4 -= AMP

# pl.plot(np.arange(len(amp0), step = 1), amp0, label = "NoFilter", color = "black")
# pl.plot(np.arange(len(amp1), step = 1), amp1, label = "Hann", color = "r")
# pl.plot(np.arange(len(amp2), step = 1), amp2, label = "Hamming", color = "c")
# pl.plot(np.arange(len(amp3), step = 1), amp3, label = "Blackman", color = "b")
# # pl.plot(np.arange(len(amp4), step = 1), amp4, label = "Flat-Top", color = "g")

# pl.xticks(np.arange(20, step = 1))
# pl.legend()
# pl.grid()

# def rating(frequency: float, arr1: np.ndarray, arr2: np.ndarray)-> float:

#     rate = 0
#     for idx in range(0, 20): rate += np.abs((arr1[idx] - arr2[idx]) / (1 - arr2[idx]) * (20 - idx) / 20)
#     rate += np.abs((frequency - 163.33) / 163.33)
#     return 100 - rate * 100

# print("%.2f" % rating(base0 * multi_Size / (multi_Size + multi_Zero), amp0, AMP), end = "%\n")
# print("%.2f" % rating(base1 * multi_Size / (multi_Size + multi_Zero), amp1, AMP), end = "%\n")
# print("%.2f" % rating(base2 * multi_Size / (multi_Size + multi_Zero), amp2, AMP), end = "%\n")
# print("%.2f" % rating(base3 * multi_Size / (multi_Size + multi_Zero), amp3, AMP), end = "%\n")
# # print("%.2f" % rating(base4 * multi_Size / (multi_Size + multi_Zero), amp4, AMP), end = "%\n")
# print("%.2f" % rating(163.33, AMP, AMP), end = "%\n")

# pl.show()

# amp0 = np.array([0.1407, 0.0954, 0.0911, 0, 0.1010, 0, 0.0692, 0, 0, 0, 0.0819, 0, 0.0665, 0, 0, 0, 0, 0.0751, 0, 0.0942]); amp0 /= np.sum(amp0)
# for idx in range(0, 20): amp0[idx] -= AMP[idx]
# amp1 = np.array([0.0895, 0.0490, 0.0482, 0, 0.0498, 0, 0.0495, 0, 0, 0, 0.0431, 0, 0.0462, 0, 0, 0, 0, 0.0449, 0, 0.0483]); amp1 /= np.sum(amp1)
# for idx in range(0, 20): amp1[idx] -= AMP[idx]
# amp2 = np.array([0.0936, 0.0527, 0.0516, 0, 0.0539, 0, 0.0534, 0, 0, 0, 0.0452, 0, 0.0491, 0, 0, 0, 0, 0.0473, 0, 0.0520]); amp2 /= np.sum(amp2)
# for idx in range(0, 20): amp2[idx] -= AMP[idx]
# amp3 = np.array([0.0771, 0.0412, 0.0409, 0, 0.0419, 0, 0.0417, 0, 0, 0, 0.0374, 0, 0.0395, 0, 0, 0, 0, 0.0386, 0, 0.0409]); amp3 /= np.sum(amp3)
# for idx in range(0, 20): amp3[idx] -= AMP[idx]

# rating0 = [0.4748, 0.4748, 0.4748, 0.4748, 0.4748, 0.4748, 0.4748,  0.4748,  0.4748,  0.4748]
# rating1 = [0.5466, 0.5511, 0.5524, 0.5575, 0.5700, 0.5608, 0.5694, -1.1986, -1.1989, -1.1989]
# rating2 = [0.5533, 0.5560, 0.5571, 0.5615, 0.5726, 0.5644, 0.5720, -0.9089, -3.8970, -3.8970]
# rating3 = [0.5252, 0.5287, 0.5299, 0.5350, 0.5477, 0.5384, 0.5470, -3.8035, -3.8035, -3.8035]

# x = np.arange(0, 100)[::-10]
# pl.plot(x, rating0, label = "NoFilter", color = "black")
# pl.plot(x, rating1, label = "Hann", color = "r")
# pl.plot(x, rating2, label = "Hamming", color = "c")
# pl.plot(x, rating3, label = "Blackman", color = "b")

# pl.legend()
# pl.grid()
# pl.gca().invert_xaxis()

# pl.show()

# import tkinter as tk
# import time

# start_time = None

# class CircularProgress:

#     def __init__(self, canvas, x, y, radius):

#         self.canvas = canvas
#         self.x = x
#         self.y = y
#         self.radius = radius
#         self.angle = 0
#         self.arc = canvas.create_arc(x - radius, y - radius, x + radius, y + radius, start=90, extent=0, style='arc', outline='cyan', width=5)

#         self.running = True

#         self.button_radius = 50
#         self.button = canvas.create_oval(x - self.button_radius, y - self.button_radius, x + self.button_radius, y + self.button_radius, fill='white', tags='button')
#         canvas.tag_bind('button', '<Button-1>', self.on_button_click)

#     def on_button_click(self, _):

#         self.running = not self.running

#         global start_time
#         start_time = time.time()

#         self.canvas.itemconfig(self.arc, extent=0)


#     def update(self, progress):

#         if self.running:

#             self.angle = 360 * progress
#             self.canvas.itemconfig(self.arc, extent=self.angle)

#         else:

#             self.angle = 0
#             self.canvas.itemconfig(self.arc, extent=self.angle)

# def play_track(duration = 1):

#     progress_indicator = CircularProgress(canva, 100, 100, 80)

#     start_time = time.time()

#     while True:

#         elapsed_time = (time.time() - start_time) % 1
#         progress = elapsed_time / duration
#         progress_indicator.update(progress)
#         root.update()
        

# root = tk.Tk()
# canva = tk.Canvas(root, width=200, height=200)
# canva.pack()

# # Example usage
# play_track(1)  # 1 second track

# root.mainloop()

# a = [1, 2, 3, 4, 23456, 7]
# print(a[0:2])

# import tkinter as tk
# import random
# import time

# START = "#C834A2"
# END = "#3400A2"

# VisualPositions = {

#     'c': (100, 100),
#     'v': (200, 100),
#     'b': (100, 200),
#     'n': (200, 200), 
# }

# TrackVisuals = {}

# ChannelDict = {
#     'c': None,
#     'v': None,
#     'b': None,
#     'n': None,
# }

# class CircularProgress:

#     def __init__(self, canva, x, y, radius, key, duration):
        
#         self.canva = canva
#         self.x = x
#         self.y = y
#         self.radius = radius
#         self.duration = duration
#         self.angle = 0
#         self.key = key
        
#         self.arc = self.canva.create_arc(
#             self.x - self.radius, self.y - self.radius, self.x + self.radius, self.y + self.radius,
#             start = 90, extent = 0, style = 'arc', outline = 'cyan', width = 5,
#         )
        
#     def draw(self, status):

#         print(f'Proceeding command {status}')

#         self.running = True
#         self.status = status
        
#         self.button_radius = self.radius * 1

#         self.bg_circle = self.canva.create_oval(
#             self.x - self.button_radius, self.y - self.button_radius,
#             self.x + self.button_radius, self.y + self.button_radius,
#             fill = self.status, outline = ''
#         )
        
#         self.button_radius = self.radius * 0.9

#         self.button = self.canva.create_oval(
#             self.x - self.button_radius, self.y - self.button_radius,
#             self.x + self.button_radius, self.y + self.button_radius,
#             fill='white', tags=f'button_{self.key}'
#         )
        
#         self.canva.tag_bind(f'button_{self.key}', '<Button-1>', self.toggle)
        
#     def update(self, progress):

#         if self.running: self.angle = 360 * progress % 360
#         else: self.angle = 0
        
#         self.canva.itemconfig(self.arc, extent = self.angle)
        
#     def toggle(self):
        
#         self.running = not self.running
#         print(f'self.running set to {self.running}')
#         self.angle = 0
#         self.canva.itemconfig(self.arc, extent = self.angle)

#         if self.running: 
            
#             self.draw(START)
#             animate(self)

            
#     def reset(self):

#         self.angle = 0
#         self.running = False
        
#         self.draw(END)

#         self.angle = 0
#         self.canva.itemconfig(self.arc, extent=self.angle)

# def animate(track: CircularProgress):

#     start_time = time.time()
    
#     def update():

#         elapsed = (time.time() - start_time) % track.duration
#         progress = elapsed / track.duration
#         track.update(progress)
#         if not track.running: return
#         # print('.')
#         track.canva.after(10, update)
    
#     update()

# def toggle_channel(key):
    
#     if key in TrackVisuals and TrackVisuals[key] != None:

#         print("Here.")
#         cp = TrackVisuals[key]
#         cp.toggle()

#     else:
#         print("There.")
#         pos = VisualPositions.get(key, (150, 150))
#         cp = CircularProgress(track_canvas, pos[0], pos[1], 80, key, duration = random.randint(1, 3) / 2)
#         TrackVisuals[key] = cp
#         cp.running = True
#         cp.draw(START)
#         animate(cp)
#         print(f"Visual created for key '{key}': Animation initializing.")

# def press(event: tk.Event) -> None:

#     key = event.keysym
#     print(f'{key} pressed.')

#     if key in ChannelDict:

#         print("Boo")
#         toggle_channel(key)

#     elif key.lower() in ChannelDict:

#         TrackVisuals[key.lower()].reset()
#         TrackVisuals[key.lower()].toggle()
#         TrackVisuals[key.lower()] = None


# window = tk.Tk()
# window.title("Loopstation")
# window.geometry("800x600")

# track_canvas = tk.Canvas(window, width=300, height=300, highlightthickness = 0)
# track_canvas.configure(bg = window.cget('background'))
# track_canvas.place(x = 450, y = 50)

# window.focus_set()
# window.bind_all("<KeyPress>", press)

# window.mainloop()

# import numpy as np
# freq = [45, 45]
# FF = [15]

# tmp = np.median([freq[len(freq) - 1], freq[len(freq) - 2], FF[0]])
# freq.append(tmp)

# print(freq)

# import time
# import pylab as pl
# import numpy as np
# import random as rd
# from playsound import playsound
# from scipy.fftpack import fft, fftfreq
# from scipy.io.wavfile import write
# from scipy.signal.windows import blackman

# def window_on(data, window, frameSize):

#     data_raw = data * window
#     FFT = fft(data_raw)
#     y = 10 * 2.0 / frameSize * np.abs(FFT)[:frameSize//2]
#     return y

# framerate = 44100

# Size = 131072

# freq = 225.28
# sinX = np.linspace(0, 2 * np.pi * freq / (framerate / Size), Size)
# waveData = np.sin(sinX) * 0.34

# for idx in range(2, 31):

#     sinX = np.linspace(0, 2 * np.pi * freq * idx / (framerate / Size), Size)
#     waveData += np.sin(sinX) * (0.1 / idx + 0.025 * (np.log2(idx) * rd.random()))

# for idx in range(0, len(waveData)):

#     if rd.random() > 0.9999: waveData[idx] += rd.random() * (-1) ** (idx % 2)

# pl.subplot(211)

# pl.plot(np.arange(len(waveData)), waveData, color = "purple")
# pl.grid()

# pl.subplot(212)

# start = time.time() * (1048576 / Size)

# x1 = fftfreq(Size, 1 / 44100)[: Size // 2]
# FFT = window_on(waveData, blackman(Size), Size)


# end = time.time() * (1048576 / Size)

# print(start)
# print(end)
# print(end - start)

# pl.plot(x1, FFT, color = "blue")

# for idx in range(1, 31):

#     pl.axvline(x = freq * idx, color = 'red', linestyle = '--')

# pl.grid()

# pl.show()

# scaled = np.int16(waveData / np.max(np.abs(waveData)) * 32767)
# write('faoSceeE.wav', framerate, scaled)

# playsound('faoSceeE.wav')

import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
from matplotlib.animation import FuncAnimation
from scipy.signal.windows import blackman
from scipy.fftpack import fft, fftfreq
from Vincent.DFSMN.NoiseReductor import DeNoiser, MODEL_PATH

fs = 44100
duration = 0.05
size = int(fs * duration)
Magnify = 2528
freqspan = fs // 2
Denoiser = DeNoiser(MODEL_PATH, 0.8, 100)

x = fftfreq(size, d = 1.0 / fs)

fig, ax = plt.subplots()
line1, = ax.plot(x, np.zeros_like(x), color = "purple", lw = 0.38)
line2, = ax.plot(x, np.zeros_like(x), color = "blue", lw = 0.38)
line3, = ax.plot(x, np.zeros_like(x), color = "green", lw = 0.38)
line4, = ax.plot(x, np.zeros_like(x), color = "orange", lw = 0.38)
ax.set_xlim(0, freqspan)
ax.set_ylim(0, 5)
ax.set_title("Real-Time FFT of Live Audio")

chunk = np.zeros(freqspan)

def audio_callback(indata, frames, time, status):

    global chunk

    if status: print(status)
    
    chunk = indata[:, 0].copy()

stream = sd.InputStream(callback = audio_callback,
                          channels = 1,
                          samplerate = fs,
                          blocksize = size)

stream.start()

def update(frame):

    global chunk

    window = blackman(len(chunk))

    data = chunk
    y = Magnify * 2.0 / size * np.abs(np.abs(fft(data.tolist())))[:freqspan]
    line1.set_ydata(y)

    data = chunk * window
    y = Magnify * 2.0 / size * np.abs(np.abs(fft(data.tolist())))[:freqspan]
    line2.set_ydata(y)

    # data = chunk
    # data = Denoiser.DeNoise(chunk, preset = "voice")
    # y = Magnify * 2.0 / size * np.abs(np.abs(fft(data.tolist())))[:freqspan]
    # line3.set_ydata(y)

    return line1, line2, line3, line4, 

animation = FuncAnimation(fig, update, interval = duration * 1000, blit = True, cache_frame_data = False)

plt.show()