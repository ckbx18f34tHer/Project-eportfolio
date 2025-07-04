
from playsound import playsound #type: ignore
from scipy.io.wavfile import write
import random as rd
import numpy as np


multi_Size = 80
multi_Zero = 0
frameSize = int(2048 * multi_Size)
zerosFrame = int(2048 * multi_Zero)
waveData0 = np.zeros(frameSize)
frequency = 163.33
framerate = 44100
AMP = [0.2, 0.1, 0.1, 0, 0.1, 0, 0.1, 0, 0, 0, 0.1, 0, 0.1, 0, 0, 0, 0.1, 0, 0.1, 0]

for idx in range(0, 20): 
    
    sinX = np.linspace(0, 2 * np.pi * frequency * (idx + 1) / (framerate / frameSize), frameSize)
    waveData0 += np.sin(sinX) * AMP[idx] 

temp = waveData0[:]

for idx in range(0, frameSize - 1):

    value = rd.random() * 5

    if value > 2.2528 and value < 2.7038: 
        
        value *= 5
        waveData0[idx] += value; waveData0[idx + 1] -= value

rate = 88200
scaled = np.int16(waveData0 / np.max(np.abs(waveData0)) * 32767)
write('盤子.wav', rate, scaled)

playsound('盤子.wav')