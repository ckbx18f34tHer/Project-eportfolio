import numpy as np

framerate = 44100
frameSize = 2048
overlap = 1024

AMP = [0.6167946080288788, 0.15585802814444288, 0.0233154887364748, 0.016160235544423782, 0.006306329878787568, 0.007873476816483569, 0.01743443729102204, 0.025100689105573802, 0.018000309470481507, 0.019513962429063607, 0.01373247341653225, 0.010655299633578714, 0.010439329470052966, 0.005068194893858414, 0.007807821351880483, 0.0021342455960532435, 0.011688467343261726, 0.010154033804421363, 0.008812165815454773, 0.013150403229273805];
octave = 2
setup = [[[82.41 * octave], 36], [[61.47 * octave], 12], [[98 * octave], 24], [[82.41 * octave], 24], [[146.83 * octave], 24], [[82.41 * octave], 12], [[138.59 * octave], 36], [[82.41 * octave], 48], [[82.41 * octave], 24], [[98 * octave], 24], [[82.41 * octave], 24], [[146.83 * octave], 24], [[82.41 * octave], 12], [[138.59 * octave], 36], [[82.41 * octave], 24]]
# mi ti so mi re mi do mi mi so me re mi do
values = []

frequency = []
Amplify = []

for count in range(2):

    for set in setup:

        for i in range(0, set[1] // 2):
            
            frequency.append(set[0])
                
            Amplify.append(1 - 1 / (i + 1))

for idx in range(0, len(frequency)):

    LineData = np.zeros(frameSize)
    ff = frequency[idx]

    for amp in range(0, len(AMP)):

        for freq in ff:

            sinX = np.linspace(0, 2 * np.pi * freq * (amp) / (framerate / frameSize), frameSize)
            LineData += np.sin(sinX) * AMP[amp] * Amplify[idx] / 2

    if (ff[len(ff) - 1] != 0):
                
        stop = int(((frameSize - overlap) // (framerate // ff[len(ff) - 1])) * (framerate // ff[len(ff)- 1]))
        values.extend(LineData[0:stop].tolist())

import matplotlib.pyplot as pl

pl.subplot(111)
pl.plot(np.arange(len(values)), values, 'r')
pl.grid()
pl.show()

from playsound import playsound
from scipy.io.wavfile import write

scaled = np.int16(values / np.max(np.abs(values)) * 32767)
write('Conversion.wav', 44100, scaled)

playsound('Conversion.wav')

print("Conversion terminated.")

# from sounddevice import *

# rate = 88200

# play(values, rate)
# wait()