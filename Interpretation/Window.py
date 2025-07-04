









import wave
import numpy as np
import pylab as pl

fw = wave.open('FluteSolo.wav','rb')
params = fw.getparams()
nchannels, sampwidth, framerate, nframes = params[:4]
strData = fw.readframes(nframes)
waveData = np.frombuffer(strData, dtype=np.int16)
waveData = waveData*1.0/max(abs(waveData))
fw.close()

frameSize = len(waveData)
overlap = 0
start = 0

Data = waveData.tolist()[start:start+frameSize]

pl.plot(np.arange(frameSize), Data, 'g', label = "Original")
pl.grid()

Data_Hamming = []
for i in range(0, frameSize):
    Data_Hamming.append(Data[i] * ((25 / 46) - (21 / 46) * np.cos(2 * np.pi * (i % 1024) / (frameSize - 1))))

pl.plot(np.arange(frameSize), Data_Hamming, 'b', label = "Hamming")
pl.grid()

pl.legend()
pl.show()











        