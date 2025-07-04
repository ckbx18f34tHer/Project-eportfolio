






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
print(framerate)

frameSize = 1024
overlap = 0

Data = waveData.tolist()
Data = Data - np.mean(Data)

pl.subplot(211)
pl.plot(np.arange(len(Data)),Data,'g')
pl.plot([161834, 161834], [-1,1], 'r')
pl.plot([(161834+frameSize), (161834+frameSize)], [-1, 1], 'r')
pl.grid()

data = Data[161834:(161834+frameSize)]

pl.subplot(212)
pl.plot(np.arange(len(data)), data, 'g')
pl.plot([63,63],[-1,1],'r')
pl.plot([367,367],[-1,1],'r')
pl.plot([667,667],[-1,1],'r')
pl.grid()

pl.show()














