







import wave
import numpy as np
import pylab as pl

fw = wave.open('FluteSolo.wav','rb')
params = fw.getparams()
nchannels, sampwidth, samplerate, nframes = params[:4]
strData = fw.readframes(nframes)
waveData = np.frombuffer(strData, dtype=np.int16)
waveData = waveData*1.0/max(abs(waveData))
fw.close()
print(samplerate)

Data = waveData.tolist()
time = []
for i in range(0, len(Data)):
    time.append(i / samplerate)

pl.subplot(211)
pl.plot(np.arange(len(Data)),Data, 'g')
pl.plot([44100, 44100], [-1, 1],'r')
pl.grid()

pl.subplot(212)
pl.plot(time, Data, 'g')
pl.plot([1, 1], [-1, 1],'r')
pl.grid()

pl.show()









