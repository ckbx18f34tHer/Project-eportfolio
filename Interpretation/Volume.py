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

frameSize = 512
overlap = 0

Data = waveData.tolist()
Data = Data - np.mean(Data)

i = 0
volume = []

while (i + frameSize < len(Data)):
    
    t = 0

    for j in range(0, frameSize + 1):
    
        t = t + (Data[i + j] ** 2)
        
    t = 10 * np.log10(t)
        
    for j in range(0, frameSize + 1):

        volume.append(t)

    i += frameSize - overlap

pl.subplot(211)
pl.plot(np.arange(len(Data)), Data, color = "black")
pl.grid()

np.sort(volume)
pr97 = np.percentile(volume, 97)
pr03 = np.percentile(volume, 3 )
th = ((pr97 - pr03) * 0.75 + pr03)

for i in range(0, len(volume) - frameSize, frameSize):

    if (volume[i] <= th and volume[i + frameSize] >= th):

        pl.plot([i, i], [-1, 1], 'r')

    elif (volume[i] >= th and volume[i + frameSize] <= th):

        pl.plot([i, i], [-1, 1], 'c')

pl.subplot(212)
pl.plot(np.arange(len(volume)), volume, color = 'purple')
pl.grid()
pl.show()






