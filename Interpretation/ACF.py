import wave
import time
import numpy as np
import pylab as pl

fw = wave.open('C:/Users/studi/OneDrive/桌面/學校的東西/獨研/PyStuff/MarimbaNotes.wav', 'rb')
params = fw.getparams()
nchannels, sampwidth, framerate, nframes = params[:4]
waveData = np.frombuffer(fw.readframes(nframes), dtype=np.int16)
waveData = waveData*1.0/max(abs(waveData))
fw.close()

frameSize = 2048
overlap = frameSize // 6 * 5
Data = waveData.tolist()
print("start")

count = int(time.time())

pl.subplot(211)
pl.plot(np.arange(len(Data)),Data,'g')
pl.grid()

acf = []
i = 0

while (i + frameSize < len(Data)):

    tmp = []
    small = Data[i:i+frameSize+1]

    for j in range(44, min(frameSize, 1000)):
    
        t = 0

        for k in range(0, frameSize - j + 1):
            
            t = t + ((small[k] * small[k + j]))
        
        tmp.append(t)

    i = i + frameSize - overlap

    maxidx = np.argmax(tmp) + 44
    frequency = 44100.0 / (maxidx - 1.0)

    for count in range(0, frameSize - overlap):
        acf.append(frequency)

count = int(time.time()) - count
print(count)

pl.subplot(212)
pl.plot(np.arange(len(acf)),acf,'g')
pl.grid()

pl.show()
