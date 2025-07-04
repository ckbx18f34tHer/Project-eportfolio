

import wave
import numpy as np
import pylab as pl
from statistics import median

fw = wave.open('FluteSolo.wav','rb')
params = fw.getparams()
nchannels, sampwidth, framerate, nframes = params[:4]
waveData = np.frombuffer(fw.readframes(nframes), dtype=np.int16)
waveData = waveData*1.0/max(abs(waveData))
fw.close()

frameSize = 768
overlap = 256
step = frameSize - overlap

Data = waveData.tolist()

pl.subplot(211)
pl.plot(np.arange(len(Data)),Data,'g')
pl.grid()

pitchtrial = []
data = np.array(Data)
i = 0

while (i + frameSize < len(Data)):
    
    small = data[i:i+frameSize]
    tmp = []

    for j in range(44, min(frameSize, 1000)):
    
        t = 0

        for k in range(0, frameSize - j):
            
            t = t + ((small[k] * small[k + j]))

        tmp.append(t)

    i += step

    maxidx = np.argmax(tmp) + 44
    frequency = 44100.0 / (maxidx - 1.0)

    for count in range(0, step):
        pitchtrial.append(frequency)

    print(f"Frame start index: {i}, Max index: {maxidx}, Frequency: {frequency}")

MedianACF = pitchtrial[0:step]
index = step // 2 + step

while (index + step < len(pitchtrial)):

    filtered = median({pitchtrial[index], pitchtrial[index - step], pitchtrial[index + step]})

    MedianACF.append(filtered for i in range(0, step))

    index += step
    print(index)

pl.subplot(212)
pl.plot(np.arange(len(pitchtrial)), pitchtrial, 'b')
pl.plot(np.arange(len(MedianACF)), MedianACF, 'r')
pl.grid()

pl.show()




