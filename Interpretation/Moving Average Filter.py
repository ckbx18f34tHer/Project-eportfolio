import wave
import numpy as np
import pylab as pl

fw = wave.open('FluteSolo.wav','rb')
params = fw.getparams()
nchannels, sampwidth, framerate, nframes = params[:4]
waveData = np.frombuffer(fw.readframes(nframes), dtype=np.int16)
waveData = waveData*1.0/max(abs(waveData))
fw.close()

frameSize = 512
overlap = 0
Data = waveData.tolist()
print("start")

pl.subplot(211)
pl.plot(np.arange(len(Data)),Data,'g')
pl.grid()

acf = []
i = 0

while (i + frameSize < len(Data)):

    small = Data[i:i+frameSize+1]
    tmp = []

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

    print(i)

pl.subplot(212)
pl.plot(np.arange(len(acf)),acf,'g')
pl.grid()

N = 20
newacf = []

for i in range(0, len(Data)):

    if (i <= N):

        newacf.append(np.mean(acf[0:i+1]))

    else:
        try:
            newacf.append(newacf[i - 1] - (acf[i - N] / N))
        except (IndexError):
            newacf.append(newacf[i - 1])
        else:
            continue

pl.plot(np.arange(len(newacf)), newacf, 'b')

pl.show()
