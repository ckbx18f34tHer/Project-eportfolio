import wave
import numpy as np
import pylab as pl
import scipy

fw = wave.open('FluteSolo.wav','rb')
params = fw.getparams()
nchannels, sampwidth, framerate, nframes = params[:4]
waveData = np.frombuffer(fw.readframes(nframes), dtype=np.int16)
waveData = waveData*1.0/max(abs(waveData))
fw.close()

frameSize = 512
overlap = 256
Data = waveData.tolist()
print("start")

pl.subplot(311)
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

pl.subplot(312)
pl.plot(np.arange(len(acf)),acf,'g')
pl.grid()

newacf = scipy.signal.savgol_filter(acf, window_length = 20, polyorder = 3)

pl.subplot(313)
pl.plot(np.arange(len(newacf)), newacf, 'g')
pl.grid()

pl.show()
