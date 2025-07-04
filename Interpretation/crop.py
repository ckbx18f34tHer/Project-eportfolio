import wave
import numpy as np
from playsound import playsound #type: ignore
from scipy.io.wavfile import write

def getdata():
    fw = wave.open('C-4-maj-chord-0.wav', 'rb')
    params = fw.getparams()
    nchannels, sampwidth, framerate, nframes = params[:4]
    strData = fw.readframes(nframes)
    waveData = np.frombuffer(strData, dtype=np.int16)
    waveData = waveData * 1.0 / max(abs(waveData))
    fw.close()
    return waveData, framerate

waveData, framerate = getdata()
waveData = waveData.tolist()
print(len(waveData))

import matplotlib.pyplot as pl

pl.subplot(111)
pl.plot(np.arange(len(waveData)), waveData, 'g')
pl.show()

waveform = waveData[0:len(waveData) // 2]
tmp = waveform[::-1]
waveform.extend(tmp)


rate = framerate * 2
scaled = np.int16(waveform / np.max(np.abs(waveform)) * 32767)
write('Crop.wav', rate, scaled)

from pydub import AudioSegment
from pydub.playback import play

song = AudioSegment.from_wav('Crop.wav')
play(song)