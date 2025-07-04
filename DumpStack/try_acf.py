import wave
import numpy as np
import pylab as pl
from scipy.io.wavfile import write
import sounddevice as sd # type: ignore
import time
from playsound import playsound # type: ignore
from scipy.interpolate import Rbf

def generate_sine_wave(frequency, duration, sample_rate=44100, amplitude=0.5):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    wave = amplitude * np.sin(2 * np.pi * frequency * t)
    return wave.astype(np.float32)

def crossfade(wave1, wave2, crossfade_duration, sample_rate=44100):
    crossfade_samples = int(sample_rate * crossfade_duration)
    fade_in = np.linspace(0, 1, crossfade_samples)
    fade_out = np.linspace(1, 0, crossfade_samples)

    wave1[-crossfade_samples:] *= fade_out
    wave2[:crossfade_samples] *= fade_in
    
    return np.concatenate((wave1, wave2))

def process_chunk(data_chunk, frameSize, framerate):
    acf = []
    pitchtrial = []
    i = 0

    while (i + frameSize < len(data_chunk)):
        small = data_chunk[i:i+frameSize+1]
        tmp = []

        for j in range(44, min(frameSize, 1000)):
            t = np.sum(small[:frameSize - j] * small[j:frameSize])
            acf.append(t)
            tmp.append(t)

        i += frameSize

        maxidx = np.argmax(tmp) + 44
        frequency = 44100.0 / (maxidx - 1.0)

        Wave = generate_sine_wave(frequency, frameSize / framerate)
        crossfade_time = frameSize / framerate * 0.1

        if len(pitchtrial) == 0:
            pitchtrial = Wave
        else:
            pitchtrial = crossfade(pitchtrial, Wave, crossfade_time)

    return acf, pitchtrial

multiple = 500
frameSize = 768
overlap = 0

fw = wave.open('C:/Users/studi/OneDrive/桌面/學校的東西/獨研/PyStuff/DumpStack/MarimbaNotes.wav', 'rb')
params = fw.getparams()
nchannels, sampwidth, framerate, nframes = params[:4]
chunk_size = frameSize * multiple

acf_total = []
pitchtrial_total = []

for i in range(0, nframes, chunk_size):
    strData = fw.readframes(chunk_size)
    waveData = np.frombuffer(strData, dtype=np.int16)
    waveData = waveData * 1.0 / max(abs(waveData))

    acf, pitchtrial = process_chunk(waveData, frameSize, framerate)
    acf_total.extend(acf)
    pitchtrial_total.extend(pitchtrial)

fw.close()

# Plotting and further processing
pl.subplot(211)
pl.plot(np.arange(len(acf_total[2000:4000:10])), acf_total[2000:4000:10], 'g')

pl.subplot(212)
pl.plot(np.arange(len(pitchtrial_total[::1000])), pitchtrial_total[::1000], 'b')

pl.show()

# Interpolation and saving
timeline = np.arange(len(acf_total))
rbf = Rbf(timeline, acf_total, function='inverse')
timeline_new = np.linspace(timeline.min(), timeline.max(), len(acf_total))
acf_smooth = rbf(timeline_new)

pl.plot(timeline_new, acf_smooth)
pl.show()

rate = 88200
scaled = np.int16(acf_total / np.max(np.abs(acf_total)) * 32767)
write('PitchTracker2.wav', rate, scaled)
playsound('PitchTracker2.wav')

scaled_smooth = np.int16(acf_smooth / np.max(np.abs(acf_smooth)) * 32767)
write('PitchTracker3.wav', rate, scaled_smooth)
playsound('PitchTracker3.wav')
