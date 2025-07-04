import wave
import numpy as np
import pylab as pl
from scipy.io.wavfile import write
import sounddevice as sd #type: ignore
import time
from playsound import playsound #type: ignore
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

multiple = 500

fw = wave.open('Record.wav','rb')
params = fw.getparams()
# print(params)
nchannels, sampwidth, framerate, nframes = params[:4]
strData = fw.readframes(nframes)
waveData = np.frombuffer(strData, dtype=np.int16)
waveData = waveData*1.0/max(abs(waveData))
fw.close()
print(framerate)
# time = np.arange(0, len(waveData)) * (1.0 / framerate)

frameSize = 768
overlap = 0
idx1 = 0
idx2 = idx1 + frameSize * multiple
index1 = idx1 * 1.0 / frameSize
index2 = idx2 * 1.0 / frameSize

Data = waveData.tolist()

pl.subplot(411)
pl.plot(np.arange(len(Data[2000:4000:10])),Data[2000:4000:10],'g')
# pl.plot([7500, 7500], [-1, 1],'r')
# pl.plot([10870, 10870], [-1, 1],'r')

# data = Data[idx1:idx2]

# index1 = idx1 * 1.0 / framerate
# index2 = idx2 * 1.0 / framerate

# acf = ACF(waveData[idx1:idx2])

# normacf = NormACF(waveData[idx1:idx2])

# amdf = AMDF(waveData[idx1:idx2])

# normamdf = NormAMDF(waveData[idx1:idx2])

# acfamdf = acf / amdf

# pl.subplot(7, 1, 1)
# pl.plot(time, waveData)
# pl.plot([index1,index1],[-1,1],'r')
# pl.plot([index2,index2],[-1,1],'r')
# pl.xlabel("time (seconds)")

# pl.subplot(312)
# pl.plot(np.arange(len(data)),data,'g')

acf = []
pitchtrial = []
i = 337
data = np.array(Data)

while (i + frameSize < len(Data)):
    
    small = data[i:i+frameSize+1]
    maxt = -10000
    maxidx = 0
    tmp = []

    for j in range(44, min(frameSize, 1000)):
    
        t = 0

        for k in range(0, frameSize - j + 1):
            
            t = t + ((small[k] * small[k + j]))
            # print(t)
        
        acf.append(t)
        tmp.append(t)

    i = i + frameSize - overlap

    maxidx = np.argmax(tmp) + 44
    frequency = 44100.0 / (maxidx - 1.0)

    Wave = generate_sine_wave(frequency, frameSize / framerate)
    crossfade_time = frameSize / framerate * 0.1

    if (len(pitchtrial) == 0):
        pitchtrial = Wave
    else:
        pitchtrial = crossfade(pitchtrial, Wave, crossfade_time)

    # print(maxidx, end = ' ')
    # print(frequency)
    # print("------")

    # time.sleep(3)

    print(f"Frame start index: {i}, Max index: {maxidx}, Frequency: {frequency}")

pl.subplot(312)
pl.plot(np.arange(len(acf[2000:4000:10])),acf[2000:4000:10],'g')

# pl.plot([0, 0], [bottom, top],'r')
# pl.plot([337, 337], [bottom, top], 'r')
# pl.plot([674, 674], [bottom, top], 'r')

# pl.subplot(7, 1, 2)
# pl.plot(np.arange(frameSize * multiple),acf,'g')

# pl.subplot(7, 1, 3)
# pl.plot(np.arange(frameSize * multiple),normacf,'g')

# pl.subplot(7, 1, 4)
# pl.plot(np.arange(frameSize * multiple),amdf,'m')

# pl.subplot(7, 1, 5)
# pl.plot(np.arange(frameSize * multiple),normamdf,'m')

# pl.subplot(7, 1, 6)
# pl.plot(np.arange(frameSize * multiple),acfamdf,'y')

# pl.subplot(3, 1, 3)
# pl.plot(np.arange(frameSize * multiple),normacfamdf,'y')
# pl.xlabel("index in 1 frame")

pl.subplot(313)
pl.plot(np.arange(len(pitchtrial[::1000])),pitchtrial[::1000],'b')

pl.show()
timeline = np.arange(len(acf))

rbf = Rbf(timeline, acf, function='inverse')  # 可以選擇不同的 RBF 函數，例如 'multiquadric', 'inverse', 'gaussian', 等等

timeline_new = np.linspace(timeline.min(), timeline.max(), len(acf))
acf_smooth = rbf(timeline_new)

pl.plot(timeline_new, acf_smooth)

pl.show()


# stuff = []
# for i in range(0, 33700 * 5):
#     stuff.append(data[i * 2 % 337])

# for i in range(0, 33700):
#     s1 = data[i % 377]
#     s2 = data[(i + 1) % 337]
#     stuff.append(s1 * 3 / 3 + s2 * 0 / 3)
#     stuff.append(s1 * 2 / 3 + s1 * 1 / 3)
#     stuff.append(s1 * 1 / 3 + s2 * 2 / 3)

# rate = 88200
# scaled = np.int16(data / np.max(np.abs(data)) * 32767)  
# write('Trial.wav', rate, scaled)

# rate = 88200
# scaled = np.int16(echo / np.max(np.abs(echo)) * 32767)  
# write('Echo.wav', rate, scaled)

rate = 88200
scaled = np.int16(acf / np.max(np.abs(acf)) * 32767)
write('PitchTracker2.wav', rate, scaled)

playsound('PitchTracker2.wav')

rate = 88200
scaled = np.int16(acf_smooth / np.max(np.abs(acf_smooth)) * 32767)
write('PitchTracker3.wav', rate, scaled)

playsound('PitchTracker3.wav')
