import wave
import glob
import os
import numpy as np
import matplotlib.pyplot as pl
from matplotlib.animation import FuncAnimation
from scipy.fft import fft, fftfreq
from scipy.signal.windows import hann
from playsound import playsound
from scipy.io.wavfile import write
from scipy.signal import find_peaks
from concurrent.futures import ThreadPoolExecutor
from pyfftw.interfaces.numpy_fft import fft


def getdata(filename):

    fw = wave.open(filename, 'rb')
    params = fw.getparams()
    nchannels, sampwidth, framerate, nframes = params[:4]
    strData = fw.readframes(nframes)
    waveData = np.frombuffer(strData, dtype=np.int16)
    waveData = waveData * 1.0 / max(abs(waveData))
    fw.close()

    return waveData, framerate

def process(folder_path):

    wav_files = glob.glob(os.path.join(folder_path, '*.wav'))

    for wav_file in wav_files:

        waveData, framerate = getdata(wav_file)
        print(f"Processing {wav_file}")

        frameSize = 2048
        overlap = frameSize // 2

        Data = waveData.tolist()
        data = Data - np.mean(Data)

        def window_on(start, end, data, window, frameSize):
            data_raw = data[start:end] * window
            FFT = fft(data_raw)
            y = 10 * 2.0 / frameSize * np.abs(FFT)[:frameSize//2]
            return y
        
        def generateTimbreAnimation(frameSize, overlap, data):
            
            fig, (ax, Ax, AX) = pl.subplots(3, 1)

            x = fftfreq(frameSize, 1/44100)[:frameSize//2]
            line, = ax.plot(x, np.zeros_like(x))
            maxPower, = ax.plot(x, np.zeros_like(x), 'g')
            time_text = ax.text(0.02, 0.85, '', transform=ax.transAxes)
            ax.set_ylim(0, 2)
            ax.set_xlim(0, 10000)

            ax2 = ax.twiny()
            ax2.set_xlim(0, len(data) // (frameSize - overlap))

            max_values = []
            max_curve, = ax2.plot([], [], 'r')

            Ax.set_ylim(-5, 5)
            X = np.linspace(0, 2 * np.pi, frameSize)
            SinWave, = Ax.plot(X, np.zeros_like(X), 'c')

            curFF = Ax.text(0.02, 0.85, '', transform=Ax.transAxes)

            values = []
            prime_values, = AX.plot([], [], 'c')
            AX.set_ylim(-5, 5)
            AX.set_xlim(0, len(Data))

            Time_Text = AX.text(0.02, 0.85, '', transform=AX.transAxes)

            window = hann(frameSize)

            def animate(i):

                start = i * (frameSize - overlap)
                end = start + frameSize

                if end > len(data):

                    return line, maxPower, time_text, max_curve, SinWave, curFF, prime_values, Time_Text

                with ThreadPoolExecutor() as executor:
                    y = executor.submit(window_on, start, end, data, window, frameSize).result()

                line.set_ydata(y)

                current_time = start / 44100 / 2
                time_text.set_text(f'Time: {current_time:.2f} s')
                Time_Text.set_text(f'Time: {current_time:.2f} s')

                green_y = np.full_like(x, y[np.argmax(y)])
                maxPower.set_ydata(green_y)

                max_values.append(np.max(y))
                max_curve.set_data(range(len(max_values)), max_values)

                def getAmplitude(y, frequency, framerate, frameSize):

                    index = frequency / (framerate / frameSize)

                    lower_bin = int(np.floor(index))
                    upper_bin = int(np.ceil(index))

                    lower_value = np.abs(y[lower_bin])
                    upper_value = np.abs(y[upper_bin])

                    interpolation = lower_value + (upper_value - lower_value) * (index - lower_bin)

                    return interpolation

                peaks, _ = find_peaks(y, height = 0.2)
                FF = x[peaks]

                LineData = np.zeros(frameSize)

                if len(FF) != 0:

                    curFF.set_text(f'FF: {FF[0]:.2f} /s')
                        
                    if FF[0] != 0:

                        sinX = np.linspace(0, 2 * np.pi * FF[0] / (framerate / frameSize), frameSize)
                        LineData += np.sin(sinX) * getAmplitude(y, FF[0], framerate, frameSize)

                    # for ff in FF:

                    #         if ff != 0:

                    #             sinX = np.linspace(0, 2 * np.pi * ff / (framerate / frameSize), frameSize)
                    #             LineData += np.sin(sinX) * getAmplitude(y, ff, framerate, frameSize)

                SinWave.set_ydata(LineData)

                values.extend(LineData[0:(frameSize - overlap)].tolist())
                prime_values.set_data(range(len(values)), values)

                return line, maxPower, time_text, max_curve, curFF, SinWave, prime_values, Time_Text

            animation = FuncAnimation(fig, animate, frames=(len(data)//(frameSize-overlap)), interval=10, blit=True, repeat=False)
            pl.show()

            rate = 88200

            scaled = np.int16(values / np.max(np.abs(values)) * 32767)
            write('faoSceeE.wav', rate, scaled)

            playsound('faoSceeE.wav')

        generateTimbreAnimation(frameSize, overlap, data)

process('C:\\Users\\studi')

