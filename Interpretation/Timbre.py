from concurrent.futures import ThreadPoolExecutor
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as pl
import numpy as np
import os
from wave import open
from glob import glob
from playsound import playsound
from scipy.io.wavfile import write
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks
from scipy.signal.windows import hann
from pyfftw.interfaces.numpy_fft import fft

def getdata(filename):

    fw = open(filename, 'rb')
    params = fw.getparams()
    nchannels, sampwidth, framerate, nframes = params[:4]
    strData = fw.readframes(nframes)
    waveData = np.frombuffer(strData, dtype=np.int16)
    waveData = waveData * 1.0 / max(abs(waveData))
    fw.close()

    return waveData, framerate

def process(folder_path):

    wav_files = glob(os.path.join(folder_path, '*.wav'))

    for wav_file in wav_files:

        waveData, framerate = getdata(wav_file)
        print(f"Processing {wav_file}")

        framerate = 44100
        frameSize = 2048
        overlap = frameSize // 2
        
        Data = waveData.tolist()
        data = Data # - np.mean(Data)

        def window_on(start, end, data, window, frameSize):
            data_raw = data[start:end] * window
            FFT = fft(data_raw)
            y = 10 * 2.0 / frameSize * np.abs(FFT)[:frameSize//2]
            return y
        
        def generateTimbreAnimation(frameSize, overlap, data):
            
            fig, ((ax, aX), (Ax, AX)) = pl.subplots(2, 2)

            x = fftfreq(frameSize, 1/44100)[:frameSize//2]
            line, = ax.plot(x, np.zeros_like(x))
            maxPower, = ax.plot(x, np.zeros_like(x), 'green')
            time_text = ax.text(0.02, 0.85, '', transform=ax.transAxes)
            ax.set_ylim(0, 2)
            ax.set_xlim(0, 10000)

            ax2 = ax.twiny()
            ax2.set_xlim(0, len(data) // (frameSize - overlap))

            max_values = []
            max_curve, = ax2.plot([], [], 'red')

            Ax.set_ylim(-2, 2)
            X = np.linspace(0, 2 * np.pi, frameSize)
            SinWave, = Ax.plot(X, np.zeros_like(X), 'purple')

            curFF = Ax.text(0.02, 0.85, '', transform=Ax.transAxes)

            values = []
            prime_values, = AX.plot([], [], 'cyan')
            AX.set_ylim(-2, 2)
            AX.set_xlim(0, frameSize)

            Time_Text = AX.text(0.02, 0.85, '', transform=AX.transAxes)

            freq = []
            amp = []
            prime_freq, = aX.plot([], [], 'black')
            aX.set_ylim(0, 1000)
            aX.set_xlim(0, len(Data) // 1024)

            window = hann(frameSize)
            
            AMP = (0.4, 0.15, 0.1, 0.18, 0.06, 0.003933103448275862, 0.0039231034482758624, 0.003913103448275862, 0.003903103448275862, 0.003893103448275862, 0.003883103448275862, 0.0038731034482758623, 0.0038631034482758623, 0.0038531034482758623, 0.0038431034482758622, 0.003833103448275862, 0.003823103448275862, 0.003813103448275862, 0.003803103448275862, 0.003793103448275862, 0.003783103448275862, 0.003773103448275862, 0.003763103448275862, 0.003753103448275862, 0.003743103448275862, 0.003733103448275862, 0.003723103448275862, 0.003713103448275862, 0.0037031034482758623, 0.0036931034482758623, 0.0036831034482758622, 0.0036731034482758622, 0.003663103448275862, 0.003653103448275862)
            # AMP = (0.5, 0.5)

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

                # max_values.append(np.max(y))
                # max_curve.set_data(range(len(max_values)), max_values)

                def getAmplitude(y, frequency, framerate, frameSize):

                    index = frequency / (framerate / frameSize)

                    lower_bin = int(np.floor(index))
                    upper_bin = int(np.ceil(index))

                    lower_value = np.abs(y[lower_bin])
                    upper_value = np.abs(y[upper_bin])

                    interpolation = lower_value + (upper_value - lower_value) * (index - lower_bin)

                    return interpolation

                peaks, _ = find_peaks(y, height = 0.1)
                FF = x[peaks]

                LineData = np.zeros(frameSize)

                if len(FF) != 0:

                    curFF.set_text(f'FF: {FF[0]:.2f} /s')
                
                else: FF = [1]
                    
                tmp = np.mean([freq[len(freq) - 1], freq[len(freq) - 2], FF[0]]) if len(freq) >= 3 else FF[0]
                amp.append(getAmplitude(y, tmp, framerate, frameSize) * 0.5)

                    # Approach 1

                    # sinX = np.linspace(0, 2 * np.pi * FF[0] / (framerate / frameSize), frameSize)
                    # LineData += np.sin(sinX) * getAmplitude(y, FF[0], framerate, frameSize)

                    # Approachh 2

                    # for ff in FF:

                    #         if ff != 0:

                    #             sinX = np.linspace(0, 2 * np.pi * ff / (framerate / frameSize), frameSize)
                    #             LineData += np.sin(sinX) * getAmplitude(y, ff, framerate, frameSize)

                    # Approach 3

                if len(freq) >= 7:

                    for idx in range(0, len(AMP)):
                        
                        sinX = np.linspace(0, 2 * np.pi * freq[len(freq) - 7] * (idx + 1) / (framerate / frameSize), frameSize)
                        LineData += np.sin(sinX) * AMP[idx] * amp[len(amp) - 7]

                SinWave.set_ydata(data[start:end])

                if (len(FF) == 0 or FF[0] == 1):

                    stop = frameSize - overlap

                else:

                    stop = int(((frameSize - overlap) // (framerate // FF[0])) * (framerate // FF[0]))

                values.extend(LineData[0:stop].tolist())
                prime_values.set_data(range(len(LineData)), LineData)

                freq.append(FF[0] if len(freq) < 7 else tmp)
                if len(freq) >= 7: freq[len(freq) - 7] = np.median(freq[len(freq) - 7 : len(freq) - 1])
                prime_freq.set_data(range(len(freq)), freq)

                return line, maxPower, time_text, max_curve, curFF, SinWave, prime_values, Time_Text, prime_freq

            animation = FuncAnimation(fig, animate, frames=(len(data)//(frameSize-overlap)), interval=10, blit=True, repeat=False)
            pl.show()

            rate = 88200

            scaled = np.int16(values / np.max(np.abs(values)) * 32767)
            write('faoSceeE.wav', rate, scaled)

            playsound('faoSceeE.wav')

            pl.subplot(211)
            pl.plot(np.arange(len(Data)), Data, 'g')
            pl.grid()

            pl.subplot(212)
            pl.plot(np.arange(len(freq)), freq, 'g')
            pl.grid()
            

        generateTimbreAnimation(frameSize, overlap, data)

if os.path.exists("C:\\Users\\studi\\OneDrive\\桌面\\PyStuff\\faoSceeE.wav"):

    os.remove("C:\\Users\\studi\\OneDrive\\桌面\\PyStuff\\faoSceeE.wav")
    print("File removed successfully")

process("C:/Users/studi/OneDrive/桌面/學校的東西/獨研/PyStuff/DumpStack")


