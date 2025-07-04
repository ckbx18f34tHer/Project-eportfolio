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

    if os.path.exists("C:\\Users\\studi\\faoSceeE.wav"):

        os.remove("C:\\Users\\studi\\faoSceeE.wav")
        print("File removed successfully")

    wav_files = glob(os.path.join(folder_path, '*.wav'))

    for wav_file in wav_files:

        waveData, framerate = getdata(wav_file)
        print(f"Processing {wav_file}")

        frameSize = 2048
        overlap = frameSize // 2

        Data = waveData.tolist()
        data = Data - np.mean(Data)

        def window_on(start, end, data, window, frameSize, Magnifier):
            
            data_raw = data[start:end] * window
            FFT = fft(data_raw)
            y = Magnifier * 2.0 / frameSize * np.abs(FFT)[:frameSize//2]
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
            prime_values, = AX.plot([], [], 'b')
            AX.set_ylim(-5, 5)
            AX.set_xlim(0, len(Data))

            Time_Text = AX.text(0.02, 0.85, '', transform=AX.transAxes)

            ratio = np.zeros(20)

            window = hann(frameSize)
            Magnifier = 10
            
            AMP = (0.40572805508245274, 0.2568626895930045, 0.1518412649770795, 0.07298023384484983, 0.04680616477466339, 0.02683350375530291, 0.014322484887685248, 0.012891759516823482, 0.004031845431828504, 0.0026062952370169284, 0.0015348292808725305, 0.0009403690640878585, 0.0006627841807982208, 0.00039414998048789226, 0.00046667109735242613, 0.00022090676329597345, 0.00019802161760374153, 0.0002578469134406851, 0.00023612719213261902, 0.000183996809221041)
            
            def animate(i):

                start = i * (frameSize - overlap)
                end = start + frameSize

                if end > len(data):

                    return line, maxPower, time_text, max_curve, SinWave, curFF, prime_values, Time_Text

                with ThreadPoolExecutor() as executor:
                    y = executor.submit(window_on, start, end, data, window, frameSize, Magnifier).result()

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

                peaks, _ = find_peaks(y, height = 0.1)
                FF = x[peaks]

                LineData = np.zeros(frameSize)

                if len(FF) != 0:

                    curFF.set_text(f'FF: {FF[0]:.2f} /s')
                    

                    # Approach 1

                    # sinX = np.linspace(0, 2 * np.pi * FF[0] / (framerate / frameSize), frameSize)
                    # LineData += np.sin(sinX) * getAmplitude(y, FF[0], framerate, frameSize)

                    # Approachh 2

                    # for ff in FF:

                    #         if ff != 0:

                    #             sinX = np.linspace(0, 2 * np.pi * ff / (framerate / frameSize), frameSize)
                    #             LineData += np.sin(sinX) * getAmplitude(y, ff, framerate, frameSize)

                    # Approach 3

                    for idx in range(0, len(AMP)):

                        sinX = np.linspace(0, 2 * np.pi * FF[0] * (idx + 1) / (framerate / frameSize), frameSize)
                        LineData += np.sin(sinX) * AMP[idx] * getAmplitude(y, FF[0], framerate, frameSize) / Magnifier * 2

                SinWave.set_ydata(LineData)

                print(len(FF))
                
                if (len(FF) == 0 or FF[0] == 0):

                    stop = frameSize - overlap

                else:

                    stop = int(((frameSize - overlap) // (framerate // FF[0])) * (framerate // FF[0]))

                values.extend(LineData[0:stop].tolist())
                prime_values.set_data(range(len(values)), values)

                if (len(FF) < 15 and len(FF) > 7 and FF[0] < 1000):
                    
                    ratios = []

                    for i in range(0, 20):

                        ratios.append(getAmplitude(y, FF[0] * (i + 1), framerate, frameSize) / Magnifier)

                    for i in range(0, 20):

                        ratio[i] = ratio[i] + (ratios[i] / np.sum(ratios))

                return line, maxPower, time_text, max_curve, curFF, SinWave, prime_values, Time_Text

            animation = FuncAnimation(fig, animate, frames=(len(data)//(frameSize-overlap)), interval=10, blit=True, repeat=False)
            pl.show()
            
            ratio = ratio / np.sum(ratio)

            print('AMP = [', end = "")
            print(", ".join(map(str, ratio)), end = "]\n")

            rate = 88200

            scaled = np.int16(values / np.max(np.abs(values)) * 32767)
            write("faoSceeE.wav", rate, scaled)

            playsound("faoSceeE.wav")

        generateTimbreAnimation(frameSize, overlap, data)

process("C:/Users/studi/OneDrive/桌面/學校的東西/獨研/PyStuff/DumpStack")

