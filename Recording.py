import pyaudio
import numpy as np
from scipy.io.wavfile import write
import time
import keyboard

def convert_and_save(audio_data, filename: str):
    
    waveform = np.array(audio_data)
    scaled = np.int16(waveform / np.max(np.abs(waveform)) * 32767)
    write(filename, RATE, scaled)
    print(f"Audio saved to {filename}")

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024

frames = np.array([])

timeStart = time.time()
print(timeStart)

audio = pyaudio.PyAudio()
stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

timeEnd = time.time()
print(timeEnd)
print(f"Open warmup: {timeEnd - timeStart}")
print("Recording... Press Ctrl+C to stop.")

recording = False

def toggle_recording(event):

    global recording

    if event.name == 'r':

        recording = True
        print("Recording started")

    elif event.name == 'p':

        recording = False
        print("Recording paused")

    elif event.name == 's':

        recording = False
        print("Recording stopped")

        stream.stop_stream()
        stream.close()
        audio.terminate()
        
        convert_and_save(frames, 'Record.wav')

keyboard.on_press(toggle_recording)

try:

    while True:

        if recording:

            data = stream.read(CHUNK)
            audio_data = np.frombuffer(data, dtype=np.int16) / 32768.0

            frames = np.append(frames, audio_data)
            print(type(audio_data))



except KeyboardInterrupt:

    pass

finally:

    stream.stop_stream()
    stream.close()
    audio.terminate()

print("Done")