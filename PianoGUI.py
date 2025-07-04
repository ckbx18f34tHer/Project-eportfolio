from tkinter import messagebox, Button, Entry, Tk, Listbox, filedialog, Event, Canvas, END, SUNKEN, RAISED
import os
import time
import random
import pygame
import pyaudio
import threading
import pylab as pl
import numpy as np
from wave import open
from scipy.fftpack import fft, fftfreq
from scipy.signal.windows import blackman
from scipy.signal import find_peaks, butter, lfilter
from Vincent.DFSMN.NoiseReductor import DeNoiser

octave = 0
mixture = 0
framerate = 44100
overtone = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
TbrDict = {'(Basic)': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], '(Prime)': [0.2, 0.1, 0.1, 0, 0.1, 0, 0.1, 0, 0, 0, 0.1, 0, 0.1, 0, 0, 0, 0.1, 0, 0.1, 0]}
track = pyaudio.PyAudio()
stream = track.open(format = pyaudio.paInt16, channels = 1, rate = 44100, input = True, frames_per_buffer = 1024)
pygame.mixer.init(frequency = 44100, size = -16, channels = 2, buffer = 512)
pygame.mixer.set_num_channels(16)
KeyDirector = {}
KitDirector = {}
LoopDirector = {}
RecordDirector = {}
ChannelDirector = {}
SoundArray = np.array([])
TrackArray = np.array([])
SynthON = False
RecordON = False
path = ""
InputEffect = "default"
OutputEffect = "default"
panel = None
RecordingStartTime = None

CHUNK = 1024
START = "#C834A2"
ERASE = "#3400A2"
OCTAVE = 2 ** (1 / 12)
MODEL_PATH = "C:/Users/studi/OneDrive/桌面/學校的東西/獨研/PyStuff/Vincent/DFSMN/checkpoint_epoch_145.pth"

window = Tk()
kbs = set()
keys = set()
selection = set()
recording_lock = threading.Lock()
Denoiser = DeNoiser(MODEL_PATH, 0.8)

if True: # Button()

    window.buttonC2 = Button(window, background = "white", command = lambda: play('F1'))
    window.buttonC2.place(x = 5, y = 390)
    window.buttonC2.configure(height = 20, width = 4)

    window.buttonD2 = Button(window, background = "white", command = lambda: play('F3'))
    window.buttonD2.place(x = 45, y = 390)
    window.buttonD2.configure(height = 20, width = 4)

    window.buttonE2 = Button(window, background = "white", command = lambda: play('F5'))
    window.buttonE2.place(x = 85, y = 390)
    window.buttonE2.configure(height = 20, width = 4)

    window.buttonF2 = Button(window, background = "white", command = lambda: play('F6'))
    window.buttonF2.place(x = 125, y = 390)
    window.buttonF2.configure(height = 20, width = 4)

    window.buttonG2 = Button(window, background = "white", command = lambda: play('F8'))
    window.buttonG2.place(x = 165, y = 390)
    window.buttonG2.configure(height = 20, width = 4)

    window.buttonA2 = Button(window, background = "white", command = lambda: play('F11'))
    window.buttonA2.place(x = 205, y = 390)
    window.buttonA2.configure(height = 20, width = 4)

    window.buttonB2 = Button(window, background = "white", command = lambda: play('Delete'))
    window.buttonB2.place(x = 245, y = 390)
    window.buttonB2.configure(height = 20, width = 4)

    window.buttonC3 = Button(window, background = "white", command = lambda: play('1'))
    window.buttonC3.place(x = 285, y = 390)
    window.buttonC3.configure(height = 20, width = 4)

    window.buttonD3 = Button(window, background = "white", command = lambda: play('3'))
    window.buttonD3.place(x = 325, y = 390)
    window.buttonD3.configure(height = 20, width = 4)

    window.buttonE3 = Button(window, background = "white", command = lambda: play('5'))
    window.buttonE3.place(x = 365, y = 390)
    window.buttonE3.configure(height = 20, width = 4)

    window.buttonF3 = Button(window, background = "white", command = lambda: play('6'))
    window.buttonF3.place(x = 405, y = 390)
    window.buttonF3.configure(height = 20, width = 4)

    window.buttonG3 = Button(window, background = "white", command = lambda: play('8'))
    window.buttonG3.place(x = 445, y = 390)
    window.buttonG3.configure(height = 20, width = 4)

    window.buttonA3 = Button(window, background = "white", command = lambda: play('0'))
    window.buttonA3.place(x = 485, y = 390)
    window.buttonA3.configure(height = 20, width = 4)

    window.buttonB3 = Button(window, background = "white", command = lambda: play('equal'))
    window.buttonB3.place(x = 525, y = 390)
    window.buttonB3.configure(height = 20, width = 4)

    window.buttonC4 = Button(window, background = "white", command = lambda: play('q'))
    window.buttonC4.place(x = 565, y = 390)
    window.buttonC4.configure(height = 20, width = 4)

    window.buttonD4 = Button(window, background = "white", command = lambda: play('e'))
    window.buttonD4.place(x = 605, y = 390)
    window.buttonD4.configure(height = 20, width = 4)

    window.buttonE4 = Button(window, background = "white", command = lambda: play('t'))
    window.buttonE4.place(x = 645, y = 390)
    window.buttonE4.configure(height = 20, width = 4)

    window.buttonF4 = Button(window, background = "white", command = lambda: play('y'))
    window.buttonF4.place(x = 685, y = 390)
    window.buttonF4.configure(height = 20, width = 4)

    window.buttonG4 = Button(window, background = "white", command = lambda: play('i'))
    window.buttonG4.place(x = 725, y = 390)
    window.buttonG4.configure(height = 20, width = 4)

    window.buttonA4 = Button(window, background = "white", command = lambda: play('p'))
    window.buttonA4.place(x = 765, y = 390)
    window.buttonA4.configure(height = 20, width = 4)

    window.buttonB4 = Button(window, background = "white", command = lambda: play('bracketright'))
    window.buttonB4.place(x = 805, y = 390)
    window.buttonB4.configure(height = 20, width = 4)

    window.buttonC5 = Button(window, background = "white", command = lambda: play('a'))
    window.buttonC5.place(x = 845, y = 390)
    window.buttonC5.configure(height = 20, width = 4)

    window.buttonD5 = Button(window, background = "white", command = lambda: play('d'))
    window.buttonD5.place(x = 885, y = 390)
    window.buttonD5.configure(height = 20, width = 4)

    window.buttonE5 = Button(window, background = "white", command = lambda: play('g'))
    window.buttonE5.place(x = 925, y = 390)
    window.buttonE5.configure(height = 20, width = 4)

    window.buttonF5 = Button(window, background = "white", command = lambda: play('h'))
    window.buttonF5.place(x = 965, y = 390)
    window.buttonF5.configure(height = 20, width = 4)

    window.buttonG5 = Button(window, background = "white", command = lambda: play('k'))
    window.buttonG5.place(x = 1005, y = 390)
    window.buttonG5.configure(height = 20, width = 4)

    window.buttonA5 = Button(window, background = "white", command = lambda: play('semicolon'))
    window.buttonA5.place(x = 1045, y = 390)
    window.buttonA5.configure(height = 20, width = 4)

    window.buttonB5 = Button(window, background = "white", command = lambda: play('return'))
    window.buttonB5.place(x = 1085, y = 390)
    window.buttonB5.configure(height = 20, width = 4)

    window.buttonDb2 = Button(window, background = "black", command = lambda: play('F2'))
    window.buttonDb2.place(x = 25, y = 390)
    window.buttonDb2.configure(height = 12, width = 4)

    window.buttonEb2 = Button(window, background = "black", command = lambda: play('F4'))
    window.buttonEb2.place(x = 65, y = 390)
    window.buttonEb2.configure(height = 12, width = 4)

    window.buttonGb2 = Button(window, background = "black", command = lambda: play('F7'))
    window.buttonGb2.place(x = 145, y = 390)
    window.buttonGb2.configure(height = 12, width = 4)

    window.buttonAb2 = Button(window, background = "black", command = lambda: play('F9'))
    window.buttonAb2.place(x = 185, y = 390)
    window.buttonAb2.configure(height = 12, width = 4)

    window.buttonBb2 = Button(window, background = "black", command = lambda: play('F12'))
    window.buttonBb2.place(x = 225, y = 390)
    window.buttonBb2.configure(height = 12, width = 4)

    window.buttonDb3 = Button(window, background = "black", command = lambda: play('2'))
    window.buttonDb3.place(x = 305, y = 390)
    window.buttonDb3.configure(height = 12, width = 4)

    window.buttonEb3 = Button(window, background = "black", command = lambda: play('4'))
    window.buttonEb3.place(x = 345, y = 390)
    window.buttonEb3.configure(height = 12, width = 4)

    window.buttonGb3 = Button(window, background = "black", command = lambda: play('7'))
    window.buttonGb3.place(x = 425, y = 390)
    window.buttonGb3.configure(height = 12, width = 4)

    window.buttonAb3 = Button(window, background = "black", command = lambda: play('9'))
    window.buttonAb3.place(x = 465, y = 390)
    window.buttonAb3.configure(height = 12, width = 4)

    window.buttonBb3 = Button(window, background = "black", command = lambda: play('minus'))
    window.buttonBb3.place(x = 505, y = 390)
    window.buttonBb3.configure(height = 12, width = 4)

    window.buttonDb4 = Button(window, background = "black", command = lambda: play('w'))
    window.buttonDb4.place(x = 585, y = 390)
    window.buttonDb4.configure(height = 12, width = 4)

    window.buttonEb4 = Button(window, background = "black", command = lambda: play('r'))
    window.buttonEb4.place(x = 625, y = 390)
    window.buttonEb4.configure(height = 12, width = 4)

    window.buttonGb4 = Button(window, background = "black", command = lambda: play('u'))
    window.buttonGb4.place(x = 705, y = 390)
    window.buttonGb4.configure(height = 12, width = 4)

    window.buttonAb4 = Button(window, background = "black", command = lambda: play('o'))
    window.buttonAb4.place(x = 745, y = 390)
    window.buttonAb4.configure(height = 12, width = 4)

    window.buttonBb4 = Button(window, background = "black", command = lambda: play('bracketleft'))
    window.buttonBb4.place(x = 785, y = 390)
    window.buttonBb4.configure(height = 12, width = 4)

    window.buttonDb5 = Button(window, background = "black", command = lambda: play('s'))
    window.buttonDb5.place(x = 865, y = 390)
    window.buttonDb5.configure(height = 12, width = 4)

    window.buttonEb5 = Button(window, background = "black", command = lambda: play('f'))
    window.buttonEb5.place(x = 905, y = 390)
    window.buttonEb5.configure(height = 12, width = 4)

    window.buttonGb5 = Button(window, background = "black", command = lambda: play('j'))
    window.buttonGb5.place(x = 985, y = 390)
    window.buttonGb5.configure(height = 12, width = 4)

    window.buttonAb5 = Button(window, background = "black", command = lambda: play('l'))
    window.buttonAb5.place(x = 1025, y = 390)
    window.buttonAb5.configure(height = 12, width = 4)

    window.buttonBb5 = Button(window, background = "black", command = lambda: play('apostrophe'))
    window.buttonBb5.place(x = 1065, y = 390)
    window.buttonBb5.configure(height = 12, width = 4)

    pass

if True: # Effect director

    def default(wave):

        return wave
    
    def rustle(wave):

        for idx in range(0, len(wave)): wave[idx] *= random.uniform(0.800, 0.900)

        return wave

    def shuffle(wave):

        # adjust = 0.1 / (float(len(wave)) // 2) ** (1 / random.randrange(50, 250) * 100)

        # for idx in range(0, len(wave) // 2): 
            
        #     wave[idx] *= (1 - (idx * adjust))
        #     wave[len(wave) - 1 - idx] *= (1 - (idx * adjust))

        CONSTANT = 28
        rev = int(np.ceil(len(wave) / CONSTANT))
        window = blackman(rev)
        print(len(window))

        for idx in range(rev, len(wave), rev): wave[idx - rev: idx] *= window

        return wave
    
    def BassShift(wave, cutoff = 1000, fs = 44100, order = 5):

        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype = 'low', analog = False)
        wave = lfilter(b, a, wave)

        return wave
    
    def TenorShift(wave, cutoff = 1000, fs = 44100, order = 5):

        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype = 'high', analog = False)
        wave = lfilter(b, a, wave)

        return wave
    
    def delay(wave, delay_ms = 50, decay = 0.38, framerate = 44100):

        delayed = np.zeros(len(wave))
        hop = int(delay_ms / 1000.0 * framerate)
        delayed[:hop] = wave[:hop]
        for idx in range(hop, len(wave)): delayed[idx] = delayed[idx - hop] * decay + wave[idx]

        # delay_samples = int(fs * (delay_ms / 1000.0)) 
        # delayed_wave = np.zeros(len(wave) + delay_samples) 
        # delayed_wave[:len(wave)] = wave 
        # delayed_wave[delay_samples:] += decay * wave

        return delayed
    
    def DoubleClash(wave):

        return wave[::-1]

    EffectDirector = {

        "default": default,
        "rustle": rustle,
        "shuffle": shuffle,
        "BassShift": BassShift,
        "TenorShift": TenorShift,
        "delay": delay, 
        "DoubleClash": DoubleClash
    }

if True: # Kit director

    KitDict = {
        
        'Q': "C://Users//studi//OneDrive//桌面//PyStuff//TrainingData//Kit//Kick.wav", # Drum
        'W': "C://Users//studi//OneDrive//桌面//PyStuff//TrainingData//Kit//Snare.wav", # Snare
        'E': "C://Users//studi//OneDrive//桌面//PyStuff//TrainingData//Kit//High-Hat.wav", # Hi-hat
        'R': "C://Users//studi//OneDrive//桌面//PyStuff//TrainingData//Kit//Clap.wav", # Clap
        'T': "C://Users//studi//OneDrive//桌面//PyStuff//TrainingData//Kit//808.wav", # 808

    }

    for KitKey, KitPath in KitDict.items():
        
        if not isinstance(KitPath, str): break

        if os.path.exists(KitPath):

            fw = open(KitPath, 'rb')
            params = fw.getparams()
            nchannels, sampwidth, framerate, nframes = params[:4]
            strData = fw.readframes(nframes)
            waveData = np.frombuffer(strData, dtype=np.int16)
            waveData = waveData * 1.0 / max(abs(waveData))
            fw.close()

            Wave = waveData.tolist()
            KitDirector[KitKey] = Wave

if True: # Modifier director

    ModDict = {

        'Up': 12,
        'Down': -12,
        'Left': -1,
        'Right': 1,
    }

if True: # Key director

    KeyDict = { # C2 ~ B5
        
        'F1': 65.41, 
        'F2': 69.3, 
        'F3': 73.42, 
        'F4': 77.78, 
        'F5': 82.41, 
        'F6': 87.31, 
        'F7': 92.5, 
        'F8': 98.0, 
        'F9': 103.83, 
        'F11': 110.0, 
        'F12': 116.54, 
        'Delete': 123.47, 
        '1': 130.81, 
        '2': 138.59, 
        '3': 146.83, 
        '4': 155.56, 
        '5': 164.81, 
        '6': 174.61, 
        '7': 185.0, 
        '8': 196.0, 
        '9': 207.65, 
        '0': 220.0, 
        'minus': 233.08, 
        'equal': 246.94, 
        'q': 261.63, 
        'w': 277.18, 
        'e': 293.66, 
        'r': 311.13, 
        't': 329.63, 
        'y': 349.23, 
        'u': 369.99, 
        'i': 392.0, 
        'o': 415.3, 
        'p': 440.0, 
        'bracketleft': 466.16, 
        'bracketright': 493.88, 
        'a': 523.25, 
        's': 554.37, 
        'd': 587.33, 
        'f': 622.25, 
        'g': 659.25,
        'h': 698.46,
        'j': 739.99,
        'k': 783.99,
        'l': 830.61,
        'semicolon': 880.0,
        'apostrophe': 932.33,
        'Return': 987.77,
    }

if True: # Channel director

    ChannelDict = {

        'c': None, 
        'v': None,
        'b': None,
        'n': None,

    }

if True: # Record director

    RecordDict = {

        'm': None,
        'comma': None,
        'period': None,
        'slash': None,

        'M': 'm',
        'lesser': 'comma',
        'greater': 'period',
        'question': 'slash',
        
    }
        
if True: # Place director

    AttrDict = {

        'c': [100, 30, 15], 
        'v': [200, 25, 15],
        'b': [300, 25, 15],
        'n': [400, 25, 15],

    }

class CircularProgress:

    def __init__(self, canva: Canvas, key: str, duration: np.float64):
        
        self.angle = 0
        self.key = key
        self.canva = canva
        self.running = True
        self.duration = duration
        self.x = AttrDict[key][0]
        self.y = AttrDict[key][1]
        self.radius = AttrDict[key][2]
        
        self.arc = self.canva.create_arc(
            self.x - self.radius, self.y - self.radius, self.x + self.radius, self.y + self.radius,
            start = 90, extent = 0, style = 'arc', outline = 'cyan', width = 5,
        )
        
    def show_arc(self, status: str):
        
        self.running = True
        self.status = status
        
        self.button_radius = self.radius * 1

        self.bg_circle = self.canva.create_oval(

            self.x - self.button_radius, self.y - self.button_radius,
            self.x + self.button_radius, self.y + self.button_radius,
            fill = self.status, outline = ''
        )
        
        self.button_radius = self.radius * 0.9

        self.button = self.canva.create_oval(

            self.x - self.button_radius, self.y - self.button_radius,
            self.x + self.button_radius, self.y + self.button_radius,
            fill='white', tags=f'button_{self.key}'
        )

        self.canva.tag_bind(f'button_{self.key}', '<Button-1>', self.toggle)
        
    def update(self, progress: np.float64):

        if self.running: self.angle = 360 * progress
        else: self.angle = 0
        
        self.canva.itemconfig(self.arc, extent=self.angle)
        
    def toggle(self):

        print(f'{self.angle} in degree' )
        self.running = not self.running
        self.angle = 0
        self.canva.itemconfig(self.arc, extent=self.angle)

        if self.running: 
            
            self.show_arc(START)
            animate(self)
            
    def reset(self):

        self.angle = 0
        self.running = False
        print("Resetting...")
        self.show_arc(ERASE)

        self.angle = 0
        self.canva.itemconfig(self.arc, extent=self.angle)

def animate(track: CircularProgress)-> None: # We will proceed to this in the future

    start_time = time.time(); print("Animating...")
    def update():

        elapsed = (time.time() - start_time) % track.duration
        progress = elapsed / track.duration
        track.update(progress)
        if not track.running: return
        
        track.canva.after(10, update)
    
    update()

def generate_wave(frequency: float, duration: float)-> np.ndarray:

    t = np.linspace(0, duration, int(44100 * duration), False)
    wave = np.zeros_like(t)

    for i in range(0, 20):

        wave += overtone[i] * 0.5 * np.sin(2 * np.pi * frequency * (i + 1) * t)

    return wave

def recKey(multiple: int, key: str, wave: np.ndarray)-> None: # OutputFX

    global SoundArray, RecordingStartTime, recording_lock

    if RecordingStartTime is None:  RecordingStartTime = time.time()
    start = int((time.time() - RecordingStartTime) * 44100)
    # index = 0

    Wave = EffectDirector[OutputEffect](wave.copy())

    while key in keys:

        end = start + len(Wave)
        
        with recording_lock:

            if (end <= len(SoundArray)):

                SoundArray[start:end] += Wave

            elif (start <= len(SoundArray)):

                SoundArray[start:] += Wave[:len(SoundArray) - start]
                SoundArray = np.append(SoundArray, Wave[len(SoundArray) - start:])

            else:

                SoundArray = np.append(SoundArray, [0] * (start - len(SoundArray)))
                SoundArray = np.append(SoundArray, Wave)
                
        start = end
        time.sleep(multiple / (KeyDict[key] * (OCTAVE ** octave)))

    # while key in keys:

    #     with recording_lock:

    #         if (start < len(SoundArray)): SoundArray[start] += Wave[index]

    #         else:

    #             SoundArray = np.append(SoundArray, [0] * (start - len(SoundArray) + 1))
    #             SoundArray[start] += Wave[index]

    #     index += 1; index %= len(Wave); 
    #     if index == 0: print(index)
    #     start += 1
    #     time.sleep(1 / 44100)

def recKit(Wave: list)-> None:

    global RecordingStartTime, SoundArray, recording_lock
    if RecordingStartTime is None: RecordingStartTime = time.time()
    now = time.time()
    start = int((now - RecordingStartTime) * 44100)
    end = start + len(Wave)
    
    with recording_lock:
        
        if (end < len(SoundArray)):

            SoundArray[start:end] += Wave

        elif (start < len(SoundArray)):

            SoundArray[start:] += Wave[:len(SoundArray) - start]
            SoundArray = np.append(SoundArray, Wave[len(SoundArray) - start:])

        else:

            SoundArray = np.append(SoundArray, [0] * (start - len(SoundArray)))
            SoundArray = np.append(SoundArray, Wave)

def recMod(effect: str = "default")-> None: # Main Focus

    global TrackArray, stream
    CHUNK = 1024

    while RecordON:

        data = stream.read(CHUNK)
        audio_data = np.frombuffer(data, dtype=np.int16) / 32768.0
        audio_data *= 100
        TrackArray = np.append(TrackArray, audio_data)

        WindowFilter = blackman(CHUNK)
        Magnifier = 10
        y = window_on(0, CHUNK, audio_data, WindowFilter, CHUNK, Magnifier)

        x = fftfreq(CHUNK, 1/44100)[:CHUNK // 2]
        peaks, _ = find_peaks(y, height = 0.1)
        FF = x[peaks]; 

        if (len(FF) != 0): print(FF[0])
        else: FF = [-1]

        global RecordingStartTime, SoundArray, recording_lock

        if RecordingStartTime is None: RecordingStartTime = time.time()
        now = time.time()
        start = int((now - RecordingStartTime) * 44100)
        
        SoundArray = np.array([])

        if (FF[0] != -1): Wave = generate_wave(FF[0], np.round((CHUNK / framerate) / (1 / FF[0])))
        else: Wave = np.zeros(CHUNK + (len(SoundArray) % CHUNK))

        with recording_lock: SoundArray = np.append(SoundArray, Wave)

    # global TrackTimbre
    # TrackTimbre = tt

def recVTs(wave: np.ndarray)-> None: # Main Focus

    VoiceVoulme = np.array([])
    pl.subplot(111)
    # pl.plot(np.arange(len(wave)), wave, color = "b", label = "Original Wave")

    global SoundArray

    length = len(wave)
    CHUNK = 1024
    
    temp = np.array([])
    
    for idx in range(0, length, CHUNK): 
        
        Wave = wave[idx % length : (idx + CHUNK) % length]

        data = stream.read(CHUNK, exception_on_overflow = False)
        audio_data = np.frombuffer(data, dtype = np.int16) / 32768.0
        audio_data *= 1000
        
        vol = 0
        for index in range(0, CHUNK): vol += (audio_data[index] ** 2)
        vol = 10 * np.log10(vol); print(vol)
        if vol <= 0: vol = 0
        
        multiple = (vol) # * 0.05
        if multiple >= 1: multiple = 1
        else: multiple = 0
        print(multiple)
        VoiceVoulme = np.append(VoiceVoulme, [multiple] * CHUNK)
        
        temp = np.append(temp, Wave * multiple)

    SoundArray = np.array(temp[:length])
    VoiceVoulme = VoiceVoulme[:length]
    pl.plot(np.arange(len(VoiceVoulme)), VoiceVoulme, color = "red", label = "Threshold")
    pl.plot(np.arange(len(SoundArray)), SoundArray, color = "cyan", label = "Voice Trigger")

    pl.grid()
    pl.legend()

    pl.show()
    
def play_sound(key: str)-> None: # InputFX

    frequency = KeyDict[key] * (OCTAVE ** octave)
    multiple = 1
    duration = multiple / frequency
    wave = generate_wave(frequency, duration)

    global InputEffect
    Wave = EffectDirector[InputEffect](wave.copy())
    stereo_Wave = np.column_stack((Wave, Wave))

    sound = pygame.sndarray.make_sound((stereo_Wave * 32767).astype(np.int16))

    for i in range(pygame.mixer.get_num_channels()):

        channel = pygame.mixer.Channel(i)
    
        if not channel.get_busy():

            KeyDirector[key] = channel
            break

    if key in KeyDirector:

        KeyDirector[key].play(sound, loops = -1)
        threading.Thread(target=recKey, args=(multiple, key, Wave, )).start()

def play(key: str)-> None: 
    
    thread = threading.Thread(target = play_sound, args=(key,))
    thread.daemon = True
    thread.start()

def playKit(channel: pygame.mixer.Channel, Wave: list)-> None:
    
    stereo_Wave = np.column_stack((Wave, Wave))
    sound = pygame.sndarray.make_sound((stereo_Wave * 32767).astype(np.int16))
    channel.play(sound)

def interrupt_sound(event: Event = None)-> None: # Don't remove the 'event' argument

    global SynthON

    if SynthON:

        pygame.mixer.stop()
        KeyDirector.clear()

def sunken(key: str)-> None:
    
    if key == 'F1': window.buttonC2.configure(relief = SUNKEN)
    if key == 'F2': window.buttonDb2.configure(relief = SUNKEN)
    if key == 'F3': window.buttonD2.configure(relief = SUNKEN)
    if key == 'F4': window.buttonEb2.configure(relief = SUNKEN)
    if key == 'F5': window.buttonE2.configure(relief = SUNKEN)
    if key == 'F6': window.buttonF2.configure(relief = SUNKEN)
    if key == 'F7': window.buttonGb2.configure(relief = SUNKEN)
    if key == 'F8': window.buttonG2.configure(relief = SUNKEN)
    if key == 'F9': window.buttonAb2.configure(relief = SUNKEN)
    if key == 'F11': window.buttonA2.configure(relief = SUNKEN)
    if key == 'F12': window.buttonBb2.configure(relief = SUNKEN)
    if key == 'Delete': window.buttonB2.configure(relief = SUNKEN)
    if key == '1': window.buttonC3.configure(relief = SUNKEN)
    if key == '2': window.buttonDb3.configure(relief = SUNKEN)
    if key == '3': window.buttonD3.configure(relief = SUNKEN)
    if key == '4': window.buttonEb3.configure(relief = SUNKEN)
    if key == '5': window.buttonE3.configure(relief = SUNKEN)
    if key == '6': window.buttonF3.configure(relief = SUNKEN)
    if key == '7': window.buttonGb3.configure(relief = SUNKEN)
    if key == '8': window.buttonG3.configure(relief = SUNKEN)
    if key == '9': window.buttonAb3.configure(relief = SUNKEN)
    if key == '0': window.buttonA3.configure(relief = SUNKEN)
    if key == 'minus': window.buttonBb3.configure(relief = SUNKEN)
    if key == 'equal': window.buttonB3.configure(relief = SUNKEN)
    if key == 'q': window.buttonC4.configure(relief = SUNKEN)
    if key == 'w': window.buttonDb4.configure(relief = SUNKEN)
    if key == 'e': window.buttonD4.configure(relief = SUNKEN)
    if key == 'r': window.buttonEb4.configure(relief = SUNKEN)
    if key == 't': window.buttonE4.configure(relief = SUNKEN)
    if key == 'y': window.buttonF4.configure(relief = SUNKEN)
    if key == 'u': window.buttonGb4.configure(relief = SUNKEN)
    if key == 'i': window.buttonG4.configure(relief = SUNKEN)
    if key == 'o': window.buttonAb4.configure(relief = SUNKEN)
    if key == 'p': window.buttonA4.configure(relief = SUNKEN)
    if key == 'bracketleft': window.buttonBb4.configure(relief = SUNKEN)
    if key == 'bracketright': window.buttonB4.configure(relief = SUNKEN)
    if key == 'a': window.buttonC5.configure(relief = SUNKEN)
    if key == 's': window.buttonDb5.configure(relief = SUNKEN)
    if key == 'd': window.buttonD5.configure(relief = SUNKEN)
    if key == 'f': window.buttonEb5.configure(relief = SUNKEN)
    if key == 'g': window.buttonE5.configure(relief = SUNKEN)
    if key == 'h': window.buttonF5.configure(relief = SUNKEN)
    if key == 'j': window.buttonGb5.configure(relief = SUNKEN)
    if key == 'k': window.buttonG5.configure(relief = SUNKEN)
    if key == 'l': window.buttonAb5.configure(relief = SUNKEN)
    if key == 'semicolon': window.buttonA5.configure(relief = SUNKEN)
    if key == 'apostrophe': window.buttonBb5.configure(relief = SUNKEN)
    if key == 'return': window.buttonB5.configure(relief = SUNKEN)
     
def raised(key: str)-> None:

    if key == 'F1': window.buttonC2.configure(relief = RAISED)
    if key == 'F2': window.buttonDb2.configure(relief = RAISED)
    if key == 'F3': window.buttonD2.configure(relief = RAISED)
    if key == 'F4': window.buttonEb2.configure(relief = RAISED)
    if key == 'F5': window.buttonE2.configure(relief = RAISED)
    if key == 'F6': window.buttonF2.configure(relief = RAISED)
    if key == 'F7': window.buttonGb2.configure(relief = RAISED)
    if key == 'F8': window.buttonG2.configure(relief = RAISED)
    if key == 'F9': window.buttonAb2.configure(relief = RAISED)
    if key == 'F11': window.buttonA2.configure(relief = RAISED)
    if key == 'F12': window.buttonBb2.configure(relief = RAISED)
    if key == 'Delete': window.buttonB2.configure(relief = RAISED)
    if key == '1': window.buttonC3.configure(relief = RAISED)
    if key == '2': window.buttonDb3.configure(relief = RAISED)
    if key == '3': window.buttonD3.configure(relief = RAISED)
    if key == '4': window.buttonEb3.configure(relief = RAISED)
    if key == '5': window.buttonE3.configure(relief = RAISED)
    if key == '6': window.buttonF3.configure(relief = RAISED)
    if key == '7': window.buttonGb3.configure(relief = RAISED)
    if key == '8': window.buttonG3.configure(relief = RAISED)
    if key == '9': window.buttonAb3.configure(relief = RAISED)
    if key == '0': window.buttonA3.configure(relief = RAISED)
    if key == 'minus': window.buttonBb3.configure(relief = RAISED)
    if key == 'equal': window.buttonB3.configure(relief = RAISED)
    if key == 'q': window.buttonC4.configure(relief = RAISED)
    if key == 'w': window.buttonDb4.configure(relief = RAISED)
    if key == 'e': window.buttonD4.configure(relief = RAISED)
    if key == 'r': window.buttonEb4.configure(relief = RAISED)
    if key == 't': window.buttonE4.configure(relief = RAISED)
    if key == 'y': window.buttonF4.configure(relief = RAISED)
    if key == 'u': window.buttonGb4.configure(relief = RAISED)
    if key == 'i': window.buttonG4.configure(relief = RAISED)
    if key == 'o': window.buttonAb4.configure(relief = RAISED)
    if key == 'p': window.buttonA4.configure(relief = RAISED)
    if key == 'bracketleft': window.buttonBb4.configure(relief = RAISED)
    if key == 'bracketright': window.buttonB4.configure(relief = RAISED)
    if key == 'a': window.buttonC5.configure(relief = RAISED)
    if key == 's': window.buttonDb5.configure(relief = RAISED)
    if key == 'd': window.buttonD5.configure(relief = RAISED)
    if key == 'f': window.buttonEb5.configure(relief = RAISED)
    if key == 'g': window.buttonE5.configure(relief = RAISED)
    if key == 'h': window.buttonF5.configure(relief = RAISED)
    if key == 'j': window.buttonGb5.configure(relief = RAISED)
    if key == 'k': window.buttonG5.configure(relief = RAISED)
    if key == 'l': window.buttonAb5.configure(relief = RAISED)
    if key == 'semicolon': window.buttonA5.configure(relief = RAISED)
    if key == 'apostrophe': window.buttonBb5.configure(relief = RAISED)
    if key == 'return': window.buttonB5.configure(relief = RAISED)

def press(event: Event)-> None: 
    
    key = event.keysym
    if key in keys: return
    kbs.add(key)
    print(key, end = " --- ")
    for k in keys: print(k, end = " ")
    print("\n")

    if key not in keys:
            
        keys.add(key)

        if key in KeyDict:

            sunken(key)
            play(key)

    global SynthON

    if key == 'Escape': 
        
        window.destroy()

    elif key in ModDict: 
        
        global octave
        octave += ModDict[key]

    elif key in RecordDict:
    
        if RecordDict[key] == None: # No track in lane
            
            global TrackArray, RecordON, track, stream

            if RecordON or len(TrackArray) > 0: # Stop recording (Currently streaming)
                
                RecordON = False

                temp = DeNoiser.DeNoise(Denoiser, TrackArray[:], preset = "music")
                stereo_Wave = np.column_stack((temp, temp))
                sound = pygame.sndarray.make_sound((stereo_Wave * 32767).astype(np.int16))

                for i in range(pygame.mixer.get_num_channels()):

                    channel = pygame.mixer.Channel(i)
                
                    if not channel.get_busy():

                        RecordDirector[key] = channel
                        RecordDict[key] = [temp, sound]
                        break

                print("Stored in " + key)

                stream.stop_stream()
                stream.close()
                track.terminate()

                TrackArray = np.array([])
                track = pyaudio.PyAudio()
                stream = track.open(format = pyaudio.paInt16, channels = 1, rate = 44100, input = True, frames_per_buffer = 1024)
                RecordDirector[key].play(sound, loops = -1)

                # global TrackTimbre

                # global TbrDict
                # name = 'Record' + (str)(random.random())
                # AMP = (TrackTimbre / np.sum(TrackTimbre)); print(AMP)
                # TbrDict[name] = AMP
                # window.timbre.insert(END, name)

            elif type(RecordDict[key]) == str: # Remove the track

                RecordDirector[RecordDict[key]].stop()
                RecordDict[RecordDict[key]] = None
                print("Removed " + RecordDict[key])

            elif type(RecordDict[key]) == list: # Restart / Stop the track
                
                if RecordDirector[key].get_busy(): RecordDirector[key].stop()
                
                else: RecordDirector[key].play(sound, -1)

            else: # Record

                RecordON = True
                effect = InputEffect
                threading.Thread(target = recMod, args = (effect, )).start()

        elif type(RecordDict[key]) == list: # Stop / Start

                if RecordDirector[key].get_busy(): 
                    
                    RecordDirector[key].stop()

                else: 
                    
                    RecordDirector[key].play(RecordDict[key][1], loops = -1)

        elif type(RecordDict[key]) == str: # Erase track
            
            if (RecordDict[key] in RecordDirector):
                
                RecordDirector[RecordDict[key]].stop()

                RecordDict[RecordDict[key]] = None
                RecordDirector[key] = [track, stream]

    elif key in ChannelDict: # Lowercase: Record/Stop/Play
        
        if 'Control_R' in kbs: # Voice trigger

            recVTs(ChannelDict[key.lower()][0])
            keys.remove('Control_L')

            global SoundArray

            temp = SoundArray[:]
            stereo_Wave = np.column_stack((temp, temp))
            sound = pygame.sndarray.make_sound((stereo_Wave * 32767).astype(np.int16))

            ChannelDirector[key.lower()].stop()
            ChannelDict[key.lower()] = [temp, sound]

            ChannelDirector[key.lower()].play(sound, loops = -1)

            print("DONE")       

        elif ChannelDict[key] == None: # No track in lane
            
            switch(None, key)

        elif ChannelDirector[key].get_busy(): # Track playing
            
            ChannelDirector[key].stop()
            ChannelDict[key][2].toggle()

        else: # Track Resuming
            
            ChannelDirector[key].play(ChannelDict[key][1], loops = -1)
            ChannelDict[key][2].toggle()

    elif key.lower() in ChannelDict and key.lower() in ChannelDirector: # Upeercase (without SHIFT): Erase
        
        if ChannelDirector[key.lower()].get_busy(): ChannelDirector[key.lower()].stop()
        if ChannelDict[key.lower()] != None: ChannelDict[key.lower()][2].reset()
        ChannelDict[key.lower()] = None

    elif SynthON and not key == 'x':
        
        if key == 'Shift_L' or key == 'Shift_R' or key == 'Win_L': 
            
            switch()
        
        elif mixture == 0:

            interrupt_sound()
            keys.clear()
            window.play.destroy()
            window.play = Button(window, text = "Play", command = None)
            window.play.place(x = 195, y = 300)
            window.play.configure(width = 10)
            SynthON = False
            messagebox.showwarning('Warning', 'Choose your timbre to play by.')
        
        elif key in KitDict:

            channel = None

            for i in range(pygame.mixer.get_num_channels()):

                channel = pygame.mixer.Channel(i)
            
                if not channel.get_busy(): break

            threading.Thread(target = recKit, args = (KitDirector[key], )).start()
            threading.Thread(target = playKit, args = (channel, KitDirector[key], )).start()

def release(event: Event)-> None: 
    
    key = event.keysym
    if key in kbs: kbs.remove(key)

    if key in keys:

        keys.remove(key)

        if key in KeyDirector: 
            
            KeyDirector[key].fadeout(10)
            KeyDirector.pop(key)

        if key in KeyDict: 
            
            raised(key)

def window_on(start: int, end: int, data: np.ndarray, WindowFilter: np.ndarray, frameSize: int = 1024, Magnify: int = 10)-> np.ndarray:

    data_raw = data[start:end] * WindowFilter
    FFT = fft(data_raw)
    y = Magnify * 2.0 / frameSize * np.abs(FFT)[:frameSize//2]

    return y

def getAmplitude(y: np.ndarray, frequency: int, framerate: int, frameSize: int)-> float:

    index = frequency / (framerate / frameSize)

    lower_bin = int(np.floor(index))
    upper_bin = int(np.ceil(index))

    lower_value = np.abs(y[lower_bin])
    upper_value = np.abs(y[upper_bin])

    return lower_value + (upper_value - lower_value) * (index - lower_bin)

def volLimit(waveData: np.ndarray, multiplier: float = 0.75, frameSize: int = 2048, overlap: int = 1024)-> list:

    waveCollection = []
    volume = []
    i = 0

    # pl.subplot(211)
    # pl.plot(np.arange(len(waveData)), waveData, color = "black")

    while (i + frameSize < len(waveData)):

        vol = 10 * np.log10(sum(waveData[i + j] ** 2 for j in range(frameSize))) 
        volume.extend([vol] * (frameSize - overlap))
        i += frameSize - overlap

    volumetemp = volume[:]
    np.sort(volume)

    Pr97 = np.percentile(volume, 97)
    Pr03 = np.percentile(volume,  3)
    gate = (Pr97 - Pr03) * multiplier + Pr03

    marker = 0

    for i in range(0, len(volume) - framerate, frameSize):

        if (volume[i] <= gate and volume[i + frameSize] >= gate):

            marker = i
            # print(i, end = " ~ ")

        elif (volume[i] >= gate and volume[i + frameSize] <= gate):

            # pl.axvline(x = marker, color = 'red', linestyle = '--')
            # pl.axvline(x = i, color = 'cyan', linestyle = '--')

            waveCollection.append(waveData[marker : i])
            marker = -1
            # print(i)

    if marker != -1: 
        
        # print(len(waveData) - 1)
        waveCollection.append(waveData[marker:])

    # pl.grid()

    # pl.subplot(212)
    # pl.plot(np.arange(len(volumetemp)), volumetemp)

    # pl.grid()

    # pl.show()

    return waveCollection

def extractAMP(waveData: np.ndarray, framerate: int, frameSize: int = 2048, limitON: bool = True)-> np.ndarray:
    
    if limitON: waveCollection = volLimit(waveData, multiplier = 0.75)
    else: waveCollection = [waveData]

    overlap = frameSize // 6 * 5
    WindowFilter = blackman(frameSize)
    Magnifier = 10
    running = True
    start = 0
    ratio = np.zeros(20)

    for data in waveCollection:

        while (running):
            
            end = start + frameSize; 
            if end > len(data): break

            if not running: WindowFilter = blackman(end - start)
            y = window_on(start, end, data, WindowFilter, frameSize, Magnifier)

            x = fftfreq(frameSize, 1/44100)[:frameSize // 2]
            peaks, _ = find_peaks(y, height = 0.1)
            FF = x[peaks]
            # print(FF)

            if len(FF) > 0 and FF[0] < 7902.13 and FF[0] > 27.50:
                
                ratios = []

                for i in range(0, 20):

                    ratios.append(getAmplitude(y, FF[0] * (i + 1), framerate, frameSize) / Magnifier)
                
                total = np.sum(ratios)
                for i in range(0, 20): ratios[i] /= total

                for i in range(0, 20):

                    ratio[i] += (ratios[i] / np.sum(ratios))

            if len(FF) == 0 or FF[0] == 0: start += (frameSize - overlap)

            else: start += int((((frameSize - overlap) // (framerate // FF[0])) + 1) * (framerate // FF[0]))

        ratio = ratio / np.sum(ratio) # Normalize (μ=0)
    print(ratio)
    return ratio

def buildAMP(path: str)-> None:

    try:
        
        window.entry.delete(0, 'end')

        if path == "None": return

        if os.path.basename(path) in window.timbre.get(0, END):

            messagebox.showinfo('Same Thoughts!', 'This file has already been processed.') 
            return

        fw = open(path, 'rb')
        params = fw.getparams()
        nchannels, sampwidth, framerate, nframes = params[:4]
        strData = fw.readframes(nframes)
        waveData = np.frombuffer(strData, dtype=np.int16)
        waveData = waveData * 1.0 / max(abs(waveData))
        fw.close()

        print(f"Processing {os.path.basename(path)}")

        AMP = extractAMP(waveData, framerate, 2048, limitON = True)
        # print('AMP = [', end = "")
        # print(", ".join(map(str, AMP)), end = "]\n")
        
        window.timbre.insert(END, os.path.basename(path))

        global TbrDict
        TbrDict[str(os.path.basename(path))] = AMP

    except PermissionError: messagebox.showerror('PermissionError', 'Access denied.') 
        
    except FileNotFoundError: messagebox.showerror('FileNotFoundError', 'No file matches the given path.') 

    except OSError: messagebox.showerror('OSError', 'Path invalid.'); # [window.after(i * 100, PathError) for i in range(1000)]

def PathError()-> None:

    import random
    from tkinter import Toplevel, Label

    top = Toplevel() 
    top.title("Uh-oh")
    top.iconbitmap('Error.ico')
    x = random.randint(0, window.winfo_screenwidth() - 200) 
    y = random.randint(0, window.winfo_screenheight() - 100) 
    top.geometry(f"200x40+{x}+{y}") 
    top.label = Label(top, text = "Your computer is sabotaged!")
    top.label.pack()
    #  messagebox.showerror('', '') 
    #  top.destroy()

def search_path()-> None:

    global path
    path = window.entry.get()
    print(path)
    # try:

        

    # except Exception as e:

    #     messagebox.showerror('Error', f'Choose .wav file to run. {e}')
    #     return
    
    if path == "": 

            path = str(filedialog.askopenfile(initialdir= os.getcwd(), defaultextension = ".wav"))
            
            for idx in range(25, len(path)):

                if path[idx] == '\'': path = path[25:idx]; break
                
    print(path)
    buildAMP(path)

def switch(event: Event = None, Channel = None):

    window.play.destroy()

    global SynthON

    if not SynthON: window.play = Button(window, text = "Stop", command = None)

    else: 
        
        window.play = Button(window, text = "Play", command = switch)

        global RecordingStartTime, SoundArray

        if not (RecordingStartTime == None):

            start = int((time.time() - RecordingStartTime) * 44100)
            SoundArray = np.append(SoundArray, [0] * (start - len(SoundArray)))

            if Channel == None:

                for channel in KeyDirector.values(): channel.stop()
                KeyDirector.clear() 

            else:
                
                temp = SoundArray[:int((time.time() - RecordingStartTime) * 44100)]
                Wave = EffectDirector[OutputEffect](temp.copy())
                # print(len(temp))

                indicator = CircularProgress(window.canvas, Channel, len(Wave) / framerate)
                indicator.show_arc(START)
                animate(indicator)

                stereo_Wave = np.column_stack((Wave, Wave))
                sound = pygame.sndarray.make_sound((stereo_Wave * 32767).astype(np.int16))

                for i in range(pygame.mixer.get_num_channels()):

                    channel = pygame.mixer.Channel(i)
                
                    if not channel.get_busy():

                        ChannelDirector[Channel] = channel
                        ChannelDict[Channel] = [temp, sound, indicator]
                        break

                ChannelDirector[Channel].play(sound, loops = -1)

        SoundArray = np.array([])
        RecordingStartTime = None

    SynthON = not SynthON

    window.play.place(x = 195, y = 250)
    window.play.configure(width = 10)

def formTBR(event: Event)-> None:

    indices = event.widget.curselection()
    marked = set([window.timbre.get(i) for i in indices])

    file = ()
    ratio = ()

    global selection, mixture, overtone

    if len(marked) > len(selection):

        file = str(marked - selection)[2 : len(file) - 2]
        if (mixture == 0): ratio = TbrDict[file]
        else: ratio = [(overtone[idx] * mixture + TbrDict[file][idx]) / (mixture + 1) for idx in range(0, 20)]
        mixture += 1

    else:

        file = str(selection - marked)[2 : len(file) - 2]
        if (mixture == 1): ratio = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        else: ratio = [(overtone[idx] * mixture - TbrDict[file][idx]) / (mixture - 1) for idx in range(0, 20)]
        mixture -= 1

    selection = marked
    overtone = ratio

def formInputFX(event: Event)-> None:

    global InputEffect
    InputEffect = window.InputFX.get(event.widget.curselection()[0])

def formOutputFX(event: Event)-> None:

    global OutputEffect
    OutputEffect = window.OutputFX.get(event.widget.curselection()[0])

def initialize()-> None:

    window.title('PianoGUI')
    window.geometry('845x1980')

    window.canvas = Canvas(window, width = 460, height = 60, bg = "black", highlightthickness = 0)
    window.canvas.place(x = 180, y = 300)
    window.canvas.configure(background = window.cget('background'))

    window.entry = Entry(window)
    window.entry.place(x = 290, y = 80)
    window.entry.configure(width = 50)

    window.search = Button(window, text = "Search", command = search_path)
    window.search.place(x = 180, y = 79)
    window.search.configure(width = 10)

    window.timbre = Listbox(window, selectmode = 'multiple', exportselection = False)
    window.timbre.place(x = 180, y = 120)
    window.timbre.configure(height = 5)
    window.timbre.insert(END, '(Basic)')
    window.timbre.insert(END, '(Prime)')

    window.InputFX = Listbox(window, selectmode = 'single', exportselection = False)
    window.InputFX.place(x = 340, y = 120)
    window.InputFX.configure(height = 5)
    window.InputFX.insert(END, 'default')
    window.InputFX.insert(END, 'rustle')
    window.InputFX.insert(END, 'shuffle')
    window.InputFX.insert(END, 'BassShift')
    window.InputFX.insert(END, 'TenorShift')
    window.InputFX.insert(END, 'delay')
    window.InputFX.insert(END, 'DoubleClash')
    window.InputFX.select_set(0); InputEffect = 'default'

    window.OutputFX = Listbox(window, selectmode = 'single', exportselection = False)
    window.OutputFX.place(x = 500, y = 120)
    window.OutputFX.configure(height = 5)
    window.OutputFX.insert(END, 'default')
    window.OutputFX.insert(END, 'rustle')
    window.OutputFX.insert(END, 'shuffle')
    window.OutputFX.insert(END, 'BassShift')
    window.OutputFX.insert(END, 'TenorShift')
    window.OutputFX.insert(END, 'delay')
    window.OutputFX.insert(END, 'DoubleClash')
    window.OutputFX.select_set(0); OutputEffect = 'default'

    window.play = Button(window, text = "Play", command = None)
    window.play.place(x = 195, y = 250)
    window.play.configure(width = 10)

    window.bind_all('<KeyPress>', press)
    window.bind_all('<KeyRelease>', release)
    window.bind_all('<x>', switch)
    window.bind_all('<X>', switch)
    window.timbre.bind('<<ListboxSelect>>', formTBR)
    window.InputFX.bind('<<ListboxSelect>>', formInputFX)
    window.OutputFX.bind('<<ListboxSelect>>', formOutputFX)

initialize()

window.mainloop()