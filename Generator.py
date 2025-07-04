# name = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B']
# num = ['2', '3', '4', '5']
# seq = []
# for b in num:
#     for a in name:
#         seq.append(a + b)
# print("seq = [", end = "")
# for c in seq: print("'" + c + "'", end = ", ")
# print("]")

# seq = ['C2', 'D2', 'E2', 'F2', 'G2', 'A2', 'B2', 'C3', 'D3', 'E3', 'F3', 'G3', 'A3', 'B3', 'C4', 'D4', 'E4', 'F4', 'G4', 'A4', 'B4', 'C5', 'D5', 'E5', 'F5', 'G5', 'A5', 'B5']
# KeyDict = ['F1', 'F3', 'F5', 'F6', 'F8', 'F11', 'delete', '1', '3', '5', '6', '8', '0', 'equal', 'q', 'e', 't', 'y', 'i', 'p', 'bracketright', 'a', 'd', 'g', 'h', 'k', 'semicolon', 'return']

# x = 5 - 40

# for idx in range(0, len(seq)):

#     x += 40
#     string = seq[idx]
#     color = "white"
    
#     print("window.button" + string + " = Button(window, background = \"" + color + "\", command = lambda: play('" + KeyDict[idx] + "'))\nwindow.button" + string + ".place(x = " +  str(x) + ", y = 390)\nwindow.button" + string + ".configure(height = " + str(20) + ", width = 4)\n")


# seq = ['Db2', 'Eb2', 'Gb2', 'Ab2', 'Bb2', 'Db3', 'Eb3', 'Gb3', 'Ab3', 'Bb3', 'Db4', 'Eb4', 'Gb4', 'Ab4', 'Bb4', 'Db5', 'Eb5', 'Gb5', 'Ab5', 'Bb5']
# KeyDict = ['F2', 'F4', 'F7', 'F9', 'F12', '2', '4', '7', '9', 'minus', 'w', 'r', 'u', 'o', 'bracketleft', 's', 'f', 'j', 'l', 'apostrophe']

# x = 5 - 40 + 20

# for idx in range(0, len(seq)):

#     x += 40
#     color = "black"

#     if (idx % 5 == 2 or (idx % 5 == 0 and idx != 0)): x += 40
    
#     print("window.button" + seq[idx] + " = Button(window, background = \"" + color + "\", command = lambda: play('" + KeyDict[idx] + "'))\nwindow.button" + seq[idx] + ".place(x = " +  str(x) + ", y = 390)\nwindow.button" + seq[idx] + ".configure(height = " + str(12) + ", width = 4)\n")


# import tkinter as tk 
# import sounddevice as sd
# import numpy as np 
# from threading import Thread

# sample_rate = 44100
# duration = 0.5
# frequency_C3 = 130.81
# t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
# wave_C3 = 0.5 * np.sin(2 * np.pi * frequency_C3 * t) * 20

# playing = False

# def play_waveform():

#     print("argh")
#     global playing

#     while playing: 

#         sd.play(wave_C3, sample_rate) 
#         sd.wait() 
        
# def key_press(event): 

#     global playing 

#     if '3' in pressed_keys and 'c' in pressed_keys and not playing: 

#         playing = True; 
#         Thread(target=play_waveform).start()
                
# def key_release(event):
                    
#     global playing 

#     if playing: playing = False 

# pressed_keys = set() 

# root = tk.Tk() 

# button3C = tk.Button(root, text="C3", background="white", command=None)
# button3C.place(x=5, y=390)
# button3C.configure(height=20, width=4)
# button3C.pack()

# root.bind_all('<KeyPress>', lambda event: (pressed_keys.add(event.keysym), key_press(event))) 
# root.bind_all('<KeyRelease>', lambda event: (pressed_keys.discard(event.keysym), key_release(event)))
# root.mainloop()

# seqKey = ["C", "Cs", "D", "Ds", "E", "F", "Fs", "G", "Gs", "A", "As", "B"]
# seqTrg = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0', '-', '=', ]

# for i in [2, 3, 4]:

#     for j in range(0, 12):
    
#         print("if '" + seqTrg[(i - 2) * 12 + j] + "' in keys: Trigger" + str(i) + seqKey[j] + "()")


# seqKey = ["C", "Cs", "D", "Ds", "E", "F", "Fs", "G", "Gs", "A", "As", "B"]

# for i in [2, 3, 4]:

#     for j in range(0, 12):

#         print("def Trigger" + str(i) + seqKey[j] + "(): pass")



# nameFirst = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B']
# KeyDict = ['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'F11', 'F12', 'delete', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0', 'minus', 'equal', 'q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p', 'bracketleft', 'bracketright', 'a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l', 'semicolon', 'apostrophe', 'return']
# frequencies = [
#     65.41,   # C2
#     69.30,   # C#2/Db2
#     73.42,   # D2
#     77.78,   # D#2/Eb2
#     82.41,   # E2
#     87.31,   # F2
#     92.50,   # F#2/Gb2
#     98.00,   # G2
#     103.83,  # G#2/Ab2
#     110.00,  # A2
#     116.54,  # A#2/Bb2
#     123.47,  # B2
#     130.81,  # C3
#     138.59,  # C#3/Db3
#     146.83,  # D3
#     155.56,  # D#3/Eb3
#     164.81,  # E3
#     174.61,  # F3
#     185.00,  # F#3/Gb3
#     196.00,  # G3
#     207.65,  # G#3/Ab3
#     220.00,  # A3
#     233.08,  # A#3/Bb3
#     246.94,  # B3
#     261.63,  # C4
#     277.18,  # C#4/Db4
#     293.66,  # D4
#     311.13,  # D#4/Eb4
#     329.63,  # E4
#     349.23,  # F4
#     369.99,  # F#4/Gb4
#     392.00,  # G4
#     415.30,  # G#4/Ab4
#     440.00,  # A4
#     466.16,  # A#4/Bb4
#     493.88,  # B4
#     523.25,  # C5
#     554.37,  # C#5/Db5
#     587.33,  # D5
#     622.25,  # D#5/Eb5
#     659.25,  # E5
#     698.46,  # F5
#     739.99,  # F#5/Gb5
#     783.99,  # G5
#     830.61,  # G#5/Ab5
#     880.00,  # A5
#     932.33,  # A#5/Bb5
#     987.77   # B5
# ]

# print("KeyDict = [")

# for i in range(0, 48):

#     print("    '" + KeyDict[i] + "': " + str(frequencies[i]) + ", ")


# for i in [2, 3, 4, 5]:

#     for j in range(0, 12):

#         print("if key == '" + KeyDict[j + (i - 2) * 12] + "': window.button" + nameFirst[j] + str(i) + ".configure(relief = RAISED)")

import numpy

SoundArray = numpy.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

def add_to_sound_array(wave, index):


    global SoundArray 

    if SoundArray.size == 0: SoundArray = wave 
    
    else: 
        
        if len(SoundArray) < index + len(wave): 
            
            SoundArray = numpy.pad(SoundArray, (0, (index + len(wave)) - len(SoundArray)), 'constant') 

    for idx in range(0, len(wave)):

        SoundArray[index + idx] += wave[idx]

    print(SoundArray)

add_to_sound_array([8, 7, 6, 5, 4, 3, 2], 5)










































































# import tkinter as tk
# import numpy as np
# import pygame
# import threading

# class App:
#     def __init__(self, root):

#         self.root = root
#         self.root.title("Sound Generator")

#         pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
#         pygame.mixer.set_num_channels(16)
#         self.key_to_channel = {}

#         self.pressed_keys = set()
#         self.root.bind('<KeyPress>', self.on_key_press)
#         self.root.bind('<KeyRelease>', self.on_key_release)
#         self.root.bind('<Delete>', self.interrupt_sound)

#         self.key_to_freq = {
#             'a': 261.63,  # C4
#             'w': 277.18,  # C#4
#             's': 293.66,  # D4
#             'e': 311.13,  # D#4
#             'd': 329.63,  # E4
#             'f': 349.23,  # F4
#             't': 369.99,  # F#4
#             'g': 392.00,  # G4
#             'y': 415.30,  # G#4
#             'h': 440.00,  # A4
#             'u': 466.16,  # A#4
#             'j': 493.88,  # B4
#             'k': 523.25,  # C5
#         }

#     def interrupt_sound(self, event): # Don't remove the 'event' argument

#         for channel in self.key_to_channel.values():

#             channel.stop()

#     def on_key_press(self, event):

#         key = event.keysym

#         if key in self.key_to_freq and key not in self.pressed_keys:

#             self.pressed_keys.add(key)
#             threading.Thread(target=self.play_sound, args=(key,)).start()
#             print(threading.active_count())

#     def on_key_release(self, event):

#         key = event.keysym

#         if key in self.pressed_keys:

#             self.pressed_keys.remove(key)

#             if key in self.key_to_channel:

#                 self.key_to_channel[key].stop()

#     def generate_wave(self, frequency, duration, sample_rate=44100):

#         t = np.linspace(0, duration, int(sample_rate * duration), False)
#         wave = 0.5 * np.sin(2 * np.pi * frequency * t)

#         # fade_in = np.linspace(0, 1, int(sample_rate * duration * 0.1))
#         # fade_out = np.linspace(1, 0, int(sample_rate * duration * 0.1))
#         # wave[:len(fade_in)] *= fade_in
#         # wave[-len(fade_out):] *= fade_out

#         stereo_wave = np.column_stack((wave, wave))
#         return stereo_wave

#     def play_sound(self, key):

#         frequency = self.key_to_freq[key]
#         sample_rate = 44100
#         duration = 10 / frequency
#         wave = self.generate_wave(frequency, duration, sample_rate)
#         sound = pygame.sndarray.make_sound((wave * 32767).astype(np.int16))

#         for i in range(pygame.mixer.get_num_channels()):

#             channel = pygame.mixer.Channel(i)
    
#             if not channel.get_busy():

#                 self.key_to_channel[key] = channel
#                 break

#         if key in self.key_to_channel:

#             self.key_to_channel[key].play(sound, -1)

# if __name__ == "__main__":

#     root = tk.Tk()
#     app = App(root)
#     root.mainloop()
