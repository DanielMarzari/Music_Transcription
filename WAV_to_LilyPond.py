# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 10:36:24 2022

@author: Daniel
"""
import os
from scipy.io import wavfile
from scipy.fftpack import rfft, rfftfreq
import matplotlib.pyplot as plt

import numpy as np
from numpy import log2, interp

import math 

import statistics 

from itertools import groupby

TEMPERMENT = 12#12 tone vs 24+ microtonal
TUNING = 440
TEMPO = 155
SUBDIVISIONS = 4
ii16 = np.iinfo(np.int16)

""" 
UPDATE IDEAS
    - Use the change in magnitude to isolate new notes
    - use the change in magnitude to calculate bpm



"""

SCALE_NAMES = {
    12: ["mga","a","mab","b","c","mcd","d","mde","e","f","mfg","g"],
    24: []
}
#Note and keys with that note
KEYS_WITH_NOTE = {
    12: {
        4 : [4, 5, 7, 9, 11, 0, 2],
        5 : [5, 6, 8, 10, 0, 1, 3],
        6 : [6, 7, 9, 11, 1, 2, 4],  
        7 : [7, 8, 10, 0, 2, 3, 5],
        8 : [8, 9, 11, 1, 3, 4, 6],
        9 : [9, 10, 0, 2, 4, 5, 7],
        10 : [10, 11, 1, 3, 5, 6, 8],
        11 : [11, 0, 2, 4, 6, 7, 9],
        0 : [0, 1, 3, 5, 7, 8, 10],
        1 : [1, 2, 4, 6, 8, 9, 11],
        2 : [2, 3, 5, 7, 9, 10, 0],
        3 : [3, 4, 6, 8, 10, 11, 1]
    }
}
#which keys to make sharp and which to make flat (True - SHARP, False - FLAT)
KEY_SHIFT = {
    0: False,
    1: True,
    2: False,
    3: True,
    4: True,
    5: False,
    6: True,
    7: False,
    8: True,
    9: False,
    10: True,
    11: True
}
RHYTHM = {
        1/16  : "16",
        2/16  : "8",
        3/16  : "8.",
        4/16  : "4",
        5/16  : "4~ 16",
        6/16  : "4.",
        7/16  : "4.~ 16",
        8/16  : "2",
        9/16  : "2~ 16",
        10/16 : "2~ 8",
        11/16 : "2~ 8.",
        12/16 : "2.",
        13/16 : "2.~ 16",
        14/16 : "2.~ 8",
        15/16 : "2.~ 8.",
        16/16 : "1"
}        

class MUSIC:
    def __init__(self, path):
        #Vars
        self.quantized_notes = []
        self.notes = []
        self.durations = []
        self.rhythms = []
        
        #Read WAV file
        self.sample_rate, self.data = wavfile.read(path)
        self.file_name = os.path.basename(path)[:-4]
        self.sample_count = self.data.shape[0]
        
        #slice (breaking up recording into smaller samples to analyze)
        self.slice_size = math.floor(self.sample_rate / (60 / TEMPO) / SUBDIVISIONS)
        self.slices = math.floor(self.sample_count / self.slice_size)
        print("%sHz %s slices" % (self.sample_rate, self.slices))
        
        #Force MONO channel 
        if(len(self.data.shape) > 1):
            self.data = np.add(self.data[:,0], self.data[:,1])
        
        #normalize data (map raw onto [-32767, 32767] (16 signed integer - np.int16)
        self.normalized_data = interp(self.data, [self.data.min(), self.data.max()], [ii16.max, ii16.min])
            
    def analyze_data(self):
        #itterate over slices of WAV file
        for i in range(self.slices):
            
            #Get fourier transform data (magnitude, frequency)
            mag = np.abs(rfft(self.normalized_data[self.slice_size*i:self.slice_size*(i+1)]))
            freq = rfftfreq(self.slice_size, 1 / self.sample_rate)
            fourier = list(zip(mag, freq))
            
            #plot for user
            plt.figure(0) #use i for separate graphs
            plt.plot(freq[:500], mag[:500])
            
            #isolate outlier frequencies (ignoreing harmonics!)
            threshold = np.percentile(mag, 99)
            filter1 = [d for d in fourier if (d[0] >= threshold) and (d[1] != 0)]            

            threshold = np.percentile(filter1, 99)
            filter2 = [d for d in fourier if (d[0] >= threshold)]
            
            threshold = np.percentile(filter2, 98)
            outliers = np.array([d for d in filter1 if d[0] >= threshold])
            
            #https://en.wikipedia.org/wiki/Equal_temperament
            #12-TET is approx 
            #Pn = Pa(2**((n-a)/12) WHERE a is A4 and n is the note index
            #n = 12 * (log(F) - log(440)) + 49
            #Get notes from frequency (adjust according to temperment and tuning)
            if(outliers.size > 0):
                self.quantized_notes.append(list(np.unique(
                    np.rint((TEMPERMENT * (log2(outliers[:,1]) - np.log2(TUNING))) + 49))
                ))
            else: #silence in audio
                self.quantized_notes.append([])
        
        #analyze notes and turn into sheet music
        self.analyze_notes()
    
    def analyze_notes(self):
        #approximate key center (most used notes will be in key)
        all_notes = [math.floor(n % TEMPERMENT) for div in self.quantized_notes for n in div]
        note_usage = [[n, all_notes.count(n)] for n in set(all_notes)] 
        note_usage.sort(reverse=True, key=lambda e: e[1])
        
        possible_keys = KEYS_WITH_NOTE[TEMPERMENT][note_usage.pop(0)[0]]
        while((len(possible_keys) > 1) and note_usage):
            remaining_keys = list(set(KEYS_WITH_NOTE[TEMPERMENT][note_usage.pop(0)[0]]) & set(possible_keys))
            if(len(remaining_keys) >= 1): #else ignore 
                possible_keys = remaining_keys
        self.key = possible_keys[0]
        
        print("Key of:", SCALE_NAMES[TEMPERMENT][self.key])
        
        #connect groups (rough fix for now)
        self.groups = [[n, sum(1 for n in group)] for n, group in groupby(self.quantized_notes)]
        
        #Lilypond writeup of notes
        for indx, g in enumerate(self.groups):
            #add a row
            self.notes.append([])
            
            #length in subdivisions 
            duration = g[1] / (SUBDIVISIONS * 4)
            #Lilypond rhythm
            self.rhythms.append("1~" * math.floor(duration) + RHYTHM[duration % 1])            
            
            #account for rests
            if(len(g[0]) == 0):
                self.notes[indx].append("r")
                #lilypond won't tie or imply a rest, we have to respecify
                self.rhythms[indx].replace("~ ", " r")
                
            #scale degree to letter and octave
            for n in g[0]:
                note_name = SCALE_NAMES[TEMPERMENT][math.floor(n % TEMPERMENT)]
                octave = math.floor((n+8)//TEMPERMENT)
                note_octave = ("'" * (octave - 3)) if (octave > 3) else ("," * (3-octave))
                #if sharp or flat
                if('m' in note_name):
                    #if flat get the second note, else get the first note
                    note_name = note_name[(2 if KEY_SHIFT[self.key] else 1)] + ("f" if KEY_SHIFT[self.key] else "s")
                #store note data
                self.notes[indx].append(note_name + note_octave)
                
            #self.music = list(zip(self.notes, self.rhythms))
            self.music = [("<%s>%s" % (" ".join(self.notes[i]), self.rhythms[i])) if (len(self.notes[i]) > 1) else
                          ("%s%s" % (self.notes[i][0], self.rhythms[i])) for i in range(len(self.rhythms))]
            
        print(self.music)
        self.write_to_LilyPond()
        
    def write_to_LilyPond(self):
        #assemble lilypond notation
        output = """
        \\version \"2.20.0\"
        \\language \"english\" 
        \\header {
            title = \"%s\"
        }
        \\absolute {
        \\key %s \\major
        \\clef treble
        %s 
        }
        """ % (self.file_name, SCALE_NAMES[TEMPERMENT][self.key], " ".join(self.music))
        
        #write to file
        f = open("C:\\Users\\Daniel\\Desktop\\WAV_to_MIDI\\test.ly", "w")
        f.write(output)
        f.close()

#TEST 2
V_test = MUSIC("C:\\Users\\Daniel\\Desktop\\WAV_to_MIDI\\Vontmer_Short.wav")

#TEST 3
#CM_test = MUSIC("C:\\Users\\Daniel\\Desktop\\WAV_to_MIDI\\C_Major.wav")

wav_file = V_test
wav_file.analyze_data()

