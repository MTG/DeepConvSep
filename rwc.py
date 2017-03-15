"""
    This file is part of DeepConvSep. 
    It contains classes to access and organize the RWC instrument sound database

    Copyright (c) 2014-2017 Marius Miron  <miron.marius at gmail.com>

    DeepConvSep is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    DeepConvSep is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with DeepConvSep.  If not, see <http://www.gnu.org/licenses/>.
 """

import os
import numpy as np
from scipy import io
import util

class Instrument:
    """
    A class for an instrument in the database RWC

    Parameters
    ----------
    path : string
        The rwc instrument sound path containing 'mat' and 'wav' subfolders
    instid : int
        The instrument id (check the RWC instrument sound instrument intex html)
    allowed_styles : list of strings, optional
        The styles of playing to choose from, e.g ['NO'] uses only normal style of playing.
        (check the RWC instrument sound instrument intex html)
    allowed_dynamics : list of strings, optional
        The dynamics of playing to choose from, e.g ['P'] uses only pianissimo style of playing.
        (check the RWC instrument sound instrument intex html)
    allowed_case : list of strings, optional
        The musicians to choose from, e.g [1] uses only the first musician (3 maximum).
        (check the RWC instrument sound instrument intex html)
    
    """
    def __init__(self, path, instid, allowed_styles=None, allowed_case=None, allowed_dynamics=None):
        self.path = path
        self.instid = instid
        self.total_notes = 0
        if allowed_styles is not None:
            self.allowed_styles = allowed_styles
            self.styles = True
        else:
            self.styles = False 
        if allowed_case is not None:
            self.allowed_case = allowed_case
        else:
            self.allowed_case = [1,2,3]
        if allowed_dynamics is not None:
            self.allowed_dynamics = allowed_dynamics
        else:
            self.allowed_dynamics = ['P','F','M']
        self.getWavs()

    def getWavs(self):
        self.wav_list=[]
        self.notes_code=[]
        self.notes_len=[]
        self.notes_nr=[]
        self.notes_d=[]
        self.notes_s=[]
        self.notes_c=[]
        self.notes=[]
        for case in self.allowed_case:
            for f in os.listdir(os.path.join(self.path,'wav',str(self.instid)+str(case))):
                if f.endswith(".WAV"):
                    if (not self.styles) or (self.styles and any(s in f for s in self.allowed_styles)):
                        if (not self.allowed_dynamics) or (self.allowed_dynamics and any(s+'.WAV' in f for s in self.allowed_dynamics)):
                            print f
                            if os.path.isfile(os.path.join(self.path,'mat',f.lower()+'.mat')):
                                self.wav_list.append(os.path.join(self.path,'wav',str(self.instid)+str(case),f))
                                self.notes_code.append(f.replace('.WAV','').lower())
                                mat = io.loadmat(os.path.join(self.path,'mat',f.lower()+'.mat'))
                                self.dynamics = mat['featureStruct'][0][0][3][0]
                                self.style = mat['featureStruct'][0][0][9][0]
                                self.notes_len.append(len(mat['featureStruct'][0][0][14][0]))
                                # self.Min = mat['ref_annotation']['noteRange']['Min']
                                # self.Max = mat['ref_annotation']['noteRange']['Max']
                                self.instrumentName = mat['featureStruct'][0][0][6][0]
                                self.instrumentSymbol = mat['featureStruct'][0][0][7][0]
                                for i in range(len(mat['featureStruct'][0][0][14][0])):
                                    note = Note(self.path, os.path.join(self.path,'wav',str(self.instid)+str(case),f), self.style+'_'+self.dynamics+'_'+str(case), f.lower(),i ,self.total_notes)
                                    self.notes.append(note)
                                    self.notes_nr.append(note.nr)
                                    self.notes_d.append(note.dynamics)
                                    self.notes_s.append(note.style)
                                    self.notes_c.append(case)
                                    self.total_notes = self.total_notes + 1
                            else:
                                print 'mat file could not be found: ' + os.path.join(self.path,'mat',f.lower()+'.mat')

    def getNote(self,nr,dynamics='F',style='NO',case=1):
        for i in range(self.total_notes):
            if self.notes_nr[i]==nr and self.notes_d[i]==dynamics and self.notes_s[i]==style and self.notes_c[i]==case:
                return self.notes[i]
        return None




class Note:
    """
    A class for a single note for a given instrument in the database RWC

    Parameters
    ----------
    path : string
        The rwc instrument sound path containing 'mat' and 'wav' subfolders
    wav_path : string
        The path inside the rwc instrument sound path where the .wav file is located
    code : string
        The name of the .mat file to read the information about the note without the extension
    piece : string
        A string code for this note 
    fid : int
        The id of the note in the .mat file
    noteid : int
        The int code for this note
    """
    def __init__(self, path, wav_path, piece, code, fid, noteid):
        self.code = code
        self.piece = piece
        self.fid = fid #note id in the wav/mat file
        self.path = path
        self.wav_path = wav_path
        self.noteid = noteid #note id in the list containing all the notes from the instrument
        self.getInfo()

    def getInfo(self):
        if os.path.isfile(os.path.join(self.path,'mat',self.code+'.mat')):
            mat = io.loadmat(os.path.join(self.path,'mat',self.code+'.mat'))
            self.sampleRate = mat['featureStruct'][0][0][0][0][0][3][0][0]
            self.style = mat['featureStruct'][0][0][9][0]
            self.dynamics = mat['featureStruct'][0][0][3][0]
            self.nr = mat['featureStruct'][0][0][14][0][self.fid]
            self.noteStart = float(mat['featureStruct'][0][0][15][0][self.fid]) / float(self.sampleRate) 
            self.noteEnd = float(mat['featureStruct'][0][0][16][0][self.fid]) / float(self.sampleRate) 
            self.instid = mat['featureStruct'][0][0][4][0][0]
            self.length = self.noteEnd-self.noteStart
        else:
            print 'mat file could not be found'

    def getAudio(self,max_duration=0,sampleRate=44100):
        if not os.path.exists(self.wav_path):
            print("file not found "+self.wav_path) 
            return False
        # Read audio data
        audio,sampleRate,bitrate = util.readAudioScipy(self.wav_path)

        if max_duration==0 or (self.noteEnd - self.noteStart) < max_duration:
            note = audio[int(self.noteStart*sampleRate):int(self.noteEnd*sampleRate)]
        else:
            note = audio[int(self.noteStart*sampleRate):int((self.noteStart+max_duration)*sampleRate)]

        #detect onset with RMS
        lengthData = len(note)
        hopsize=128
        lengthWindow=1024
        numberFrames = int(np.ceil(lengthData / np.double(hopsize)))
        
        energy = np.zeros([int(numberFrames), 1], dtype=float)
        for n in np.arange(numberFrames):
            beginFrame = int(n*hopsize)
            endFrame = int(beginFrame+lengthWindow)
            segment = note[int(beginFrame):int(endFrame)]
            energy[n] = np.sqrt( np.sum( np.power(segment,2)) / len(segment))
  
        onset = np.maximum(0,np.argmax(energy>0.01)-1)
        energy=None
        
        if onset>0:
            self.noteStart=self.noteStart+onset*float(hopsize)/sampleRate
            if max_duration==0 or (self.noteEnd - self.noteStart) < max_duration:
                note = audio[int(self.noteStart*sampleRate):int(self.noteEnd*sampleRate)]
            else:
                note = audio[int(self.noteStart*sampleRate):int((self.noteStart+max_duration)*sampleRate)]
        return note


