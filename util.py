"""
    This file is part of DeepConvSep.

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

import numpy as np
import scipy
from scipy import io
import os
import sys
from os import listdir
from os.path import isfile, join
import cPickle as pickle
from bisect import bisect_left, bisect_right
import itertools as it
from scipy.interpolate import interp1d,InterpolatedUnivariateSpline

import scipy.io.wavfile

import Queue
import time
import threading


#routines to read and write audio
def infoAudioScipy(filein):
    sampleRate, audioObj = scipy.io.wavfile.read(filein)  
    bitrate = audioObj.dtype    
    nsamples = len(audioObj)
    audioObj = None
    return nsamples, sampleRate, bitrate

def readAudioScipy(filein):
    sampleRate, audioObj = scipy.io.wavfile.read(filein)  
    bitrate = audioObj.dtype    
    try:
        maxv = np.finfo(bitrate).max
    except:
        maxv = np.iinfo(bitrate).max
    return audioObj.astype('float')/maxv, sampleRate, bitrate

def writeAudioScipy(fileout,audio_out,sampleRate,bitrate="int16"):
    maxn = np.iinfo(bitrate).max  
    scipy.io.wavfile.write(filename=fileout, rate=sampleRate, data=(audio_out*maxn).astype(bitrate))


#circular shift audio with 'cs'
def circular_shift(audio,min_size,cs=0.1,sampleRate=44100):
    if cs == 0:
        if len(audio) > min_size:
            segment = audio[:min_size]
        else:
            segment = np.zeros(min_size)
            segment[:len(audio)] = audio
    elif cs < 0:
        seg_idx = int(abs(cs*sampleRate))
        segment = np.pad(audio,((0,seg_idx+np.maximum(0,min_size-len(audio)))), mode='constant')
        if len(segment)<(min_size+seg_idx):
            segment = np.pad(segment,((0,min_size+seg_idx - len(segment))), mode='constant')
        segment = segment[seg_idx:min_size+seg_idx]
    else:
        segment = np.pad(audio,((int(cs*sampleRate),0)), mode='constant')
        if len(segment)<min_size:
            segment = np.pad(segment,((0,min_size - len(segment))), mode='constant')
        segment = segment[:min_size]

    return segment


class interpolate:
    def __init__(self,cqt,Ls):
        from scipy.interpolate import interp1d
        self.intp = [interp1d(np.linspace(0,Ls,len(r)),r) for r in cqt]
    def __call__(self,x):
        try:
            len(x)
        except:
            return np.array([i(x) for i in self.intp])
        else:
            return np.array([[i(xi) for i in self.intp] for xi in x])

class interpolate1d:
    def __init__(self,cqt,Ls):
        from scipy.interpolate import interp1d
        self.intp = interp1d(np.linspace(0,Ls,len(cqt)),cqt) 
    def __call__(self,x):
        try:
            len(x)
        except:
            return np.array([self.intp(xi) for xi in x],dtype='complex128')
        else:
            return np.array([self.intp(xi) for xi in x],dtype='complex128')

def saveObj(obj,filename):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, -1)

def loadObj(filename):
    with open(filename, 'rb') as input:
        obj= pickle.load(input)
    return obj

def emptyDir(dirPath):
    fileList = os.listdir(dirPath)
    for fileName in fileList:
        os.remove(os.path.join(dirPath,fileName))

# Useful constants
MIDI_A4 = 69   # MIDI Pitch number
FREQ_A4 = 440. # Hz
SEMITONE_RATIO = 2. ** (1. / 12.) # Ascending
def midi2freq(midi_number, tuning_freq=440., MIDI_A4=69.):
  return float(tuning_freq) * 2.0 ** ((float(midi_number) - float(MIDI_A4)) * (1./12.))


def gaussian1d(height, center_x, width_x):
    """Returns a gaussian function with the given parameters"""
    width_x = float(width_x)
    return lambda x,y: height*np.exp(
                -(((center_x-x)/width_x)**2)/2)

def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

#removes overlap between two intervals
def remove_overlap(ranges):
    result = []
    current_start = -1
    current_stop = -1 

    for start, stop in sorted(ranges):
        if start > current_stop:
            # this segment starts after the last segment stops
            # just add a new segment
            result.append( (start, stop) )
            current_start, current_stop = start, stop
        else:
            # segments overlap, replace
            result[-1] = (current_start, stop)
            # current_start already guaranteed to be lower
            current_stop = max(current_stop, stop)

    return result


#code to compute frequency bands of the harmonic partials for a given pitch
def slicefft(pitch,size,interval=30,tuning_freq=440,nharmonics=20,fmin=25,fmax=18000,iscale = 'lin',sampleRate=44100):
    if pitch>0:
        binfactor = size/float(sampleRate)
        fups,fdowns = getfreqs(pitch,interval=interval,tuning_freq=tuning_freq,nharmonics=nharmonics)
        slices_y = np.hstack(tuple([range(int(np.floor(fdowns[f]*binfactor)),int(np.ceil(fups[f]*binfactor)+2)) for f in range(len(fups)) \
            if int(fdowns[f]*binfactor)>=0 and int(fdowns[f]*binfactor)<=(size/2+1) and int(fdowns[f]*binfactor)>0]))
    else:
        slices_y = []
    return slices_y

def slicefft_slices(pitch,size,interval=30,tuning_freq=440,nharmonics=20,fmin=25,fmax=18000,iscale = 'lin',sampleRate=44100):
    if pitch>0:
        binfactor = float(size)/float(sampleRate)
        fups,fdowns = getfreqs(pitch,interval=interval,tuning_freq=tuning_freq,nharmonics=nharmonics)
        ranges = tuple((1+int(np.floor(fdowns[f]*binfactor)), 1+int(np.ceil(fups[f]*binfactor))) for f in range(len(fdowns)))
        ranges = remove_overlap(ranges)
        slices_y = [slice(ranges[f][0],ranges[f][1]) for f in range(len(ranges)) if ranges[f][1]<=(size/2+1)]
    else:
        slices_y = []
    return slices_y


def getfreqs(midinote,interval=30,tuning_freq=440,nharmonics=20,ismidi=True):
    factor = 2.0**(interval/1200.0)
    if ismidi:
        f0 = float(midi2freq(midinote,tuning_freq=tuning_freq))
    else:
        f0 = midinote
    fdowns = [f * f0 / float(factor) for f in range(1,nharmonics)]
    fups = [f * f0 * float(factor) for f in range(1,nharmonics)] 
    return fups,fdowns

def fixoverlap(slices):
    for k in range(1,len(slices)): #harmonics
        if slices[k-1][1]>slices[k][0]:
            v = (slices[k-1][1]-slices[k][0])/2.0 
            slices[k-1][1] = slices[k-1][1] - v 
            slices[k][0] = slices[k][0] + v + 1
    if (slices[-1][1]-slices[-1][0])>(slices[-2][1]-slices[-2][0]):
        slices[-1][1] = slices[-1][0] + slices[-2][1]-slices[-2][0]
    return slices

def getPitches(pitch,shape_time,interp='zero'):
    if len(pitch.shape)<2:
        print 'shape of pitches should be (ninst,npitches,values)'
    npitches = pitch.shape[1]
    ninst = pitch.shape[0]
    pitchr = np.zeros((ninst,npitches,shape_time))
    for i in range(ninst):
        for p in range(npitches):
            #use np.interp to obtain the pitch values for the correct hop size 
            x = np.linspace(0, pitch.shape[-1], num=pitch.shape[-1])
            x_new = np.linspace(0, pitch.shape[-1], num=shape_time)
            f = interp1d(x, pitch[i,p,:], kind=interp)
            pitchr[i,p,:] = f(x_new)
    return pitchr



def generate_overlapadd(allmix,input_size=513,time_context=30, overlap=10,batch_size=32,sampleRate=44100):

    if len(allmix.shape)>2:
        nchannels=allmix.shape[0]
    else:
        nchannels=1

    if input_size == allmix.shape[-1]:
        
        i=0
        start=0  
        while (start + time_context) < allmix.shape[-2]:
            i = i + 1
            start = start - overlap + time_context 
        fbatch = np.empty([int(np.ceil(float(i)/batch_size)),batch_size,nchannels,time_context,input_size])   
       
        i=0
        start=0   
        while (start + time_context) < allmix.shape[-2]:
            if len(allmix.shape)>2:
                fbatch[int(i/batch_size),int(i%batch_size),:,:,:]=allmix[:,start:start+time_context,:]
            else:
                fbatch[int(i/batch_size),int(i%batch_size),:,:,:]=allmix[start:start+time_context,:]
            #data_in[i] = allm[0]
            i = i + 1 #index for each block
            start = start - overlap + time_context #starting point for each block
    return fbatch,i


def overlapadd(fbatch,obatch,nchunks,overlap=10):

    input_size=fbatch.shape[-1]
    time_context=fbatch.shape[-2]
    batch_size=fbatch.shape[2]
    #print time_context
    #print batch_size

    #window = np.sin((np.pi*(np.arange(2*overlap+1)))/(2.0*overlap))
    window = np.linspace(0., 1.0, num=overlap)
    window = np.concatenate((window,window[::-1]))
    #time_context = net.network.find('hid2', 'hh').size
    # input_size = net.layers[0].size  #input_size is the number of spectral bins in the fft
    window = np.repeat(np.expand_dims(window, axis=1),input_size,axis=1)

    sep1 = np.zeros((nchunks*(time_context-overlap)+time_context,input_size))
    sep2 = np.zeros((nchunks*(time_context-overlap)+time_context,input_size)) #allocate for output of prediction
    #import pdb;pdb.set_trace()
    i=0
    start=0 
    while i < nchunks:
        # import pdb;pdb.set_trace()
        fbatch1=fbatch[:,0,:,:,:]
        fbatch2=fbatch[:,1,:,:,:]
        s1= fbatch1[int(i/batch_size),int(i%batch_size),0,:,:]
        s2= fbatch2[int(i/batch_size),int(i%batch_size),0,:,:]
        # m1=s1p/(s1p+s2p)
        # m2=s1p/(s1p+s2p)
        # s1 = m1 * obatch[int(i/batch_size),int(i%batch_size),0,:,:]
        # s2 = m2 * obatch[int(i/batch_size),int(i%batch_size),0,:,:]
        #print s1.shape
        if start==0:
            sep1[0:time_context] = s1
            sep2[0:time_context] = s2
        else:
            #print start+overlap
            #print start+time_context
            sep1[start+overlap:start+time_context] = s1[overlap:time_context]
            sep1[start:start+overlap] = window[overlap:]*sep1[start:start+overlap] + window[:overlap]*s1[:overlap]
            sep2[start+overlap:start+time_context] = s2[overlap:time_context]
            sep2[start:start+overlap] = window[overlap:]*sep2[start:start+overlap] + window[:overlap]*s2[:overlap]
        i = i + 1 #index for each block
        start = start - overlap + time_context #starting point for each block
    return sep1,sep2 


def overlapadd_multi(fbatch,obatch,nchunks,overlap=10):

    input_size=fbatch.shape[-1]
    time_context=fbatch.shape[-2]
    batch_size=fbatch.shape[2]
    nsources = fbatch.shape[1]
    #print time_context
    #print batch_size

    #window = np.sin((np.pi*(np.arange(2*overlap+1)))/(2.0*overlap))
    window = np.linspace(0., 1.0, num=overlap)
    window = np.concatenate((window,window[::-1]))
    #time_context = net.network.find('hid2', 'hh').size
    # input_size = net.layers[0].size  #input_size is the number of spectral bins in the fft
    window = np.repeat(np.expand_dims(window, axis=1),input_size,axis=1)
    sep = np.zeros((nsources, nchunks*(time_context-overlap)+time_context, input_size)) #allocate for output of prediction 
    for s in range(nsources):
        i=0
        start=0 
        while i < nchunks:
            # import pdb;pdb.set_trace()
            fbatch1=fbatch[:,s,:,:,:]
            source= fbatch1[int(i/batch_size),int(i%batch_size),0,:,:]
            if start==0:
                sep[s,0:time_context] = source
            else:
                sep[s,start+overlap:start+time_context] = source[overlap:time_context]
                sep[s,start:start+overlap] = window[overlap:]*sep[s,start:start+overlap] + window[:overlap]*source[:overlap]
            i = i + 1 #index for each block
            start = start - overlap + time_context #starting point for each block
    return sep


## read a txt containing midi notes : onset,offset,midinote
def getMidi(instrument,FilePath,beginTime,finishTime,samplerate,hop,window,timeSpan_on,timeSpan_off,nframes,nlines=1,fermata=0.):
    fermata = np.maximum(timeSpan_off,fermata)
    midifile = os.path.join(FilePath,instrument + '.txt')
    melodyFromFile = np.genfromtxt(midifile, comments='!', \
      delimiter=',',names="a,b,c",dtype=["f","f","S3"])
    melTimeStampsBeginO = melodyFromFile['a'].tolist()
    melTimeStampsEndO = melodyFromFile['b'].tolist()
    #startTime = np.maximum(bisect_right(melTimeStampsBegin,beginTime),bisect_right(melTimeStampsEnd,beginTime))
    #endTime = np.minimum(bisect_left(melTimeStampsBegin,finishTime),bisect_left(melTimeStampsEnd,finishTime))
    startTime = bisect_right(melTimeStampsEndO,beginTime)
    endTime = bisect_left(melTimeStampsBeginO,finishTime)

    if melTimeStampsEndO[startTime]<float(beginTime):
        startTime=startTime+1
    if endTime>=len(melTimeStampsBeginO):
        endTime = len(melTimeStampsBeginO) - 1 
    elif melTimeStampsBeginO[endTime]>float(finishTime):
        endTime=endTime-1     
    
    if (startTime<endTime):
        melTimeStampsBeginO = melTimeStampsBeginO[startTime:endTime+1] 
        melTimeStampsBegin = [x - beginTime for x in melTimeStampsBeginO]
        melTimeStampsEndO = melTimeStampsEndO[startTime:endTime+1]
        melTimeStampsEnd = [x - beginTime for x in melTimeStampsEndO]
        
        for i in range(len(melTimeStampsBegin)):
            if (melTimeStampsBegin[i] < 0):
               melTimeStampsBegin[i] = 0.0
            if (melTimeStampsEnd[i] < 0):
               melTimeStampsEnd[i] = 0.0
            if (melTimeStampsEnd[i] > (finishTime-beginTime)):
               melTimeStampsEnd[i] = finishTime-beginTime
            if (melTimeStampsBegin[i] > (finishTime-beginTime)):
               melTimeStampsBegin[i] = finishTime-beginTime

        #get the midi   
        melNotesMIDI = melodyFromFile['c'].tolist()
        melNotesMIDI = melNotesMIDI[startTime:endTime+1]
        melIndex=[k for k in range(startTime,endTime+1)]

        tframes = float(nframes)*float(hop) / float(samplerate)
        #eliminate short notes
        lenlist = len(melTimeStampsBegin)
        i=0
        while i<lenlist:
            if (melTimeStampsEnd[i]<=0) \
                or (melTimeStampsEnd[i]<=melTimeStampsBegin[i]) \
                or (melTimeStampsBegin[i]>=tframes) \
                or ((melTimeStampsEnd[i]-melTimeStampsBegin[i])<0.01) :
                melTimeStampsBegin.pop(i)
                melTimeStampsEnd.pop(i)
                melNotesMIDI.pop(i)
                melIndex.pop(i)
                lenlist=lenlist-1
                i=i-1
            i=i+1

        melody = np.zeros((nlines,nframes))
        maxAllowed_on = int(round(timeSpan_on * float(samplerate / hop)))
        maxAllowed_off = int(round(timeSpan_off * float(samplerate / hop)))
        endMelody = int((finishTime-beginTime) * round(float(samplerate / hop)))
        w = window/2/hop
        for i in range(len(melTimeStampsEnd)):
            melodyBegin = np.maximum(0,int(melTimeStampsBegin[i] * round(float(samplerate / hop))) - maxAllowed_on)
            intersect = [mb for mb,me in zip(melTimeStampsBegin,melTimeStampsEnd) if (mb>melTimeStampsBegin[i]) and (me+timeSpan_off)>=(melTimeStampsBegin[i]-timeSpan_on) and (mb-timeSpan_on)<=(melTimeStampsEnd[i]+timeSpan_off) ]
            if len(intersect)==0:
                notesafter = filter(lambda x: (x-timeSpan_on)>(melTimeStampsEnd[i] +timeSpan_off), melTimeStampsBegin)
                if len(notesafter)>0:
                    #import pdb;pdb.set_trace()
                    newoffset= np.minimum(melTimeStampsEnd[i] + fermata, np.maximum(0,min(notesafter)-timeSpan_on) )
                else:
                    newoffset= melTimeStampsEnd[i] + fermata
                melodyEnd = np.minimum(nframes,np.minimum(endMelody,int(newoffset * round(float(samplerate / hop)))))
            else:
                melodyEnd = np.minimum(nframes,np.minimum(endMelody,int(melTimeStampsEnd[i] * round(float(samplerate / hop))) + maxAllowed_off))
            
            l=0
            if nlines>1:
                while l<nlines and np.sum(melody[l,melodyBegin:melodyEnd])>0:
                    l=l+1 
                if l>=nlines:
                    l=nlines-1
                    print "no space to store note: "+str(i)
            melody[l,melodyBegin:melodyEnd]=str2midi(melNotesMIDI[i])
        
        melodyBegin = [np.maximum(0,mel - timeSpan_on) for mel in melTimeStampsBegin]
        melodyEnd = [np.minimum(finishTime-beginTime,mel + timeSpan_off) for mel in melTimeStampsEnd]
        melNotes = [str2midi(n) for n in melNotesMIDI]
        return melody,melodyBegin,melodyEnd,melNotes
    else:
        return []

def getMidiLength(instrument,FilePath):
    #midifile = FilePath+instrument.replace('.txt','_1.txt')
    midifile = os.path.join(FilePath,instrument + '.txt')
    melodyFromFile = np.genfromtxt(midifile, comments='!', \
      delimiter=',',names="a,b,c",dtype=["f","f","S3"])
    melTimeStampsBeginO = melodyFromFile['a'].tolist()
    melTimeStampsEndO = melodyFromFile['b'].tolist()
    return max(melTimeStampsEndO)
    
def str2midi(note_string):
  """
  Given a note string name (e.g. "Bb4"), returns its MIDI pitch number.
  """
  if note_string == "?":
    return nan
  data = note_string.strip().lower()
  name2delta = {"c": -9, "d": -7, "e": -5, "f": -4, "g": -2, "a": 0, "b": 2}
  accident2delta = {"b": -1, "#": 1, "x": 2}
  accidents = list(it.takewhile(lambda el: el in accident2delta, data[2:]))
  octave_delta = int(data[1]) - 4
  return (MIDI_A4 +
          name2delta[data[0]] + # Name
          sum(accident2delta[ac] for ac in accidents) + # Accident
          12 * octave_delta # Octave
         )
