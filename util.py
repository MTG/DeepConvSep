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
import itertools as it
from scipy.interpolate import interp1d,InterpolatedUnivariateSpline

import scipy.io.wavfile

import Queue
import time
import threading



def buffered_gen(source_gen, buffer_size=2, sleep_time=1):
    """
    Generator that runs a slow source generator in a separate thread.
    buffer_size: the maximal number of items to pre-generate (length of the buffer)
    """
    buffer = Queue.Queue(maxsize=buffer_size)

    def _buffered_generation_thread(source_gen, buffer):
        while True:
            # we block here when the buffer is full. There's no point in generating more data
            # when the buffer is full, it only causes extra memory usage and effectively
            # increases the buffer size by one.
            while buffer.full():
                #print "DEBUG: buffer is full, waiting to generate more data."
                time.sleep(sleep_time)

            try:
                data = source_gen.next()
            except StopIteration:
                break

            buffer.put(data)
    
    thread = threading.Thread(target=_buffered_generation_thread, args=(source_gen, buffer))
    thread.setDaemon(True)
    thread.start()
    
    while True:
        yield buffer.get()
        buffer.task_done()


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
  return tuning_freq * 2 ** ((midi_number - MIDI_A4) * (1./12.))


def gaussian1d(height, center_x, width_x):
    """Returns a gaussian function with the given parameters"""
    width_x = float(width_x)
    return lambda x,y: height*np.exp(
                -(((center_x-x)/width_x)**2)/2)

def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

#code to compute frequency bands of the harmonic partials for a given pitch
def slicefft(pitch,size,interval=30,tuning_freq=440,nharmonics=20,fmin=25,fmax=18000,iscale = 'lin',sampleRate=44100):
    if pitch>0:
        binfactor = size/float(sampleRate)
        fups,fdowns = getfreqs(pitch,interval=interval,tuning_freq=tuning_freq,nharmonics=nharmonics)
        slices_y = np.hstack(tuple([range(int(fdowns[f]*binfactor),int(fups[f]*binfactor)+2) for f in range(len(fups)) \
            if int(fdowns[f]*binfactor)>=0 and int(fdowns[f]*binfactor)<=(size/2+1) and int(fdowns[f]*binfactor)>0]))
    else:
        slices_y = []
    return slices_y

def slicefft_slices(pitch,size,interval=30,tuning_freq=440,nharmonics=20,fmin=25,fmax=18000,iscale = 'lin',sampleRate=44100):
    if pitch>0:
        binfactor = size/float(sampleRate)
        fups,fdowns = getfreqs(pitch,interval=interval,tuning_freq=tuning_freq,nharmonics=nharmonics)
        slices_y = [slice(int(fdowns[f]*binfactor),int(fups[f]*binfactor)+2) for f in range(len(fups)) \
            if int(fdowns[f]*binfactor)>=0 and int(fdowns[f]*binfactor)<=(size/2+1) and int(fdowns[f]*binfactor)>0]
    else:
        slices_y = []
    return slices_y

def slicecqt(pitch,size,interval=70,tuning_freq=440,nharmonics=20,fmin=25,fmax=18000,iscale = 'lin',sampleRate=44100):
    if pitch>0:
        c2f = (2.0**(interval/1200.0)) 
        from nsgt import LinScale,LogScale,OctScale
        scales = {'log':LogScale,'lin':LinScale,'oct':OctScale}
        scale = scales[iscale]
        scl = scale(fmin,fmax,size)
        if iscale=='lin':
            df = (fmax-fmin)/float(size-1)
            centsperbin = 1200.0*np.log2(float(fmax)/fmin)/(size-1)
            band=np.maximum(1,int(interval/centsperbin))
        elif iscale=='log':
            odiv = np.log2(fmax/fmin)/(float(size))
            pow2n = 2**odiv          
            fups,fdowns = getfreqs(pitch,interval=interval,tuning_freq=tuning_freq,nharmonics=nharmonics)
            if iscale=='lin':
                slices_y = [slice(int(np.maximum(0,np.floor((fdowns[h]-fmin)/df))), int(np.ceil((fups[h]-fmin)/df))) \
                    for h in range(len(fups)) if int(np.ceil((fups[h]-fmin)/df))<size]
            elif iscale=='log':
                slices_t = [slice(int(np.maximum(0,np.log(fdowns[h]/fmin)/np.log(pow2n))), \
                    int(np.ceil(np.log(fups[h]/fmin)/np.log(pow2n)))) \
                    for h in range(len(fups)) if int(np.ceil(np.log(fups[h]/fmin)/np.log(pow2n)))<size \
                    and  int(np.ceil(np.log(fups[h]/fmin)/np.log(pow2n)))>0]                     
                fact = interval/1200.0 
                slices_y = np.array([[int(np.maximum(0,slices_t[h].start-(size-slices_t[h].start)*fact)), \
                    int(np.ceil(slices_t[h].stop+(size-slices_t[h].stop)*fact))] \
                    for h in range(len(slices_t)) if int(np.ceil(slices_t[h].stop+(size-slices_t[h].stop)*fact))<size],dtype=int)                    
                #slices_y = fixoverlap(slices_y)
                slices_y = np.hstack(tuple([range(int(sl[0]),int(sl[1])) for sl in slices_y]))             
                #ss = [slice(int(sl[0]),int(sl[1])) for sl in slices_y]
                # if not isinstance(slices_y[0],int):
                #   import pdb;pdb.set_trace()
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
    for k in range(2,len(fups)):
        if fups[k-1]>fdowns[k]:
            v = (fups[k-1]-fdowns[k])/2.0 
            fups[k-1] = fups[k-1] - v
            fdowns[k] = fdowns[k] + v 
    if (fups[-1]-fdowns[-1])>(fups[-2]-fdowns[-2]):
        fups[-1] = fdowns[-1] + fups[-2]-fdowns[-2]
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

def getBands1(bins=48, interval=20, iscale='lin', frameSize=1024, fmin=50, fmax=14000, ttype='fft',sampleRate=44100):
        iscale=iscale
        from nsgt import LinScale,LogScale,OctScale
        scales = {'log':LogScale,'lin':LinScale,'oct':OctScale}
        scale = scales[iscale]
        scl = scale(fmin,fmax-1,8)
        if ttype == 'cqt':
            fsampled = scl.F()
        else:
            fsampled = octaves(8,fmin,fmax-1)
        if ttype == 'cqt':
            if iscale=='lin':
                freqs = [fmin]+[2*fmin*k for k in range(1,100) if fmin/2*k<=fmax]
            elif iscale=='log':
                # freqs = [55*k for k in range(1,2)]
                freqs = [fmin]+[2*fmin*k for k in range(1,100) if fmin/2*k<=fmax]
                #freqs = scl.F()
        else: 
            freqs = [fmin]+[2*fmin*k for k in range(1,100)]
        factor = 2.0**(interval/1200.0)
        fdowns = [f / float(factor) for f in freqs]
        fups = [f * float(factor) for f in freqs]
        import pdb;pdb.set_trace()
        for k in range(2,len(fups)):
            if fups[k-1]>fdowns[k]:
                v = (fups[k-1]-fdowns[k])/2.0 
                fups[k-1] = fups[k-1] - v
                fdowns[k] = fdowns[k] + v 
        if (fups[-1]-fdowns[-1])>(fups[-2]-fdowns[-2]):
                fups[-1] = fdowns[-1] + fups[-2]-fdowns[-2]

        if iscale=='lin':
            df = (fmax-fmin)/float(bins-1)
        elif iscale=='log':
            odiv = np.log2(fmax/fmin)/(float(bins))
            pow2n = 2**odiv
        if ttype == 'cqt':
            if iscale=='lin':
                bands = [np.maximum(1,int(np.ceil((fups[h]-fmin)/df - (fdowns[h]-fmin)/df))) for h in range(len(fups))]
            elif iscale=='log':
                slices_t = [slice(int(np.maximum(0,np.log(fdowns[h]/fmin)/np.log(pow2n))), \
                    int(np.ceil(np.log(fups[h]/fmin)/np.log(pow2n)))) \
                    for h in range(len(fups)) if int(np.ceil(np.log(fups[h]/fmin)/np.log(pow2n)))<bins]                     
                fact = interval/1200.0 
                slices_y = [[int(np.maximum(0,slices_t[h].start-(bins-slices_t[h].start)*fact)), \
                    int(np.ceil(slices_t[h].stop+(bins-slices_t[h].stop)*fact))] \
                    for h in range(len(slices_t)) if int(np.ceil(slices_t[h].stop+(bins-slices_t[h].stop)*fact))<bins]                     
                slices_y = fixoverlap(slices_y)
                bands = [np.maximum(1,int(np.ceil(sl[1]-sl[0]))) for sl in slices_y]
        else:
            binfactor = frameSize/float(sampleRate)
            bands = [np.maximum(1,int(np.ceil(fups[h]*binfactor - fdowns[h]*binfactor))) for h in range(len(fups))]
        freqs = freqs[:len(bands)]
        if iscale=='log' and bands[0]<bands[1]:
            bands[1]=(2*bands[0]+bands[1])/2
            freqs[-2] = (freqs[-2] + freqs[-1])/2
            bands = bands[1:len(bands)]
            freqs = freqs[1:len(freqs)]
        elif iscale=='lin' and bands[-1]<bands[-2]:
            bands[-2]=(2*bands[-1]+bands[-2])/2
            freqs[-2] = (freqs[-2] + freqs[-1])/2
            bands = bands[:len(bands)-1]
            freqs = freqs[:len(freqs)-1]
        
        from bisect import bisect_left
        sel_freqs =  [freqs[np.minimum(len(freqs)-1,bisect_left(freqs, f))] for f in fsampled]
        sel_bands =  [bands[np.minimum(len(freqs)-1,bisect_left(freqs, f))] for f in fsampled]
        usf=list(set(sel_bands))
        idx = [sel_bands.index(b) for b in usf]
        sfreqs = [sel_freqs[i] for i in idx]
        sbands = [sel_bands[i] for i in idx]
        return sfreqs,sbands

def getBands(bins=48, interval=20, iscale='lin', frameSize=1024, fmin=50, fmax=14000, ttype='fft',sampleRate=44100):
        from nsgt import LinScale,LogScale,OctScale
        scales = {'log':LogScale,'lin':LinScale,'oct':OctScale}
        scale = scales[iscale]
        scl = scale(fmin,fmax-1,8)
        if ttype == 'cqt':
            fsampled = scl.F()
        else:
            fsampled = octaves(8,fmin,fmax-1)
        if ttype == 'cqt':
            if iscale=='lin':
                freqs = [fmin]+[2*fmin*k for k in range(1,100) if fmin/2*k<=fmax]
            elif iscale=='log':
                # freqs = [55*k for k in range(1,2)]
                freqs = [fmin]+[2*fmin*k for k in range(1,100) if fmin/2*k<=fmax]
                #freqs = scl.F()
        else: 
            freqs = [fmin]+[2*fmin*k for k in range(1,100)]
        factor = 2.0**(interval/1200.0)
        fdowns = [f / float(factor) for f in freqs]
        fups = [f * float(factor) for f in freqs]
        for k in range(2,len(fups)):
            if fups[k-1]>fdowns[k]:
                v = (fups[k-1]-fdowns[k])/2.0 
                fups[k-1] = fups[k-1] - v
                fdowns[k] = fdowns[k] + v 
        if (fups[-1]-fdowns[-1])>(fups[-2]-fdowns[-2]):
                fups[-1] = fdowns[-1] + fups[-2]-fdowns[-2]
        if iscale=='lin':
            df = (fmax-fmin)/float(bins-1)
        elif iscale=='log':
            odiv = np.log2(fmax/fmin)/(float(bins))
            pow2n = 2**odiv
        if ttype == 'cqt':
            if iscale=='lin':
                bands = [np.maximum(1,int(np.ceil((fups[h]-fmin)/df - (fdowns[h]-fmin)/df))) for h in range(len(fups))]
            elif iscale=='log':
                slices_t = [slice(int(np.maximum(0,np.log(fdowns[h]/fmin)/np.log(pow2n))), \
                    int(np.ceil(np.log(fups[h]/fmin)/np.log(pow2n)))) \
                    for h in range(len(fups)) if int(np.ceil(np.log(fups[h]/fmin)/np.log(pow2n)))<bins]                     
                fact = interval/1200.0 
                slices_y = [[int(np.maximum(0,slices_t[h].start-(bins-slices_t[h].start)*fact)), \
                    int(np.ceil(slices_t[h].stop+(bins-slices_t[h].stop)*fact))] \
                    for h in range(len(slices_t)) if int(np.ceil(slices_t[h].stop+(bins-slices_t[h].stop)*fact))<bins]                     
                slices_y = fixoverlap(slices_y)
                bands = [np.maximum(1,int(np.ceil(sl[1]-sl[0]))) for sl in slices_y]
        else:
            binfactor = frameSize/float(sampleRate)
            bands = [np.maximum(1,int(np.ceil(fups[h]*binfactor - fdowns[h]*binfactor))) for h in range(len(fups))]
        freqs = freqs[:len(bands)]
        if iscale=='log' and bands[0]<bands[1]:
            bands[1]=(2*bands[0]+bands[1])/2
            freqs[-2] = (freqs[-2] + freqs[-1])/2
            bands = bands[1:len(bands)]
            freqs = freqs[1:len(freqs)]
        elif iscale=='lin' and bands[-1]<bands[-2]:
            bands[-2]=(2*bands[-1]+bands[-2])/2
            freqs[-2] = (freqs[-2] + freqs[-1])/2
            bands = bands[:len(bands)-1]
            freqs = freqs[:len(freqs)-1]
        
        from bisect import bisect_left
        sel_freqs =  [freqs[np.minimum(len(freqs)-1,bisect_left(freqs, f))] for f in fsampled]
        sel_bands =  [bands[np.minimum(len(freqs)-1,bisect_left(freqs, f))] for f in fsampled]
        usf=list(set(sel_bands))
        idx = [sel_bands.index(b) for b in usf]
        sfreqs = [sel_freqs[i] for i in idx]
        sbands = [sel_bands[i] for i in idx]
        return sfreqs,sbands


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
