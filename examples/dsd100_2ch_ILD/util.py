"""
    This file is part of DeepConvSep.

    Copyright (c) 2014-2017 Marius Miron  <miron.marius at gmail.com>
    Copyright (c) 2017 Gerard Erruz  <gerard.erruz at upf.edu>

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
import scipy.io.wavfile

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

def generate_overlapadd(allmix,input_size=513,time_context=30, overlap=10,batch_size=32,sampleRate=44100):

    if len(allmix.shape)>2:
        nchannels=allmix.shape[0]
    else:
        nchannels=1

    assert input_size == allmix.shape[-1], "Feature size must be the same as the last dimension of the spectrogram"
    i=0
    start=0
    while (start + overlap) < allmix.shape[-2]:
        i = i + 1
        start = start - overlap + time_context
    fbatch = np.zeros([int(np.ceil(float(i)/batch_size)),batch_size,nchannels,time_context,input_size])

    i=0
    start=0
    while (start + overlap) < allmix.shape[-2]:
        fbatchend = np.minimum(time_context,allmix.shape[-2]-start)
        end = np.minimum(start+time_context,allmix.shape[-2])
        if len(allmix.shape)>2:
            fbatch[int(i/batch_size),int(i%batch_size),:,:fbatchend,:]=allmix[:,start:end,:]
        else:
            fbatch[int(i/batch_size),int(i%batch_size),:,:fbatchend,:]=allmix[start:end,:]
        i = i + 1 #index for each block
        start = start - overlap + time_context #starting point for each block

    return fbatch,i


def overlapadd(fbatch,obatch,nchunks,overlap=10):

    input_size=fbatch.shape[-1]
    time_context=fbatch.shape[-2]
    batch_size=fbatch.shape[2]

    window = np.linspace(0., 1.0, num=overlap)
    window = np.concatenate((window,window[::-1]))
    window = np.repeat(np.expand_dims(window, axis=1),input_size,axis=1)

    sep1 = np.zeros((nchunks*(time_context-overlap)+time_context,input_size))
    sep2 = np.zeros((nchunks*(time_context-overlap)+time_context,input_size)) #allocate for output of prediction
    i=0
    start=0
    while i < nchunks:
        fbatch1=fbatch[:,0,:,:,:]
        fbatch2=fbatch[:,1,:,:,:]
        s1= fbatch1[int(i/batch_size),int(i%batch_size),0,:,:]
        s2= fbatch2[int(i/batch_size),int(i%batch_size),0,:,:]
        if start==0:
            sep1[0:time_context] = s1
            sep2[0:time_context] = s2
        else:
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

    window = np.linspace(0., 1.0, num=overlap)
    window = np.concatenate((window,window[::-1]))
    window = np.repeat(np.expand_dims(window, axis=1),input_size,axis=1)
    sep = np.zeros((nsources, nchunks*(time_context-overlap)+time_context, input_size)) #allocate for output of prediction
    for s in range(nsources):
        i=0
        start=0
        while i < nchunks:
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
