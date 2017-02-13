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

import scipy
import numpy as np
from scipy import io
from collections import defaultdict
import os
import sys
from os import listdir
from os.path import isfile, join
import itertools
import math
import random
import re
import util
from util import *

def sinebell(lengthWindow):
    """
    window = sinebell(lengthWindow)
    
    Computes a \"sinebell\" window function of length L=lengthWindow
    
    The formula is:

    .. math::
    
        window(t) = sin(\pi \\frac{t}{L}), t=0..L-1
        
    """
    window = np.sin((np.pi*(np.arange(lengthWindow)))/(1.0*lengthWindow))
    return window


class Transforms(object):
    """
    A general class which can be extended to compute features from audio (STFT,CQT)

    Parameters
    ----------
    frameSize : int, optional
        The frame size for the analysis in samples
    hopSize : int, optional
        The hop size for the analysis in samples
    sampleRate : int, optional
        The sample rate at which to read the signals
    window : function, optional
        The window function for the analysis
    
    """
    def __init__(self, ttype='fft', bins=48, frameSize=1024, hopSize=256, tffmin=25, tffmax=18000, iscale = 'lin', suffix='', sampleRate=44100, window=np.hanning):
        self.bins = bins
        self.frameSize = frameSize
        self.hopSize = hopSize
        self.fmin = tffmin
        self.fmax = tffmax
        self.iscale = iscale
        self.suffix=suffix
        self.sampleRate = sampleRate
        self.ttype = ttype
        self.window = window(self.frameSize)

    def compute_transform(self,audio, out_path=None, score=None, pitch=None, phase=False, save=True, pitch_interp='linear'):
        """
        Compute the features for a given set of audio signals.
            The audio signal \"audio\" is a numpy array with the shape (t,i) - t is time and i is the id of signal
            Depending on the variable \"save\", it can save the features to a binary file, accompanied by a shape file,
            which is useful for loading the binary data afterwards
        
        Parameters
        ----------
        audio : 2D numpy array
            The array comprising the audio signals
        out_path : string, optional
            The path of the directory where to save the audio.
        save : bool, optional
            To return or to save in the out_path the computed features
        phase : bool, optional
            To return/save the phase 
        pitch : 3D numpy array, optional
            Give as input the multiple pitch contours for each of the audio files (ninst,npitches,values)
            The value of the pitch is interpolated so it matches the time dimension of other computed features
        Yields
        ------
        mag : 3D numpy array
            The features computed for each of the signals in the audio array, e.g. magnitude spectrograms
        phs: 3D numpy array
            The features computed for each of the signals in the audio array, e.g. phase spectrograms
        """
        self.out_path = out_path
        for i in range(audio.shape[1]):
            if phase:
                mag,ph=self.compute_file(audio[:,i], phase=True, sampleRate=self.sampleRate)
            else:
                mag=self.compute_file(audio[:,i], phase=False, sampleRate=self.sampleRate)
            if i==0:
                mags = np.zeros((audio.shape[1],mag.shape[0],mag.shape[1])) #This line will be used when using without phase
                if phase:
                    if len(ph.shape)==3:
                        phs = np.zeros((audio.shape[1],ph.shape[0],ph.shape[1],ph.shape[2]))
                    else: 
                        phs = np.zeros((audio.shape[1],ph.shape[0],ph.shape[1]))
            mags[i]=mag
            if phase:
                phs[i]=ph

        
        if pitch is not None:
            pitchr=util.getPitches(pitch,mags.shape[1],interp=pitch_interp)

        if save and self.out_path is not None:
            self.saveTensor(mags,'_'+self.suffix+'_m_')
            if phase:
                self.saveTensor(phs,'_'+self.suffix+'_p_')
            if score is not None:
                saveObj(score, self.out_path.replace('.data','_'+self.suffix+'_s_'+'.data'))
            if pitch is not None:
                self.saveTensor(pitchr, '_'+self.suffix+'_p_')
            mags = None
            phase = None
            pitch = None
            score = None
        else:
            if phase:
                return mags,phs
            else:
                return mags

    def compute_playing(self,audio, out_path):
        """
        Function to compute playing/not playing labels for the audio files 

        Parameters
        ----------
        audio : 2D numpy array
            The array comprising the audio signals
        out_path : string, optional
            The path of the directory where to save the audio.
        """
        self.out_path = out_path
        ndim=audio.shape[1]/self.sampleRate*self.hopSize
        labels=np.zeros((ndim,audio.shape[1]))
        for i in range(audio.shape[1]):
            for j in range(ndim-1):
                if abs(audio[self.hopSize*i:self.hopSize*(i+1)]).mean()>0.02:
                    labels[j,i]=1
        self.saveTensor(labels,'_'+self.suffix+'_l_')

    def compute_file(self,audio, phase=False):
        return None

    def compute_inverse(self, mag, phase):
        return None

    def saveTensor(self, t, name='_cqt_m_'):
        """
        Saves a numpy array as a binary file
        """
        t.tofile(self.out_path.replace('.data',name+'.data'))
        #save shapes
        self.shape = t.shape
        self.save_shape(self.out_path.replace('.data',name+'.shape'),t.shape)

    def loadTensor(self, name='_cqt_m_'):
        """
        Loads a binary .data file
        """
        f_in = np.fromfile(self.out_path.replace('.data',name+'.data'))
        shape = self.get_shape(self.out_path.replace('.data','.shape'))
        if self.shape == shape:
            f_in = f_in.reshape(shape)    
            return f_in
        else:
            print 'Shape of loaded array does not match with the original shape of the transform'

    def save_shape(self,shape_file,shape):
        """
        Saves the shape of a numpy array
        """
        with open(shape_file, 'w') as fout:
            fout.write(u'#'+'\t'.join(str(e) for e in shape)+'\n')

    def get_shape(self,shape_file):
        """
        Reads a .shape file
        """
        with open(shape_file, 'rb') as f:
            line=f.readline().decode('ascii')
            if line.startswith('#'):
                shape=tuple(map(int, re.findall(r'(\d+)', line)))
                return shape
            else:
                raise IOError('Failed to find shape in file') 



class transformFFT(Transforms):
    """
    A class to help computing the short time Fourier transform (STFT) 
    
    Examples
    --------
    ### 1. Computing the STFT of a matrix of signals \"audio\" and writing the STFT data in \"path\" (except the phase)
    tt1=transformFFT(frameSize=2048, hopSize=512, sampleRate=44100)
    tt1.compute_transform(audio,out_path=path, phase=False)

    ### 2. Computing the STFT of a single signal \"audio\" and returning the magnitude and phase
    tt1=transformFFT(frameSize=2048, hopSize=512, sampleRate=44100)
    mag,ph = tt1.compute_file(audio,phase=True)

    ### 3. Computing the inverse STFT using the magnitude and phase and returning the audio data
    #we use the tt1 from 2.
    audio = tt1.compute_inverse(mag,phase)
    
    """

    def __init__(self, ttype='fft', bins=48, frameSize=1024, hopSize=256, tffmin=25, tffmax=18000, iscale = 'lin', suffix='', sampleRate=44100, window=np.hanning):
        super(transformFFT, self).__init__(ttype='fft', bins=bins, frameSize=frameSize, hopSize=hopSize, tffmin=tffmin, tffmax=tffmax, iscale = iscale, suffix=suffix, sampleRate=sampleRate, window=window)

    def compute_file(self,audio, phase=False, sampleRate=44100):
        """
        Compute the STFT for a single audio signal

        Parameters
        ----------
        audio : 1D numpy array
            The array comprising the audio signals
        phase : bool, optional
            To return the phase 
        sampleRate : int, optional
            The sample rate at which to read the signals
        Yields
        ------
        mag : 3D numpy array
            The features computed for each of the signals in the audio array, e.g. magnitude spectrograms
        phs: 3D numpy array
            The features computed for each of the signals in the audio array, e.g. phase spectrograms
        """
        X = stft_norm(audio, window=self.window, hopsize=float(self.hopSize), nfft=float(self.frameSize), fs=float(sampleRate))
        mag = np.abs(X)
        mag = mag  / np.sqrt(self.frameSize) #normalization
        if phase:
            ph = np.angle(X)
            X = None
            return mag,ph
        else:
            X = None
            return mag

    def compute_inverse(self, mag, phase, sampleRate=44100):
        """
        Compute the inverse STFT for a given magnitude and phase 

        Parameters
        ----------
        mag : 3D numpy array
            The features computed for each of the signals in the audio array, e.g. magnitude spectrograms
        phs: 3D numpy array
            The features computed for each of the signals in the audio array, e.g. phase spectrograms
        sampleRate : int, optional
            The sample rate at which to read the signals
        Yields
        ------
        audio : 1D numpy array
            The array comprising the audio signals
        """
        mag = mag  * np.sqrt(self.frameSize) #normalization
        Xback = mag * np.exp(1j*phase)
        data = istft_norm(Xback, window=self.window, analysisWindow=self.window, hopsize=float(self.hopSize), nfft=float(self.frameSize))
        return data                


def stft_norm(data, window=sinebell(2048),
         hopsize=256.0, nfft=2048.0, fs=44100.0):
    """
    X = stft_norm(data,window=sinebell(2048),hopsize=1024.0,
                   nfft=2048.0,fs=44100)
                   
    Computes the short time Fourier transform (STFT) of data.
    
    Inputs:
        data                  :
            one-dimensional time-series to be analyzed
        window=sinebell(2048) :
            analysis window
        hopsize=1024.0        :
            hopsize for the analysis
        nfft=2048.0           :
            number of points for the Fourier computation
            (the user has to provide an even number)
        fs=44100.0            :
            sampling rate of the signal
        
    Outputs:
        X                     :
            STFT of data
    """
    
    # window defines the size of the analysis windows
    lengthWindow = window.size
    
    lengthData = data.size
    
    # should be the number of frames by YAAFE:
    numberFrames = int(np.ceil(lengthData / np.double(hopsize)) + 2)
    # to ensure that the data array s big enough,
    # assuming the first frame is centered on first sample:
    newLengthData = int((numberFrames-1) * hopsize + lengthWindow)
    
    # !!! adding zeros to the beginning of data, such that the first window is
    # centered on the first sample of data
    data = np.concatenate((np.zeros(int(lengthWindow/2.0)), data))
    
    # zero-padding data such that it holds an exact number of frames
    data = np.concatenate((data, np.zeros(newLengthData - data.size)))
    
    # the output STFT has nfft/2+1 rows. Note that nfft has to be an even
    # number (and a power of 2 for the fft to be fast)
    numberFrequencies = int(nfft / 2 + 1)
    
    STFT = np.zeros([numberFrequencies, numberFrames], dtype=complex)
    
    # storing FT of each frame in STFT:
    for n in np.arange(numberFrames):
        beginFrame = int(n*hopsize)
        endFrame = beginFrame+lengthWindow
        frameToProcess = window*data[beginFrame:endFrame]
        STFT[:,n] = np.fft.rfft(frameToProcess, np.int32(nfft))
        frameToProcess = None
    
    return STFT.T

def istft_norm(X, window=sinebell(2048),
          analysisWindow=None,
          hopsize=256.0, nfft=2048.0):
    """
    data = istft_norm(X,window=sinebell(2048),hopsize=1024.0,nfft=2048.0,fs=44100)

    Computes an inverse of the short time Fourier transform (STFT),
    here, the overlap-add procedure is implemented.

    Inputs:
        X                     :
            STFT of the signal, to be \"inverted\"
        window=sinebell(2048) :
            synthesis window
            (should be the \"complementary\" window
            for the analysis window)
        hopsize=1024.0        :
            hopsize for the analysis
        nfft=2048.0           :
            number of points for the Fourier computation
            (the user has to provide an even number)

    Outputs:
        data                  :
            time series corresponding to the given STFT
            the first half-window is removed, complying
            with the STFT computation given in the
            function stft
    
    """
    X=X.T
    if analysisWindow is None:
        analysisWindow = window
    
    lengthWindow = np.array(window.size)
    numberFrequencies, numberFrames = X.shape
    lengthData = int(hopsize*(numberFrames-1) + lengthWindow)
    
    normalisationSeq = np.zeros(lengthData)
    
    data = np.zeros(lengthData)
    
    for n in np.arange(numberFrames):
        beginFrame = int(n * hopsize)
        endFrame = beginFrame + lengthWindow
        frameTMP = np.fft.irfft(X[:,n], np.int32(nfft))
        frameTMP = frameTMP[:lengthWindow]
        normalisationSeq[beginFrame:endFrame] = (
            normalisationSeq[beginFrame:endFrame] +
            window * analysisWindow)
        data[beginFrame:endFrame] = (
            data[beginFrame:endFrame] + window * frameTMP)
    
    data = data[int(lengthWindow/2.0):]
    normalisationSeq = normalisationSeq[int(lengthWindow/2.0):]
    normalisationSeq[normalisationSeq==0] = 1.
 
    data = data / normalisationSeq
    
    return data



