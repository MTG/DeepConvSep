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

import os,sys
import transform
import util
from transform import transformFFT
import numpy as np
import re
from scipy.signal import blackmanharris as blackmanharris
import climate


if __name__ == "__main__": 
    if len(sys.argv)>-1:
        climate.add_arg('--db', help="the dataset path")
    kwargs = climate.parse_args()
    if kwargs.__getattribute__('db'):
        db = kwargs.__getattribute__('db')
    # else:
    #     db='/home/marius/Documents/Database/iKala/'  
    assert os.path.isdir(db), "Please input the directory for the iKala dataset with --db path_to_iKala"

    tt = None
    for f in os.listdir(os.path.join(db,"Wavfile")):
        if f.endswith(".wav"):
            #read the audio file
            audioObj, sampleRate, bitrate = util.readAudioScipy(os.path.join(db,"Wavfile",f))
            if tt is None:
                #initialize the transform object which will compute the STFT
                tt=transformFFT(frameSize=2048, hopSize=512, sampleRate=sampleRate, window=blackmanharris)
                pitchhop=0.032*float(sampleRate) #seconds to frames
            assert sampleRate == 44100,"Sample rate needs to be 44100"
    
            audio = np.zeros((audioObj.shape[0],3))

            audio[:,0] = audioObj[:,0] + audioObj[:,1] #create mixture voice + accompaniment
            audio[:,1] = audioObj[:,1] #voice
            audio[:,2] = audioObj[:,0] #accompaniment
            audioObj=None

            #read pitches so they can be written as separate features 
            lines = np.loadtxt(db+"PitchLabel/"+f.replace('wav','pv'), comments="#", delimiter="\n", unpack=False)
            
            if not os.path.exists(db+"transforms/t1/"):
                os.makedirs(db+"transforms/t1/")
            #compute the STFT and write the .data file in the subfolder /transform/t1/ of the iKala folder
            tt.compute_transform(audio,db+"transforms/t1/"+f.replace('.wav','.data'),pitch=lines[np.newaxis,np.newaxis,:],phase=False)
       
