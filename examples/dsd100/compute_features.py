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
        climate.add_arg('--feature_path', help="the path where to save the features")
    db=None
    kwargs = climate.parse_args()
    if kwargs.__getattribute__('db'):
        db = kwargs.__getattribute__('db')
    else:
        db='/home/marius/Documents/Database/DSD100/'  
    if kwargs.__getattribute__('feature_path'):
        feature_path = kwargs.__getattribute__('feature_path')
    else:
        feature_path=os.path.join(db,'transforms','t1') 
    assert os.path.isdir(db), "Please input the directory for the DSD100 dataset with --db path_to_DSD"
    
    mixture_directory=os.path.join(db,'Mixtures')
    source_directory=os.path.join(db,'Sources')

    tt = None
    dirlist = os.listdir(os.path.join(mixture_directory,"Dev"))
    for f in sorted(dirlist):
        if not f.startswith('.'):
            #read the mix audio file
            mix_raw, sampleRate, bitrate = util.readAudioScipy(os.path.join(mixture_directory,"Dev",f,"mixture.wav"))
            
            if mix_raw.shape[1]>1:
                mix_raw[:,0] = (mix_raw[:,0] + mix_raw[:,1]) / 2
                mix_raw = mix_raw[:,0]

            number_of_blocks=int(len(mix_raw)/(float(sampleRate)*30.0))
            last_block=int(len(mix_raw)%float(sampleRate))

            #read the sources audio files
            vocals, sampleRate, bitrate = util.readAudioScipy(os.path.join(source_directory,"Dev",f,"vocals.wav"))
            if vocals.shape[1]>1:
                vocals[:,0] = (vocals[:,0] + vocals[:,1]) / 2
                vocals = vocals[:,0]
            bass, sampleRate, bitrate = util.readAudioScipy(os.path.join(source_directory,"Dev",f,"bass.wav"))
            if bass.shape[1]>1:
                bass[:,0] = (bass[:,0] + bass[:,1]) / 2
                bass = bass[:,0]
            drums, sampleRate, bitrate = util.readAudioScipy(os.path.join(source_directory,"Dev",f,"drums.wav"))
            if drums.shape[1]>1:
                drums[:,0] = (drums[:,0] + drums[:,1]) / 2
                drums = drums[:,0]
            others, sampleRate, bitrate = util.readAudioScipy(os.path.join(source_directory,"Dev",f,"other.wav"))
            if others.shape[1]>1:
                others[:,0] = (others[:,0] + others[:,1]) / 2
                others = others[:,0]

            if tt is None:
                #initialize the transform object which will compute the STFT
                tt=transformFFT(frameSize=1024, hopSize=512, sampleRate=sampleRate, window=blackmanharris)
 
            assert sampleRate == 44100,"Sample rate needs to be 44100"
    
            #Take chunks of 30 secs
            for i in range(number_of_blocks):
                audio = np.zeros((sampleRate*30,5))
                audio[:,0]=mix_raw[i*30*sampleRate:(i+1)*30*sampleRate] 
                audio[:,1]=vocals[i*sampleRate*30:(i+1)*30*sampleRate]
                audio[:,2]=bass[i*sampleRate*30:(i+1)*sampleRate*30]
                audio[:,3]=drums[i*sampleRate*30:(i+1)*sampleRate*30]
                audio[:,4]=others[i*sampleRate*30:(i+1)*sampleRate*30]
              
                if not os.path.exists(feature_path):
                    os.makedirs(feature_path)
                #compute the STFT and write the .data file in the subfolder /transform/t1/ of the iKala folder
                tt.compute_transform(audio,os.path.join(feature_path,f+"_"+str(i)+'.data'),phase=False)
                audio = None

            #rest of file
            rest=mix_raw[(i+1)*30*sampleRate:]
            audio = np.zeros((len(rest),5))
            audio[:,0]=rest
            audio[:,1]=vocals[(i+1)*30*sampleRate:]
            audio[:,2]=bass[sampleRate*30*(i+1):]
            audio[:,3]=drums[sampleRate*30*(i+1):]
            audio[:,4]=others[sampleRate*30*(i+1):]
            
            #compute the STFT and write the .data file in the subfolder /transform/t1/ of the iKala folder
            tt.compute_transform(audio,os.path.join(feature_path,f+"_"+str(i+1)+'.data'),phase=False)
            audio = None
            rest = None 
            mix_raw = None
            vocals = None 
            bass = None 
            drums = None
            others = None 