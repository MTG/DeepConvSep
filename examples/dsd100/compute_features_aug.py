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
import itertools as it


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
    sources = ['vocals','bass','drums','other']

    #possible augmentations
    time_shifts=[0.,0.2]
    intensity_shifts=[1.]
    cc=[(time_shifts[i], intensity_shifts[j]) for i in xrange(len(time_shifts)) for j in xrange(len(intensity_shifts))]
    if len(cc)<len(sources):
        combo1 = list(it.product(cc,repeat=len(sources)))
        combo = []    
        for i in range(len(combo1)):
          c = np.array(combo1[i])
          #if (all(x == c[0,0] for x in c[:,0]) or all(x == c[0,1] for x in c[:,1])) \
          if (len(intensity_shifts)==1 and not(all(x == c[0,0] for x in c[:,0]))) \
            or (len(time_shifts)==1 and not(all(x == c[0,1] for x in c[:,1]))):
              combo.append(c)
    else:
        combo = list(it.permutations(cc,len(sources)))
    if len(combo)==0:
        combo = [[[time_shifts[0],intensity_shifts[0]] for s in sources]]
    
    tt = None
    dirlist = os.listdir(os.path.join(mixture_directory,"Dev"))
    dirlist.extend(os.listdir(os.path.join(mixture_directory,"Test")))
    dev_directory = os.listdir(os.path.join(mixture_directory,"Dev"))
    test_directory = os.listdir(os.path.join(mixture_directory,"Test"))
    for f in sorted(dirlist):
        if not f.startswith('.'):
            for co in combo:       
                c = np.array(co)  

                if f in dev_directory:
                    song_dir=os.path.join(source_directory,"Dev",f)
                else:
                    song_dir=os.path.join(source_directory,"Test",f)               

                #read the sources audio files
                vocals, sampleRate, bitrate = util.readAudioScipy(os.path.join(song_dir,"vocals.wav"))
                if vocals.shape[1]>1:
                    vocals[:,0] = (vocals[:,0] + vocals[:,1]) / 2
                    vocals = vocals[:,0]

                #initialize variables
                number_of_blocks=int(len(vocals)/(float(sampleRate)*30.0))
                last_block=int(len(vocals)%float(sampleRate))
                if tt is None:
                    #initialize the transform object which will compute the STFT
                    tt=transformFFT(frameSize=1024, hopSize=512, sampleRate=sampleRate, window=blackmanharris)
                nframes = int(np.ceil(len(vocals) / np.double(tt.hopSize))) + 2
                size = int(len(vocals)-int(np.max(np.array(c[:,0]))*sampleRate))

                bass, sampleRate, bitrate = util.readAudioScipy(os.path.join(song_dir,"bass.wav"))
                if bass.shape[1]>1:
                    bass[:,0] = (bass[:,0] + bass[:,1]) / 2
                    bass = bass[:,0]
                drums, sampleRate, bitrate = util.readAudioScipy(os.path.join(song_dir,"drums.wav"))
                if drums.shape[1]>1:
                    drums[:,0] = (drums[:,0] + drums[:,1]) / 2
                    drums = drums[:,0]
                others, sampleRate, bitrate = util.readAudioScipy(os.path.join(song_dir,"other.wav"))
                if others.shape[1]>1:
                    others[:,0] = (others[:,0] + others[:,1]) / 2
                    others = others[:,0]
                
                assert sampleRate == 44100,"Sample rate needs to be 44100"

                #apply circular shift
                vocals = c[0,1]*util.circular_shift(vocals,min_size=size,cs=c[0,0],sampleRate=sampleRate)
                bass = c[1,1]*util.circular_shift(bass,min_size=size,cs=c[1,0],sampleRate=sampleRate)
                drums = c[2,1]*util.circular_shift(drums,min_size=size,cs=c[2,0],sampleRate=sampleRate)
                others = c[3,1]*util.circular_shift(others,min_size=size,cs=c[3,0],sampleRate=sampleRate)
                mix_raw = vocals + bass + drums + others

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
                    tt.compute_transform(audio,os.path.join(feature_path,f+"_"+str(i)+str(c[:,0])+'.data'),phase=False)
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
                tt.compute_transform(audio,os.path.join(feature_path,f+"_"+str(i+1)+str(c[:,0])+'.data'),phase=False)
                audio = None
                rest = None 
                mix_raw = None
                vocals = None 
                bass = None 
                drums = None
                others = None 