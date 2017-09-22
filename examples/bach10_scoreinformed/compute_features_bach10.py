"""
    This file is part of DeepConvSep.

    Copyright (c) 2014-2017 Marius Miron  <miron.marius at gmail.com>

    DeepConvSep is free software: you can redistribute it and/or modify
    it under the terms of the Affero GPL License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    DeepConvSep is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the Affero GPL License
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
        climate.add_arg('--db', help="the Bach10 dataset path")
        climate.add_arg('--feature_path', help="the path where to save the features")
    db=None
    kwargs = climate.parse_args()
    if kwargs.__getattribute__('db'):
        db = kwargs.__getattribute__('db')
    else:
        db='/home/marius/Documents/Database/Bach10/Sources/'
        db='/Volumes/Macintosh HD 2/Documents/Database/Bach10/Sources/'
    if kwargs.__getattribute__('feature_path'):
        feature_path = kwargs.__getattribute__('feature_path')
    else:
        feature_path=os.path.join(db,'transforms','t3')
    assert os.path.isdir(db), "Please input the directory for the Bach10 dataset with --db path_to_Bach10"

    nharmonics=20
    interval=50 #cents
    tuning_freq=440 #Hz
    sources = ['bassoon','clarinet','saxphone','violin']
    sources_midi = ['bassoon','clarinet','saxophone','violin']

    if not os.path.exists(feature_path):
        os.makedirs(feature_path)

    #compute transform
    for f in os.listdir(db):
        if os.path.isdir(os.path.join(db,f)) and f[0].isdigit() :
            if not f.startswith('.'):
                nelem_g=1
                nelem_b=1
                for i in range(len(sources)):
                    ng = util.getMidiNum(sources_midi[i]+'_g',os.path.join(db,f),0,40.0)
                    nelem_g = np.maximum(ng,nelem_g)
                    nb = util.getMidiNum(sources_midi[i]+'_b',os.path.join(db,f),0,40.0)
                    nelem_b = np.maximum(nb,nelem_b)
                melody_g = np.zeros((len(sources),int(nelem_g),2*nharmonics+3))
                melody_b = np.zeros((len(sources),int(nelem_b),2*nharmonics+3))
                melody_e = np.zeros((len(sources),int(nelem_b),2*nharmonics+3))

                for i in range(len(sources)):
                    #read the audio file
                    audioObj, sampleRate, bitrate = util.readAudioScipy(os.path.join(db,f,f+'-'+sources[i]+'.wav'))

                    if i==0:
                        tt=transformFFT(frameSize=4096, hopSize=512, sampleRate=44100, window=blackmanharris)
                        nframes = int(len(audioObj)/tt.hopSize)
                        audio = np.zeros((audioObj.shape[0],len(sources)+1))

                    audio[:,0] = audio[:,0] + audioObj
                    audio[:,i+1] = audioObj
                    audioObj=None

                    tmp = util.expandMidi(sources_midi[i]+'_g',os.path.join(db,f),0,40.0,interval,tuning_freq,nharmonics,sampleRate,tt.hopSize,tt.frameSize,0.,0.,nframes)
                    melody_g[i,:tmp.shape[0],:] = tmp
                    tmp = None
                    tmp = util.expandMidi(sources_midi[i]+'_b',os.path.join(db,f),0,40.0,interval,tuning_freq,nharmonics,sampleRate,tt.hopSize,tt.frameSize,0.,0.,nframes,fermata=0.5)
                    melody_b[i,:tmp.shape[0],:] = tmp
                    tmp = None
                    tmp = util.expandMidi(sources_midi[i]+'_b',os.path.join(db,f),0,40.0,interval,tuning_freq,nharmonics,sampleRate,tt.hopSize,tt.frameSize,0.2,0.2,nframes,fermata=0.5)
                    melody_e[i,:tmp.shape[0],:] = tmp
                    tmp = None

                #masks_temp = testfilterSpec(audio,tt,melody_b,0,nframes)

                tt.compute_transform(audio,os.path.join(feature_path,f+'.data'),phase=False)
                tt.saveTensor(melody_g, '_'+tt.suffix+'_g_')
                tt.saveTensor(melody_b, '_'+tt.suffix+'_b_')
                tt.saveTensor(melody_e, '_'+tt.suffix+'_e_')

