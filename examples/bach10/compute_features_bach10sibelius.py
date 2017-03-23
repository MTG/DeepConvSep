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
import itertools as it
from scipy.signal import blackmanharris as blackmanharris
import climate


if __name__ == "__main__": 
    if len(sys.argv)>-1:
        climate.add_arg('--db', help="the Bach10 Sibelius dataset path")
        climate.add_arg('--feature_path', help="the path where to save the features")
        climate.add_arg('--gt', help="compute features for the ground truth aligned rendition or the others")
    db=None
    kwargs = climate.parse_args()
    if kwargs.__getattribute__('db'):
        db = kwargs.__getattribute__('db')
    else:
        db='/home/marius/Documents/Database/Bach10/Source separation/' 
        # db='/Volumes/Macintosh HD 2/Documents/Database/Bach10/Source separation/'  
    if kwargs.__getattribute__('feature_path'):
        feature_path = kwargs.__getattribute__('feature_path')
    else:
        feature_path=os.path.join(db,'transforms','t3_synth_aug_more') 
    if kwargs.__getattribute__('gt'):
        gt = int(kwargs.__getattribute__('gt')) 
    else:
        gt = True
    assert os.path.isdir(db), "Please input the directory for the Bach10 Sibelius dataset with --db path_to_Bach10"
    
    sources = ['bassoon','clarinet','saxophone','violin']
    sources_midi = ['bassoon','clarinet','saxophone','violin']

    if gt:
        style = ['gt']
        style_midi = ['']    
        time_shifts=[0.]
        intensity_shifts=[1.]
    else:
        style = ['fast','slow','original']
        style_midi = ['_fast20','_slow20','_original']    
        time_shifts=[0.,0.1,0.2]
        intensity_shifts=[1.]

    cc=[(time_shifts[i], intensity_shifts[j]) for i in xrange(len(time_shifts)) for j in xrange(len(intensity_shifts))]
    if len(cc)<len(sources):
        combo1 = list(it.product(cc,repeat=len(sources)))
        combo = []    
        for i in range(len(combo1)):
            c = np.array(combo1[i])
            if (len(intensity_shifts)==1 and not(all(x == c[0,0] for x in c[:,0]))) \
                or (len(time_shifts)==1 and not(all(x == c[0,1] for x in c[:,1]))):
                combo.append(c)
    else:
        combo = list(it.permutations(cc,len(sources)))
    if len(combo)==0:
        combo = [[[time_shifts[0],intensity_shifts[0]] for s in sources]]
    #print len(combo)

    #compute transform
    for f in os.listdir(db):
        if os.path.isdir(os.path.join(db,f)) and f[0].isdigit() :
            if not f.startswith('.'):
                for s in range(len(style)):
                    if not os.path.exists(os.path.join(feature_path,style[s])):
                        os.makedirs(os.path.join(feature_path,style[s]))
                    for co in combo:       
                        c = np.array(co)  
                        for i in range(len(sources)):
                            #read the audio file
                            sounds,sampleRate,bitrate = util.readAudioScipy(os.path.join(db,f,f+'_'+style[s]+'_'+sources[i]+'.wav'))
                            
                            if sampleRate != 44100:
                                print 'sample rate is not consistent'

                            if i==0:
                                tt=transformFFT(frameSize=4096, hopSize=512, sampleRate=44100, window=blackmanharris)
                                nframes = int(np.ceil(len(sounds) / np.double(tt.hopSize))) + 2
                                size = int(len(sounds)-int(np.max(np.array(c[:,0]))*sampleRate))
                                audio = np.zeros((size,len(sources)+1))
                            
                            if c[i,0] == 0:
                                if len(sounds) > size:
                                    segment = sounds[:size]
                                else:
                                    segment = np.zeros(size)
                                    segment[:len(sounds)] = sounds
                            elif c[i,0] < 0:
                                seg_idx = int(abs(c[i,0]*sampleRate))
                                segment = np.pad(sounds,((0,seg_idx+np.maximum(0,size-len(sounds)))), mode='constant')
                                if len(segment)<(size+seg_idx):
                                    segment = np.pad(segment,((0,size+seg_idx - len(segment))), mode='constant')
                                segment = segment[seg_idx:size+seg_idx]
                            else:
                                segment = np.pad(sounds,((int(c[i,0]*sampleRate),0)), mode='constant')
                                if len(segment)<size:
                                    segment = np.pad(segment,((0,size - len(segment))), mode='constant')
                                segment = segment[:size]
                            
                            audio[:,0] = audio[:,0] + c[i,1] * segment[:size]
                            audio[:,i+1] = c[i,1] * segment[:size]

                            segment = None
                            sounds = None

                        tt.compute_transform(audio,os.path.join(feature_path,style[s],f+'.data'),phase=False)
                    