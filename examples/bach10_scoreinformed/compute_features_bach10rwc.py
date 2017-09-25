"""
    This file is part of DeepConvSep.

    Copyright (c) 2014-2017 Marius Miron  <miron.marius at gmail.com>

    DeepConvSep is free software: you can redistribute it and/or modify
    it under the terms of the Afero GPL License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    DeepConvSep is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the Afero GPL License
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
import rwc
import multiprocessing
from multiprocessing import Pool

# Useful constants
MIDI_A4 = 69   # MIDI Pitch number
FREQ_A4 = 440. # Hz
SEMITONE_RATIO = 2. ** (1. / 12.) # Ascending


class Engine(object):
    def __init__(self,db,feature_path,instruments,allowed_styles,allowed_dynamics,allowed_case,time_shifts,rwc_path,chunk_size,sample_size,style,style_midi,nharmonics,interval,tuning_freq):

      self.allowed_styles = allowed_styles
      self.allowed_dynamics = allowed_dynamics
      self.allowed_case = allowed_case
      self.time_shifts = time_shifts
      #self.allowed_case = [1]
      self.sampleRate=44100
      self.db = db
      self.chunk_size = chunk_size
      self.instruments = instruments

      self.sources = ['bassoon','clarinet','saxophone','violin']
      self.sources_midi = ['bassoon','clarinet','saxophone','violin']
      self.style = style
      self.style_midi = style_midi

      self.nharmonics = nharmonics
      self.tuning_freq = tuning_freq
      self.interval=interval
      self.feature_path = feature_path

      #time_shifts=[0.]
      intensity_shifts=list(range(len(self.allowed_dynamics)))
      timbre_shifts=self.allowed_case
      style_shifts=list(range(len(self.allowed_styles)))

      cc=[(time_shifts[i], intensity_shifts[j], style_shifts[l], timbre_shifts[k]) for i in xrange(len(time_shifts)) for j in xrange(len(intensity_shifts)) for l in xrange(len(style_shifts)) for k in xrange(len(timbre_shifts))]
      #import pdb;pdb.set_trace()
      if len(cc)<len(self.sources):
        combo1 = list(it.product(cc,repeat=len(self.sources)))
        combo = []
        for i in range(len(combo1)):
          c = np.array(combo1[i])
          #if (all(x == c[0,0] for x in c[:,0]) or all(x == c[0,1] for x in c[:,1])) \
          if (len(intensity_shifts)==1 and not(all(x == c[0,0] for x in c[:,0]))) \
            or (len(time_shifts)==1 and not(all(x == c[0,1] for x in c[:,1]))):
              combo.append(c)
        combo = np.array(combo)
      else:
        combo = np.array(list(it.permutations(cc,len(self.sources))))
      if len(combo)==0:
        combo = np.array([[[time_shifts[0],intensity_shifts[0],style_shifts[0],timbre_shifts[0]] for s in self.sources]])

      if sample_size<len(combo):
        sampled_combo = combo[np.random.choice(len(combo),size=sample_size, replace=False)]
      else:
        sampled_combo = combo

      combo = None
      self.combo = sampled_combo
      sampled_combo=None

    def getCombos(self):
      return self.combo

    def __call__(self, combo):
      c = np.array(combo)
      chunk_size = self.chunk_size
      db = self.db
      feature_path = self.feature_path

      tt=transformFFT(frameSize=4096, hopSize=512, sampleRate=44100, window=blackmanharris)
      maxLength=0
      for i in range(len(self.sources)):
        instlen = util.getMidiLength(self.sources_midi[i]+'_g'+self.style_midi,db)
        if instlen>maxLength:
          maxLength = instlen
      if chunk_size>maxLength:
        chunk_size = maxLength

      for chnk in range(int(np.floor(maxLength/chunk_size))):
        chunk_start = float(chunk_size * chnk)
        chunk_end = float((chnk+1) * chunk_size)
        if not os.path.isfile(os.path.join(feature_path,self.style,str(c).encode('base64','strict')+'_'+str(chnk)+'.data')):
          try:
            nelem_g=1
            for i in range(len(self.sources)):
                ng = util.getMidiNum(self.sources_midi[i]+'_g'+self.style_midi,db,chunk_start,chunk_end)
                nelem_g = np.maximum(ng,nelem_g)
            melody_g = np.zeros((len(self.sources),int(nelem_g),2*self.nharmonics+3))
            melody_e = np.zeros((len(self.sources),int(nelem_g),2*self.nharmonics+3))

            for i in range(len(self.sources)):

              nframes = int(np.ceil(chunk_size*self.sampleRate / np.double(tt.hopSize))) + 2
              size = int(chunk_size*self.sampleRate-int(np.max( c[:,0].astype(float))*self.sampleRate))

              if self.sampleRate != 44100:
                  print 'sample rate is not consistent'
              if i==0:
                  audio = np.zeros((size,len(self.sources)+1))

              tmp = util.expandMidi(self.sources_midi[i]+'_g'+self.style_midi,db,chunk_start,chunk_end,self.interval,self.tuning_freq,self.nharmonics,self.sampleRate,tt.hopSize,tt.frameSize,c[i,0],c[i,0],nframes)
              melody_g[i,:tmp.shape[0],:] = tmp
              tmp = None
              tmp = util.expandMidi(self.sources_midi[i]+'_g'+self.style_midi,db,chunk_start,chunk_end,self.interval,self.tuning_freq,self.nharmonics,self.sampleRate,tt.hopSize,tt.frameSize,c[i,0]+0.2,c[i,0]+0.2,nframes,fermata=c[i,0]+0.5)
              melody_e[i,:tmp.shape[0],:] = tmp
              tmp = None
              #generate the audio, note by note
              for m in range(nelem_g):
                if melody_g[i,m,2]>0:
                  note = self.instruments[i].getNote(melody_g[i,m,2],self.allowed_dynamics[int(c[i,1])],self.allowed_styles[int(c[i,2])],int(c[i,3]))
                  if note is None:
                    raise GetOutOfLoop
                  else:
                    segment = note.getAudio(max_duration=float(melody_g[i,m,1]-melody_g[i,m,0])*tt.hopSize/self.sampleRate)
                    if len(segment)>(len(audio)-int(np.floor(melody_g[i,m,0]*tt.hopSize))):
                      audio[int(np.floor(melody_g[i,m,0]*tt.hopSize)):int(np.floor(melody_g[i,m,0]*tt.hopSize)+len(segment)),i+1] = segment[:len(audio)-int(np.floor(melody_g[i,m,0]*tt.hopSize))]
                    else:
                      audio[int(np.floor(melody_g[i,m,0]*tt.hopSize)):int(np.floor(melody_g[i,m,0]*tt.hopSize)+len(segment)),i+1] = segment
                    segment = None
                  note = None
                  segment = None

            audio[:,0] = np.sum(audio[:,1:len(self.sources)+1],axis=1)
            tt.compute_transform(audio,os.path.join(feature_path,self.style,str(c).encode('base64','strict')+'_'+str(chnk)+'.data'),phase=False)
            tt.saveTensor(melody_g, '__g_')
            tt.saveTensor(melody_e, '__e_')
            audio = None
            melody_g = None
            melody_e = None
          except GetOutOfLoop:
            pass

class GetOutOfLoop( Exception):
  pass


if __name__ == "__main__":
  if len(sys.argv)>-1:
    climate.add_arg('--db', help="the Bach 10 Sibelius dataset path")
    climate.add_arg('--rwc', help="the rwc instrument sound path with mat and wav subfolders")
    climate.add_arg('--chunk_size', help="the chunk size to split the midi")
    climate.add_arg('--sample_size', help="sample this number of combinations of possible cases")
    climate.add_arg('--nprocs', help="the number of processors")
    climate.add_arg('--feature_path', help="the path where to save the features")
    climate.add_arg('--original', help="compute features for the original score or ground truth aligned score")
    kwargs = climate.parse_args()
    if kwargs.__getattribute__('db'):
      db = kwargs.__getattribute__('db')
    else:
      db='/home/marius/Documents/Database/Bach10/Source separation/'
      # db='/Volumes/Macintosh HD 2/Documents/Database/Bach10/Source separation/'
    if kwargs.__getattribute__('rwc'):
      rwc_path = kwargs.__getattribute__('rwc')
    else:
      rwc_path='/home/marius/Documents/Database/RWC/'
      # rwc_path='/Volumes/Macintosh HD 2/Documents/Database/RWC/'
    if kwargs.__getattribute__('chunk_size'):
        chunk_size = float(kwargs.__getattribute__('chunk_size'))
    else:
        chunk_size = 45.0

    if kwargs.__getattribute__('nprocs'):
        nprocs = int(kwargs.__getattribute__('nprocs'))
    else:
        nprocs = multiprocessing.cpu_count()-1
    if kwargs.__getattribute__('sample_size'):
        sample_size = float(kwargs.__getattribute__('sample_size'))
    else:
        sample_size = 400

    if kwargs.__getattribute__('feature_path'):
        feature_path = kwargs.__getattribute__('feature_path')
    else:
        feature_path = os.path.join(db,'transforms','t3_rwc')

    if kwargs.__getattribute__('original'):
        original = int(kwargs.__getattribute__('original'))
    else:
        original = True

    assert os.path.isdir(db), "Please input the directory for the Bach10 Sibelius dataset with --db path_to_Bach10"
    assert os.path.isdir(rwc_path), "Please input the directory for the RWC instrument sound with --db path_to_RWC"

    if original:
      style = ['original']
      style_midi = ['_original']
      time_shifts=[0.,0.1,0.2]
    else:
      style = ['gt']
      style_midi = ['']
      time_shifts=[0.]
      sample_size = np.minimum(50,sample_size)

    allowed_styles = ['NO']
    allowed_dynamics = ['F','M','P']
    allowed_case = [1,2,3]
    instrument_nums=[30,31,27,15]

    nharmonics=20
    interval=50 #cents
    tuning_freq=440 #Hz

    instruments = []
    for ins in range(len(instrument_nums)):
        instruments.append(rwc.Instrument(rwc_path,instrument_nums[ins],allowed_styles,allowed_case,allowed_dynamics))

    #compute transform
    for f in sorted(os.listdir(db)):
        if os.path.isdir(os.path.join(db,f)) and f[0].isdigit() :

          for s in range(len(style)):

              if not os.path.exists(os.path.join(feature_path,f,style[s])):
                  os.makedirs(os.path.join(feature_path,f,style[s]))

              engine = Engine(os.path.join(db,f),os.path.join(feature_path,f),instruments,allowed_styles,allowed_dynamics,allowed_case,time_shifts,rwc_path,chunk_size,sample_size,style[s],style_midi[s],nharmonics,interval,tuning_freq)
              combos = engine.getCombos()
              print len(combos)
              try:
                pool = Pool(nprocs) # on nprocs processors
                pool.map(engine, combos)
              except Exception as e:
                print str(e)
                pool.terminate()
              finally: # To make sure processes are closed in the end, even if errors happen
                pool.close()
                pool.join()
