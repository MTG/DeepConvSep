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
import rwc
import multiprocessing
from multiprocessing import Pool

# Useful constants
MIDI_A4 = 69   # MIDI Pitch number
FREQ_A4 = 440. # Hz
SEMITONE_RATIO = 2. ** (1. / 12.) # Ascending


class Engine(object):
    def __init__(self,db,feature_path,instruments,allowed_styles,allowed_dynamics,allowed_case,time_shifts,rwc_path,chunk_size,sample_size,style,style_midi):
      
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

      tt=transformFFT(frameSize=4096, hopSize=512, sampleRate=44100, window=blackmanharris)

      maxLength=0
      for i in range(len(self.sources)):
        instlen = int(util.getMidiLength(self.sources_midi[i]+'_g'+self.style_midi[s],os.path.join(db,f)))
        if instlen>maxLength:
          maxLength = instlen
      if chunk_size>maxLength:
        chunk_size = maxLength

      for chnk in range(int(np.floor(maxLength/chunk_size))):
        chunk_start = chunk_size * chnk
        chunk_end = (chnk+1) * chunk_size
        if not os.path.isfile(os.path.join(feature_path,f,self.style[s],f+'_'+str(c)+'_'+str(chnk)+'.data')):
          try:
            for i in range(len(self.sources)):
         
              nframes = int(np.ceil(chunk_size*self.sampleRate / np.double(tt.hopSize))) + 2
              size = int(chunk_size*self.sampleRate-int(np.max( c[:,0].astype(float))*self.sampleRate))

              if self.sampleRate != 44100:
                  print 'sample rate is not consistent'
              if i==0:
                  audio = np.zeros((size,len(self.sources)+1))
                  melody = np.zeros((len(self.sources),1,int(nframes)))
              
              melody[i,0:1,:],melodyBegin,melodyEnd,melNotes = util.getMidi(self.sources_midi[i]+'_g'+self.style_midi[s],os.path.join(db,f),chunk_start,chunk_end,self.sampleRate,tt.hopSize,tt.frameSize,c[i,0],c[i,0],nframes)
  
              #generate the audio, note by note
              for m in range(len(melNotes)):
                note = self.instruments[i].getNote(melNotes[m],self.allowed_dynamics[int(c[i,1])],self.allowed_styles[int(c[i,2])],int(c[i,3]))
                if note is None:
                  raise GetOutOfLoop
                else:
                  segment = note.getAudio(max_duration=melodyEnd[m]-melodyBegin[m])
                  if len(segment)>(len(audio)-int(np.floor(melodyBegin[m]*self.sampleRate))):
                    audio[int(np.floor(melodyBegin[m]*self.sampleRate)):int(np.floor(melodyBegin[m]*self.sampleRate)+len(segment)),i+1] = segment[:len(audio)-int(np.floor(melodyBegin[m]*self.sampleRate))]
                  else:
                    audio[int(np.floor(melodyBegin[m]*self.sampleRate)):int(np.floor(melodyBegin[m]*self.sampleRate)+len(segment)),i+1] = segment
                  segment = None
                note = None
                segment = None

            audio[:,0] = np.sum(audio[:,1:len(self.sources)+1],axis=1)
                
            tt.compute_transform(audio,os.path.join(feature_path,f,self.style[s],f+'_'+str(c)+'_'+str(chnk)+'.data'),pitch=melody,phase=False,pitch_interp='zero')
           
            audio = None
            melody = None
          except GetOutOfLoop:
            pass

class GetOutOfLoop( Exception ):
  pass


if __name__ == "__main__": 
  if len(sys.argv)>-1:
    climate.add_arg('--db', help="the dataset path")
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
      db='/home/user/Documents/Database/Bach10 Sibelius/' 
      
    if kwargs.__getattribute__('rwc'):
      rwc_path = kwargs.__getattribute__('rwc')
    else:
      rwc_path='/home/user/Documents/Database/RWC/'  
     
    if kwargs.__getattribute__('chunk_size'):
        chunk_size = float(kwargs.__getattribute__('chunk_size'))
    else:
        chunk_size = 45

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
      sample_size = np.minimum(100,sample_size)

    allowed_styles = ['NO']
    allowed_dynamics = ['F','M','P']
    allowed_case = [1,2,3]
    instrument_nums=[30,31,27,15]
            
    instruments = []
    for ins in range(len(instrument_nums)):
        instruments.append(rwc.Instrument(rwc_path,instrument_nums[ins],allowed_styles,allowed_case,allowed_dynamics))
    
    #compute transform
    for f in sorted(os.listdir(os.path.join(db))):
        if os.path.isdir(os.path.join(db,f)) and f[0].isdigit() :

          for s in range(len(style)):

              if not os.path.exists(os.path.join(feature_path,f,style[s])):
                  os.makedirs(os.path.join(feature_path,f,style[s])) 
        
              engine = Engine(db,feature_path,instruments,allowed_styles,allowed_dynamics,allowed_case,time_shifts,rwc_path,chunk_size,sample_size,style,style_midi)
              combos = engine.getCombos() 
              print len(combos)
              try:
                pool = Pool(nprocs) # on nprocs processors
                pool.map(engine, combos)
              except:
                pool.terminate()
              finally: # To make sure processes are closed in the end, even if errors happen
                pool.close()
                pool.join()   
