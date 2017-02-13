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
from scipy import io
import os
import sys
from os import listdir
from os.path import isfile, join
import cPickle as pickle
import random
import re
import multiprocessing 
import util
import climate
import itertools as it
logging = climate.get_logger('dataset')
climate.enable_default_logging()


"""
Routines for multiprocessing which can be used inside a class, e.g. loading many files from hard drive in parallel 
"""
def fun(f,q_in,q_out):
    while True:
        i,x = q_in.get()
        if i is None:
            break
        q_out.put((i,f(x)))

def parmap(f, X, nprocs = multiprocessing.cpu_count()-1):
    """
    Paralellize the function f with the list X, using a number of CPU of nprocs
    """
    nprocs = np.maximum(multiprocessing.cpu_count()-1,nprocs)
    q_in   = multiprocessing.Queue(1)
    q_out  = multiprocessing.Queue()

    proc = [multiprocessing.Process(target=fun,args=(f,q_in,q_out)) for _ in range(nprocs)]
    for p in proc:
        p.daemon = True
        p.start()

    sent = [q_in.put((i,x)) for i,x in enumerate(X)]
    [q_in.put((None,None)) for _ in range(nprocs)]
    res = [q_out.get() for _ in range(len(sent))]

    [p.join() for p in proc]

    return [x for i,x in sorted(res)]


"""
Classes to load features which have been computed with one of the functions in transform.py, 
and yield batches necessary for training neural networks. 
These classes are useful when the data does not fit into memory, and the batches can be loaded in chunks.
"""
class LargeDataset(object):
    """
    The parent class to load data in chunks and prepare batches for training neural networks

    Parameters
    ----------
    path_transform_in : list of strings or string
        The paths for the directories from where to load the input data to the network
    path_transform_out : list of strings or string, optional
        The paths for the directories from where to load the output data to the network
        If ommited is considered to be the same as path_transform_in
    exclude_list : list of strings or string, optional
        This list contains strings which are used to filter the data. Files containing these strings are excluded.
    batch_size : int, optional
        The number of examples in a batch
    batch_memory : int, optional
        The number of batches to load in memory at once
    time_context : int, optional
        The time context modeled by the network. 
        The data files are split into segments of this size
    overlap : int, optional
        The number of overlapping frames between adjacent segments
    nsources : int, optional
        In the case of source separation, this is the number of sources to separate 
    nprocs : int, optional
        The number of CPU to use when loading the data in parallel: the more, the faster
    log_in : bool, optional
        Apply log10 to the input 
    log_out : bool, optional
        Apply log10 to the output
    mult_factor_in : float, optional
        Multiply the input with factor
    mult_factor_out : float, optional
        Multiply the output with factor
    scratch_path : string, optional
        To speed up batch fetching, the resulting batches are written to a scratch path (e.g. SSD disk)
    
    """
    def __init__(self, path_transform_in=None, path_transform_out=None, sampleRate=44100, exclude_list=[], nsamples=0,
        batch_size=64, batch_memory=8000, time_context=-1, overlap=5, weighted=False, tensortype=float, scratch_path=None, 
        log_in=False, log_out=False, mult_factor_in=1., mult_factor_out=1.,nsources=2,pitched=False,save_mask=False,pitch_norm=127,nprocs=2):
    
        self.batch_size = batch_size
        self.nsources = nsources
        self.tensortype = tensortype
        if path_transform_in is not None:
            if not isinstance(path_transform_in, (list, tuple)):
                self.path_transform_in = [path_transform_in]
            else:
                self.path_transform_in = path_transform_in
        else:
            self.path_transform_in = None
        if path_transform_out is not None:
            if not isinstance(path_transform_out, (list, tuple)):
                self.path_transform_out = [path_transform_out]
            else:
                self.path_transform_out = path_transform_out
        else:
            self.path_transform_out = None

        self.weighted = weighted 
        self.batch_memory = batch_memory #number of batches to keep in memory
        self.mult_factor_in = mult_factor_in
        self.mult_factor_out = mult_factor_out
        self.log_in = log_in
        self.log_out = log_out
        #self.iteration_size = int(self.batch_memory / self.batch_size)
        self.sampleRate = sampleRate
        self.pitched = pitched
        self.save_mask = save_mask 
        self.pitch_norm = pitch_norm   
        self.nprocs = nprocs  
        self.exclude_list = exclude_list
        self.nsamples = nsamples

        if time_context != -1:
            self.time_context = int(time_context)
        else: 
            self.time_context = 10 #frames 
        if overlap > self.time_context:
            self.overlap = int(0.5 * self.time_context)
        else:
            self.overlap = overlap

        #checks if the path where the .data files are store exists
        if self.path_transform_in is not None:
            if self.path_transform_out is None:
                self.path_transform_out = self.path_transform_in
            #if path exists, this routine reads total number of batches, initializes batches and variables
            self.updatePath(self.path_transform_in,self.path_transform_out)
        #to save a created batch to maybe a ssd fast drive 
        if scratch_path is not None:
            if not os.path.exists(scratch_path):
                os.makedirs(scratch_path)
            self.scratch_path = scratch_path
        else: 
            self.scratch_path = None
        self._index = 0

    def iterate(self):
        """
        This is called whenever you need to return a batch, e.g. the callable that generates numpy arrays
        """
        if self._index>=self.iteration_size or self.findex>=self.total_points:
            self._index = 0
            self.findex = 0
            self.nindex = 1
            self.idxbegin = 0
            self.idxend = 0
            self.foffset = 0 
            self.mini_index = 0
            self.scratch_index = 0

        #checks if there are enough batches in memory, if not loads more batches from the disk
        if self.batch_memory<self.iteration_size and self.mini_index>=self.batch_memory:
            self.loadBatches()  
        #logging.info('loaded batch %s from %s',str(self._index+1),str(self.iteration_size))
        self._index = self._index + 1
        idx0=self.mini_index*self.batch_size
        idx1=(self.mini_index+1)*self.batch_size
        self.mini_index = self.mini_index + 1
        return self.returns(idx0,idx1)

    def returns(self, idx0, idx1): 
        """
        This is a wrapper used by the callable \"iterate\" to return a batch between the indices idx0,idx1
        """       
        if self.pitched:   
            if self.weighted:
                return [self.batch_inputs[idx0:idx1],self.batch_outputs[idx0:idx1],self.batch_pitches[idx0:idx1],self.batch_weights[idx0:idx1]] 
            else:
                return [self.batch_inputs[idx0:idx1],self.batch_outputs[idx0:idx1],self.batch_pitches[idx0:idx1]] 
        elif self.save_mask: 
            if self.weighted:
                return [self.batch_inputs[idx0:idx1],self.batch_outputs[idx0:idx1],self.batch_masks[idx0:idx1],self.batch_weights[idx0:idx1]]
            else:
                return [self.batch_inputs[idx0:idx1],self.batch_outputs[idx0:idx1],self.batch_masks[idx0:idx1]]
        else:
            return [self.batch_inputs[idx0:idx1],self.batch_outputs[idx0:idx1]]

    def loadBatches(self):
        """
        Loads more batches from the disk, if the batches from the memory are exhausted
        """
        if hasattr(self, 'scratch_path') and self.scratch_path is not None:
            batch_file = os.path.join(self.scratch_path,'batch'+str(self.scratch_index))
            if os.path.exists(batch_file+'_inputs.data') and os.path.exists(batch_file+'_outputs.data'):
                self.batch_inputs = self.loadTensor(batch_file+'_inputs.data')
                self.batch_outputs = self.loadTensor(batch_file+'_outputs.data')
                if self.pitched:    
                    self.batch_pitches = self.loadTensor(batch_file+'_pitches.data')
                elif self.save_mask: 
                    self.batch_masks = self.loadTensor(batch_file+'_masks.data')
                if self.weighted:
                    self.batch_weights = self.loadTensor(batch_file+'_weights.data')
                self.shuffleBatches()   
            else: 
                #generate and save
                self.genBatches()
                self.saveBatches(batch_file)
                self.scratch_index = self.scratch_index + 1
        else:
            self.genBatches()
            self.scratch_index = self.scratch_index + 1
        #logging.info('read %s more batches from hdd',str(self.batch_memory))
        self.mini_index = 0

    def genBatches(self):
        """
        This function is called by \"loadBatches\" to generate batches from the disk
        """
        #getNextIndex sets the time indices corresponding to the next batch 
        self.getNextIndex()

        #no multiprocessing
        if self.nindex==self.findex:
            x = self.loadFile(self.findex, idxbegin=self.idxbegin, idxend=self.idxend)
            self.batch_inputs[0:self.idxend-self.idxbegin] = x[0]
            self.batch_outputs[0:self.idxend-self.idxbegin] = x[1]
            if self.pitched:    
                self.batch_pitches[0:self.idxend-self.idxbegin] = x[2]
            elif self.save_mask: 
                self.batch_masks[0:self.idxend-self.idxbegin] = x[2]     
            if self.weighted:
                self.batch_weights[0:self.idxend-self.idxbegin] = x[3]
            x=None
        else:
            x = self.loadFile(self.findex, idxbegin=self.idxbegin)
            self.batch_inputs[0:self.num_points[self.findex+1]-self.num_points[self.findex]-self.idxbegin] = x[0]
            self.batch_outputs[0:self.num_points[self.findex+1]-self.num_points[self.findex]-self.idxbegin] = x[1]
            if self.pitched:   
                self.batch_pitches[0:self.num_points[self.findex+1]-self.num_points[self.findex]-self.idxbegin] = x[2] 
            elif self.save_mask: 
                self.batch_masks[0:self.num_points[self.findex+1]-self.num_points[self.findex]-self.idxbegin] = x[2]
            if self.weighted:
                self.batch_weights[0:self.num_points[self.findex+1]-self.num_points[self.findex]-self.idxbegin] = x[3]
            x=None

        #this is where multiprocessing happens
        if (self.nindex-self.findex) > 2:
            i = self.findex + 1
            xall = parmap(self.loadFile, list(range(self.findex+1,self.nindex)),nprocs=self.nprocs)
            for i in range(self.findex+1,self.nindex):
                #x = self.loadFile(i)
                x=xall[i-self.findex-1]
                idx0=self.num_points[i]-self.foffset
                idx1=self.num_points[i+1]-self.foffset
                self.batch_inputs[idx0:idx1] = x[0]
                self.batch_outputs[idx0:idx1] = x[1]
                if self.pitched: 
                    self.batch_pitches[idx0:idx1] = x[2]   
                elif self.save_mask: 
                    self.batch_masks[idx0:idx1] = x[2]
                if self.weighted:
                    self.batch_weights[idx0:idx1] = x[3]
                x=None
            xall=None
        
        #no multiprocessing
        if (self.nindex-self.findex) > 0:
            x = self.loadFile(self.nindex,idxend=self.idxend)
            idx0=self.num_points[self.nindex] - self.foffset
            idx1=self.num_points[self.nindex] + self.idxend - self.foffset
            self.batch_inputs[idx0:idx1] = x[0]
            self.batch_outputs[idx0:idx1] = x[1]
            if self.pitched:    
                self.batch_pitches[idx0:idx1] = x[2]
            elif self.save_mask: 
                self.batch_masks[idx0:idx1] = x[2]
            if self.weighted:
                self.batch_weights[idx0:idx1] = x[3]
            x=None
               
        #shuffle batches 
        self.shuffleBatches()

        if self.idxend == (self.num_points[self.nindex+1]-self.num_points[self.nindex]):
            self.findex = self.nindex + 1
            self.idxbegin = 0
            self.foffset = self.num_points[self.findex] 
        else:
            self.findex = self.nindex 
            self.idxbegin = self.idxend 
            self.foffset = self.num_points[self.findex] + self.idxbegin 
        self.idxend = -1

    def getNextIndex(self):
        """
        Returns how many batches/sequences to load from each .data file
        """
        target_value = (self.scratch_index+1)*(self.batch_memory*self.batch_size)
        idx_target = np.searchsorted(self.num_points,target_value, side='right')
        if target_value>self.num_points[-1] or idx_target>=len(self.num_points):
            idx_target = idx_target - 2
            target_value = self.num_points[idx_target]
            self.idxend = self.num_points[idx_target] - self.num_points[idx_target-1]
            self.nindex = idx_target
        else:
            while target_value<=self.num_points[idx_target]:
                idx_target = idx_target - 1
            self.idxend = target_value - self.num_points[idx_target] 
            self.nindex = idx_target

    def loadPitch(self,id):
        return self.loadTensor(os.path.join(self.path_transform_in[self.dirid[id]],self.file_list[id].replace('_m_','_p_')))

    def loadInputOutput(self,id):
        """
        Loads the .data fft file from the hard drive
        """
        allmixinput = self.loadTensor(os.path.join(self.path_transform_in[self.dirid[id]],self.file_list[id]))
        if self.path_transform_in==self.path_transform_out:
            allmixoutput = allmixinput[1:]
        else:
            allmixoutput = self.loadTensor(os.path.join(self.path_transform_out[self.dirid[id]],self.file_list[id]))
            if self.nsources>1:
                allmixoutput = allmixoutput[1:]
        allmixinput = np.expand_dims(allmixinput[0], axis=0)
        return allmixinput,allmixoutput 
 
    def loadFile(self,id,idxbegin=None,idxend=None):
        """
        reads a .data file and splits into batches
        """
        if self.path_transform_in is not None and self.path_transform_out is not None:
            if idxbegin is None: 
                idxbegin = 0 
            if idxend is None or idxend==-1:
                idxend = self.num_points[id+1] - self.num_points[id] 
            
            if self.pitched:  
                if self.weighted:
                    inputs,outputs,pitches,weights = self.initOutput(idxend - idxbegin)
                else:
                    inputs,outputs,pitches = self.initOutput(idxend - idxbegin)
            elif self.save_mask: 
                if self.weighted:
                    inputs,outputs,masks,weights = self.initOutput(idxend - idxbegin)
                else:
                    inputs,outputs,masks = self.initOutput(idxend - idxbegin)
            else:
                inputs,outputs = self.initOutput(idxend - idxbegin)         
       
            #loads the .data fft file from the hard drive
            allmixinput,allmixoutput = self.loadInputOutput(id)

            if self.pitched or self.save_mask:
                allpitch = self.loadPitch(id)

            #apply a scaled log10(1+value) function to make sure larger values are eliminated 
            if self.log_in==True:
                allmixinput = self.mult_factor_in*np.log10(1.0+allmixinput)
            else:
                allmixinput = self.mult_factor_in*allmixinput
            if self.log_out==True:
                allmixoutput = self.mult_factor_out*np.log10(1.0+allmixoutput)
            else:
                allmixoutput = self.mult_factor_out*allmixoutput

            i = 0
            start = 0
                
            if self.time_context > allmixinput.shape[1]:
                inputs[0,:allmixinput.shape[1],:] = allmixinput[0]
                outputs[0, :allmixoutput.shape[1], :allmixoutput.shape[-1]] = allmixoutput[0]
                if self.weighted:
                    weights[0, :allmixoutput.shape[1], :] = self.getWeighted(allmixinput[0],allpitch)  
                
                for j in range(1,self.nsources):
                    outputs[0, :allmixoutput.shape[1], j*allmixoutput.shape[-1]:(j+1)*allmixoutput.shape[-1]] = allmixoutput[j]
                    #import pdb;pdb.set_trace()
                    # if self.weighted == True:
                    #     weights[0, :allmixoutput.shape[1], j*allmixoutput.shape[-1]:(j+1)*allmixoutput.shape[-1]] = self.getWeighted(allmixinput[0],allpitch)   

                
                if self.pitched:
                    for j in range(self.ninst): #for all the inputed pitched instrument get their mask for the corresponding pitch contours
                        pitches[0, :allmixinput.shape[1], j*self.npitches:(j+1)*self.npitches] = allpitch[j] / float(self.pitch_norm) 
                elif self.save_mask:
                    masks[0, :allmixinput.shape[1],:] = self.filterSpec(allmixinput[0],allpitch)   
            else:
                while (start + self.time_context) < allmixinput.shape[1]:
                    if i>=idxbegin and i<idxend:
                        allminput = allmixinput[:,start:start+self.time_context,:] #truncate on time axis so it would match the actual context
                        allmoutput = allmixoutput[:,start:start+self.time_context,:]
                       
                        inputs[i-idxbegin] = allminput[0]
                        #import pdb;pdb.set_trace()
                        outputs[i-idxbegin, :, :allmoutput.shape[-1]] = allmoutput[0]
                        if self.weighted:
                            weights[i-idxbegin, :, :] = self.getWeighted(allminput[0],allpitch[:,:,start:start+self.time_context])   
                       
                        for j in range(1,self.nsources):
                            outputs[i-idxbegin,:, j*allmoutput.shape[-1]:(j+1)*allmoutput.shape[-1]] = allmoutput[j,:,:]
                         
                        if self.pitched:
                            for j in range(self.ninst): #for all the inputed instrument pitches
                                pitches[i-idxbegin, :allmixinput.shape[1], j*self.npitches:(j+1)*self.npitches] = allpitch[j,:,start:start+self.time_context].T / float(self.pitch_norm)
                        elif self.save_mask: 
                            masks[i-idxbegin, :allmixinput.shape[1], :] = self.filterSpec(allminput[0],allpitch[:,:,start:start+self.time_context])   
                       
                    i = i + 1
                    start = start - self.overlap + self.time_context
                    #clear memory
                    allminput=None
                    allmoutput=None
      
            #clear memory
            allmixinput=None
            allmixoutput=None
            i=None
            j=None
            start=None
            if self.pitched or self.save_mask:
                allpitch=None
            
            if self.pitched:  
                if self.weighted:
                    return [inputs, outputs, pitches, weights]
                else:
                    return [inputs, outputs, pitches]
            elif self.save_mask: 
                if self.weighted:
                    return [inputs, outputs, masks, weights]
                else: 
                    return [inputs, outputs, masks]                    
            else:
                return [inputs, outputs]


    def shuffleBatches(self):
        """
        Shuffle batches 
        """
        idxstop = self.num_points[self.nindex] + self.idxend - self.num_points[self.findex] - self.idxbegin
        if idxstop>=self.batch_inputs.shape[0]:
            idxstop=self.batch_inputs.shape[0]
        idxrand = np.random.permutation(idxstop)
        self.batch_inputs[:idxstop] = self.batch_inputs[idxrand] 
        self.batch_outputs[:idxstop] = self.batch_outputs[idxrand] 
        if self.pitched: 
            self.batch_pitches[:idxstop] = self.batch_pitches[idxrand]    
        elif self.save_mask: 
            self.batch_masks[:idxstop] = self.batch_masks[idxrand]                             
        if self.weighted:
            self.batch_weights[:idxstop] = self.batch_weights[idxrand] 


    def initOutput(self,size):
        """
        Allocate memory for read data, where \"size\" is the number of examples of size \"time_context\"
        """
        inp = np.zeros((size, self.time_context, self.input_size), dtype=self.tensortype)
        out = np.zeros((size, self.time_context, self.output_size), dtype=self.tensortype)
        if self.pitched:  
            ptc = np.zeros((size, self.time_context, self.npitches*self.ninst), dtype=self.tensortype)  
        elif self.save_mask:  
            msk = np.zeros((size, self.time_context, self.input_size*self.ninst), dtype=self.tensortype)
       
        if self.weighted:
            weight = np.zeros((size, self.time_context, self.output_size), dtype=self.tensortype)
        
     
        if self.pitched:  
            if self.weighted:
                return inp,out,ptc,weight
            else:
                return inp,out,ptc
        elif self.save_mask:  
            if self.weighted:
                return inp,out,msk,weight
            else:
                return inp,out,msk
        else:
            return inp,out


    def saveBatches(self,batch_file):
        """
        If set, save the batches to the \"scratch_path\"
        """
        self.saveTensor(self.batch_inputs, batch_file+'_inputs.data')
        self.saveTensor(self.batch_outputs, batch_file+'_outputs.data')
        if self.pitched:    
            self.saveTensor(self.batch_outputs, batch_file+'_pitches.data')
        if self.save_mask: 
            self.saveTensor(self.batch_outputs, batch_file+'_masks.data')    
        if self.weighted:
            self.saveTensor(self.batch_weights, batch_file+'_weights.data')

    def getFeatureSize(self):
        """
        Returns the feature size of the input and of the output to the neural network 
        """
        if self.path_transform_in is not None and self.path_transform_out is not None:
            for i in range(len(self.file_list)):
                if os.path.isfile(os.path.join(self.path_transform_in[self.dirid[i]],self.file_list[i])): 

                    allmix = self.loadTensor(os.path.join(self.path_transform_in[self.dirid[i]],self.file_list[i]))
                    if self.pitched or self.save_mask:
                        pitch = self.loadTensor(os.path.join(self.path_transform_in[self.dirid[i]],self.file_list[i].replace('_m_','_p_')))
                        self.ninst = pitch.shape[0] #number of pitched instruments (inst for which pitch is defined)
                        self.npitches = pitch.shape[1] #numer of pitch contours for each pitched instrument
                        pitch=None
                    if self.path_transform_in==self.path_transform_out:
                        return allmix.shape[-1], self.nsources * allmix.shape[-1]
                    else:
                        allmixoutput = self.loadTensor(os.path.join(self.path_transform_out[self.dirid[i]],self.file_list[0]))
                        return allmix.shape[-1], self.nsources * allmixoutput.shape[-1]

    def getMean(self):
        if self.path_transform_in is not None:
            return np.mean(self.batch_inputs)

    def getStd(self):
        if self.path_transform_in is not None:  
            return np.std(self.batch_inputs)

    def getMax(self):
        if self.path_transform_in is not None:
            return self.batch_inputs.max()

    def getMin(self):
        if self.path_transform_in is not None:
            return self.batch_inputs.min()
            
    def getNum(self,id):
        """
        For a single .data file computes the number of examples of size \"time_context\" that can be created
        """
        shape = self.get_shape(os.path.join(self.path_transform_in[self.dirid[id]],self.file_list[id].replace('.data','.shape')))
        time_axis = shape[1]
        return np.maximum(1,int(np.floor((time_axis + (np.floor(float(time_axis)/self.time_context) * self.overlap))  / self.time_context)))


    def updatePath(self, path_in, path_out=None):
        """
        Read the list of .data files in path, compute how many examples we can create from each file, and initialize the output variables
        """
        self.path_transform_in = path_in
        if path_out is None:
            self.path_transform_out = self.path_transform_in
        else:
            self.path_transform_out = path_out

      
        #we read the file_list from the path_transform_in directory
        self.file_list = [f for k in range(len(self.path_transform_in)) for f in os.listdir(self.path_transform_in[k]) \
            if f.endswith('_m_.data') and os.path.isfile(os.path.join(self.path_transform_out[k],f)) and\
            f.split('_',1)[0] not in self.exclude_list]

        self.dirid = [k for k in range(len(self.path_transform_in)) for f in os.listdir(self.path_transform_in[k]) \
            if f.endswith('_m_.data') and os.path.isfile(os.path.join(self.path_transform_out[k],f)) and\
            f.split('_',1)[0] not in self.exclude_list]

        if self.nsamples>2 and self.nsamples < len(self.file_list):
            ids = np.squeeze(np.random.choice(len(self.file_list), size=self.nsamples, replace=False))
            self.file_list = list([self.file_list[iids] for iids in ids])
            self.dirid = list([self.dirid[iids] for iids in ids])
            ids = None

        
        self.total_files = len(self.file_list)
        if self.total_files<1:
            raise Exception('Could not find any file in the input directory! Files must end with _m_.data')
        logging.info("found %s files",str(self.total_files)) 
        self.num_points = np.cumsum(np.array([0]+[self.getNum(i) for i in range(self.total_files)], dtype=int))
        self.total_points = self.num_points[-1]
        #print self.num_points
        self.input_size,self.output_size = self.getFeatureSize()
        self.initBatches()

    def updateBatch(self, batch_size):
        """
        This function is called when batch_size changes
        """
        self.batch_size = batch_size
        self.initBatches()


    def initBatches(self):
        """
        Allocates memory for the output
        """
        self.batch_size = np.minimum(self.batch_size,self.num_points[-1])
        self.iteration_size = int(self.total_points / self.batch_size) 
        self.batch_memory = np.minimum(self.batch_memory,self.iteration_size)
        logging.info("iteration size %s",str(self.iteration_size)) 
        self._index = 0
        self.findex = 0
        self.nindex = 1
        self.idxbegin = 0
        self.idxend = 0
        self.foffset = 0 
        self.mini_index = 0
        self.scratch_index = 0
        self.batch_inputs = np.zeros((self.batch_memory*self.batch_size,self.time_context,self.input_size), dtype=self.tensortype)
        self.batch_outputs = np.zeros((self.batch_memory*self.batch_size,self.time_context,self.output_size), dtype=self.tensortype)
        if self.pitched:  
            self.batch_pitches = np.zeros((self.batch_memory*self.batch_size,self.time_context,self.npitches*self.ninst), dtype=self.tensortype)
         
        if self.save_mask: 
            self.batch_masks = np.zeros((self.batch_memory*self.batch_size,self.time_context,self.input_size*self.ninst), dtype=self.tensortype)
        
        if self.weighted == True:
            self.batch_weights = np.zeros((self.batch_memory*self.batch_size,self.time_context,self.output_size), dtype=self.tensortype)
        
        self.loadBatches()


    def loadTensor(self, path, name=''):
        """
        Loads a binary .data file
        """
        if os.path.isfile(path):
            f_in = np.fromfile(path)
            shape = self.get_shape(path.replace('.data','.shape'))
            f_in = f_in.reshape(shape)    
            
            return f_in
        else:    
            logging.info('File does not exist: %s'+path)
            return -1

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

    def saveTensor(self, t, out_path):
        """
        Saves a numpy array as a binary file
        """
        t.tofile(out_path)
        #save shapes
        self.shape = t.shape
        self.save_shape(out_path.replace('.data','.shape'),t.shape)

    def save_shape(self,shape_file,shape):
        """
        Saves the shape of a numpy array
        """
        with open(shape_file, 'w') as fout:
            fout.write(u'#'+'\t'.join(str(e) for e in shape)+'\n')

    def __len__(self):
        return self.iteration_size

    def __call__(self):
        return self.iterate()

    def __iter__(self):
        return self.iterate()

    def next(self):
        return self.iterate()

    def batches(self):
        return self.iterate()



class LargeDatasetPitch1(LargeDataset):
    """
    The child class of LargeDataset for loading multi-pitch contours and returning a mask with the same shape
    as the spectrogram, which has 1 in the bins corresponding to the harmonic partials.

    The pitch needs to be given in a float 3D array in MIDI (see parent class)

    Additional parameters
    ----------
    nharmonics : int, optional
        The number of harmonic partials to include in the mask
    interval : int, optional
        The allowed interval in cents above and below each partial
    tuning_freq : float, optional
        The tuning frequency necessary to pass from MIDI to frequency
    weighted : bool, optional
        If this is true then the mask is more complex, returning gaussians centered in the harmonic partials.

    """
    def __init__(self, path_transform_in=None, path_transform_out=None, sampleRate=44100, exclude_list=[],nsamples=0,
        batch_size=64, batch_memory=1000, time_context=-1, overlap=5, weighted=False, tensortype=float, scratch_path=None, 
        log_in=False, log_out=False, mult_factor_in=1., mult_factor_out=1., pitched=True,save_mask=False,pitch_norm=120.,nsources=2,
        fmin=25, fmax=18000, ttype='fft', iscale = 'lin', patch_size=10, nharmonics=20, interval=30, tuning_freq=440,nprocs=2): 
        self.fmin=fmin
        self.fmax=fmax
        self.iscale=iscale
        self.patch_size = patch_size
        self.nharmonics = nharmonics
        self.interval = interval
        self.tuning_freq = tuning_freq
        self.ttype = ttype
        #self.pitched = True
        self.save_mask = save_mask
        self.pitch_norm = pitch_norm  
        self.weighted = weighted      
        self.nsamples = nsamples
        super(LargeDatasetPitch1, self).__init__(path_transform_in=path_transform_in, path_transform_out=path_transform_out, sampleRate=sampleRate, exclude_list=exclude_list, nsamples=nsamples,
            batch_size=batch_size, batch_memory=batch_memory, time_context=time_context, overlap=overlap, weighted=weighted, tensortype=tensortype, scratch_path=scratch_path, nsources=nsources,
            log_in=log_in, log_out=log_out, mult_factor_in=mult_factor_in, mult_factor_out=mult_factor_out,pitched=pitched,save_mask=save_mask,pitch_norm=pitch_norm,nprocs=nprocs)    

    def filterSpec(self,mag,pitches):
        """
        Creates the mask with the shape of \"mag\" and the pitches in \"pitches\" 
        """
        filtered = np.zeros((self.ninst,mag.shape[0],mag.shape[1]))
        for j in range(self.ninst): #for all the inputed instrument pitches
            for p in range(len(pitches[j])): #for each pitch contour
                for t in range(len(pitches[j,p])):
                    if pitches[j,p,t] > 0:
                        slices_y = util.slicefft_slices(pitches[j,p,t],size=(mag.shape[-1]-1)*2,interval=self.interval,tuning_freq=self.tuning_freq,nharmonics=self.nharmonics,fmin=self.fmin,fmax=self.fmax,iscale=self.iscale,sampleRate=self.sampleRate)
                        for k in range(len(slices_y)):
                            filtered[j,t,slices_y[k]] = 1.
                        slices_y = None
        filtered /= np.expand_dims(np.maximum(1e-18,filtered.max(axis=2)),axis=2)
        mask = np.zeros((mag.shape[0],self.ninst*mag.shape[1]))
        for j in range(self.ninst): #for all the inputed instrument pitches
            mask[:,j*mag.shape[1]:(j+1)*mag.shape[1]] = filtered[j,:,:]
        filtered = None

        j=None
        p=None
        t=None
        k=None
        return mask

    def getWeighted(self,mag,pitches):
        """
        Creates the mask with the shape of \"mag\" and the pitches in \"pitches\" 
        and having gaussians centered in the frequency bands
        """
        filtered = np.zeros((self.ninst,mag.shape[0],mag.shape[1]))
        for j in range(self.ninst): #for all the inputed instrument pitches
            for p in range(len(pitches[j])): #for each pitch contour
                for t in range(len(pitches[j,p])):
                    if pitches[j,p,t] > 0:
                        slices_y = util.slicefft_slices(pitches[j,p,t],size=(mag.shape[-1]-1)*2,interval=self.interval,tuning_freq=self.tuning_freq,nharmonics=self.nharmonics,fmin=self.fmin,fmax=self.fmax,iscale=self.iscale,sampleRate=self.sampleRate)
                        gss = [util.gaussian(np.linspace(-1,1,slices_y[k].stop-slices_y[k].start), 1/(slices_y[k].stop-slices_y[k].start), (slices_y[k].stop-slices_y[k].start)) for k in range(len(slices_y))]
                        for k in range(len(slices_y)):
                            filtered[j,t,slices_y[k]] = filtered[j,t,slices_y[k]] + (gss[k]-min(gss[k]))/(max(gss[k])-min(gss[k]))
                        gss = None
                        slices_y = None
        filtered /= np.expand_dims(np.maximum(1e-18,filtered.max(axis=2)),axis=2)
        mask = np.zeros((mag.shape[0],self.ninst*mag.shape[1]))
        for j in range(self.ninst): #for all the inputed instrument pitches
            mask[:,j*mag.shape[1]:(j+1)*mag.shape[1]] = filtered[j,:,:]
        filtered = None

        j=None
        p=None
        t=None
        k=None
        return mask


class LargeDatasetPitch2(LargeDataset):
    """
    The child class of LargeDataset for loading multi-pitch contours and returning a mask with the same shape
    as the spectrogram, which has 1 in the bins corresponding to the harmonic partials
    One has the option to initialize each band corresponding to a harmonic partial to a number which 
    corresponds to the weight of that harmonic partial (the weights are previously learned)

    Additional parameters
    ----------
    nharmonics : int, optional
        The number of harmonic partials to include in the mask
    interval : int, optional
        The allowed interval in cents above and below each partial
    tuning_freq : float, optional
        The tuning frequency necessary to pass from MIDI to frequency
    weighted : bool, optional
        If this is true then the mask is more complex, returning gaussians centered in the harmonic partials.
        
    """
    def __init__(self, path_transform_in=None, path_transform_out=None, sampleRate=44100, exclude_list=[],nsamples=0,
        batch_size=64, batch_memory=1000, time_context=-1, overlap=5, weighted=False, tensortype=float, scratch_path=None, timbre_model_path=None,
        log_in=False, log_out=False, mult_factor_in=1., mult_factor_out=1., pitched=True,save_mask=False,pitch_norm=120.,nsources=2,
        fmin=25, fmax=18000, ttype='fft', iscale = 'lin', patch_size=10, nharmonics=20, interval=30, tuning_freq=440,nprocs=2): 
        self.fmin=fmin
        self.fmax=fmax
        self.iscale=iscale
        self.patch_size = patch_size
        self.nharmonics = nharmonics
        self.interval = interval
        self.tuning_freq = tuning_freq
        self.ttype = ttype
        #self.pitched = True
        self.save_mask = save_mask
        self.pitch_norm = pitch_norm
        self.timbre_model_path=timbre_model_path
        self.weighted = weighted 
        self.nsamples = nsamples
        #load timbre models or initialize with 1s
        if self.timbre_model_path is not None:
            self.harmonics = util.loadObj(self.timbre_model_path)
        else: 
            self.harmonics = np.ones((nsources,127,self.nharmonics))
        super(LargeDatasetPitch2, self).__init__(path_transform_in=path_transform_in, path_transform_out=path_transform_out, sampleRate=sampleRate, exclude_list=exclude_list, nsamples=nsamples,
            batch_size=batch_size, batch_memory=batch_memory, time_context=time_context, overlap=overlap, weighted=weighted, tensortype=tensortype, scratch_path=scratch_path, nsources=nsources,
            log_in=log_in, log_out=log_out, mult_factor_in=mult_factor_in, mult_factor_out=mult_factor_out,pitched=pitched,save_mask=save_mask,pitch_norm=pitch_norm,nprocs=nprocs)    

    def filterSpec(self,mag,pitches):
        """
        Creates the mask with the shape of \"mag\" and the pitches in \"pitches\" 
        """
        filtered = np.ones((self.ninst,mag.shape[0],mag.shape[1])) * 0.001
        for j in range(self.ninst): #for all the inputed instrument pitches
            for p in range(len(pitches[j])): #for each pitch contour
                for t in range(len(pitches[j,p])):
                    if pitches[j,p,t] > 0:
                        slices_y = util.slicefft_slices(pitches[j,p,t],size=(mag.shape[-1]-1)*2,interval=self.interval,tuning_freq=self.tuning_freq,nharmonics=self.nharmonics,fmin=self.fmin,fmax=self.fmax,iscale=self.iscale,sampleRate=self.sampleRate)
                        for k in range(len(slices_y)):
                            filtered[j,t,slices_y[k]] = filtered[j,t,slices_y[k]] + self.harmonics[j,int(pitches[j,p,t]),k]
                        slices_y = None
        filtered[filtered < 0.001] = 0.001
        mask = np.zeros((mag.shape[0],self.ninst*mag.shape[1]))
        for j in range(self.ninst): #for all the inputed instrument pitches
            mask[:,j*mag.shape[1]:(j+1)*mag.shape[1]] = filtered[j,:,:]
        filtered = None
        j=None
        p=None
        t=None
        k=None
        return filtered

    def getWeighted(self,mag,pitches):
        """
        Creates the mask with the shape of \"mag\" and the pitches in \"pitches\" 
        and having gaussians centered in the frequency bands
        """
        filtered = np.zeros((self.ninst,mag.shape[0],mag.shape[1]))
        for j in range(self.ninst): #for all the inputed instrument pitches
            for p in range(len(pitches[j])): #for each pitch contour
                for t in range(len(pitches[j,p])):
                    if pitches[j,p,t] > 0:
                        slices_y = util.slicefft_slices(pitches[j,p,t],size=(mag.shape[-1]-1)*2,interval=self.interval,tuning_freq=self.tuning_freq,nharmonics=self.nharmonics,fmin=self.fmin,fmax=self.fmax,iscale=self.iscale,sampleRate=self.sampleRate)
                        gss = [util.gaussian(np.linspace(-1,1,slices_y[k].stop-slices_y[k].start), 1/(slices_y[k].stop-slices_y[k].start), (slices_y[k].stop-slices_y[k].start)) for k in range(len(slices_y))]
                        for k in range(len(slices_y)):
                            filtered[j,t,slices_y[k]] = filtered[j,t,slices_y[k]] + (gss[k]-min(gss[k]))/(max(gss[k])-min(gss[k]))*self.harmonics[j,int(pitches[j,p,t]),k] 
                        gss = None
                        slices_y = None
        filtered /= np.expand_dims(np.maximum(1e-18,filtered.max(axis=2)),axis=2)
        mask = np.zeros((mag.shape[0],self.ninst*mag.shape[1]))
        for j in range(self.ninst): #for all the inputed instrument pitches
            mask[:,j*mag.shape[1]:(j+1)*mag.shape[1]] = filtered[j,:,:]
        filtered = None
        j=None
        p=None
        t=None
        k=None
        return mask


class LargeDatasetPitch3(LargeDataset):
    """
    The child class of LargeDataset for loading multi-pitch contours and returning a mask with the same shape
    as the spectrogram, which has 1 in the bins corresponding to the harmonic partials
    One has the option to initialize each band corresponding to a harmonic partial to a number which 
    corresponds to the weight of that harmonic partial (the weights are previously learned)

    Additional parameters
    ----------
    nharmonics : int, optional
        The number of harmonic partials to include in the mask
    interval : int, optional
        The allowed interval in cents above and below each partial
    tuning_freq : float, optional
        The tuning frequency necessary to pass from MIDI to frequency
    weighted : bool, optional
        If this is true then the mask is more complex, returning gaussians centered in the harmonic partials.
        

    """
    def __init__(self, path_transform_in=None, path_transform_out=None, sampleRate=44100, exclude_list=[],nsamples=0,
        batch_size=64, batch_memory=1000, time_context=-1, overlap=5, weighted=False, tensortype=float, scratch_path=None, timbre_model_path=None,
        log_in=False, log_out=False, mult_factor_in=1., mult_factor_out=1., pitched=True,save_mask=False,pitch_norm=120.,nsources=2,
        fmin=25, fmax=18000, ttype='fft', iscale = 'lin', patch_size=10, nharmonics=20, interval=30, tuning_freq=440,nprocs=2): 
        self.fmin=fmin
        self.fmax=fmax
        self.iscale=iscale
        self.patch_size = patch_size
        self.nharmonics = nharmonics
        self.interval = interval
        self.tuning_freq = tuning_freq
        self.ttype = ttype
        #self.pitched = True
        self.save_mask = save_mask
        self.pitch_norm = pitch_norm
        self.weighted = weighted 
        self.timbre_model_path=timbre_model_path
        self.nsamples = nsamples
        #load timbre models or initialize with 1s
        if self.timbre_model_path is not None:
            self.harmonics = util.loadObj(self.timbre_model_path)
        else: 
            self.harmonics = np.ones((nsources,127,self.nharmonics))
        super(LargeDatasetPitch3, self).__init__(path_transform_in=path_transform_in, path_transform_out=path_transform_out, sampleRate=sampleRate, exclude_list=exclude_list, nsamples=nsamples,
            batch_size=batch_size, batch_memory=batch_memory, time_context=time_context, overlap=overlap, weighted=weighted, tensortype=tensortype, scratch_path=scratch_path, nsources=nsources,
            log_in=log_in, log_out=log_out, mult_factor_in=mult_factor_in, mult_factor_out=mult_factor_out,pitched=pitched,save_mask=save_mask,pitch_norm=pitch_norm,nprocs=nprocs)    

    def filterSpec(self,mag,pitches):
        """
        Creates the mask with the shape of \"mag\" and the pitches in \"pitches\" 
        """
        filtered = np.ones((self.ninst,mag.shape[0],mag.shape[1])) * 0.001
        for j in range(self.ninst): #for all the inputed instrument pitches
            for p in range(len(pitches[j])): #for each pitch contour
                for t in range(len(pitches[j,p])):
                    if pitches[j,p,t] > 0:
                        slices_y = util.slicefft_slices(pitches[j,p,t],size=(mag.shape[-1]-1)*2,interval=self.interval,tuning_freq=self.tuning_freq,nharmonics=self.nharmonics,fmin=self.fmin,fmax=self.fmax,iscale=self.iscale,sampleRate=self.sampleRate)
                        for k in range(len(slices_y)):
                            filtered[j,t,slices_y[k]] = filtered[j,t,slices_y[k]] + self.harmonics[j,int(pitches[j,p,t]),k]
                        slices_y = None
        filtered[filtered < 0.001] = 0.001
        
        mask = np.zeros((mag.shape[0],self.ninst*mag.shape[1]))
        for j in range(self.ninst): #for all the inputed instrument pitches
            mask[:,j*mag.shape[1]:(j+1)*mag.shape[1]] = filtered[j,:,:] / np.sum(filtered,axis=0)
        filtered = None

        j=None
        p=None
        t=None
        k=None
        return mask


    def getWeighted(self,mag,pitches):
        """
        Creates the mask with the shape of \"mag\" and the pitches in \"pitches\" 
        and having gaussians centered in the frequency bands
        """
        filtered = np.zeros((self.ninst,mag.shape[0],mag.shape[1]))
        for j in range(self.ninst): #for all the inputed instrument pitches
            for p in range(len(pitches[j])): #for each pitch contour
                for t in range(len(pitches[j,p])):
                    if pitches[j,p,t] > 0:
                        slices_y = util.slicefft_slices(pitches[j,p,t],size=(mag.shape[-1]-1)*2,interval=self.interval,tuning_freq=self.tuning_freq,nharmonics=self.nharmonics,fmin=self.fmin,fmax=self.fmax,iscale=self.iscale,sampleRate=self.sampleRate)
                        gss = [util.gaussian(np.linspace(-1,1,slices_y[k].stop-slices_y[k].start), 1/(slices_y[k].stop-slices_y[k].start), (slices_y[k].stop-slices_y[k].start)) for k in range(len(slices_y))]
                        for k in range(len(slices_y)):
                            filtered[j,t,slices_y[k]] = filtered[j,t,slices_y[k]] + (gss[k]-min(gss[k]))/(max(gss[k])-min(gss[k]))*self.harmonics[j,int(pitches[j,p,t]),k] 
                        gss = None
                        slices_y = None
        filtered /= np.expand_dims(np.maximum(1e-18,filtered.max(axis=2)),axis=2)
        
        mask = np.zeros((mag.shape[0],self.ninst*mag.shape[1]))
        for j in range(self.ninst): #for all the inputed instrument pitches
            mask[:,j*mag.shape[1]:(j+1)*mag.shape[1]] = filtered[j,:,:]
        
        filtered = None
        

        j=None
        p=None
        t=None
        k=None
        return mask




class LargeDatasetClassification(LargeDataset):

    def getFeatureSize(self):
        if self.path_transform_in is not None and self.path_transform_out is not None:
            for i in range(len(self.file_list)):
                if os.path.isfile(os.path.join(self.path_transform_in[self.dirid[i]],self.file_list[i])): 
                    allmix = self.loadTensor(os.path.join(self.path_transform_in[self.dirid[i]],self.file_list[i]))
                    labels = self.loadTensor(os.path.join(self.path_transform_in[self.dirid[i]],self.file_list[i].replace('_m_','_l_')))
                    return allmix.shape[-1], labels.shape[-1]

    #reads a .data fft file and splits into batches
    def loadFile(self,id,idxbegin=None,idxend=None):
        if self.path_transform_in is not None and self.path_transform_out is not None:
            if idxbegin is None: 
                idxbegin = 0 
            if idxend is None or idxend==-1:
                idxend = self.num_points[id+1] - self.num_points[id] 
            if self.weighted == True:
                inputs,outputs,weights = self.initOutput(idxend - idxbegin)
            else:
                inputs,outputs = self.initOutput(idxend - idxbegin)
            #import pdb;pdb.set_trace()          
            i = 0
            start = 0
            #loads the .data fft file from the hard drive
            allmixinput = self.loadTensor(os.path.join(self.path_transform_in[self.dirid[i]],self.file_list[id]))
            allmixoutput = self.loadTensor(os.path.join(self.path_transform_in[self.dirid[i]],self.file_list[id].replace('_m_','_l_')))
           
            allmixinput = np.expand_dims(allmixinput[0], axis=0)
            #allmixinput is not getting set to the input
            #apply a scaled log10(1+value) function to make sure larger values are eliminated 
            if self.log_in==True:
                allmixinput = self.mult_factor_in*np.log10(1.0+allmixinput)
            else:
                allmixinput = self.mult_factor_in*allmixinput
            
            if self.time_context > allmixinput.shape[1]:
                inputs[0,:allmixinput.shape[1],:] = allmixinput[0]
                outputs[0, :allmixoutput.shape[1], :] = allmixoutput[0]
                if self.weighted == True:
                    weights[0, :allmixoutput.shape[1], :] = np.ones_like(allmixoutput[0])
            else:
                while (start + self.time_context) < allmixinput.shape[1]:
                    if i>=idxbegin and i<idxend:
                        allminput = allmixinput[:,start:start+self.time_context,:] #truncate on time axis so it would match the actual context
                        allmoutput = allmixoutput[:,start:start+self.time_context,:]
                       
                        inputs[i-idxbegin] = allminput[0]
                        #import pdb;pdb.set_trace()
                        outputs[i-idxbegin] = allmoutput[0]
                        if self.weighted == True:
                            weights[i-idxbegin] = np.ones_like(allmoutput[0])
                      
                    i = i + 1
                    start = start - self.overlap + self.time_context
            
                    
            if self.weighted == True:
                return [inputs, outputs, weights]
            else:
                return [inputs, outputs]


