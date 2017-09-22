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
        batch_size=64, batch_memory=8000, time_context=-1, overlap=5, tensortype=float, scratch_path=None, extra_features=False, model="", context=5,
        log_in=False, log_out=False, mult_factor_in=1., mult_factor_out=1.,nsources=2,pitched=False,save_mask=False,pitch_norm=127,nprocs=2,jump=0):

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

        self.extra_features = extra_features
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

        self.extra_features = extra_features
        self.model = model
        self.context = context
        if jump<1:
            self.jump = int(np.ceil(float(self.time_context)/(self.time_context - self.overlap)))
        else:
            self.jump = int(jump)

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
            if self.save_mask:
                if self.extra_features:
                    return [self.batch_inputs[idx0:idx1],self.batch_outputs[idx0:idx1],self.batch_pitches[idx0:idx1],self.batch_masks[idx0:idx1],self.batch_features[idx0:idx1]]
                else:
                    return [self.batch_inputs[idx0:idx1],self.batch_outputs[idx0:idx1],self.batch_pitches[idx0:idx1],self.batch_masks[idx0:idx1]]
            else:
                if self.extra_features:
                    return [self.batch_inputs[idx0:idx1],self.batch_outputs[idx0:idx1],self.batch_pitches[idx0:idx1],self.batch_features[idx0:idx1]]
                else:
                    return [self.batch_inputs[idx0:idx1],self.batch_outputs[idx0:idx1],self.batch_pitches[idx0:idx1]]
        else:
            if self.save_mask:
                if self.extra_features:
                    return [self.batch_inputs[idx0:idx1],self.batch_outputs[idx0:idx1],self.batch_masks[idx0:idx1],self.batch_features[idx0:idx1]]
                else:
                    return [self.batch_inputs[idx0:idx1],self.batch_outputs[idx0:idx1],self.batch_masks[idx0:idx1]]
            else:
                if self.extra_features:
                    return [self.batch_inputs[idx0:idx1],self.batch_outputs[idx0:idx1],self.batch_features[idx0:idx1]]
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
                if self.save_mask:
                    self.batch_masks = self.loadTensor(batch_file+'_masks.data')
                if self.extra_features:
                    self.batch_features = self.loadTensor(batch_file+'_features.data')
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
            self.batch_inputs[0:self.idxend-self.idxbegin] = x['inputs']
            self.batch_outputs[0:self.idxend-self.idxbegin] = x['outputs']
            if self.pitched:
                self.batch_pitches[0:self.idxend-self.idxbegin] = x['pitches']
            if self.save_mask:
                self.batch_masks[0:self.idxend-self.idxbegin] = x['masks']
            if self.extra_features:
                self.batch_features[0:self.idxend-self.idxbegin] = x['features']
            x=None
        else:
            x = self.loadFile(self.findex, idxbegin=self.idxbegin)
            self.batch_inputs[0:self.num_points[self.findex+1]-self.num_points[self.findex]-self.idxbegin] = x['inputs']
            self.batch_outputs[0:self.num_points[self.findex+1]-self.num_points[self.findex]-self.idxbegin] = x['outputs']
            if self.pitched:
                self.batch_pitches[0:self.num_points[self.findex+1]-self.num_points[self.findex]-self.idxbegin] = x['pitches']
            if self.save_mask:
                self.batch_masks[0:self.num_points[self.findex+1]-self.num_points[self.findex]-self.idxbegin] = x['masks']
            if self.extra_features:
                self.batch_features[0:self.num_points[self.findex+1]-self.num_points[self.findex]-self.idxbegin] = x['features']
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
                self.batch_inputs[idx0:idx1] = x['inputs']
                self.batch_outputs[idx0:idx1] = x['outputs']
                if self.pitched:
                    self.batch_pitches[idx0:idx1] = x['pitches']
                if self.save_mask:
                    self.batch_masks[idx0:idx1] = x['masks']
                if self.extra_features:
                    self.batch_features[idx0:idx1] = x['features']
                x=None
            xall=None

        #no multiprocessing
        if (self.nindex-self.findex) > 0:
            idx0=self.num_points[self.nindex] - self.foffset
            idx1=self.num_points[self.nindex] + self.idxend - self.foffset
            if idx1>len(self.batch_inputs):
                self.idxend = self.idxend - (idx1-len(self.batch_inputs))
                idx1=len(self.batch_inputs)

            x = self.loadFile(self.nindex,idxend=self.idxend)

            self.batch_inputs[idx0:idx1] = x['inputs']
            self.batch_outputs[idx0:idx1] = x['outputs']
            if self.pitched:
                self.batch_pitches[idx0:idx1] = x['pitches']
            if self.save_mask:
                self.batch_masks[idx0:idx1] = x['masks']
            if self.extra_features:
                self.batch_features[idx0:idx1] = x['features']
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
        if self.pitch_code is None:
            self.pitch_code = 'g'
        return self.loadTensor(os.path.join(self.path_transform_in[self.dirid[id]],self.file_list[id].replace('_m_','_'+self.pitch_code+'_')))

    def load_extra_features(self,id):
        return self.loadTensor(os.path.join(self.path_transform_in[self.dirid[id]],self.file_list[id].replace('_m_','_'+self.model+'_')))

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

            inputs,outputs = self.initOutput(idxend - idxbegin)
            if self.pitched:
                pitches = self.initPitches(idxend - idxbegin)
            else:
                pitches = []
            if self.save_mask:
                masks = self.initMasks(idxend - idxbegin)
            else:
                masks = []
            if self.extra_features:
                features = self.initFeatures(idxend - idxbegin)
            else:
                features = []

            #loads the .data fft file from the hard drive
            allmixinput,allmixoutput = self.loadInputOutput(id)

            if self.pitched or self.save_mask:
                allpitch = self.loadPitch(id)

            if self.extra_features:
                allfeatures = self.load_extra_features(id)

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
                if self.extra_features:
                    features[0,-1, :] = allfeatures

                for j in range(1,self.nsources):
                    outputs[0, :allmixoutput.shape[1], j*allmixoutput.shape[-1]:(j+1)*allmixoutput.shape[-1]] = allmixoutput[j]

                if self.pitched:
                    pitches[0, :allmixinput.shape[1],:] = self.buildPitch(allmixinput[0],allpitch,start,start+self.time_context)
                if self.save_mask:
                    masks[0, :allmixinput.shape[1],:] = self.filterSpec(allmixinput[0],allpitch,start,start+self.time_context)
            else:
                while (start + self.time_context) < allmixinput.shape[1]:
                    if i>=idxbegin and i<idxend:
                        allminput = allmixinput[:,start:start+self.time_context,:] #truncate on time axis so it would match the actual context
                        allmoutput = allmixoutput[:,start:start+self.time_context,:]

                        inputs[i-idxbegin] = allminput[0]
                        outputs[i-idxbegin, :, :allmoutput.shape[-1]] = allmoutput[0]

                        if self.extra_features:
                            j=0
                            while (i-j*self.jump-1)>=0 and j<self.context:
                                features[i-idxbegin,self.context-j-1, :] = allfeatures[i-j*self.jump-1,:]
                                j=j+1

                        for j in range(1,self.nsources):
                            outputs[i-idxbegin,:, j*allmoutput.shape[-1]:(j+1)*allmoutput.shape[-1]] = allmoutput[j,:,:]

                        if self.pitched:
                            pitches[i-idxbegin, :allmixinput.shape[1], :] = self.buildPitch(allminput[0],allpitch,start,start+self.time_context)
                        if self.save_mask:
                            masks[i-idxbegin, :allmixinput.shape[1], :] = self.filterSpec(allminput[0],allpitch,start,start+self.time_context)

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
            if self.extra_features:
                allfeatures = None

            result = {'inputs':inputs, 'outputs':outputs, 'pitches':pitches, 'masks':masks, 'features':features}
            inputs = None
            outputs = None
            pitches = None
            masks = None
            features = None
            return result


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
        if self.save_mask:
            self.batch_masks[:idxstop] = self.batch_masks[idxrand]
        if self.extra_features:
            self.batch_features[:idxstop] = self.batch_features[idxrand]


    def initOutput(self,size):
        """
        Allocate memory for read data, where \"size\" is the number of examples of size \"time_context\"
        """
        inp = np.zeros((size, self.time_context, self.input_size), dtype=self.tensortype)
        out = np.zeros((size, self.time_context, self.output_size), dtype=self.tensortype)

        return inp,out

    def initPitches(self,size):
        ptc = np.zeros((size, self.time_context, self.npitches*self.ninst), dtype=self.tensortype)
        return ptc

    def initMasks(self,size):
        msk = np.zeros((size, self.time_context, self.input_size*self.ninst), dtype=self.tensortype)
        return msk

    def initFeatures(self,size):
        features = np.zeros((size, self.context, self.extra_feat_size), dtype=self.tensortype)
        return features

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
        if self.extra_features:
            self.saveTensor(self.batch_features, batch_file+'_features.data')

    def getFeatureSize(self):
        """
        Returns the feature size of the input and of the output to the neural network
        """
        if self.path_transform_in is not None and self.path_transform_out is not None:
            for i in range(len(self.file_list)):
                if os.path.isfile(os.path.join(self.path_transform_in[self.dirid[i]],self.file_list[i])):
                    # import pdb;pdb.set_trace()
                    allmix = self.loadTensor(os.path.join(self.path_transform_in[self.dirid[i]],self.file_list[i]))
                    if self.pitched or self.save_mask:
                        pitch = self.loadTensor(os.path.join(self.path_transform_in[self.dirid[i]],self.file_list[i].replace('_m_','_'+self.pitch_code+'_')))
                        self.ninst = pitch.shape[0] #number of pitched instruments (inst for which pitch is defined)
                        self.npitches = 127 #midi notes/pitch granularity
                        pitch=None
                    if self.extra_features:
                        feat = self.loadTensor(os.path.join(self.path_transform_in[self.dirid[i]],self.file_list[i].replace('_m_','_'+self.model+'_')))
                        self.extra_feat_size = feat.shape[-1]
                        feat=None
                    if self.path_transform_in==self.path_transform_out:
                        return allmix.shape[-1], self.nsources * allmix.shape[-1]
                    else:
                        allmixoutput = self.loadTensor(os.path.join(self.path_transform_out[self.dirid[i]],self.file_list[0]))
                        return allmix.shape[-1], self.nsources * allmixoutput.shape[-1]

    def getMean(self,inputs=True):
        if self.path_transform_in is not None:
            if inputs:
                return np.mean(self.batch_inputs)
            else:
                return np.mean(self.batch_outputs)

    def getStd(self,inputs=True):
        if self.path_transform_in is not None:
            if inputs:
                return np.std(self.batch_inputs)
            else:
                return np.std(self.batch_outputs)

    def getMax(self,inputs=True):
        if self.path_transform_in is not None:
            if inputs:
                return self.batch_inputs.max()
            else:
                return self.batch_outputs.max()

    def getMin(self,inputs=True):
        if self.path_transform_in is not None:
            if inputs:
                return self.batch_inputs.min()
            else:
                return self.batch_outputs.min()


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

        if self.extra_features == True:
            self.batch_features = np.zeros((self.batch_memory*self.batch_size,self.context,self.extra_feat_size), dtype=self.tensortype)

        self.loadBatches()


    def loadTensor(self, path, name=''):
        """
        Loads a binary .data file
        """
        if os.path.isfile(path):
            f_in = np.fromfile(path)
            shape = self.get_shape(path.replace('.data','.shape'))
            # import pdb;pdb.set_trace()
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


class LargeDatasetMask1(LargeDataset):
    def __init__(self, path_transform_in=None, path_transform_out=None, sampleRate=44100, exclude_list=[],nsamples=0, timbre_model_path=None,
        batch_size=64, batch_memory=1000, time_context=-1, overlap=5, tensortype=float, scratch_path=None, extra_features=False, model="", context=5,
        log_in=False, log_out=False, mult_factor_in=1., mult_factor_out=1., pitched=False,save_mask=True,pitch_norm=127.,nsources=2,
        nharmonics=20, nprocs=2,pitch_code='g',jump=0):

        self.nharmonics = nharmonics
        self.timbre_model_path=timbre_model_path
        self.save_mask = save_mask
        self.pitch_norm = pitch_norm
        self.extra_features = extra_features
        self.nsamples = nsamples
        self.pitch_code = pitch_code
        #load timbre models or initialize with 1s
        if self.timbre_model_path is not None:
            self.harmonics = util.loadObj(self.timbre_model_path)
        super(LargeDatasetMask1, self).__init__(path_transform_in=path_transform_in, path_transform_out=path_transform_out, sampleRate=sampleRate, exclude_list=exclude_list, nsamples=nsamples, extra_features=extra_features, model=model, context=context,
            batch_size=batch_size, batch_memory=batch_memory, time_context=time_context, overlap=overlap, tensortype=tensortype, scratch_path=scratch_path, nsources=nsources,jump=jump,
            log_in=log_in, log_out=log_out, mult_factor_in=mult_factor_in, mult_factor_out=mult_factor_out,pitched=pitched,save_mask=save_mask,pitch_norm=pitch_norm,nprocs=nprocs)

    def filterSpec(self,mag,notes,start,stop):
        if not hasattr(self, 'ninst'):
            self.ninst = notes.shape[0]
        filtered = np.ones((self.ninst,mag.shape[0],mag.shape[1]), dtype=self.tensortype) * 1e-18
        for j in range(self.ninst): #for all the inputed instrument notes
            for p in range(len(notes[j])): #for all notes
                if notes[j,p,2] > 0 and np.maximum(0, np.minimum(notes[j,p,1], stop) - np.maximum(notes[j,p,0], start))>0:
                    begin = int(np.maximum(notes[j,p,0], start))-start
                    end = int(np.minimum(notes[j,p,1], stop))-start
                    slice_x = slice(begin,end,None)
                    slices_y_start = notes[j,p,3::2]
                    slices_y_stop = notes[j,p,4::2]
                    if self.timbre_model_path is None:
                        slices_y = np.hstack(tuple([range(int(slices_y_start[f]),int(slices_y_stop[f])) for f in range(np.minimum(len(slices_y_start),len(slices_y_stop))) if slices_y_stop[f]>0]))
                        filtered[j,slice_x,slices_y] = 1.
                        slices_y = None
                    else:
                        for k in range(len(slices_y_start)):
                            filtered[j,slice_x,slice(int(slices_y_start[k]),int(slices_y_stop[k]),None)] = filtered[j,slice_x,slice(int(slices_y_start[k]),int(slices_y_stop[k]),None)] + self.harmonics[j,int(notes[j,p,2]),k]
                    slice_x = None
        mask = np.zeros((mag.shape[0],self.ninst*mag.shape[1]), dtype=self.tensortype)
        for j in range(self.ninst): #for all the inputed instrument pitches
            mask[:,j*mag.shape[1]:(j+1)*mag.shape[1]] = filtered[j,:,:] / np.max(filtered[j,:,:])
        filtered = None
        # import matplotlib.pyplot as plt
        # for j in range(self.ninst):
        #     plt.subplot(211)
        #     plt.imshow(mag,interpolation='none')
        #     plt.subplot(212)
        #     plt.imshow(mask[:,j*mag.shape[1]:(j+1)*mag.shape[1]],interpolation='none')
        #     plt.show()
        #import pdb;pdb.set_trace()
        j=None
        p=None
        f=None
        return mask

    def buildPitch(self,mag,notes,start,stop):
        if not hasattr(self, 'ninst'):
            self.ninst = notes.shape[0]
        filtered = np.zeros((self.ninst,mag.shape[0],self.npitches), dtype=self.tensortype)
        for j in range(self.ninst): #for all the inputed instrument notes
            for p in range(len(notes[j])): #for all notes
                if notes[j,p,2] > 0 and np.maximum(0, np.minimum(notes[j,p,1], stop) - np.maximum(notes[j,p,0], start))>0:
                    begin = int(np.maximum(notes[j,p,0], start))-start
                    end = int(np.minimum(notes[j,p,1], stop))-start
                    slice_x = slice(begin,end,None)
                    filtered[j,slice_x,int(notes[j,p,2])] = 1.
                    slice_x = None

        mask = np.zeros((mag.shape[0],self.ninst*self.npitches), dtype=self.tensortype)
        for j in range(self.ninst): #for all the inputed instrument pitches
            mask[:,j*self.npitches:(j+1)*self.npitches] = filtered[j,:,:]
        filtered = None

        j=None
        p=None
        return mask


class LargeDatasetMask2(LargeDataset):
    def __init__(self, path_transform_in=None, path_transform_out=None, sampleRate=44100, exclude_list=[],nsamples=0, timbre_model_path=None,
        batch_size=64, batch_memory=1000, time_context=-1, overlap=5, tensortype=float, scratch_path=None, extra_features=False, model="", context=5,
        log_in=False, log_out=False, mult_factor_in=1., mult_factor_out=1., pitched=False,save_mask=True,pitch_norm=127.,nsources=2,
        nharmonics=20, nprocs=2,pitch_code='g',jump=0):

        self.nharmonics = nharmonics
        self.timbre_model_path=timbre_model_path
        self.save_mask = save_mask
        self.pitch_norm = pitch_norm
        self.extra_features = extra_features
        self.nsamples = nsamples
        self.pitch_code = pitch_code
        #load timbre models or initialize with 1s
        if self.timbre_model_path is not None:
            self.harmonics = util.loadObj(self.timbre_model_path)
        super(LargeDatasetMask2, self).__init__(path_transform_in=path_transform_in, path_transform_out=path_transform_out, sampleRate=sampleRate, exclude_list=exclude_list, nsamples=nsamples, extra_features=extra_features, model=model, context=context,
            batch_size=batch_size, batch_memory=batch_memory, time_context=time_context, overlap=overlap, tensortype=tensortype, scratch_path=scratch_path, nsources=nsources,jump=jump,
            log_in=log_in, log_out=log_out, mult_factor_in=mult_factor_in, mult_factor_out=mult_factor_out,pitched=pitched,save_mask=save_mask,pitch_norm=pitch_norm,nprocs=nprocs)

    def filterSpec(self,mag,notes,start,stop):
        if not hasattr(self, 'ninst'):
            self.ninst = notes.shape[0]
        filtered = np.ones((self.ninst,mag.shape[0],mag.shape[1]), dtype=self.tensortype) * 1e-18
        for j in range(self.ninst): #for all the inputed instrument notes
            for p in range(len(notes[j])): #for all notes
                if notes[j,p,2] > 0 and np.maximum(0, np.minimum(notes[j,p,1], stop) - np.maximum(notes[j,p,0], start))>0:
                    begin = int(np.maximum(notes[j,p,0], start))-start
                    end = int(np.minimum(notes[j,p,1], stop))-start
                    slice_x = slice(begin,end,None)
                    slices_y_start = notes[j,p,3::2]
                    slices_y_stop = notes[j,p,4::2]
                    if self.timbre_model_path is None:
                        slices_y = np.hstack(tuple([range(int(slices_y_start[f]),int(slices_y_stop[f])) for f in range(np.minimum(len(slices_y_start),len(slices_y_stop))) if slices_y_stop[f]>0]))
                        filtered[j,slice_x,slices_y] = 1.
                        slices_y = None
                    else:
                        for k in range(len(slices_y_start)):
                            filtered[j,slice_x,slice(int(slices_y_start[k]),int(slices_y_stop[k]),None)] = filtered[j,slice_x,slice(int(slices_y_start[k]),int(slices_y_stop[k]),None)] + self.harmonics[j,int(notes[j,p,2]),k]
                    slice_x = None

        mask = np.zeros((mag.shape[0],self.ninst*mag.shape[1]), dtype=self.tensortype)
        for j in range(self.ninst): #for all the inputed instrument pitches
            mask[:,j*mag.shape[1]:(j+1)*mag.shape[1]] = filtered[j,:,:] / np.sum(filtered,axis=0)
        filtered = None

        # import matplotlib.pyplot as plt
        # for j in range(self.ninst):
        #     plt.subplot(211)
        #     plt.imshow(mag.T,interpolation='none')
        #     plt.ylim([0,200])
        #     plt.subplot(212)
        #     #plt.imshow(mask[:,j*mag.shape[1]:(j+1)*mag.shape[1]],interpolation='none')
        #     plt.imshow(filtered[j,:,:].T,interpolation='none')
        #     plt.ylim([0,200])
        #     plt.show()
        # import pdb;pdb.set_trace()
        j=None
        p=None
        f=None
        return mask

    def buildPitch(self,mag,notes,start,stop):
        if not hasattr(self, 'ninst'):
            self.ninst = notes.shape[0]
        filtered = np.zeros((self.ninst,mag.shape[0],self.npitches))
        for j in range(self.ninst): #for all the inputed instrument notes
            for p in range(len(notes[j])): #for all notes
                if notes[j,p,2] > 0 and np.maximum(0, np.minimum(notes[j,p,1], stop) - np.maximum(notes[j,p,0], start))>0:
                    begin = int(np.maximum(notes[j,p,0], start))-start
                    end = int(np.minimum(notes[j,p,1], stop))-start
                    slice_x = slice(begin,end,None)
                    filtered[j,slice_x,int(notes[j,p,2])] = 1.
                    slice_x = None

        mask = np.zeros((mag.shape[0],self.ninst*self.npitches), dtype=self.tensortype)
        for j in range(self.ninst): #for all the inputed instrument pitches
            mask[:,j*self.npitches:(j+1)*self.npitches] = filtered[j,:,:]
        filtered = None

        j=None
        p=None
        return mask

class LargeDatasetMulti(LargeDataset):
    def __init__(self, prefix_in="in",prefix_out="out", path_transform_in=None, path_transform_out=None, sampleRate=44100, exclude_list=[],nsamples=0, timbre_model_path=None,
        batch_size=64, batch_memory=1000, time_context=-1, overlap=5, tensortype=float, scratch_path=None, extra_features=False, model="", context=5,pitched=False,save_mask=False,
        log_in=False, log_out=False, mult_factor_in=1., mult_factor_out=1.,nsources=2, pitch_norm=127,nprocs=2,jump=0):
        self.prefix_in = prefix_in
        self.prefix_out = prefix_out
        super(LargeDatasetMulti, self).__init__(path_transform_in=path_transform_in, path_transform_out=path_transform_out, sampleRate=sampleRate, exclude_list=exclude_list, nsamples=nsamples, extra_features=extra_features, model=model, context=context,
            batch_size=batch_size, batch_memory=batch_memory, time_context=time_context, overlap=overlap, tensortype=tensortype, scratch_path=scratch_path, nsources=nsources,jump=jump,
            log_in=log_in, log_out=log_out, mult_factor_in=mult_factor_in, mult_factor_out=mult_factor_out,pitched=pitched,save_mask=save_mask,pitch_norm=pitch_norm,nprocs=nprocs)

    def loadPitch(self,id):
        if self.pitch_code is None:
            self.pitch_code = 'g'
        return self.loadTensor(os.path.join(self.path_transform_in[self.dirid[id]],self.file_list[id].replace(self.prefix_in+'_m_','_'+self.pitch_code+'_')))

    def load_extra_features(self,id):
        return self.loadTensor(os.path.join(self.path_transform_in[self.dirid[id]],self.file_list[id].replace(self.prefix_in+'_m_','_'+self.model+'_')))

    def loadInputOutput(self,id):
        """
        Loads the .data fft file from the hard drive
        """
        allmixinput = self.loadTensor(os.path.join(self.path_transform_in[self.dirid[id]],self.file_list[id]))
        allmixoutput = self.loadTensor(os.path.join(self.path_transform_out[self.dirid[id]],self.file_list[id].replace(self.prefix_in+'_m_',self.prefix_out+'_m_')))

        #allmixinput = np.expand_dims(allmixinput[0], axis=0)
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

            inputs,outputs = self.initOutput(idxend - idxbegin)
            if self.pitched:
                pitches = self.initPitches(idxend - idxbegin)
            else:
                pitches = []
            if self.save_mask:
                masks = self.initMasks(idxend - idxbegin)
            else:
                masks = []
            if self.extra_features:
                features = self.initFeatures(idxend - idxbegin)
            else:
                features = []

            #loads the .data fft file from the hard drive
            allmixinput,allmixoutput = self.loadInputOutput(id)

            if self.pitched or self.save_mask:
                allpitch = self.loadPitch(id)

            if self.extra_features:
                allfeatures = self.load_extra_features(id)

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
                inputs[0,:,:allmixinput.shape[1],:] = allmixinput
                outputs[0,:,:allmixoutput.shape[1],:] = allmixoutput
                if self.extra_features:
                    features[0,-1, :] = allfeatures
                if self.pitched:
                    pitches[0, :, :allmixinput.shape[1],:] = self.buildPitch(allmixinput[0],allpitch,start,start+self.time_context)
                if self.save_mask:
                    masks[0, :, :allmixinput.shape[1],:] = self.filterSpec(allmixinput[0],allpitch,start,start+self.time_context)
            else:
                while (start + self.time_context) < allmixinput.shape[1]:
                    if i>=idxbegin and i<idxend:
                        allminput = allmixinput[:,start:start+self.time_context,:] #truncate on time axis so it would match the actual context
                        allmoutput = allmixoutput[:,start:start+self.time_context,:]

                        inputs[i-idxbegin] = allminput
                        outputs[i-idxbegin] = allmoutput

                        if self.extra_features:
                            j=0
                            while (i-j*self.jump-1)>=0 and j<self.context:
                                features[i-idxbegin,self.context-j-1, :] = allfeatures[i-j*self.jump-1,:]
                                j=j+1

                        if self.pitched:
                            pitches[i-idxbegin, :, :allmixinput.shape[1], :] = self.buildPitch(allminput,allpitch,start,start+self.time_context)
                        if self.save_mask:
                            masks[i-idxbegin, :, :allmixinput.shape[1], :] = self.filterSpec(allminput,allpitch,start,start+self.time_context)

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
            if self.extra_features:
                allfeatures = None

            result = {'inputs':inputs, 'outputs':outputs, 'pitches':pitches, 'masks':masks, 'features':features}
            inputs = None
            outputs = None
            pitches = None
            masks = None
            features = None
            return result

    def initOutput(self,size):
        """
        Allocate memory for read data, where \"size\" is the number of examples of size \"time_context\"
        """
        inp = np.zeros((size, self.channels_in, self.time_context, self.input_size), dtype=self.tensortype)
        out = np.zeros((size, self.channels_out, self.time_context, self.output_size), dtype=self.tensortype)

        return inp,out

    def initPitches(self,size):
        ptc = np.zeros((size, self.channels_out, self.time_context, self.npitches), dtype=self.tensortype)
        return ptc

    def initMasks(self,size):
        msk = np.zeros((size, self.channels_out, self.time_context, self.input_size), dtype=self.tensortype)
        return msk

    def getFeatureSize(self):
        """
        Returns the feature size of the input and of the output to the neural network
        """
        if self.path_transform_in is not None and self.path_transform_out is not None:
            for i in range(len(self.file_list)):
                if os.path.isfile(os.path.join(self.path_transform_in[self.dirid[i]],self.file_list[i])):

                    allmix = self.loadTensor(os.path.join(self.path_transform_in[self.dirid[i]],self.file_list[i]))
                    self.channels_in = allmix.shape[0]
                    self.input_size=allmix.shape[-1]
                    allmix = None

                    allmixoutput = self.loadTensor(os.path.join(self.path_transform_out[self.dirid[i]],self.file_list[0].replace(self.prefix_in+'_m_',self.prefix_out+'_m_')))
                    self.channels_out = allmixoutput.shape[0]
                    self.output_size = allmixoutput.shape[-1]
                    allmixoutput = None

                    assert self.channels_out % self.channels_in == 0, "number of outputs is not multiple of number of inputs"

                    if self.pitched or self.save_mask:
                        pitch = self.loadTensor(os.path.join(self.path_transform_in[self.dirid[i]],self.file_list[i].replace(self.prefix_in+'_m_','_'+self.pitch_code+'_')))
                        if len(pitch.shape)>3:
                            self.nchan = np.minimum(pitch.shape[0],self.channels_in)
                            self.total_inst = int(np.floor(self.channels_out/self.nchan))
                            self.ninst = np.minimum(pitch.shape[1],self.total_inst)#number of pitched instruments (inst for which pitch is defined)
                        else:
                            self.ninst = np.minimum(pitch.shape[0],self.channels_out)
                            self.nchan = 1
                            self.total_inst = self.channels_out
                        self.npitches = 127 #midi notes/pitch granularity
                        pitch=None
                    if self.extra_features:
                        feat = self.loadTensor(os.path.join(self.path_transform_in[self.dirid[i]],self.file_list[i].replace(self.prefix_in+'_m_','_'+self.model+'_')))
                        self.extra_feat_size = feat.shape[-1]
                        feat=None



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
            if f.endswith(self.prefix_in+'_m_.data') and os.path.isfile(os.path.join(self.path_transform_out[k],f.replace(self.prefix_in+'_m_',self.prefix_out+'_m_'))) and\
            f.split('_',1)[0] not in self.exclude_list]

        self.dirid = [k for k in range(len(self.path_transform_in)) for f in os.listdir(self.path_transform_in[k]) \
            if f.endswith(self.prefix_in+'_m_.data') and os.path.isfile(os.path.join(self.path_transform_out[k],f.replace(self.prefix_in+'_m_',self.prefix_out+'_m_'))) and\
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
        self.getFeatureSize()
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
        self.batch_inputs = np.zeros((self.batch_memory*self.batch_size,self.channels_in,self.time_context,self.input_size), dtype=self.tensortype)
        self.batch_outputs = np.zeros((self.batch_memory*self.batch_size,self.channels_out,self.time_context,self.output_size), dtype=self.tensortype)
        if self.pitched:
            self.batch_pitches = np.zeros((self.batch_memory*self.batch_size,self.channels_out,self.time_context,self.npitches), dtype=self.tensortype)

        if self.save_mask:
            self.batch_masks = np.zeros((self.batch_memory*self.batch_size,self.channels_out,self.time_context,self.input_size), dtype=self.tensortype)

        if self.extra_features == True:
            self.batch_features = np.zeros((self.batch_memory*self.batch_size,self.context,self.extra_feat_size), dtype=self.tensortype)

        self.loadBatches()


class LargeDatasetMultiMask1(LargeDatasetMulti):
    def __init__(self, prefix_in="in", prefix_out="out",path_transform_in=None, path_transform_out=None, sampleRate=44100, exclude_list=[],nsamples=0, timbre_model_path=None,
        batch_size=64, batch_memory=1000, time_context=-1, overlap=5, tensortype=float, scratch_path=None, extra_features=False, model="", context=5,
        log_in=False, log_out=False, mult_factor_in=1., mult_factor_out=1., pitched=False,save_mask=True,pitch_norm=127.,nsources=2,
        nharmonics=20, nprocs=2,pitch_code='g',jump=0):

        self.nharmonics = nharmonics
        self.timbre_model_path=timbre_model_path
        self.save_mask = save_mask
        self.pitch_norm = pitch_norm
        self.extra_features = extra_features
        self.nsamples = nsamples
        self.pitch_code = pitch_code
        #load timbre models or initialize with 1s
        if self.timbre_model_path is not None:
            self.harmonics = util.loadObj(self.timbre_model_path)
        super(LargeDatasetMultiMask1, self).__init__(path_transform_in=path_transform_in, path_transform_out=path_transform_out, sampleRate=sampleRate, exclude_list=exclude_list, nsamples=nsamples, extra_features=extra_features, model=model, context=context,
            batch_size=batch_size, batch_memory=batch_memory, time_context=time_context, overlap=overlap, tensortype=tensortype, scratch_path=scratch_path, nsources=nsources,jump=jump,prefix_in=prefix_in, prefix_out=prefix_out,
            log_in=log_in, log_out=log_out, mult_factor_in=mult_factor_in, mult_factor_out=mult_factor_out,pitched=pitched,save_mask=save_mask,pitch_norm=pitch_norm,nprocs=nprocs)

    def filterSpec(self,mag,notes,start,stop):
        if not hasattr(self, 'ninst'):
            if len(notes.shape)>3:
                self.nchan = np.minimum(notes.shape[0],self.channels_in)
                self.total_inst = int(np.floor(self.channels_out/self.nchan))
                self.ninst = np.minimum(notes.shape[1],self.total_inst)
            else:
                self.ninst = np.minimum(notes.shape[0],self.channels_out)
                self.total_inst = self.channels_out
                self.nchan = 1
        filtered = np.ones((self.total_inst*self.nchan,mag.shape[-2],mag.shape[-1]), dtype=self.tensortype) * 1e-14
        for c in range(self.nchan):
            if len(notes.shape)>3:
                notes_=notes[c]
            else:
                notes_ = notes
            for j in range(self.ninst): #for all the inputed instrument notes_
                for p in range(len(notes_[j])): #for all notes_
                    if notes_[j,p,2] > 0 and np.maximum(0, np.minimum(notes_[j,p,1], stop) - np.maximum(notes_[j,p,0], start))>0:
                        begin = int(np.maximum(notes_[j,p,0], start))-start
                        end = int(np.minimum(notes_[j,p,1], stop))-start
                        slice_x = slice(begin,end,None)
                        slices_y_start = notes_[j,p,3::2]
                        slices_y_stop = notes_[j,p,4::2]
                        if self.timbre_model_path is None:
                            slices_y = np.hstack(tuple([range(int(slices_y_start[f]),int(slices_y_stop[f])) for f in range(np.minimum(len(slices_y_start),len(slices_y_stop))) if slices_y_stop[f]>0]))
                            filtered[self.nchan*j+c,slice_x,slices_y] = 1.
                            slices_y = None
                        else:
                            for k in range(len(slices_y_start)):
                                filtered[self.nchan*j+c,slice_x,slice(int(slices_y_start[k]),int(slices_y_stop[k]),None)] = filtered[self.nchan*j+c,slice_x,slice(int(slices_y_start[k]),int(slices_y_stop[k]),None)] + self.harmonics[j,int(notes_[j,p,2]),k]
                        slice_x = None
        notes_=None
        for j in range(self.total_inst*self.nchan): #normalize per output
            filtered[j,:,:] = filtered[j,:,:] / np.max(filtered[j,:,:])

        # import matplotlib.pyplot as plt
        # for j in range(self.ninst):
        #     plt.subplot(211)
        #     plt.imshow(mag,interpolation='none')
        #     plt.subplot(212)
        #     plt.imshow(filtered[j,:,:],interpolation='none')
        #     plt.show()
        # import pdb;pdb.set_trace()

        j=None
        p=None
        f=None
        return filtered

    def getClassWeights(self):
        if self.path_transform_in is not None and self.path_transform_out is not None:
            unique_paths = list(set(self.path_transform_in))
            for ids in range(len(unique_paths)):
                i = self.path_transform_in.index(unique_paths[ids])
                if os.path.isfile(os.path.join(self.path_transform_in[self.dirid[i]],self.file_list[i])):
                    notes = self.loadTensor(os.path.join(self.path_transform_in[self.dirid[i]],self.file_list[i].replace(self.prefix_in+'_m_','_'+self.pitch_code+'_')))
                    if ids==0:
                        if not hasattr(self, 'ninst'):
                            if len(notes.shape)>3:
                                self.nchan = np.minimum(notes.shape[0],self.channels_in)
                                self.total_inst = int(np.floor(self.channels_out/self.nchan))
                                self.ninst = np.minimum(notes.shape[1],self.total_inst)
                            else:
                                self.ninst = np.minimum(notes.shape[0],self.channels_out)
                                self.nchan = 1
                                self.total_inst = self.channels_out
                        self.weights = np.zeros(self.total_inst)
                    for c in range(notes.shape[0]):
                        if len(notes.shape)>3:
                            notes_= notes[c]
                        else:
                            notes_ = notes
                        for j in range(notes.shape[1]): #for all the inputed instrument notes_
                            for p in range(len(notes_[j])): #for all notes_
                                if notes_[j,p,2] > 0:
                                    begin = notes_[j,p,0]
                                    end = notes_[j,p,1]
                                    self.weights[j] += end-begin
                    self.weights = np.clip(1.-self.weights / self.weights.max(),0.25,0.99)
                else:
                    self.weights = np.ones(int(self.channels_out/self.channels_in))
        else:
            self.weights = np.ones(int(self.channels_out/self.channels_in))
        return self.weights

    def buildPitch(self,mag,notes,start,stop):
        if not hasattr(self, 'ninst'):
            if len(notes.shape)>3:
                self.nchan = np.minimum(notes.shape[0],self.channels_in)
                self.ninst = np.minimum(notes.shape[1],int(np.floor(self.channels_out/self.nchan)))
            else:
                self.ninst = np.minimum(notes.shape[0],self.channels_out)
                self.nchan = 1
        filtered = np.zeros((self.ninst,mag.shape[-2],self.npitches), dtype=self.tensortype)
        for j in range(self.ninst): #for all the inputed instrument notes
            for p in range(len(notes[j])): #for all notes
                if notes[j,p,2] > 0 and np.maximum(0, np.minimum(notes[j,p,1], stop) - np.maximum(notes[j,p,0], start))>0:
                    begin = int(np.maximum(notes[j,p,0], start))-start
                    end = int(np.minimum(notes[j,p,1], stop))-start
                    slice_x = slice(begin,end,None)
                    filtered[j,slice_x,int(notes[j,p,2])] = 1.
                    slice_x = None
        j=None
        p=None
        return mask


class LargeDatasetMultiMask2(LargeDatasetMulti):
    def __init__(self, prefix_in="in", prefix_out="out", path_transform_in=None, path_transform_out=None, sampleRate=44100, exclude_list=[],nsamples=0, timbre_model_path=None,
        batch_size=64, batch_memory=1000, time_context=-1, overlap=5, tensortype=float, scratch_path=None, extra_features=False, model="", context=5,
        log_in=False, log_out=False, mult_factor_in=1., mult_factor_out=1., pitched=False,save_mask=True,pitch_norm=127.,nsources=2,
        nharmonics=20, nprocs=2,pitch_code='g',jump=0):

        self.nharmonics = nharmonics
        self.timbre_model_path=timbre_model_path
        self.save_mask = save_mask
        self.pitch_norm = pitch_norm
        self.extra_features = extra_features
        self.nsamples = nsamples
        self.pitch_code = pitch_code
        #load timbre models or initialize with 1s
        if self.timbre_model_path is not None:
            self.harmonics = util.loadObj(self.timbre_model_path)
        super(LargeDatasetMultiMask2, self).__init__(path_transform_in=path_transform_in, path_transform_out=path_transform_out, sampleRate=sampleRate, exclude_list=exclude_list, nsamples=nsamples, extra_features=extra_features, model=model, context=context,
            batch_size=batch_size, batch_memory=batch_memory, time_context=time_context, overlap=overlap, tensortype=tensortype, scratch_path=scratch_path, nsources=nsources,jump=jump,prefix_in=prefix_in, prefix_out=prefix_out,
            log_in=log_in, log_out=log_out, mult_factor_in=mult_factor_in, mult_factor_out=mult_factor_out,pitched=pitched,save_mask=save_mask,pitch_norm=pitch_norm,nprocs=nprocs)

    def filterSpec(self,mag,notes,start,stop):
        if not hasattr(self, 'ninst'):
            if len(notes.shape)>3:
                self.nchan = np.minimum(notes.shape[0],self.channels_in)
                self.total_inst = int(np.floor(self.channels_out/self.nchan))
                self.ninst = np.minimum(notes.shape[1],self.total_inst)
            else:
                self.ninst = np.minimum(notes.shape[0],self.channels_out)
                self.nchan = 1
                self.total_inst = self.channels_out
        if self.timbre_model_path is None:
            filtered = np.ones((self.total_inst*self.nchan,mag.shape[-2],mag.shape[-1]), dtype=self.tensortype) * 1e-14
        else:
            filtered = np.ones((self.total_inst*self.nchan,mag.shape[-2],mag.shape[-1]), dtype=self.tensortype) * 1e-4
        for c in range(self.nchan):
            if len(notes.shape)>3:
                notes_= notes[c]
            else:
                notes_ = notes
            for j in range(self.ninst): #for all the inputed instrument notes_
                for p in range(len(notes_[j])): #for all notes_
                    if notes_[j,p,2] > 0 and np.maximum(0, np.minimum(notes_[j,p,1], stop) - np.maximum(notes_[j,p,0], start))>0:
                        begin = int(np.maximum(notes_[j,p,0], start))-start
                        end = int(np.minimum(notes_[j,p,1], stop))-start
                        slice_x = slice(begin,end,None)
                        slices_y_start = notes_[j,p,3::2]
                        slices_y_stop = notes_[j,p,4::2]
                        if self.timbre_model_path is None:
                            slices_y = np.hstack(tuple([range(int(slices_y_start[f]),int(slices_y_stop[f])) for f in range(np.minimum(len(slices_y_start),len(slices_y_stop))) if slices_y_stop[f]>0]))
                            filtered[self.nchan*j+c,slice_x,slices_y] = 1.
                            slices_y = None
                        else:
                            for k in range(len(slices_y_start)):
                                filtered[self.nchan*j+c,slice_x,slice(int(slices_y_start[k]),int(slices_y_stop[k]),None)] = filtered[self.nchan*j+c,slice_x,slice(int(slices_y_start[k]),int(slices_y_stop[k]),None)] + self.harmonics[j,int(notes_[j,p,2]),k]
                        slice_x = None
            notes_=None
        for c in range(self.nchan):
            allsum = np.sum(filtered[c::self.nchan,:,:],axis=0)
            for j in range(self.total_inst): #build a soft-mask per channel(divide with sum)
                filtered[self.nchan*j+c,:,:] = filtered[self.nchan*j+c,:,:] / allsum

        # import matplotlib.pyplot as plt
        # for j in range(self.ninst):
        #     plt.subplot(211)
        #     plt.imshow(mag,interpolation='none')
        #     plt.subplot(212)
        #     plt.imshow(filtered[j,:,:100],interpolation='none')
        #     plt.show()
        # import pdb;pdb.set_trace()

        j=None
        p=None
        f=None
        c=None
        return filtered

    def getClassWeights(self,imin=0,imax=1.):
        if self.path_transform_in is not None and self.path_transform_out is not None:
            unique_paths = list(set(self.path_transform_in))
            for ids in range(len(unique_paths)):
                i = self.path_transform_in.index(unique_paths[ids])
                if os.path.isfile(os.path.join(self.path_transform_in[self.dirid[i]],self.file_list[i])):
                    notes = self.loadTensor(os.path.join(self.path_transform_in[self.dirid[i]],self.file_list[i].replace(self.prefix_in+'_m_','_'+self.pitch_code+'_')))
                    if ids==0:
                        if not hasattr(self, 'ninst'):
                            if len(notes.shape)>3:
                                self.nchan = np.minimum(notes.shape[0],self.channels_in)
                                self.total_inst = int(np.floor(self.channels_out/self.nchan))
                                self.ninst = np.minimum(notes.shape[1],self.total_inst)
                            else:
                                self.ninst = np.minimum(notes.shape[0],self.channels_out)
                                self.nchan = 1
                                self.total_inst = self.channels_out
                        self.weights = np.zeros(self.total_inst)
                    for c in range(notes.shape[0]):
                        if len(notes.shape)>3:
                            notes_= notes[c]
                        else:
                            notes_ = notes
                        for j in range(notes.shape[1]): #for all the inputed instrument notes_
                            for p in range(len(notes_[j])): #for all notes_
                                if notes_[j,p,2] > 0:
                                    begin = notes_[j,p,0]
                                    end = notes_[j,p,1]
                                    self.weights[j] += end-begin
                    a = (imax-imin)/(self.weights.max() - self.weights.min())
                    b = imax - a * self.weights.max()
                    self.weights = a * self.weights + b
                else:
                    self.weights = np.ones(int(self.channels_out/self.channels_in))
        else:
            self.weights = np.ones(int(self.channels_out/self.channels_in))
        return self.weights



    def buildPitch(self,mag,notes,start,stop):
        if not hasattr(self, 'ninst'):
            if len(notes.shape)>3:
                self.nchan = np.minimum(notes.shape[0],self.channels_in)
                self.ninst = np.minimum(notes.shape[1],int(np.floor(self.channels_out/self.nchan)))
            else:
                self.ninst = np.minimum(notes.shape[0],self.channels_out)
                self.nchan = 1
        filtered = np.zeros((self.ninst,mag.shape[-2],self.npitches),dtype=self.tensortype)
        for j in range(self.ninst): #for all the inputed instrument notes
            for p in range(len(notes[j])): #for all notes
                if notes[j,p,2] > 0 and np.maximum(0, np.minimum(notes[j,p,1], stop) - np.maximum(notes[j,p,0], start))>0:
                    begin = int(np.maximum(notes[j,p,0], start))-start
                    end = int(np.minimum(notes[j,p,1], stop))-start
                    slice_x = slice(begin,end,None)
                    filtered[j,slice_x,int(notes[j,p,2])] = 1.
                    slice_x = None
        j=None
        p=None
        return mask

    # def initPitches(self,size):
    #     ptc = np.zeros((size, self.channels_out, self.time_context, self.npitches), dtype=self.tensortype)
    #     return ptc
    #     #super(LargeDatasetMulti, self).initPitches(size)

    # def initMasks(self,size):
    #     msk = np.zeros((size, self.channels_out, self.time_context, self.input_size), dtype=self.tensortype)
    #     return msk
    #     #super(LargeDatasetMulti, self).initMasks(size)
