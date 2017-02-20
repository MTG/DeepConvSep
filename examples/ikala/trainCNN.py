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
from transform import transformFFT
import dataset
from dataset import LargeDataset
import util

import numpy as np
import re
from scipy.signal import blackmanharris as blackmanharris
import shutil
import time
import cPickle
import re
import climate
import ConfigParser

import theano
import theano.tensor as T
import theano.sandbox.rng_mrg

import lasagne
from lasagne.layers import ReshapeLayer,Layer
from lasagne.init import Normal
from lasagne.regularization import regularize_layer_params_weighted, l2, l1
from lasagne.regularization import regularize_layer_params


logging = climate.get_logger('trainer')

climate.enable_default_logging()


def load_model(filename):
    f=file(filename,'rb')
    params=cPickle.load(f)
    f.close()
    return params

def save_model(filename, model):
    params=lasagne.layers.get_all_param_values(model)
    f = file(filename, 'wb')
    cPickle.dump(params,f,protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()
    return None

def build_ca(input_var=None, batch_size=32,time_context=30,feat_size=513):
    """
    Builds a network with lasagne
    
    Parameters
    ----------
    input_var : Theano tensor
        The input for the network
    batch_size : int, optional
        The number of examples in a batch   
    time_context : int, optional
        The time context modeled by the network. 
    feat_size : int, optional
        The feature size modeled by the network (last dimension of the feature vector)
    Yields
    ------
    l_out : Theano tensor
        The output of the network
    """

    input_shape=(batch_size,1,time_context,feat_size)

    #input layer
    l_in_1 = lasagne.layers.InputLayer(shape=input_shape, input_var=input_var)

    #vertical convolution layer
    l_conv1 = lasagne.layers.Conv2DLayer(l_in_1, num_filters=30, filter_size=(1,30),stride=(1,3), pad='valid', nonlinearity=None)
    l_conv1b= lasagne.layers.BiasLayer(l_conv1)

    #max-pool layer
    l_pool1 = lasagne.layers.MaxPool2DLayer(l_conv1b, pool_size=(1, 4))

    #horizontal convolution layer
    l_conv2 = lasagne.layers.Conv2DLayer(l_pool1, num_filters=30, filter_size=(10,20),stride=(1,1), pad='valid', nonlinearity=None)
    l_conv2b= lasagne.layers.BiasLayer(l_conv2)

    #bottlneck layer
    l_fc=lasagne.layers.DenseLayer(l_conv2b,256)

    #build output for source1
    l_fc11=lasagne.layers.DenseLayer(l_fc,l_conv2.output_shape[1]*l_conv2.output_shape[2]*l_conv2.output_shape[3])
    l_reshape1 = lasagne.layers.ReshapeLayer(l_fc11,(batch_size,l_conv2.output_shape[1],l_conv2.output_shape[2], l_conv2.output_shape[3]))
    l_inverse11=lasagne.layers.InverseLayer(l_reshape1, l_conv2)
    l_inverse31=lasagne.layers.InverseLayer(l_inverse11, l_pool1)
    l_inverse41=lasagne.layers.InverseLayer(l_inverse31, l_conv1)

    #build output for source1
    l_fc12=lasagne.layers.DenseLayer(l_fc,l_conv2.output_shape[1]*l_conv2.output_shape[2]*l_conv2.output_shape[3])
    l_reshape2 = lasagne.layers.ReshapeLayer(l_fc12,(batch_size,l_conv2.output_shape[1],l_conv2.output_shape[2], l_conv2.output_shape[3]))
    l_inverse12=lasagne.layers.InverseLayer(l_reshape2, l_conv2)
    l_inverse32=lasagne.layers.InverseLayer(l_inverse12, l_pool1)
    l_inverse42=lasagne.layers.InverseLayer(l_inverse32, l_conv1)

    #build final output 
    l_merge=lasagne.layers.ConcatLayer([l_inverse41,l_inverse42],axis=1)
    l_out = lasagne.layers.NonlinearityLayer(lasagne.layers.BiasLayer(l_merge), nonlinearity=lasagne.nonlinearities.rectify)
   
    return l_out


def train_auto(train,fun,transform,testdir,outdir,num_epochs=30,model="1.pkl",scale_factor=0.3,load=False,skip_train=False,skip_sep=False):
    """
    Trains a network built with \"fun\" with the data generated with \"train\"
    and then separates the files in \"testdir\",writing the result in \"outdir\"

    Parameters
    ----------
    train : Callable, e.g. LargeDataset object
        The callable which generates training data for the network: inputs, target = train()
    fun : lasagne network object, Theano tensor
        The network to be trained  
    transform : transformFFT object
        The Transform object which was used to compute the features (see compute_features.py)
    testdir : string, optional
        The directory where the files to be separated are located
    outdir : string, optional
        The directory where to write the separated files
    num_epochs : int, optional
        The number the epochs to train for (one epoch is when all examples in the dataset are seen by the network)
    model : string, optional
        The path where to save the trained model (theano tensor containing the network) 
    scale_factor : float, optional
        Scale the magnitude of the files to be separated with this factor
    Yields
    ------
    losser : list
        The losses for each epoch, stored in a list
    """

    logging.info("Building Autoencoder")
    input_var2 = T.tensor4('inputs')
    target_var2 = T.tensor4('targets')
    rand_num = T.tensor4('rand_num')
    
    eps=1e-8
    alpha=0.9
    beta_acc=0.005
    beta_voc=0.02

    network2 = fun(input_var=input_var2,batch_size=train.batch_size,time_context=train.time_context,feat_size=train.input_size)
    
    if load:
        params=load_model(model)
        lasagne.layers.set_all_param_values(network2,params)

    prediction2 = lasagne.layers.get_output(network2, deterministic=True)

    rand_num = np.random.uniform(size=(train.batch_size,1,train.time_context,train.input_size))

    voc=prediction2[:,0:1,:,:]+eps*rand_num
    acco=prediction2[:,1:2,:,:]+eps*rand_num

    mask1=voc/(voc+acco)
    mask2=acco/(voc+acco)

    vocals=mask1*input_var2[:,0:1,:,:]
    acc=mask2*input_var2[:,0:1,:,:]
    
    train_loss_recon_vocals = lasagne.objectives.squared_error(vocals,target_var2[:,0:1,:,:])
    train_loss_recon_acc = alpha * lasagne.objectives.squared_error(acc,target_var2[:,1:2,:,:])    
    train_loss_recon_neg_voc = beta_voc * lasagne.objectives.squared_error(vocals,target_var2[:,1:2,:,:])
    train_loss_recon_neg_acc = beta_acc * lasagne.objectives.squared_error(acc,target_var2[:,0:1,:,:])

    vocals_error=train_loss_recon_vocals.sum()  
    acc_error=train_loss_recon_acc.sum()  
    negative_error_voc=train_loss_recon_neg_voc.sum()
    negative_error_acc=train_loss_recon_neg_acc.sum()
    
    loss=abs(vocals_error+acc_error-negative_error_voc)

    params1 = lasagne.layers.get_all_params(network2, trainable=True)

    updates = lasagne.updates.adadelta(loss, params1)

    train_fn = theano.function([input_var2,target_var2], loss, updates=updates,allow_input_downcast=True)

    train_fn1 = theano.function([input_var2,target_var2], [vocals_error,acc_error,negative_error_voc,negative_error_acc], allow_input_downcast=True)

    predict_function2=theano.function([input_var2],[vocals,acc],allow_input_downcast=True)
    predict_function3=theano.function([input_var2],[prediction2[:,0:1,:,:],prediction2[:,1:2,:,:]],allow_input_downcast=True)

    losser=[]
    loss2=[]

    if not skip_train:

        logging.info("Training...")
        for epoch in range(num_epochs):

            train_err = 0
            train_batches = 0
            vocals_err=0
            acc_err=0        
            beta_voc=0
            beta_acc=0
            start_time = time.time()
            for batch in range(train.iteration_size): 
                inputs, target = train()
                
                jump = inputs.shape[2]
                targets=np.ndarray(shape=(inputs.shape[0],2,inputs.shape[1],inputs.shape[2]))
                inputs=np.reshape(inputs,(inputs.shape[0],1,inputs.shape[1],inputs.shape[2]))          

                targets[:,0,:,:]=target[:,:,:jump]
                targets[:,1,:,:]=target[:,:,jump:jump*2]         
                target=None
        
                train_err+=train_fn(inputs,targets)
                [vocals_erre,acc_erre,betae_voc,betae_acc]=train_fn1(inputs,targets)
                vocals_err += vocals_erre
                acc_err += acc_erre           
                beta_voc+= betae_voc
                beta_acc+= betae_acc
                train_batches += 1
            
            logging.info("Epoch {} of {} took {:.3f}s".format(
                epoch + 1, num_epochs, time.time() - start_time))
            logging.info("  training loss:\t\t{:.6f}".format(train_err/train_batches))
            logging.info("  training loss for vocals:\t\t{:.6f}".format(vocals_err/train_batches))
            logging.info("  training loss for acc:\t\t{:.6f}".format(acc_err/train_batches))
            logging.info("  Beta component for voice:\t\t{:.6f}".format(beta_voc/train_batches))
            logging.info("  Beta component for acc:\t\t{:.6f}".format(beta_acc/train_batches))
            losser.append(train_err / train_batches)
            save_model(model,network2)

    if not skip_sep:

        logging.info("Separating")
        for f in os.listdir(testdir):
            if f.endswith(".wav"):
                audioObj, sampleRate, bitrate = util.readAudioScipy(os.path.join(testdir,f))
                
                assert sampleRate == 44100,"Sample rate needs to be 44100"

                audio = audioObj[:,0] + audioObj[:,1]
                audioObj = None
                mag,ph=transform.compute_file(audio,phase=True)
         
                mag=scale_factor*mag.astype(np.float32)

                batches,nchunks = util.generate_overlapadd(mag,input_size=mag.shape[-1],time_context=train.time_context,overlap=train.overlap,batch_size=train.batch_size,sampleRate=sampleRate)
                output=[]

                batch_no=1
                for batch in batches:
                    batch_no+=1
                    start_time=time.time()
                    output.append(predict_function2(batch))

                output=np.array(output)
                bmag,mm=util.overlapadd(output,batches,nchunks,overlap=train.overlap)
                
                audio_out=transform.compute_inverse(bmag[:len(ph)]/scale_factor,ph)
                if len(audio_out)>len(audio):
                    audio_out=audio_out[:len(audio)]
                audio_out=essentia.array(audio_out)
                audio_out2= transform.compute_inverse(mm[:len(ph)]/scale_factor,ph) 
                if len(audio_out2)>len(audio):
                    audio_out2=audio_out2[:len(audio)]  
                audio_out2=essentia.array(audio_out2) 
                #write audio files
                util.writeAudioScipy(os.path.join(outdir,f.replace(".wav","-voice.wav")),audio_out,sampleRate,bitrate)
                util.writeAudioScipy(os.path.join(outdir,f.replace(".wav","-music.wav")),audio_out2,sampleRate,bitrate)
                audio_out=None 
                audio_out2=None   

    return losser  




if __name__ == "__main__": 
    """
    Source separation for the iKala dataset.
    2nd place MIREX Singing voice separation 2016
    http://www.music-ir.org/mirex/wiki/2016:Singing_Voice_Separation_Results

    More details in the following article:
    P. Chandna, M. Miron, J. Janer, and E. Gomez,
    \“Monoaural audio source separation using deep convolutional neural networks,\” 
    International Conference on Latent Variable Analysis and Signal Separation, 2017.

    Given the features computed previusly with compute_features, train a network and perform the separation.
    
    Parameters
    ----------
    db : string
        The path to the iKala dataset  
    nepochs : int, optional
        The number the epochs to train for (one epoch is when all examples in the dataset are seen by the network)
    model : string, optional
        The name of the trained model 
    scale_factor : float, optional
        Scale the magnitude of the files to be separated with this factor
    batch_size : int, optional
        The number of examples in a batch (see LargeDataset in dataset.py)  
    batch_memory : int, optional
        The number of batches to load in memory at once (see LargeDataset in dataset.py)
    time_context : int, optional
        The time context modeled by the network
    overlap : int, optional
        The number of overlapping frames between adjacent segments (see LargeDataset in dataset.py)
    nprocs : int, optional
        The number of CPU to use when loading the data in parallel: the more, the faster (see LargeDataset in dataset.py)
    """
    if len(sys.argv)>-1:
        climate.add_arg('--db', help="the ikala dataset path")
        climate.add_arg('--model', help="the name of the model to test/save")
        climate.add_arg('--nepochs', help="number of epochs to train the net")
        climate.add_arg('--time_context', help="number of frames for the recurrent/lstm/conv net")
        climate.add_arg('--batch_size', help="batch size for training")
        climate.add_arg('--batch_memory', help="number of big batches to load into memory")
        climate.add_arg('--overlap', help="overlap time context for training")
        climate.add_arg('--nprocs', help="number of processor to parallelize file reading")
        climate.add_arg('--scale_factor', help="scale factor for the data")
        climate.add_arg('--feature_path', help="the path where to load the features from")
        db=None
        kwargs = climate.parse_args()
        if kwargs.__getattribute__('db'):
            db = kwargs.__getattribute__('db')
        # else:
        #     db='/home/marius/Documents/Database/iKala/'  
        if kwargs.__getattribute__('feature_path'):
            feature_path = kwargs.__getattribute__('feature_path')
        else:
            feature_path=os.path.join(db,'transforms','t1') 
        assert os.path.isdir(db), "Please input the directory for the iKala dataset with --db path_to_iKala"  
        if kwargs.__getattribute__('model'):
            model = kwargs.__getattribute__('model')
        else:
            model="fft_1024"    
        if kwargs.__getattribute__('batch_size'):
            batch_size = int(kwargs.__getattribute__('batch_size')) 
        else:
            batch_size = 32
        if kwargs.__getattribute__('batch_memory'):
            batch_memory = int(kwargs.__getattribute__('batch_memory')) 
        else:
            batch_memory = 200
        if kwargs.__getattribute__('time_context'):
            time_context = int(kwargs.__getattribute__('time_context')) 
        else:
            time_context = 30
        if kwargs.__getattribute__('overlap'):
            overlap = int(kwargs.__getattribute__('overlap')) 
        else:
            overlap = 20
        if kwargs.__getattribute__('nprocs'):
            nprocs = int(kwargs.__getattribute__('nprocs')) 
        else:
            nprocs = 7
        if kwargs.__getattribute__('nepochs'):
            nepochs = int(kwargs.__getattribute__('nepochs')) 
        else:
            nepochs = 40
        if kwargs.__getattribute__('scale_factor'):
            scale_factor = int(kwargs.__getattribute__('scale_factor')) 
        else:
            scale_factor = 0.3

    #tt object needs to be the same as the one in compute_features
    tt = transformFFT(frameSize=1024, hopSize=512, sampleRate=44100, window=blackmanharris)
    pitchhop=0.032*float(tt.sampleRate) #seconds to frames   

    ld1 = LargeDataset(path_transform_in=feature_path, batch_size=batch_size, batch_memory=batch_memory, time_context=time_context, overlap=overlap, nprocs=nprocs,mult_factor_in=scale_factor,mult_factor_out=scale_factor)
    logging.info("  Maximum:\t\t{:.6f}".format(ld1.getMax()))
    logging.info("  Mean:\t\t{:.6f}".format(ld1.getMean()))
    logging.info("  Standard dev:\t\t{:.6f}".format(ld1.getStd()))

    if not os.path.exists(db+'output/'):
        os.makedirs(db+'output/')
    if not os.path.exists(db+'output/'+model):
        os.makedirs(db+'output/'+model)
    if not os.path.exists(db+'models/'):
        os.makedirs(db+'models/')

    train_errs=train_auto(train=ld1,fun=build_ca,transform=tt,outdir=db+'output/'+model+"/",testdir=db+'Wavfile/',model=db+"models/"+"model_"+model+".pkl",num_epochs=nepochs,scale_factor=scale_factor)      
    f = file(db+"models/"+"loss_"+model+".data", 'wb')
    cPickle.dump(train_errs,f,protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()

