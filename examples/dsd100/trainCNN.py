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
    l_conv1 = lasagne.layers.Conv2DLayer(l_in_1, num_filters=50, filter_size=(1,feat_size),stride=(1,1), pad='valid', nonlinearity=None)
    l_conv1b= lasagne.layers.BiasLayer(l_conv1)

    #horizontal convolution layer
    l_conv2 = lasagne.layers.Conv2DLayer(l_conv1b, num_filters=50, filter_size=(int(time_context/2),1),stride=(1,1), pad='valid', nonlinearity=None)
    l_conv2b= lasagne.layers.BiasLayer(l_conv2)

    #bottlneck layer
    l_fc=lasagne.layers.DenseLayer(l_conv2b,128)

    #build output for source1
    l_fc11=lasagne.layers.DenseLayer(l_fc,l_conv2.output_shape[1]*l_conv2.output_shape[2]*l_conv2.output_shape[3])
    l_reshape1 = lasagne.layers.ReshapeLayer(l_fc11,(batch_size,l_conv2.output_shape[1],l_conv2.output_shape[2], l_conv2.output_shape[3]))
    l_inverse11=lasagne.layers.InverseLayer(l_reshape1, l_conv2)
    l_inverse41=lasagne.layers.InverseLayer(l_inverse11, l_conv1)

    #build output for source2
    l_fc12=lasagne.layers.DenseLayer(l_fc,l_conv2.output_shape[1]*l_conv2.output_shape[2]*l_conv2.output_shape[3])
    l_reshape2 = lasagne.layers.ReshapeLayer(l_fc12,(batch_size,l_conv2.output_shape[1],l_conv2.output_shape[2], l_conv2.output_shape[3]))
    l_inverse12=lasagne.layers.InverseLayer(l_reshape2, l_conv2)
    l_inverse42=lasagne.layers.InverseLayer(l_inverse12, l_conv1)

    #build output for source3
    l_fc13=lasagne.layers.DenseLayer(l_fc,l_conv2.output_shape[1]*l_conv2.output_shape[2]*l_conv2.output_shape[3])
    l_reshape3 = lasagne.layers.ReshapeLayer(l_fc13,(batch_size,l_conv2.output_shape[1],l_conv2.output_shape[2], l_conv2.output_shape[3]))
    l_inverse13=lasagne.layers.InverseLayer(l_reshape3, l_conv2)
    l_inverse43=lasagne.layers.InverseLayer(l_inverse13, l_conv1)

    #build output for source4
    l_fc14=lasagne.layers.DenseLayer(l_fc,l_conv2.output_shape[1]*l_conv2.output_shape[2]*l_conv2.output_shape[3])
    l_reshape4 = lasagne.layers.ReshapeLayer(l_fc12,(batch_size,l_conv2.output_shape[1],l_conv2.output_shape[2], l_conv2.output_shape[3]))
    l_inverse14=lasagne.layers.InverseLayer(l_reshape4, l_conv2)
    l_inverse44=lasagne.layers.InverseLayer(l_inverse14, l_conv1)

    #build final output 
    l_merge=lasagne.layers.ConcatLayer([l_inverse41,l_inverse42,l_inverse43,l_inverse44],axis=1)
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
    alpha=0.001
    beta=0.01
    beta_voc=0.03
   
    network2 = fun(input_var=input_var2,batch_size=train.batch_size,time_context=train.time_context,feat_size=train.input_size)
    
    if load:
        params=load_model(model)
        lasagne.layers.set_all_param_values(network2,params)

    prediction2 = lasagne.layers.get_output(network2, deterministic=True)

    rand_num = np.random.uniform(size=(train.batch_size,1,train.time_context,train.input_size))

    voc=prediction2[:,0:1,:,:]+eps*rand_num
    bas=prediction2[:,1:2,:,:]+eps*rand_num
    dru=prediction2[:,2:3,:,:]+eps*rand_num
    oth=prediction2[:,3:4,:,:]+eps*rand_num

    mask1=voc/(voc+bas+dru+oth)
    mask2=bas/(voc+bas+dru+oth)
    mask3=dru/(voc+bas+dru+oth)
    mask4=oth/(voc+bas+dru+oth)

    vocals=mask1*input_var2
    bass=mask2*input_var2
    drums=mask3*input_var2
    others=mask4*input_var2

    train_loss_recon_vocals = lasagne.objectives.squared_error(vocals,target_var2[:,0:1,:,:])
    alpha_component = alpha*lasagne.objectives.squared_error(vocals,target_var2[:,1:2,:,:])
    alpha_component += alpha*lasagne.objectives.squared_error(vocals,target_var2[:,2:3,:,:])    
    train_loss_recon_neg_voc = beta_voc*lasagne.objectives.squared_error(vocals,target_var2[:,3:4,:,:])

    train_loss_recon_bass = lasagne.objectives.squared_error(bass,target_var2[:,1:2,:,:])
    alpha_component += alpha*lasagne.objectives.squared_error(bass,target_var2[:,0:1,:,:])
    alpha_component += alpha*lasagne.objectives.squared_error(bass,target_var2[:,2:3,:,:])
    train_loss_recon_neg = beta*lasagne.objectives.squared_error(bass,target_var2[:,3:4,:,:])

    train_loss_recon_drums = lasagne.objectives.squared_error(drums,target_var2[:,2:3,:,:])
    alpha_component += alpha*lasagne.objectives.squared_error(drums,target_var2[:,0:1,:,:])
    alpha_component += alpha*lasagne.objectives.squared_error(drums,target_var2[:,1:2,:,:])
    train_loss_recon_neg += beta*lasagne.objectives.squared_error(drums,target_var2[:,3:4,:,:])

    vocals_error=train_loss_recon_vocals.sum()
    drums_error=train_loss_recon_drums.sum()
    bass_error=train_loss_recon_bass.sum()
    negative_error=train_loss_recon_neg.sum()
    negative_error_voc=train_loss_recon_neg_voc.sum()
    alpha_component=alpha_component.sum()

    loss=abs(vocals_error+drums_error+bass_error-negative_error-alpha_component-negative_error_voc)

    params1 = lasagne.layers.get_all_params(network2, trainable=True)

    updates = lasagne.updates.adadelta(loss, params1)

    # val_updates=lasagne.updates.nesterov_momentum(loss1, params1, learning_rate=0.00001, momentum=0.7)

    train_fn = theano.function([input_var2,target_var2], loss, updates=updates,allow_input_downcast=True)

    train_fn1 = theano.function([input_var2,target_var2], [vocals_error,bass_error,drums_error,negative_error,alpha_component,negative_error_voc], allow_input_downcast=True)

    predict_function2=theano.function([input_var2],[vocals,bass,drums,others],allow_input_downcast=True)

    losser=[]
    loss2=[]

    if not skip_train:

        logging.info("Training...")
        for epoch in range(num_epochs):

            train_err = 0
            train_batches = 0
            vocals_err=0
            drums_err=0
            bass_err=0
            negative_err=0
            alpha_component=0
            beta_voc=0
            start_time = time.time()
            for batch in range(train.iteration_size): 
                inputs, target = train()
                jump = inputs.shape[2]
                inputs=np.reshape(inputs,(inputs.shape[0],1,inputs.shape[1],inputs.shape[2]))
                targets=np.ndarray(shape=(inputs.shape[0],4,inputs.shape[2],inputs.shape[3]))
                #import pdb;pdb.set_trace()
                targets[:,0,:,:]=target[:,:,:jump]
                targets[:,1,:,:]=target[:,:,jump:jump*2]
                targets[:,2,:,:]=target[:,:,jump*2:jump*3]
                targets[:,3,:,:]=target[:,:,jump*3:jump*4]
                target = None

                train_err+=train_fn(inputs,targets)
                [vocals_erre,bass_erre,drums_erre,negative_erre,alpha,betae_voc]=train_fn1(inputs,targets)
                vocals_err +=vocals_erre
                bass_err +=bass_erre
                drums_err +=drums_erre
                negative_err +=negative_erre
                beta_voc+=betae_voc
                alpha_component+=alpha
                train_batches += 1
        
            print("Epoch {} of {} took {:.3f}s".format(
                epoch + 1, num_epochs, time.time() - start_time))
            print("  training loss:\t\t{:.6f}".format(train_err/train_batches))
            losser.append(train_err / train_batches)
            print("  training loss for vocals:\t\t{:.6f}".format(vocals_err/train_batches))
            print("  training loss for bass:\t\t{:.6f}".format(bass_err/train_batches))
            print("  training loss for drums:\t\t{:.6f}".format(drums_err/train_batches))
            print("  Beta component:\t\t{:.6f}".format(negative_err/train_batches))
            print("  Beta component for voice:\t\t{:.6f}".format(beta_voc/train_batches))
            print("  alpha component:\t\t{:.6f}".format(alpha_component/train_batches))
            losser.append(train_err / train_batches)
            save_model(model,network2)

    if not skip_sep:

        logging.info("Separating")
        source = ['vocals','bass','drums','other']
        dev_directory = os.listdir(os.path.join(testdir,"Dev"))
        test_directory = os.listdir(os.path.join(testdir,"Test")) #we do not include the test dir
        dirlist = []
        dirlist.extend(dev_directory)
        dirlist.extend(test_directory)
        for f in sorted(dirlist):
            if not f.startswith('.'):
                if f in dev_directory:
                    song=os.path.join(testdir,"Dev",f,"mixture.wav")
                else:
                    song=os.path.join(testdir,"Test",f,"mixture.wav")
                audioObj, sampleRate, bitrate = util.readAudioScipy(song)
                
                assert sampleRate == 44100,"Sample rate needs to be 44100"

                audio = (audioObj[:,0] + audioObj[:,1])/2
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
                mm=util.overlapadd_multi(output,batches,nchunks,overlap=train.overlap)

                #write audio files
                if f in dev_directory:
                    dirout=os.path.join(outdir,f,"Dev")
                else:
                    dirout=os.path.join(outdir,f,"Test")
                for i in range(mm.shape[0]):
                    audio_out=transform.compute_inverse(mm[i,:len(ph)]/scale_factor,ph)
                    if len(audio_out)>len(audio):
                        audio_out=audio_out[:len(audio)]
                    util.writeAudioScipy(os.path.join(dirout,source[i]),audio_out,sampleRate,bitrate)
                    audio_out=None 
                audio = None

    return losser  




if __name__ == "__main__": 
    """
    Separating Professionally Produced Music
    https://sisec.inria.fr/home/2016-professionally-produced-music-recordings/

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
        else:
            db='/home/marius/Documents/Database/DSD100/'  
        if kwargs.__getattribute__('feature_path'):
            feature_path = kwargs.__getattribute__('feature_path')
        else:
            feature_path=os.path.join(db,'transforms','t1') 
        assert os.path.isdir(db), "Please input the directory for the DSD100 dataset with --db path_to_iKala"  
        if kwargs.__getattribute__('model'):
            model = kwargs.__getattribute__('model')
        else:
            model="dsd_fft_1024"    
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
            overlap = 25
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

    ld1 = LargeDataset(path_transform_in=feature_path, nsources=4, batch_size=batch_size, batch_memory=batch_memory, time_context=time_context, overlap=overlap, nprocs=nprocs,mult_factor_in=scale_factor,mult_factor_out=scale_factor)
    logging.info("  Maximum:\t\t{:.6f}".format(ld1.getMax()))
    logging.info("  Mean:\t\t{:.6f}".format(ld1.getMean()))
    logging.info("  Standard dev:\t\t{:.6f}".format(ld1.getStd()))

    if not os.path.exists(db+'output/'):
        os.makedirs(db+'output/')
    if not os.path.exists(db+'output/'+model):
        os.makedirs(db+'output/'+model)
    if not os.path.exists(db+'models/'):
        os.makedirs(db+'models/')

    train_errs=train_auto(train=ld1,fun=build_ca,transform=tt,outdir=db+'output/'+model+"/",testdir=db+'Mixtures/',model=db+"models/"+"model_"+model+".pkl",num_epochs=nepochs,scale_factor=scale_factor)      
    f = file(db+"models/"+"loss_"+model+".data", 'wb')
    cPickle.dump(train_errs,f,protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()

