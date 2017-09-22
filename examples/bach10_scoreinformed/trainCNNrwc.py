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
from transform import transformFFT
import dataset
from dataset import LargeDatasetMask1,LargeDatasetMask2
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
import multiprocessing

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

def build_ca_1x1(input_var=None, batch_size=32,time_context=30,feat_size=1025,nchannels=4):

    input_shape=(batch_size,nchannels,time_context,feat_size)
    #scaled_tanh = lasagne.nonlinearities.ScaledTanH(scale_in=1, scale_out=0.5)
    nfilt_conv = 200

    l_in_1 = lasagne.layers.InputLayer(shape=input_shape, input_var=input_var)

    l_conv1 = lasagne.layers.Conv2DLayer(l_in_1, num_filters=30, filter_size=(1,5),stride=(1,2), pad='valid', nonlinearity=lasagne.nonlinearities.rectify)
    l_conv1b= lasagne.layers.BiasLayer(l_conv1)

    l_conv2 = lasagne.layers.Conv2DLayer(l_conv1b, num_filters=50, filter_size=(1,5),stride=(1,2), pad='valid', nonlinearity=lasagne.nonlinearities.rectify)
    l_conv2b= lasagne.layers.BiasLayer(l_conv2)

    l_conv3 = lasagne.layers.Conv2DLayer(l_conv2b, num_filters=70, filter_size=(1,5),stride=(1,2), pad='valid', nonlinearity=lasagne.nonlinearities.rectify)
    l_conv3b= lasagne.layers.BiasLayer(l_conv3)

    l_conv4 = lasagne.layers.Conv2DLayer(l_conv3b, num_filters=100, filter_size=(1,5),stride=(1,2), pad='valid', nonlinearity=lasagne.nonlinearities.rectify)
    l_conv4b= lasagne.layers.BiasLayer(l_conv4)

    l_conv5 = lasagne.layers.Conv2DLayer(l_conv4b, num_filters=200, filter_size=(10,5),stride=(1,2), pad='valid', nonlinearity=lasagne.nonlinearities.rectify)
    l_conv5b= lasagne.layers.BiasLayer(l_conv5)

    l_conv6 = lasagne.layers.Conv2DLayer(l_conv5b, num_filters=nfilt_conv, filter_size=(10,5),stride=(1,2), pad='valid', nonlinearity=lasagne.nonlinearities.rectify)
    l_conv6b= lasagne.layers.BiasLayer(l_conv6)

    l_conv = lasagne.layers.Conv2DLayer(l_conv6b, num_filters=nfilt_conv*4, filter_size=(1,1),stride=(1,1), pad='valid', nonlinearity=lasagne.nonlinearities.rectify)
    l_convb= lasagne.layers.BiasLayer(l_conv)

    l_source1 = lasagne.layers.SliceLayer(l_convb, indices=slice(0, nfilt_conv), axis=1)
    l_source2 = lasagne.layers.SliceLayer(l_convb, indices=slice(nfilt_conv, 2*nfilt_conv), axis=1)
    l_source3 = lasagne.layers.SliceLayer(l_convb, indices=slice(2*nfilt_conv, 3*nfilt_conv), axis=1)
    l_source4 = lasagne.layers.SliceLayer(l_convb, indices=slice(3*nfilt_conv, 4*nfilt_conv), axis=1)

    l_inverse11=lasagne.layers.InverseLayer(l_source1, l_conv6)
    l_inverse21=lasagne.layers.InverseLayer(l_inverse11, l_conv5)
    l_inverse31=lasagne.layers.InverseLayer(l_inverse21, l_conv4)
    l_inverse41=lasagne.layers.InverseLayer(l_inverse31, l_conv3)
    l_inverse51=lasagne.layers.InverseLayer(l_inverse41, l_conv2)
    l_inverse61=lasagne.layers.InverseLayer(l_inverse51, l_conv1)

    l_inverse12=lasagne.layers.InverseLayer(l_source2, l_conv6)
    l_inverse22=lasagne.layers.InverseLayer(l_inverse12, l_conv5)
    l_inverse32=lasagne.layers.InverseLayer(l_inverse22, l_conv4)
    l_inverse42=lasagne.layers.InverseLayer(l_inverse32, l_conv3)
    l_inverse52=lasagne.layers.InverseLayer(l_inverse42, l_conv2)
    l_inverse62=lasagne.layers.InverseLayer(l_inverse52, l_conv1)

    l_inverse13=lasagne.layers.InverseLayer(l_source3, l_conv6)
    l_inverse23=lasagne.layers.InverseLayer(l_inverse13, l_conv5)
    l_inverse33=lasagne.layers.InverseLayer(l_inverse23, l_conv4)
    l_inverse43=lasagne.layers.InverseLayer(l_inverse33, l_conv3)
    l_inverse53=lasagne.layers.InverseLayer(l_inverse43, l_conv2)
    l_inverse63=lasagne.layers.InverseLayer(l_inverse53, l_conv1)

    l_inverse14=lasagne.layers.InverseLayer(l_source4, l_conv6)
    l_inverse24=lasagne.layers.InverseLayer(l_inverse14, l_conv5)
    l_inverse34=lasagne.layers.InverseLayer(l_inverse24, l_conv4)
    l_inverse44=lasagne.layers.InverseLayer(l_inverse34, l_conv3)
    l_inverse54=lasagne.layers.InverseLayer(l_inverse44, l_conv2)
    l_inverse64=lasagne.layers.InverseLayer(l_inverse54, l_conv1)

    l_merge=lasagne.layers.ConcatLayer([l_inverse61,l_inverse62,l_inverse63,l_inverse64],axis=1)

    l_out = lasagne.layers.NonlinearityLayer(lasagne.layers.BiasLayer(l_merge), nonlinearity=lasagne.nonlinearities.rectify)

    return l_out

def build_ca(input_var=None, batch_size=32,time_context=30,feat_size=513,nchannels=4):
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

    input_shape=(batch_size,nchannels,time_context,feat_size)
    #scaled_tanh = lasagne.nonlinearities.ScaledTanH(scale_in=1, scale_out=0.5)

    l_in_1 = lasagne.layers.InputLayer(shape=input_shape, input_var=input_var)

    fsize = int(feat_size/6)

    l_conv1 = lasagne.layers.Conv2DLayer(l_in_1, num_filters=30, filter_size=(1,30),stride=(1,4), pad='valid', nonlinearity=None)
    l_conv1b= lasagne.layers.BiasLayer(l_conv1)

    l_conv2 = lasagne.layers.Conv2DLayer(l_conv1b, num_filters=30, filter_size=(int(2*time_context/3),1),stride=(1,1), pad='valid', nonlinearity=None)
    l_conv2b= lasagne.layers.BiasLayer(l_conv2)

    l_fc=lasagne.layers.DenseLayer(l_conv2b,256)

    l_fc11=lasagne.layers.DenseLayer(l_fc,l_conv2.output_shape[1]*l_conv2.output_shape[2]*l_conv2.output_shape[3])
    l_reshape1 = lasagne.layers.ReshapeLayer(l_fc11,(batch_size,l_conv2.output_shape[1],l_conv2.output_shape[2], l_conv2.output_shape[3]))
    l_inverse11=lasagne.layers.InverseLayer(l_reshape1, l_conv2)
    l_inverse41=lasagne.layers.InverseLayer(l_inverse11, l_conv1)

    l_fc12=lasagne.layers.DenseLayer(l_fc,l_conv2.output_shape[1]*l_conv2.output_shape[2]*l_conv2.output_shape[3])
    l_reshape2 = lasagne.layers.ReshapeLayer(l_fc12,(batch_size,l_conv2.output_shape[1],l_conv2.output_shape[2], l_conv2.output_shape[3]))
    l_inverse12=lasagne.layers.InverseLayer(l_reshape2, l_conv2)
    l_inverse42=lasagne.layers.InverseLayer(l_inverse12, l_conv1)

    l_fc13=lasagne.layers.DenseLayer(l_fc,l_conv2.output_shape[1]*l_conv2.output_shape[2]*l_conv2.output_shape[3])
    l_reshape3 = lasagne.layers.ReshapeLayer(l_fc13,(batch_size,l_conv2.output_shape[1],l_conv2.output_shape[2], l_conv2.output_shape[3]))
    l_inverse13=lasagne.layers.InverseLayer(l_reshape3, l_conv2)
    l_inverse43=lasagne.layers.InverseLayer(l_inverse13, l_conv1)

    l_fc14=lasagne.layers.DenseLayer(l_fc,l_conv2.output_shape[1]*l_conv2.output_shape[2]*l_conv2.output_shape[3])
    l_reshape4 = lasagne.layers.ReshapeLayer(l_fc14,(batch_size,l_conv2.output_shape[1],l_conv2.output_shape[2], l_conv2.output_shape[3]))
    l_inverse14=lasagne.layers.InverseLayer(l_reshape4, l_conv2)
    l_inverse44=lasagne.layers.InverseLayer(l_inverse14, l_conv1)

    l_merge=lasagne.layers.ConcatLayer([l_inverse41,l_inverse42,l_inverse43,l_inverse44],axis=1)

    l_out = lasagne.layers.NonlinearityLayer(lasagne.layers.BiasLayer(l_merge), nonlinearity=lasagne.nonlinearities.rectify)

    return l_out


def train_auto(train,fun,transform,testdir,outdir,testfile_list,testdir1,outdir1,testfile_list1,num_epochs=30,model="1.pkl",scale_factor=0.3,load=False,skip_train=False,skip_sep=False):
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

    #parameters for the score-informed separation
    nharmonics=20
    interval=50 #cents
    tuning_freq=440 #Hz

    eps=1e-18
    alpha=0.001

    network2 = fun(input_var=input_var2,batch_size=train.batch_size,time_context=train.time_context,feat_size=train.input_size,nchannels=4)

    if load:
        params=load_model(model)
        lasagne.layers.set_all_param_values(network2,params)

    prediction2 = lasagne.layers.get_output(network2, deterministic=True)

    rand_num = np.random.uniform(size=(train.batch_size,1,train.time_context,train.input_size))

    s1=prediction2[:,0:1,:,:]
    s2=prediction2[:,1:2,:,:]
    s3=prediction2[:,2:3,:,:]
    s4=prediction2[:,3:4,:,:]

    mask1=s1/(s1+s2+s3+s4+eps*rand_num)
    mask2=s2/(s1+s2+s3+s4+eps*rand_num)
    mask3=s3/(s1+s2+s3+s4+eps*rand_num)
    mask4=s4/(s1+s2+s3+s4+eps*rand_num)

    input_var = input_var2[:,0:1,:,:] + input_var2[:,1:2,:,:] + input_var2[:,2:3,:,:] + input_var2[:,3:4,:,:]

    source1=mask1*input_var[:,0:1,:,:]
    source2=mask2*input_var[:,0:1,:,:]
    source3=mask3*input_var[:,0:1,:,:]
    source4=mask4*input_var[:,0:1,:,:]

    train_loss_recon1 = lasagne.objectives.squared_error(source1,target_var2[:,0:1,:,:])
    train_loss_recon2 = lasagne.objectives.squared_error(source2,target_var2[:,1:2,:,:])
    train_loss_recon3 = lasagne.objectives.squared_error(source3,target_var2[:,2:3,:,:])
    train_loss_recon4 = lasagne.objectives.squared_error(source4,target_var2[:,3:4,:,:])

    error1=train_loss_recon1.sum()
    error2=train_loss_recon2.sum()
    error3=train_loss_recon3.sum()
    error4=train_loss_recon4.sum()

    loss=abs(error1+error2+error3+error4)

    params1 = lasagne.layers.get_all_params(network2, trainable=True)

    updates = lasagne.updates.adadelta(loss, params1)

    train_fn = theano.function([input_var2,target_var2], loss, updates=updates,allow_input_downcast=True)

    train_fn1 = theano.function([input_var2,target_var2], [error1,error2,error3,error4], allow_input_downcast=True)

    predict_function2=theano.function([input_var2],[source1,source2,source3,source4],allow_input_downcast=True)

    losser=[]
    min_loss = 1e14

    training_steps = 0

    while not skip_train and training_steps<2:

        logging.info("Training...")
        for epoch in range(num_epochs):

            train_err = 0
            train_batches = 0
            err1=0
            err2=0
            err3=0
            err4=0
            start_time = time.time()
            for batch in range(train.iteration_size):

                inputs, target, masks = train()
                jump = inputs.shape[2]

                mask=np.empty(shape=(inputs.shape[0],4,inputs.shape[1],inputs.shape[2]),dtype=theano.config.floatX)
                mask[:,0,:,:]=masks[:,:,:jump] * inputs
                mask[:,1,:,:]=masks[:,:,jump:jump*2] * inputs
                mask[:,2,:,:]=masks[:,:,jump*2:jump*3] * inputs
                mask[:,3,:,:]=masks[:,:,jump*3:jump*4] * inputs
                masks=None

                targets=np.empty(shape=(inputs.shape[0],4,inputs.shape[1],inputs.shape[2]),dtype=theano.config.floatX)
                targets[:,0,:,:]=target[:,:,:jump]
                targets[:,1,:,:]=target[:,:,jump:jump*2]
                targets[:,2,:,:]=target[:,:,jump*2:jump*3]
                targets[:,3,:,:]=target[:,:,jump*3:jump*4]
                target=None

                inputs=None

                train_err+=train_fn(mask,targets)
                [e1,e2,e3,e4]=train_fn1(mask,targets)
                err1 += e1
                err2 += e2
                err3 += e3
                err4 += e4
                train_batches += 1

            logging.info("Epoch {} of {} took {:.3f}s".format(
                epoch + 1, num_epochs, time.time() - start_time))
            logging.info("  training loss:\t\t{:.6f}".format(train_err/train_batches))
            logging.info("  training loss for bassoon:\t\t{:.6f}".format(err1/train_batches))
            logging.info("  training loss for clarinet:\t\t{:.6f}".format(err2/train_batches))
            logging.info("  training loss for saxophone:\t\t{:.6f}".format(err3/train_batches))
            logging.info("  training loss for violin:\t\t{:.6f}".format(err4/train_batches))
            losser.append(train_err / train_batches)
            #save_model(model,network2)
            if (train_err/train_batches) < min_loss:
                min_loss = train_err/train_batches
                save_model(model,network2)

        training_steps = training_steps + 1
        num_epochs = int(np.ceil(float(num_epochs)/5.))

        if losser[-1] > min_loss:
            params=load_model(model)
            lasagne.layers.set_all_param_values(network2,params,learning_rate=0.0001)

        updates = lasagne.updates.adam(loss, params1)
        train_fn = theano.function([input_var2,target_var2], loss, updates=updates,allow_input_downcast=True)


    if not skip_sep:

        logging.info("Separating")
        sources = ['bassoon','clarinet','saxphone','violin']
        sources_midi = ['bassoon','clarinet','saxophone','violin']

        for f in testfile_list:
            nelem_g=1
            for i in range(len(sources)):
                ng = util.getMidiNum(sources_midi[i]+'_b',os.path.join(testdir,f),0,40.0)
                nelem_g = np.maximum(ng,nelem_g)
            melody = np.zeros((len(sources),int(nelem_g),2*nharmonics+3))
            for i in range(len(sources)):
                filename=os.path.join(testdir,f,f+'-'+sources[i]+'.wav')
                audioObj, sampleRate, bitrate = util.readAudioScipy(filename)

                assert sampleRate == 44100,"Sample rate needs to be 44100"

                nframes = int(np.ceil(len(audioObj) / np.double(tt.hopSize))) + 2
                if i==0:
                    audio = np.zeros(audioObj.shape[0])
                audio = audio + audioObj
                audioObj=None

                tmp = util.expandMidi(sources_midi[i]+'_b',os.path.join(testdir,f),0,40.0,interval,tuning_freq,nharmonics,sampleRate,tt.hopSize,tt.frameSize,0.2,0.2,nframes,0.5)
                melody[i,:tmp.shape[0],:] = tmp
                tmp = None

            mag,ph=transform.compute_file(audio,phase=True)
            mag=scale_factor*mag.astype(np.float32)

            jump = mag.shape[-1]

            masks_temp = train.filterSpec(mag,melody,0,nframes)
            masks = np.ones((train.ninst,mag.shape[0],mag.shape[1]))
            masks[0,:,:]=masks_temp[:,:jump] * mag
            masks[1,:,:]=masks_temp[:,jump:jump*2] * mag
            masks[2,:,:]=masks_temp[:,jump*2:jump*3] * mag
            masks[3,:,:]=masks_temp[:,jump*3:jump*4] * mag
            mag = None
            masks_temp = None

            batches,nchunks = util.generate_overlapadd(masks,input_size=masks.shape[-1],time_context=train.time_context,overlap=train.overlap,batch_size=train.batch_size,sampleRate=44100)
            masks = None

            batch_no=1
            output=[]
            for batch in batches:
                batch_no+=1
                #start_time=time.time()
                output.append(predict_function2(batch))

            output=np.array(output)
            mm=util.overlapadd_multi(output,batches,nchunks,overlap=train.overlap)
            for i in range(len(sources)):
                audio_out=transform.compute_inverse(mm[i,:len(ph)]/scale_factor,ph)
                if len(audio_out)>len(audio):
                    audio_out=audio_out[:len(audio)]
                util.writeAudioScipy(os.path.join(outdir,f+'-'+sources[i]+'.wav'),audio_out,sampleRate,bitrate)
                audio_out=None

        # style = ['fast','slow','original']
        # style_midi = ['_fast20','_slow20','_original']
        # if not os.path.exists(outdir1):
        #     os.makedirs(outdir1)
        # for s in range(len(style)):
        #     for f in testfile_list1:
        #         nelem_g=1
        #         for i in range(len(sources)):
        #             ng = util.getMidiNum(sources_midi[i]+'_g'+style_midi[s],os.path.join(testdir1,f),0,40.0)
        #             nelem_g = np.maximum(ng,nelem_g)
        #         melody = np.zeros((len(sources),int(nelem_g),2*nharmonics+3))
        #         for i in range(len(sources)):
        #             filename=os.path.join(testdir1,f,f+'_'+style[s]+'_'+sources_midi[i]+'.wav')

        #             audioObj, sampleRate, bitrate = util.readAudioScipy(filename)

        #             assert sampleRate == 44100,"Sample rate needs to be 44100"

        #             nframes = int(np.ceil(len(audioObj) / np.double(tt.hopSize))) + 2

        #             if i==0:
        #                 audio = np.zeros(audioObj.shape[0])

        #             audio = audio + audioObj
        #             audioObj=None

        #             tmp = util.expandMidi(sources_midi[i]+'_g'+style_midi[s],os.path.join(testdir1,f),0,40.0,interval,tuning_freq,nharmonics,sampleRate,tt.hopSize,tt.frameSize,0.2,0.2,nframes)
        #             melody[i,:tmp.shape[0],:] = tmp
        #             tmp = None

        #         mag,ph=transform.compute_file(audio,phase=True)
        #         mag=scale_factor*mag.astype(np.float32)

        #         jump = mag.shape[-1]

        #         masks_temp = train.filterSpec(mag,melody,0,nframes)
        #         masks = np.ones((train.ninst,mag.shape[0],mag.shape[1]))
        #         masks[0,:,:]=masks_temp[:,:jump] * mag
        #         masks[1,:,:]=masks_temp[:,jump:jump*2] * mag
        #         masks[2,:,:]=masks_temp[:,jump*2:jump*3] * mag
        #         masks[3,:,:]=masks_temp[:,jump*3:jump*4] * mag
        #         mag = None
        #         masks_temp = None

        #         batches,nchunks = util.generate_overlapadd(masks,input_size=masks.shape[-1],time_context=train.time_context,overlap=train.overlap,batch_size=train.batch_size,sampleRate=44100)
        #         masks = None

        #         batch_no=1
        #         output=[]
        #         for batch in batches:
        #             batch_no+=1
        #             #start_time=time.time()
        #             output.append(predict_function2(batch))

        #         output=np.array(output)
        #         mm=util.overlapadd_multi(output,batches,nchunks,overlap=train.overlap)
        #         for i in range(len(sources)):
        #             audio_out=transform.compute_inverse(mm[i,:len(ph)]/scale_factor,ph)
        #             if len(audio_out)>len(audio):
        #                 audio_out=audio_out[:len(audio)]
        #             filename=os.path.join(outdir1,f+'_'+style[s]+'_'+sources_midi[i]+'.wav')
        #             util.writeAudioScipy(filename,audio_out,sampleRate,bitrate)
        #             audio_out=None

    return losser




if __name__ == "__main__":
    """
    Separating Bach10 chorales using the synthesized version with Sibelius
    http://music.cs.northwestern.edu/data/Bach10.html

    More details in the following article:
    Marius Miron, Jordi Janer, Emilia Gomez, "Generating data to train convolutional neural networks for low latency classical music source separation", Sound and Music Computing Conference 2017 (submitted)

    Given the features computed previusly with compute_features_bach10sibelius, train a network and perform the separation.

    Parameters
    ----------
    db : string
        The path to the Bach10 dataset
    dbs : string
        The path to the Bach10 Sibelius dataset
    feature_path : string
        The path where to load the features from
    output : string
        The path where to save the output nepochs : int, optional
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
        climate.add_arg('--db', help="the Bach10 dataset path")
        climate.add_arg('--dbs', help="the Bach10 Sibelius dataset path")
        climate.add_arg('--output', help="the path where to save the model and the output")
        climate.add_arg('--model', help="the name of the model to test/save")
        climate.add_arg('--nepochs', help="number of epochs to train the net")
        climate.add_arg('--time_context', help="number of frames for the recurrent/lstm/conv net")
        climate.add_arg('--batch_size', help="batch size for training")
        climate.add_arg('--batch_memory', help="number of big batches to load into memory")
        climate.add_arg('--overlap', help="overlap time context for training")
        climate.add_arg('--nprocs', help="number of processor to parallelize file reading")
        climate.add_arg('--scale_factor', help="scale factor for the data")
        climate.add_arg('--feature_path', help="the path where to load the features from")
        climate.add_arg('--scale_factor_test', help="scale factor for the test data")
        climate.add_arg('--nsamples', help="max number of files to train on")
        climate.add_arg('--gt', help="compute features for the ground truth aligned rendition or the others")
        climate.add_arg('--load', help="load external model")
        climate.add_arg('--skip', help="skip training")
        climate.add_arg('--function', help="build function for the neural network; default build_ca")
        climate.add_arg('--timbre_model_path', help="the path to the timbre model pkl")
        db=None
        kwargs = climate.parse_args()
        if kwargs.__getattribute__('db'):
            db = kwargs.__getattribute__('db')
        else:
            db='/home/marius/Documents/Database/Bach10/Sources/'
            # db='/Volumes/Macintosh HD 2/Documents/Database/Bach10/Sources/'
        if kwargs.__getattribute__('dbs'):
            dbs = kwargs.__getattribute__('dbs')
        else:
            dbs='/home/marius/Documents/Database/Bach10/Source separation/'
            # dbs='/Volumes/Macintosh HD 2/Documents/Database/Bach10/Source separation/'
        if kwargs.__getattribute__('output'):
            output = kwargs.__getattribute__('output')
        else:
            output='/home/marius/Documents/Database/Bach10/'
            # output='/Volumes/Macintosh HD 2/Documents/Database/Bach10/'
        if kwargs.__getattribute__('feature_path'):
            feature_path = kwargs.__getattribute__('feature_path')
        else:
            feature_path=os.path.join(dbs,'transforms','t3_rwc')
        assert os.path.isdir(db), "Please input the directory for the Bach10 dataset with --db path_to_Bach10"
        assert os.path.isdir(dbs), "Please input the directory for the Bach10 Sibelius dataset with --dbs path_to_Bach10Sibelius"
        assert os.path.isdir(feature_path), "Please input the directory where you stored the training features --feature_path path_to_features"
        assert os.path.isdir(output), "Please input the output directory --output path_to_output"
        if kwargs.__getattribute__('model'):
            model = kwargs.__getattribute__('model')
        else:
            model="CNNrwc_se1"
        if kwargs.__getattribute__('batch_size'):
            batch_size = int(kwargs.__getattribute__('batch_size'))
        else:
            batch_size = 32
        if kwargs.__getattribute__('batch_memory'):
            batch_memory = int(kwargs.__getattribute__('batch_memory'))
        else:
            batch_memory = 100
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
            nprocs = multiprocessing.cpu_count()-1
        if kwargs.__getattribute__('nepochs'):
            nepochs = int(kwargs.__getattribute__('nepochs'))
        else:
            nepochs = 40
        if kwargs.__getattribute__('scale_factor'):
            scale_factor = int(kwargs.__getattribute__('scale_factor'))
        else:
            scale_factor = 0.3
        if kwargs.__getattribute__('scale_factor_test'):
            scale_factor_test = int(kwargs.__getattribute__('scale_factor_test'))
        else:
            scale_factor_test = 0.2
        if kwargs.__getattribute__('gt'):
            gt = int(kwargs.__getattribute__('gt'))
        else:
            gt = True
        if kwargs.__getattribute__('load'):
            load = int(kwargs.__getattribute__('load'))
        else:
            load = False
        if kwargs.__getattribute__('skip'):
            skip = int(kwargs.__getattribute__('skip'))
        else:
            skip = False
        if kwargs.__getattribute__('nsamples'):
            nsamples = int(kwargs.__getattribute__('nsamples'))
        else:
            nsamples = 0
        if kwargs.__getattribute__('timbre_model_path'):
            timbre_model_path = kwargs.__getattribute__('timbre_model_path')
        else:
            timbre_model_path = None
            #timbre_model_path = '/home/marius/Documents/Database/Bach10/harmonics.pkl'
        if kwargs.__getattribute__('function'):
            function = kwargs.__getattribute__('function')
        else:
            function = 'build_ca'
        funcs = {'build_ca_1x1': build_ca_1x1, 'build_ca': build_ca}
        if function not in funcs:
            function = 'build_ca'

    if gt:
        style = ['gt']
    else:
        style = ['original']
    if function=='build_ca_1x1':
        model = model + '_x'
    model = model + '_' + style[0]
    if timbre_model_path is not None:
        model = model + '_tm'

    path_in = []
    testfile_list = []

    for f in sorted(os.listdir(db)):
        if os.path.isdir(os.path.join(db,f)) and f[0].isdigit():
            testfile_list.append(f)
            for s in style:
                if os.path.exists(os.path.join(feature_path,f,s)):
                    path_in.append(os.path.join(feature_path,f,s))


    #tt object needs to be the same as the one in compute_features
    tt=transformFFT(frameSize=4096, hopSize=512, sampleRate=44100, window=blackmanharris)

    ld1 = LargeDatasetMask2(path_transform_in=path_in, nsources=4, nsamples=nsamples, batch_size=batch_size, batch_memory=batch_memory, time_context=time_context, overlap=overlap, nprocs=nprocs,mult_factor_in=scale_factor,mult_factor_out=scale_factor,\
        sampleRate=tt.sampleRate,pitch_code='e', nharmonics=20, pitch_norm=127.,tensortype=theano.config.floatX,timbre_model_path=timbre_model_path)
    logging.info("  Maximum:\t\t{:.6f}".format(ld1.getMax()))
    logging.info("  Mean:\t\t{:.6f}".format(ld1.getMean()))
    logging.info("  Standard dev:\t\t{:.6f}".format(ld1.getStd()))

    if not os.path.exists(os.path.join(output,'output',model)):
        os.makedirs(os.path.join(output,'output',model))
    if not os.path.exists(os.path.join(output,'models')):
        os.makedirs(os.path.join(output,'models'))
    if not os.path.exists(os.path.join(output,'output',model+"_original")):
        os.makedirs(os.path.join(output,'output',model+"_original"))

    train_errs=train_auto(train=ld1,fun=funcs[function],transform=tt,outdir=os.path.join(output,'output',model),testdir=db,testfile_list=testfile_list,\
        outdir1=os.path.join(output,'output',model+"_original"),testdir1=dbs,testfile_list1=testfile_list,
        model=os.path.join(output,"models","model_"+model+".pkl"),num_epochs=nepochs,scale_factor=scale_factor_test,load=load,skip_train=skip)
    f = file(os.path.join(output,"models","loss_"+model+".data"), 'wb')
    cPickle.dump(train_errs,f,protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()

