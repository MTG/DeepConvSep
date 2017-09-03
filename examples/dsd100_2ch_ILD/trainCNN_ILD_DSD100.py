"""
    This file is part of DeepConvSep.

    Copyright (c) 2014-2017 Marius Miron  <miron.marius at gmail.com>
    Copyright (c) 2017 Gerard Erruz  <gerard.erruz at upf.edu>

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

import transform
from transform import transformFFT
import dataset
from dataset import LargeDatasetMulti
import util

import os,sys
import numpy as np
import scipy
from scipy.signal import hanning as hanning
import time
import cPickle
import climate

import theano
import theano.tensor as T
import theano.sandbox.rng_mrg
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

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


def build_ca(input_var=None, batch_size=32,time_context=15,feat_size=513,nchannels=2,nsources=4):
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
    net = {}
    net['l_in_1'] = lasagne.layers.InputLayer(shape=input_shape, input_var=input_var)

    net['l_conv1'] = lasagne.layers.Conv2DLayer(net['l_in_1'], num_filters=50, filter_size=(1,feat_size),stride=(1,1), pad='valid', nonlinearity=None)
    net['l_conv1b']= lasagne.layers.BiasLayer(net['l_conv1'])

    net['l_conv2'] = lasagne.layers.Conv2DLayer(net['l_conv1b'], num_filters=50, filter_size=(int(time_context/2),1),stride=(1,1), pad='valid', nonlinearity=None)
    net['l_conv2b']= lasagne.layers.BiasLayer(net['l_conv2'])

    net['l_fc']=lasagne.layers.DenseLayer(net['l_conv2b'],256)

    net['ls'] = []
    for s in range(nsources):
        net['l_fc'+str(s)]=lasagne.layers.DenseLayer(net['l_fc'],net['l_conv2'].output_shape[1]*net['l_conv2'].output_shape[2]*net['l_conv2'].output_shape[3])
        net['l_reshape'+str(s)] = lasagne.layers.ReshapeLayer(net['l_fc'+str(s)],(batch_size,net['l_conv2'].output_shape[1],net['l_conv2'].output_shape[2], net['l_conv2'].output_shape[3]))
        net['l_inverse1'+str(s)]=lasagne.layers.InverseLayer(net['l_reshape'+str(s)], net['l_conv2'])
        net['l_inverse2'+str(s)]=lasagne.layers.InverseLayer(net['l_inverse1'+str(s)], net['l_conv1'])
        net['ls'].append(net['l_inverse2'+str(s)])

    net['l_merge']=lasagne.layers.ConcatLayer(net['ls'],axis=1)
    # example for 2 sources in 2 channels:
    # 0, 1 source 0 in channel 0 and 1
    # 2, 3 source 1 in channel 0 and 1

    net['l_out'] = lasagne.layers.NonlinearityLayer(lasagne.layers.BiasLayer(net['l_merge']), nonlinearity=lasagne.nonlinearities.rectify)

    return net



def train_auto(fun,train,transform,testdir,outdir,num_epochs=30,model="1.pkl",scale_factor=0.3,load=False,skip_train=False,skip_sep=False, chunk_size=60,chunk_overlap=2,
    nsamples=40,batch_size=32, batch_memory=50, time_context=30, overlap=25, nprocs=4,mult_factor_in=0.3,mult_factor_out=0.3):
    """
    Trains a network built with \"fun\" with the data generated with \"train\"
    and then separates the files in \"testdir\",writing the result in \"outdir\"

    Parameters
    ----------
    fun : lasagne network object, Theano tensor
        The network to be trained
    transform : transformFFT object
        The Transform object which was used to compute the features (see compute_features_DSD100.py)
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
    input_var = T.tensor4('inputs')
    input_mask = T.tensor4('input_mask')
    target_var = T.tensor4('targets')

    theano_rng = RandomStreams(128)

    eps=1e-12

    sources = ['vocals','bass','drums','other']

    nchannels = int(train.channels_in)
    nsources = int(train.channels_out/train.channels_in)

    print 'nchannels: ', nchannels
    print 'nsources: ', nsources

    input_size = int(float(transform.frameSize) / 2 + 1)

    rand_num = theano_rng.normal( size = (batch_size,nsources,time_context,input_size), avg = 0.0, std = 0.1, dtype=theano.config.floatX)

    net = fun(input_var=input_var,batch_size=batch_size,time_context=time_context,feat_size=input_size,nchannels=nchannels,nsources=nsources)
    network = net['l_out']
    if load:
        params=load_model(model)
        lasagne.layers.set_all_param_values(network,params)

    prediction = lasagne.layers.get_output(network, deterministic=True)

    sourceall=[]
    errors_insts = []
    loss = 0

    sep_chann = []

    # prediction example for 2 sources in 2 channels:
    # 0, 1 source 0 in channel 0 and 1
    # 2, 3 source 1 in channel 0 and 1
    for j in range(nchannels):
        #print "j: ", j
        masksum = T.sum(prediction[:,j::nchannels,:,:],axis=1)
        temp = T.tile(masksum.dimshuffle(0,'x', 1,2),(1,nsources,1,1))
        mask = prediction[:,j::nchannels,:,:] / (temp + eps*rand_num)
        source=mask*T.tile(input_var[:,j:j+1,:,:],(1,nsources,1,1)) + eps*rand_num
        sourceall.append(source)

        sep_chann.append(source)
        train_loss_recon = lasagne.objectives.squared_error(source,target_var[:,j::nchannels,:,:])

        errors_inst=abs(train_loss_recon.sum(axis=(0,2,3)))

        errors_insts.append(errors_inst)

        loss=loss+abs(train_loss_recon.sum())

    params1 = lasagne.layers.get_all_params(network, trainable=True)

    updates = lasagne.updates.adadelta(loss, params1)

    train_fn_mse = theano.function([input_var,target_var], loss, updates=updates,allow_input_downcast=True)

    train_fn1 = theano.function([input_var,target_var], errors_insts, allow_input_downcast=True)

    #----------NEW ILD LOSS CONDITION----------

    rand_num2 = theano_rng.normal( size = (batch_size,nsources,time_context,input_size), avg = 0.0, std = 0.1, dtype=theano.config.floatX) #nsources a primera dim?

    #estimate

    interaural_spec_est = sep_chann[0] / (sep_chann[1] + eps*rand_num2)

    alpha_est = 20*np.log10(abs(interaural_spec_est + eps*rand_num2))
    alpha_est_mean = alpha_est.mean(axis=(0,1,2))

    #groundtruth

    interaural_spec_gt = target_var[:,0::nchannels,:,:] / (target_var[:,1::nchannels,:,:] + eps*rand_num2)

    alpha_gt = 20*np.log10(abs(interaural_spec_gt + eps*rand_num2))
    alpha_gt_mean = alpha_gt.mean(axis=(0,1,2)) #aixo hauria de ser un vector d'una dimensio

    train_loss_ild = lasagne.objectives.squared_error(alpha_est_mean,alpha_gt_mean)

    loss = loss + (abs(train_loss_ild.sum())/500)

    #------------------------------------------

    predict_function=theano.function([input_var],sourceall,allow_input_downcast=True)

    losser=[]

    if not skip_train:
        logging.info("Training stage 1 (mse)...")
        for epoch in range(num_epochs):

            train_err = 0
            train_batches = 0
            errs=np.zeros((nchannels,nsources))
            start_time = time.time()
            for batch in range(train.iteration_size):
                inputs, target = train()
                train_err+=train_fn_mse(inputs, target)
                errs+=np.array(train_fn1(inputs, target))
                train_batches += 1

            logging.info("Epoch {} of {} took {:.3f}s".format(
                epoch + 1, num_epochs, time.time() - start_time))
            logging.info("  training loss:\t\t{:.6f}".format(train_err/train_batches))
            for j in range(nchannels):
                for i in range(nsources):
                    logging.info("  training loss for "+sources[i]+" in mic "+str(j)+":\t\t{:.6f}".format(errs[j][i]/train_batches))

            model_noILD = model[:-4] + '_noILD' + model[-4:]
            print 'model_noILD: ', model_noILD
            save_model(model_noILD,network)
            losser.append(train_err/train_batches)

#NEW ILD TRAINING---------------------------------------------------------

        params=load_model(model_noILD)
        lasagne.layers.set_all_param_values(network,params)
        params1 = lasagne.layers.get_all_params(network,trainable=True)
        updates = lasagne.updates.adadelta(loss,params1)
        train_fn_ILD = theano.function([input_var,target_var],loss,updates=updates,allow_input_downcast=True)

        logging.info("Training stage 2 (ILD)...")

        for epoch in range(int(num_epochs/2)):

            train_err = 0
            train_batches = 0

            start_time = time.time()
            for batch in range(train.iteration_size):
                inputs,target = train()

                train_err+=train_fn_ILD(inputs,target)
                train_batches+=1

            logging.info("Epoch {} of {} took {:.3f}s".format(
                epoch + 1, num_epochs, time.time() - start_time))
            logging.info("  training loss:\t\t{:.6f}".format(train_err/train_batches))

            save_model(model,network)
            losser.append(train_err/train_batches)

    if not skip_sep:

        logging.info("Separating")

        subsets = ['Dev','Test']
        for sub in subsets:
            for d in sorted(os.listdir(os.path.join(db,'Mixtures',sub))):
                print os.path.join(os.path.sep,db,'Mixtures',sub,d,'mixture.wav')
                audio, sampleRate, bitrate = util.readAudioScipy(os.path.join(os.path.sep,db,'Mixtures',sub,d,'mixture.wav'))
                nsamples = audio.shape[0]
                sep_audio = np.zeros((nsamples,len(sources),audio.shape[1]))

                mag,ph=transform.compute_transform(audio,phase=True)
                mag=scale_factor*mag.astype(np.float32)
                #print 'mag.shape: ', mag.shape, 'batch_size: ', train.batch_size
                nframes = mag.shape[-2]

                batches_mag,nchunks = util.generate_overlapadd(mag,input_size=mag.shape[-1],time_context=train.time_context,overlap=train.overlap,batch_size=train.batch_size,sampleRate=sampleRate)
                mag = None

                output=[]
                for b in range(len(batches_mag)):
                    output.append(predict_function(batches_mag[b]))
                output=np.array(output)

                for j in range(audio.shape[1]):
                    mm=util.overlapadd_multi(np.swapaxes(output[:,j:j+1,:,:,:,:],1,3),batches_mag,nchunks,overlap=train.overlap)
                    for i in range(len(sources)):
                        audio_out=transform.compute_inverse(mm[i,:ph.shape[1],:]/scale_factor,ph[j])
                        # if len(sep_audio[:i,j])<len(audio_out):
                        #     print len(sep_audio), len(audio_out), len(audio_out)-len(sep_audio[:i,j])
                        #     sep_audio = np.concatenate(sep_audio,np.zeros(len(audio_out)-len(sep_audio[:i,j])))
                        #     print len(sep_audio), len(audio_out), len(audio_out)-len(sep_audio[:i,j])
                        sep_audio[:,i,j] = audio_out[:len(sep_audio)]

                print 'Saving separation: ', outdir
                if not os.path.exists(os.path.join(outdir)):
                    os.makedirs(os.path.join(outdir))
                    print 'Creating model folder'
                if not os.path.exists(os.path.join(outdir,'Sources')):
                    os.makedirs(os.path.join(outdir,'Sources'))
                    print 'Creating Sources folder: ', os.path.join(outdir,'Sources')
                if not os.path.exists(os.path.join(outdir,'Sources',sub)):
                    os.makedirs(os.path.join(outdir,'Sources',sub))
                    print 'Creating subset folder'
                if not os.path.exists(os.path.join(outdir,'Sources',sub,d)):
                    os.makedirs(os.path.join(outdir,'Sources',sub,d))
                    print 'Creating song folder', os.path.join(outdir,'Sources',sub,d)
                for i in range(len(sources)):
                    print 'Final audio file: ', i, os.path.join(outdir,'Sources',sub,d,sources[i]+'.wav'), 'nsamples: ', nsamples,'len sep_audio :', len(sep_audio)
                    util.writeAudioScipy(os.path.join(outdir,'Sources',sub,d,sources[i]+'.wav'),sep_audio[:nsamples,i,:],sampleRate,bitrate)

    return losser




if __name__ == "__main__":
    """
    Separating DSD100 stereo and binaural mixtures

    More details in the following master thesis:
    Gerard Erruz, (2017), "Binaural Source Separation with Convolutional Neural Networks" (not yet published)

    And in the following article:
    Marius Miron, Jordi Janer, Emilia Gomez, "Generating data to train convolutional neural networks for low latency classical music source separation", Sound and Music Computing Conference 2017 (submitted)

    Given the features computed previusly with compute_features_DSD100, train a network and perform the separation.

    Parameters
    ----------
    db : string
        The path to the DSD100 dataset
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
        climate.add_arg('--db', help="the DSD100 dataset path")
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
        climate.add_arg('--chunk_size', help="split large files at separation stage")
        climate.add_arg('--chunk_overlap', help="overlap for splitting large files at separation stage")
        db=None
        kwargs = climate.parse_args()
        if kwargs.__getattribute__('db'):
            db = kwargs.__getattribute__('db')
        else:
            db='path_to_DSD100'
        if kwargs.__getattribute__('output'):
            output = kwargs.__getattribute__('output')
        else:
            output='path_to_output'
        if kwargs.__getattribute__('feature_path'):
            feature_path = kwargs.__getattribute__('feature_path')
        else:
            feature_path=os.path.join(db,'transforms','feature_folder')
        if kwargs.__getattribute__('chunk_size'):
            chunk_size = int(kwargs.__getattribute__('chunk_size'))
        else:
            chunk_size = 30
        if kwargs.__getattribute__('chunk_overlap'):
            chunk_overlap = int(kwargs.__getattribute__('chunk_overlap'))
        else:
            chunk_overlap = 2
        assert os.path.isdir(db), "Please input the directory for the DSD100 dataset with --db path_to_DSD100"
        assert os.path.isdir(feature_path), "Please input the directory where you stored the training features --feature_path path_to_features"
        assert os.path.isdir(output), "Please input the output directory --output path_to_output"
        if kwargs.__getattribute__('model'):
            model = kwargs.__getattribute__('model')
        else:
            model="model_name"
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
            nepochs = 30
        if kwargs.__getattribute__('scale_factor'):
            scale_factor = int(kwargs.__getattribute__('scale_factor'))
        else:
            scale_factor = 0.3
        if kwargs.__getattribute__('scale_factor_test'):
            scale_factor_test = int(kwargs.__getattribute__('scale_factor_test'))
        else:
            scale_factor_test = 0.3
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
        if kwargs.__getattribute__('function'):
            function = kwargs.__getattribute__('function')
        else:
            function = 'build_ca'
        funcs = { 'build_ca': build_ca}
        if function not in funcs:
            function = 'build_ca'

    path_in = [feature_path]

    #tt object needs to be the same as the one in compute_features
    tt = transformFFT(frameSize=1024, hopSize=512, sampleRate=44100, window=hanning)

    ld1 = LargeDatasetMulti(path_transform_in=path_in, nsources=4, nsamples=nsamples, batch_size=batch_size, batch_memory=batch_memory, time_context=time_context, overlap=overlap, nprocs=nprocs,mult_factor_in=scale_factor,mult_factor_out=scale_factor,\
        sampleRate=tt.sampleRate,tensortype=theano.config.floatX)
    logging.info("  Maximum input:\t\t{:.6f}".format(ld1.getMax()))
    logging.info("  Minimum input:\t\t{:.6f}".format(ld1.getMin()))
    logging.info("  Mean input:\t\t{:.6f}".format(ld1.getMean()))
    logging.info("  Standard dev input:\t\t{:.6f}".format(ld1.getStd()))
    logging.info("  Maximum:\t\t{:.6f}".format(ld1.getMax(inputs=False)))
    logging.info("  Minimum:\t\t{:.6f}".format(ld1.getMin(inputs=False)))
    logging.info("  Mean:\t\t{:.6f}".format(ld1.getMean(inputs=False)))
    logging.info("  Standard dev:\t\t{:.6f}".format(ld1.getStd(inputs=False)))


    if not os.path.exists(os.path.join(output,'output',model)):
        os.makedirs(os.path.join(output,'output',model))
    if not os.path.exists(os.path.join(output,'models')):
        os.makedirs(os.path.join(output,'models'))

    train_errs=train_auto(fun=funcs[function],train=ld1,transform=tt,outdir=os.path.join(output,'output',model),
        testdir=db,chunk_size=chunk_size,chunk_overlap=chunk_overlap,
        model=os.path.join(output,"models","model_"+model+".pkl"),num_epochs=nepochs,scale_factor=scale_factor_test,load=False,skip_train=False,
        nsamples=nsamples,batch_size=batch_size, batch_memory=batch_memory, time_context=time_context, overlap=overlap,
        nprocs=nprocs,mult_factor_in=scale_factor,mult_factor_out=scale_factor,skip_sep=False)
    f = file(os.path.join(output,"models","loss_"+model+".data"), 'wb')
    cPickle.dump(train_errs,f,protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()
