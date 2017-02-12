import os,sys
import transform
from transform_general import transformFFT2,transformCQT
import numpy as np
import re
import essentia
from essentia.standard import *
from scipy.signal import blackmanharris as blackmanharris
import shutil
import time
import cPickle
import re
import math
import climate
import ConfigParser

import theano
import theano.tensor as T

import lasagne
from lasagne.layers import ReshapeLayer,Layer

import dataset_general
from dataset_general import LargeDatasetPitch1
from lasagne.init import Normal
from lasagne.regularization import regularize_layer_params_weighted, l2, l1
from lasagne.regularization import regularize_layer_params

def distFunction(x,y,d=0,weight=None):
    if d == 0:
        err = T.abs_(y - x)
        if weight is not None:
            return (weight * err * err).sum() / weight.sum()
        return (err * err).mean()
    else:
        eps = 1e-8
        t = T.clip(y, eps, 1 - eps)
        u = T.clip(x, eps, 1 - eps)
        kl = t * T.log(t / u) - t + u
        if weight is not None:
            return abs(weight * kl).sum() / weight.sum()
        #return abs(kl).mean()
        return kl

def load_model(filename):
    f=file(filename,'rb')
    params=cPickle.load(f)
    f.close()
    #lasagne.layers.set_all_param_values(model,params)
    return params

def save_model(filename, model):
    params=lasagne.layers.get_all_param_values(model)
    #import pdb;pdb.set_trace()
    f = file(filename, 'wb')
    cPickle.dump(params,f,protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()
    return None

def build_ca(input_var=None, batch_size=32,time_context=30,feat_size=1025):

    input_shape=(batch_size,1,time_context,feat_size)
    scaled_tanh = lasagne.nonlinearities.ScaledTanH(scale_in=1, scale_out=0.5)

    l_in_1 = lasagne.layers.InputLayer(shape=input_shape, input_var=input_var)

    l_conv1 = lasagne.layers.Conv2DLayer(l_in_1, num_filters=30, filter_size=(1,60),stride=(1,5), pad='valid', nonlinearity=None)
    
    l_conv1b= lasagne.layers.BiasLayer(l_conv1)

    l_pool1 = lasagne.layers.MaxPool2DLayer(l_conv1b, pool_size=(1, 2))  

    l_conv2 = lasagne.layers.Conv2DLayer(l_pool1, num_filters=30, filter_size=(10,20),stride=(1,1), pad='valid', nonlinearity=None)

    l_conv2b= lasagne.layers.BiasLayer(l_conv2)

    l_conv3 = lasagne.layers.Conv2DLayer(l_conv2, num_filters=30, filter_size=(20,10),stride=(1,1), pad='valid', nonlinearity=None)

    l_conv3b= lasagne.layers.BiasLayer(l_conv3)

    l_fc=lasagne.layers.DenseLayer(l_conv3b,512)
    # l1a = lasagne.layers.DenseLayer(l_conv3b, nonlinearity=None, num_units=512)
    # l_fc = lasagne.layers.FeaturePoolLayer(l1a, pool_size=256)

    l_fc11=lasagne.layers.DenseLayer(l_fc,l_conv3.output_shape[1]*l_conv3.output_shape[2]*l_conv3.output_shape[3])

    l_reshape1 = lasagne.layers.ReshapeLayer(l_fc11,(batch_size,l_conv3.output_shape[1],l_conv3.output_shape[2], l_conv3.output_shape[3]))

    l_inverse11=lasagne.layers.InverseLayer(l_reshape1, l_conv3)

    l_inverse21=lasagne.layers.InverseLayer(l_inverse11, l_conv2)

    l_inverse31=lasagne.layers.InverseLayer(l_inverse21, l_pool1)

    l_inverse41=lasagne.layers.InverseLayer(l_inverse31, l_conv1)

    # l_fc12=lasagne.layers.DenseLayer(l_fc,l_conv3.output_shape[1]*l_conv3.output_shape[2]*l_conv3.output_shape[3])

    # l_reshape2 = lasagne.layers.ReshapeLayer(l_fc12,(batch_size,l_conv3.output_shape[1],l_conv3.output_shape[2], l_conv3.output_shape[3]))

    # l_inverse12=lasagne.layers.InverseLayer(l_reshape2, l_conv3)

    # l_inverse22=lasagne.layers.InverseLayer(l_inverse12, l_conv2)

    # l_inverse32=lasagne.layers.InverseLayer(l_inverse22, l_pool1)

    # l_inverse42=lasagne.layers.InverseLayer(l_inverse32, l_conv1)

    # l_merge=lasagne.layers.ConcatLayer([l_inverse41,l_inverse42],axis=1)

    # l_abstract2=lasagne.layers.DenseLayer(l_abstract1,50)

    l_out = lasagne.layers.NonlinearityLayer(lasagne.layers.BiasLayer(l_inverse41), nonlinearity=lasagne.nonlinearities.rectify)
   
    return l_out

def train_auto(train,fun,num_epochs=25,model="1.pkl",load=False):
    print("Building Autoencoder")
    input_var2 = T.tensor4('inputs')
    target_var2 = T.tensor4('targets')

    eps=1e-7
    alpha=0.003
    
    network2 = fun(input_var=input_var2,batch_size=train.batch_size,time_context=train.time_context,feat_size=train.input_size)
    
    if load:
        params=load_model(model)
        lasagne.layers.set_all_param_values(network2,params)

    prediction2 = lasagne.layers.get_output(network2, deterministic=True)

    vocals=prediction2[:,0:1,:,:]+eps

    # train_loss_recon = distFunction(vocals,target_var2[:,0:1,:,:],d=1)
    # train_loss_recon -= alpha*distFunction(vocals,target_var2[:,1:2,:,:],d=1)
    # train_loss_recon += distFunction(acc,target_var2[:,1:2,:,:],d=1)
    # train_loss_recon -= alpha*distFunction(acc,target_var2[:,0:1,:,:],d=1)

    train_loss_recon = lasagne.objectives.squared_error(vocals,target_var2[:,0:1,:,:])
   
 
    #loss=train_loss_recon.sum()
    loss = lasagne.objectives.aggregate(train_loss_recon, mode='sum')

    params1 = lasagne.layers.get_all_params(network2, trainable=True)

    updates = lasagne.updates.adadelta(loss, params1)

    # val_updates=lasagne.updates.nesterov_momentum(loss1, params1, learning_rate=0.00001, momentum=0.7)

    train_fn = theano.function([input_var2,target_var2], loss, updates=updates,allow_input_downcast=True)

    #train_fn1 = theano.function([input_var2,target_var2], updates=updates, allow_input_downcast=True)

    predict_function2=theano.function([input_var2],[vocals],allow_input_downcast=True)

    # val_fn=theano.function([input_var2], loss1,updates=val_updates,allow_input_downcast=True)

    losser=[]
    loss2=[]

    print("Training...")
    #import pdb;pdb.set_trace()
    for epoch in range(num_epochs):

        train_err = 0
        train_batches = 0
        start_time = time.time()
        for batch in range(train.iteration_size): #Have to change this line, train is not a generator, just a callabale
        #for batch in range(2):
            inputs, target, masks = train()
            
            #import pdb;pdb.set_trace()
            jump = inputs.shape[2]
            # targets=np.log10(1.+np.ndarray(shape=(inputs.shape[0],2,inputs.shape[1],inputs.shape[2])))
            # inputs=np.log10(1.+np.reshape(inputs,(inputs.shape[0],1,inputs.shape[1],inputs.shape[2])))
            targets=np.ndarray(shape=(inputs.shape[0],1,inputs.shape[1],inputs.shape[2]))
            masks=np.reshape(masks,(inputs.shape[0],1,inputs.shape[1],inputs.shape[2]))
            inputs=np.reshape(inputs,(inputs.shape[0],1,inputs.shape[1],inputs.shape[2]))
          
            

            # inputs=np.ndarray(shape=(inputs1.shape[0],2,inputs1.shape[1],inputs1.shape[2]))
            # targets=np.ndarray(shape=(inputs1.shape[0],2,inputs1.shape[1],inputs1.shape[2]))
            # jump = inputs1.shape[2]
            targets[:,0,:,:]=target[:,:,:jump]
            # targets[:,1,:,:]=target[:,:,jump:jump*2]
            # inputs[:,0,:,:]=inputs1
            inputs[:,0,:,:]=inputs*masks+0.0000001
            targets[:,0,:,:]=targets*masks+0.0000001
            # inputs1=None
            # inputs1=None
            target=None
            masks=None
            train_err+=train_fn(inputs,targets)
            train_batches += 1
        
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err/train_batches))
        losser.append(train_err / train_batches)
        save_model(model,network2)

    return losser  


db='/home/marius/Documents/Database/iKala/'
pitchhop=0.032*44100.0 #seconds to frames
tt1=transformFFT2(frameSize=2048, hopSize=512, sampleRate=44100, window=blackmanharris)
tt2=transformCQT(frameSize=2048, hopSize=512, bins=1025, sampleRate=44100, window=blackmanharris, tffmin=100, tffmax=14000, iscale = 'log')

# import util
# ff,bb=util.getBands(bins=tt1.bins,interval=50,iscale=tt1.iscale,frameSize=tt1.frameSize,fmin=tt1.fmin,fmax=tt1.fmax,ttype=tt1.ttype,sampleRate=tt1.sampleRate)
# import pdb;pdb.set_trace()

dinf='/home/marius/Documents/Database/iKala/transforms/t1/'
dinc='/home/marius/Documents/Database/iKala/transforms/t2/'
batch_size=32
time_context=30

ld1 = LargeDatasetPitch1(path_transform_in=dinf, batch_size=batch_size, batch_memory=2, time_context=time_context, overlap=10, nprocs=8,
    sampleRate=tt1.sampleRate,fmin=tt1.fmin, fmax=tt1.fmax,ttype=tt1.ttype,iscale=tt1.iscale, 
    nharmonics=20, interval=50, tuning_freq=440, pitched=True, save_mask=True, pitch_norm=120.)
model1="fft1"

# ld2 = LargeDatasetPitch1(path_transform_in=dinc, batch_size=batch_size, batch_memory=10, time_context=time_context, overlap=10,
#     sampleRate=tt2.sampleRate,fmin=tt2.fmin, fmax=tt2.fmax,ttype=tt2.ttype,iscale=tt2.iscale, 
#     nharmonics=20, interval=50, tuning_freq=440, pitched=True, save_mask=True, pitch_norm=120.)


train_errs=train_auto(train=ld1,fun=build_ca,model=db+"models/"+"model_"+model1+".pkl",load=False)        
f = file(db+"models/"+"loss_"+model1+".data", 'wb')
cPickle.dump(train_errs,f,protocol=cPickle.HIGHEST_PROTOCOL)
f.close()

