from __future__ import print_function

import sys
import os
import shutil
import time

import cPickle
import re

import numpy as np
import theano
import theano.tensor as T

from transform import transformFFT2
import essentia
from essentia.standard import *
from scipy.io.wavfile import write

import lasagne
from lasagne.layers import ReshapeLayer,Layer
import math
import matplotlib.pyplot as plt
import climate
import os, sys
# import theanets
# import sepNets
import numpy as np
import theano
import ConfigParser
import downhill
import dataset_general
from dataset_general import LargeDataset
from lasagne.init import Normal
from lasagne.regularization import regularize_layer_params_weighted, l2, l1
from lasagne.regularization import regularize_layer_params

batch_size=30
time_context=30
jump=513
#min=0


print("FFT")



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

def generate_overlapadd(allmix,input_size=jump,time_context=time_context, overlap=time_context/2,batch_size=batch_size,sampleRate=44100):
    #window = np.sin((np.pi*(np.arange(2*overlap+1)))/(2.0*overlap))
    window = np.linspace(0., 1.0, num=overlap)
    window = np.concatenate((window,window[::-1]))
    #time_context = net.network.find('hid2', 'hh').size
    # input_size = net.layers[0].size  #input_size is the number of spectral bins in the fft
    window = np.repeat(np.expand_dims(window, axis=1),input_size,axis=1)
    #self.transform[id_transform].out_path = os.path.join(in_path,self.fcode)
    #import pdb;pdb.set_trace()
    
    if input_size == allmix.shape[-1]:
        
        i=0
        start=0  
        while (start + time_context) < allmix.shape[0]:
            i = i + 1
            start = start - overlap + time_context 
        fbatch = np.empty([int(np.ceil(float(i)/batch_size)),batch_size,1,time_context,input_size])
        #data_in = np.zeros((i+1,time_context,input_size))
        #data_out = np.zeros((i+1,time_context,output_size))
        sep1 = np.zeros((start+time_context,input_size))
        sep2 = np.zeros((start+time_context,input_size)) #allocate for output of prediction
        i=0
        start=0  
 
        while (start + time_context) < allmix.shape[0]:
            #batch[i,:,:] = allmix[start:start+time_context,:] #truncate on time axis so it would match the actual context
            fbatch[int(i/batch_size),int(i%batch_size),:,:,:]=allmix[start:start+time_context,:]
            #data_in[i] = allm[0]
            i = i + 1 #index for each block
            start = start - overlap + time_context #starting point for each block
    return fbatch,i

def overlapadd(fbatch,nchunks,overlap=time_context/2):

    input_size=fbatch.shape[-1]
    time_context=fbatch.shape[-2]
    batch_size=fbatch.shape[2]
    #print time_context
    #print batch_size

    #window = np.sin((np.pi*(np.arange(2*overlap+1)))/(2.0*overlap))
    window = np.linspace(0., 1.0, num=overlap)
    window = np.concatenate((window,window[::-1]))
    #time_context = net.network.find('hid2', 'hh').size
    # input_size = net.layers[0].size  #input_size is the number of spectral bins in the fft
    window = np.repeat(np.expand_dims(window, axis=1),input_size,axis=1)

    sep1 = np.zeros((nchunks*(time_context-overlap)+time_context,input_size))
    sep2 = np.zeros((nchunks*(time_context-overlap)+time_context,input_size)) #allocate for output of prediction
    #import pdb;pdb.set_trace()
    i=0
    start=0 
    while i < nchunks:
        # import pdb;pdb.set_trace()
        fbatch1=fbatch[:,0,:,:,:]
        fbatch2=fbatch[:,1,:,:,:]
        s1= fbatch1[int(i/batch_size),int(i%batch_size),0,:,:]
        s2= fbatch2[int(i/batch_size),int(i%batch_size),0,:,:]
        #print s1.shape
        if start==0:
            sep1[0:time_context] = s1
            sep2[0:time_context] = s2
        else:
            #print start+overlap
            #print start+time_context
            sep1[start+overlap:start+time_context] = s1[overlap:time_context]
            sep1[start:start+overlap] = window[overlap:]*sep1[start:start+overlap] + window[:overlap]*s1[:overlap]
            sep2[start+overlap:start+time_context] = s2[overlap:time_context]
            sep2[start:start+overlap] = window[overlap:]*sep2[start:start+overlap] + window[:overlap]*s2[:overlap]
        i = i + 1 #index for each block
        start = start - overlap + time_context #starting point for each block
    return sep1,sep2 

find = re.compile(r"^[^_]*")    
dev_directory=os.listdir('/home/pritish/pc_bss/datasets/MSD100/Mixtures/Dev')
test_directory=os.listdir('/home/pritish/pc_bss/datasets/MSD100/Mixtures/Test')    

def build_cnn(input_var=None, input_shape=(batch_size,1,time_context, jump), num_filters=55, filter_size=(20, 30),time_context=time_context):

    l_in_1 = lasagne.layers.InputLayer(shape=input_shape, input_var=input_var)

    # conv layer 1
    #l_conv1_1 = lasagne.layers.Conv2DLayer(l_in_1, num_filters=num_filters, filter_size=filter_size, W=lasagne.init.Normal(std=std, mean=mean), pad='valid', nonlinearity=None) 
    l_conv1_1 = lasagne.layers.Conv2DLayer(l_in_1, num_filters=num_filters, filter_size=filter_size, pad='valid', nonlinearity=None) 
    l_conv1_n = lasagne.layers.NonlinearityLayer(lasagne.layers.BiasLayer(l_conv1_1),nonlinearity=lasagne.nonlinearities.rectify)  

    l_pool1 = lasagne.layers.MaxPool2DLayer(l_conv1_n, pool_size=(1, 2))                                      

    l_conv2_1 = lasagne.layers.Conv2DLayer(l_pool1, num_filters=num_filters, filter_size=(1,10), pad='valid', nonlinearity=None) 
    
    l_conv2_n = lasagne.layers.NonlinearityLayer(lasagne.layers.BiasLayer(l_conv2_1),nonlinearity=lasagne.nonlinearities.rectify)      

    l_pool2 = lasagne.layers.MaxPool2DLayer(l_conv2_n, pool_size=(1, 2)) 

    l_fc2= lasagne.layers.DenseLayer(l_pool2,num_units=256)

    network = lasagne.layers.DenseLayer(l_fc2,num_units=2,nonlinearity=lasagne.nonlinearities.softmax)

    return network

def build_ca(kaka,input_var=None, input_shape=(batch_size,1,time_context, jump),time_context=time_context):

    l_in_1 = lasagne.layers.InputLayer(shape=input_shape, input_var=input_var)

    l_conv1 = lasagne.layers.Conv2DLayer(l_in_1, num_filters=50, filter_size=(1,450),stride=(1,1), pad='valid', nonlinearity=None)

    l_conv1b= lasagne.layers.BiasLayer(l_conv1)

    l_pool1 = lasagne.layers.MaxPool2DLayer(l_conv1b, pool_size=(1, 2))

    l_conv2 = lasagne.layers.Conv2DLayer(l_pool1, num_filters=30, filter_size=(10,1),stride=(1,1), pad='valid', nonlinearity=None)

    l_conv2b= lasagne.layers.BiasLayer(l_conv2)

    # l_conv3 = lasagne.layers.Conv2DLayer(l_conv2, num_filters=20, filter_size=(5,5),stride=(2,2), pad='valid', nonlinearity=None)

    # l_conv3b= lasagne.layers.BiasLayer(l_conv3)

    l_fc=lasagne.layers.DenseLayer(l_conv2b,512)

    l_fc11=lasagne.layers.DenseLayer(l_fc,l_conv2.output_shape[1]*l_conv2.output_shape[2]*l_conv2.output_shape[3])

    l_reshape1 = lasagne.layers.ReshapeLayer(l_fc11,(batch_size,l_conv2.output_shape[1],l_conv2.output_shape[2], l_conv2.output_shape[3]))

    l_inverse11=lasagne.layers.InverseLayer(l_reshape1, l_conv2)

    # l_inverse21=lasagne.layers.InverseLayer(l_inverse11, l_conv2)

    l_inverse31=lasagne.layers.InverseLayer(l_inverse11, l_pool1)

    l_inverse41=lasagne.layers.InverseLayer(l_inverse31, l_conv1)

    l_fc12=lasagne.layers.DenseLayer(l_fc,l_conv2.output_shape[1]*l_conv2.output_shape[2]*l_conv2.output_shape[3])

    l_reshape2 = lasagne.layers.ReshapeLayer(l_fc12,(batch_size,l_conv2.output_shape[1],l_conv2.output_shape[2], l_conv2.output_shape[3]))

    l_inverse12=lasagne.layers.InverseLayer(l_reshape2, l_conv2)

    # l_inverse22=lasagne.layers.InverseLayer(l_inverse12, l_conv2)

    l_inverse32=lasagne.layers.InverseLayer(l_inverse12, l_pool1)

    l_inverse42=lasagne.layers.InverseLayer(l_inverse32, l_conv1)

    l_fc13=lasagne.layers.DenseLayer(l_fc,l_conv2.output_shape[1]*l_conv2.output_shape[2]*l_conv2.output_shape[3])

    l_reshape3 = lasagne.layers.ReshapeLayer(l_fc13,(batch_size,l_conv2.output_shape[1],l_conv2.output_shape[2], l_conv2.output_shape[3]))

    l_inverse13=lasagne.layers.InverseLayer(l_reshape3, l_conv2)

    # l_inverse23=lasagne.layers.InverseLayer(l_inverse13, l_conv2)

    l_inverse33=lasagne.layers.InverseLayer(l_inverse13, l_pool1)

    l_inverse43=lasagne.layers.InverseLayer(l_inverse33, l_conv1)

    l_fc14=lasagne.layers.DenseLayer(l_fc,l_conv2.output_shape[1]*l_conv2.output_shape[2]*l_conv2.output_shape[3])

    l_reshape4 = lasagne.layers.ReshapeLayer(l_fc14,(batch_size,l_conv2.output_shape[1],l_conv2.output_shape[2], l_conv2.output_shape[3]))

    l_inverse14=lasagne.layers.InverseLayer(l_reshape4, l_conv2)

    # l_inverse24=lasagne.layers.InverseLayer(l_inverse14, l_conv2)

    l_inverse34=lasagne.layers.InverseLayer(l_inverse14, l_pool1)

    l_inverse44=lasagne.layers.InverseLayer(l_inverse34, l_conv1)

    l_merge=lasagne.layers.ConcatLayer([l_inverse41,l_inverse42,l_inverse43,l_inverse44],axis=1)

    # l_abstract2=lasagne.layers.DenseLayer(l_abstract1,50)

    l_out = lasagne.layers.NonlinearityLayer(lasagne.layers.BiasLayer(l_merge), nonlinearity=lasagne.nonlinearities.rectify)
   
    return l_out

def train_classifier(kaka,train_dir,test_dir,valid_dir,num_epochs=30):
    train=LargeDataset(train_dir,batch_size=batch_size, batch_memory=100, time_context=time_context, nsources=2)
    test=LargeDataset(test_dir,batch_size=batch_size, batch_memory=100, time_context=time_context, nsources=2)
    valid=LargeDataset(valid_dir,batch_size=batch_size, batch_memory=100, time_context=time_context, nsources=2)

    # Prepare Theano variables for inputs and targets
    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')

    # Create neural network model (depending on first command line parameter)
    print("Building model and compiling functions...")
    network = build_cnn(input_var)
    #import pdb;pdb.set_trace()

    # Create a loss expression for training, i.e., a scalar objective we want
    # to minimize (for our multi-class problem, it is the cross-entropy loss):
    prediction = lasagne.layers.get_output(network)
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.mean()
    # We could add some weight decay as well here, see lasagne.regularization.

    # Create update expressions for training, i.e., how to modify the
    # parameters at each training step. Here, we'll use Stochastic Gradient
    # Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate=0.0005, momentum=0.9)

    # Create a loss expression for validation/testing. The crucial difference
    # here is that we do a deterministic forward pass through the network,
    # disabling dropout layers.
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,target_var)
    test_loss = test_loss.mean()
    # As a bonus, also create an expression for the classification accuracy:
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),dtype=theano.config.floatX)

    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    train_fn = theano.function([input_var, target_var], loss, updates=updates,allow_input_downcast=True)

    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input_var, target_var], [test_loss, test_acc],allow_input_downcast=True)

    # Finally, launch the training loop.
    print("Starting training...")
    # We iterate over epochs:
    for epoch in range(num_epochs):
    # val_acc=0
    # while(val_acc<0.93):
        # In each epoch, we do a full pass over the training data:
        val_accs=[]    
        train_err = 0
        train_batches = 0
        start_time = time.time()
        # for batch in range(train.iteration_size): #Have to change this line, train is not a generator, just a callabale
        for batch in range(train.iteration_size):
        #for batch in range(2):
            inputs, target = train()
            inputs=np.reshape(inputs,(inputs.shape[0],1,inputs.shape[1],inputs.shape[2]))
            targets=np.ndarray(shape=(inputs.shape[0]*2,1,inputs.shape[2],inputs.shape[3]+1))
            targets1=np.ndarray(shape=(inputs.shape[0]*2,1,inputs.shape[2],inputs.shape[3]))
            labels=np.ndarray(inputs.shape[0]*2)
            #print(target.shape)
            targets[:inputs.shape[0],0,:,:jump]=target[:,:,:jump]
            targets[:inputs.shape[0],0,:,jump]=0
            targets[inputs.shape[0]:inputs.shape[0]*2,0,:,:jump]=target[:,:,jump:]
            targets[inputs.shape[0]:inputs.shape[0]*2,0,:,jump]=1
            import pdb;pdb.set_trace()
            # targets[inputs.shape[0]*2:,0,:,:jump]=inputs[:,0,:,:]
            # targets[inputs.shape[0]*2:,0,:,jump]=1            

            np.random.shuffle(targets)
            labels=targets[:,:,0,jump]
            #import pdb;pdb.set_trace()
            labels=np.reshape(labels,(inputs.shape[0]*2))
            targets1=np.log10(1+targets[:,:,:,:jump])
            train_err += train_fn(targets1, labels)
            train_batches += 1
            save_model("/home/pritish/pc_bss/datasets/MSD100/Data5/class_models/classifier-d_i5_"+str(kaka)+".pkl",network)
        # And a full pass over the validation data:
        val_err = 0
        val_acc = 0
        val_batches = 0
        for batch in range(valid.iteration_size): #Have to change this line, train is not a generator, just a callabale
        #for batch in range(2):
            inputs, target = valid()
            inputs=np.reshape(inputs,(inputs.shape[0],1,inputs.shape[1],inputs.shape[2]))
            targets=np.ndarray(shape=(inputs.shape[0]*2,1,inputs.shape[2],inputs.shape[3]+1))
            targets1=np.ndarray(shape=(inputs.shape[0]*2,1,inputs.shape[2],inputs.shape[3]))
            labels=np.ndarray(inputs.shape[0]*2)
            #print(target.shape)
            targets[:inputs.shape[0],0,:,:jump]=target[:,:,:jump]
            targets[:inputs.shape[0],0,:,jump]=0
            targets[inputs.shape[0]:inputs.shape[0]*2,0,:,:jump]=target[:,:,jump:]
            targets[inputs.shape[0]:inputs.shape[0]*2,0,:,jump]=1

            # targets[inputs.shape[0]*2:,0,:,:jump]=inputs[:,0,:,:]
            # targets[inputs.shape[0]*2:,0,:,jump]=1            

            np.random.shuffle(targets)
            labels=targets[:,:,0,jump]
            #import pdb;pdb.set_trace()
            labels=np.reshape(labels,(inputs.shape[0]*2))
            targets1=np.log10(1+targets[:,:,:,:jump])
            
            err, acc = val_fn(targets1, labels)
            val_err += err
            val_acc += acc
            val_batches += 1

        # Then we print the results for this epoch:
          
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training "+str(kaka)+" loss:\t\t{:.6f}".format(train_err / train_batches))
        print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
        print("  validation accuracy:\t\t{:.2f} %".format(
            val_acc / val_batches * 100))
        val_accs.append(val_acc / val_batches * 100)
    test_err = 0
    test_acc = 0
    test_batches = 0
    for batch in range(test.iteration_size): #Have to change this line, train is not a generator, just a callabale
    #for batch in range(2):
        inputs, target = test()
        inputs=np.reshape(inputs,(inputs.shape[0],1,inputs.shape[1],inputs.shape[2]))
        targets=np.ndarray(shape=(inputs.shape[0]*2,1,inputs.shape[2],inputs.shape[3]+1))
        targets1=np.ndarray(shape=(inputs.shape[0]*2,1,inputs.shape[2],inputs.shape[3]))
        labels=np.ndarray(inputs.shape[0]*2)
        #print(target.shape)
        targets[:inputs.shape[0],0,:,:jump]=target[:,:,:jump]
        targets[:inputs.shape[0],0,:,jump]=0
        targets[inputs.shape[0]:inputs.shape[0]*2,0,:,:jump]=target[:,:,jump:]
        targets[inputs.shape[0]:inputs.shape[0]*2,0,:,jump]=1

        # targets[inputs.shape[0]*2:,0,:,:jump]=inputs[:,0,:,:]
        # targets[inputs.shape[0]*2:,0,:,jump]=1            

        np.random.shuffle(targets)
        labels=targets[:,:,0,jump]
        #import pdb;pdb.set_trace()
        labels=np.reshape(labels,(inputs.shape[0]*2))
        targets1=np.log10(1+targets[:,:,:,:jump])
        
        err, acc = val_fn(targets1, labels)
        test_err += err
        test_acc += acc
        test_batches += 1
    print("Final results:")
    print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
    print("  test accuracy:\t\t{:.2f} %".format(
        test_acc / test_batches * 100))
    return val_accs,test_acc / test_batches * 100,network

def train_GAN(kaka,train_dir,test_dir,valid_dir,num_epochs=30):
    train=LargeDataset(train_dir,batch_size=batch_size, batch_memory=100, time_context=time_context, nsources=2)
    test=LargeDataset(test_dir,batch_size=batch_size, batch_memory=100, time_context=time_context, nsources=2)
    valid=LargeDataset(valid_dir,batch_size=batch_size, batch_memory=100, time_context=time_context, nsources=2)

def kld(x,y):
    eps = 1e-8
    t = T.clip(y, eps, 1 - eps)
    u = T.clip(x, eps, 1 - eps)
    kl = abs(T.log(t / u))
    return kl

def train_auto(kaka,train_dir,valid_dir,network,num_epochs=30):
    train=LargeDataset(train_dir,batch_size=batch_size, batch_memory=300, time_context=time_context, nsources=4)
    # valid=LargeDataset(valid_dir,batch_size=batch_size, batch_memory=100, time_context=time_context, nsources=4)
    print("Building Autoencoder")
    input_var2 = T.tensor4('inputs')
    target_var2 = T.tensor4('targets')
    # mix=T.tensor4('mix')
    input_var=input_var2/T.sqrt(T.sum(T.sqr(input_var2)))


    network2 = build_ca(kaka=kaka,input_var=input_var)

    # params=load_model("/home/pritish/pc_bss/datasets/MSD100/Data5/auto_models/autoclass-phase2_fft14_"+str(kaka)+".pkl")

    # lasagne.layers.set_all_param_values(network2,params)


    prediction2 = lasagne.layers.get_output(network2, deterministic=True)


    #Change code, prediction2 should be a mask - done
    # ins=prediction2*input_var2
    # outs=(1-prediction2)*input_var2

    #prediction=T.tensor4('prediction')
    #m=abs(prediction[:,0:1,:,:])/(abs(prediction[:,0:1,:,:])+abs(prediction[:,1:2,:,:]))
    #a=m*input_var
    #b=(T.ones_like(input_var)-m)*input_var
    #c=T.concatenate((a,b),axis=1)
    alpha=0.001
    beta=0.01
    beta_voc=0.03


    voc=prediction2[:,0:1,:,:]+0.00001

    bas=prediction2[:,1:2,:,:]+0.00001

    dru=prediction2[:,2:3,:,:]+0.00001

    oth=prediction2[:,3:4,:,:]+0.00001

    mask1=voc/(voc+bas+dru+oth)

    mask2=bas/(voc+bas+dru+oth)

    mask3=dru/(voc+bas+dru+oth)

    mask4=oth/(voc+bas+dru+oth)

    # input_sum=input_var2-target_var2[:,3:4,:,:]

    vocals=mask1*input_var2

    bass=mask2*input_var2

    drums=mask3*input_var2

    others=mask4*input_var2


    train_loss_recon_vocals = kld(vocals,target_var2[:,0:1,:,:])

    alpha_component = alpha*kld(vocals,target_var2[:,1:2,:,:])

    alpha_component += alpha*kld(vocals,target_var2[:,2:3,:,:])

    train_loss_recon_neg_voc = beta_voc*kld(vocals,target_var2[:,3:4,:,:])

    train_loss_recon_bass = kld(bass,target_var2[:,1:2,:,:])

    alpha_component += alpha*kld(bass,target_var2[:,0:1,:,:])

    alpha_component += alpha*kld(bass,target_var2[:,2:3,:,:])

    train_loss_recon_neg = beta*kld(bass,target_var2[:,3:4,:,:])

    train_loss_recon_drums = kld(drums,target_var2[:,2:3,:,:])

    alpha_component += alpha*kld(drums,target_var2[:,0:1,:,:])

    alpha_component += alpha*kld(drums,target_var2[:,1:2,:,:])

    train_loss_recon_neg += beta*kld(drums,target_var2[:,3:4,:,:])


    vocals_error=train_loss_recon_vocals.sum()
    drums_error=train_loss_recon_drums.sum()
    bass_error=train_loss_recon_bass.sum()
    negative_error=train_loss_recon_neg.sum()
    negative_error_voc=train_loss_recon_neg_voc.sum()
    alpha_component=alpha_component.sum()

    # l1_penalty = regularize_layer_params(layer2, l1)
    
    loss=abs(vocals_error+drums_error+bass_error-negative_error-alpha_component-negative_error_voc)
    # loss=alpha*(loss1+loss2)+beta*loss3
    #loss=train_loss_recon.sum(axis=1).mean()+beta*loss3

    params1 = lasagne.layers.get_all_params(network2, trainable=True)

    updates = lasagne.updates.adadelta(loss, params1)

    # val_updates=lasagne.updates.nesterov_momentum(loss1, params1, learning_rate=0.00001, momentum=0.7)

    train_fn = theano.function([input_var2,target_var2], loss, updates=updates,allow_input_downcast=True)

    train_fn1 = theano.function([input_var2,target_var2], [vocals_error,bass_error,drums_error,negative_error,alpha_component,negative_error_voc], allow_input_downcast=True)

    predict_function2=theano.function([input_var2],[vocals,bass,drums,others],allow_input_downcast=True)

    # val_fn=theano.function([input_var2], loss1,updates=val_updates,allow_input_downcast=True)

    losser=[]
    loss2=[]



    print("Training...")
    #import pdb;pdb.set_trace()
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
        for batch in range(train.iteration_size): #Have to change this line, train is not a generator, just a callabale
        #for batch in range(2):
            inputs, target = train()
            
            inputs=np.reshape(inputs,(inputs.shape[0],1,inputs.shape[1],inputs.shape[2]))
            targets=np.ndarray(shape=(inputs.shape[0],4,inputs.shape[2],inputs.shape[3]))
            #print(target.shape)
            targets[:,0,:,:]=target[:,:,:jump]
            targets[:,1,:,:]=target[:,:,jump:jump*2]
            targets[:,2,:,:]=target[:,:,jump*2:jump*3]
            targets[:,3,:,:]=target[:,:,jump*3:jump*4]
            # mask=abs(targets[:,0:1,:,:])/(abs(targets[:,0:1,:,:])+abs(targets[:,1:2,:,:])) 
            # where_are_NaNs = np.isnan(mask)
            # mask[where_are_NaNs] = 0
            #import pdb;pdb.set_trace()
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
        # print("  validation loss:\t\t{:.6f}".format(val_err/val_batches))
        # loss2.append(val_err / val_batches)
        save_model("/home/pritish/pc_bss/datasets/MSD100/Data5/auto_models/autoclass-phase2_fft4_"+str(kaka)+".pkl",network2)
        # Testing for a file, to see if the output is good    

        #     And a full pass over the validation data:
        # if (epoch+1)%5==0:
    val_batches = 0
    print("now seperating")
    print(kaka)
    f=file('/home/pritish/pc_bss/datasets/MSD100/Data5/Test_dirs/Test_file_ffto_'+str(kaka)+".file",'r')
    b=tuple(f)
    # import pdb;pdb.set_trace()
    f.close()
    c=[re.search(find, l).group(0) for l in b]
    d=list(set(c))

    print(d)

    for song_name in d:
        # print("Testing file "+song_name)

        output=[]

        start_time=time.time()
        if song_name in dev_directory:
            song='/home/pritish/pc_bss/datasets/MSD100/Mixtures/Dev/'+song_name+"/mixture.wav"
        elif song_name in test_directory:
            song='/home/pritish/pc_bss/datasets/MSD100/Mixtures/Test/'+song_name+"/mixture.wav"    
        loader=essentia.standard.MonoLoader(filename = song,sampleRate=44100)
        audio=loader()
        megatron=transformFFT2()

        mag,ph=megatron.compute_file(audio,phase=True)

        mag=mag.astype(np.float32)

        batches,nchunks = generate_overlapadd(mag)
        
        batch_no=1
        for batch in batches:
            # print("batch {} of {}".format(batch_no,len(batches)))
            # batch=batch.astype('float32')
            batch_no+=1
            # ee=0
            # val_err=50
            # params=load_model("/home/pritish/pc_bss/datasets/MSD100/Data5/auto_models/autoclass-phase2_fft_"+str(kaka)+".pkl")
            # lasagne.layers.set_all_param_values(network2,params)
            start_time=time.time()

            output.append(predict_function2(batch))
        output=np.array(output)
        # import pdb;pdb.set_trace()
        output1=output[:,:2]
        output2=output[:,2:]
        bmag,mm=overlapadd(output1,nchunks)
        # import pdb;pdb.set_trace()
        audio_out=megatron.compute_inverse(bmag[:len(ph)],ph)
        audio_out=essentia.array(audio_out)
        audio_out2= megatron.compute_inverse(mm[:len(ph)],ph)    
        audio_out2=essentia.array(audio_out2)
        writer1=essentia.standard.MonoWriter(bitrate=32,filename="/home/pritish/pc_bss/datasets/MSD100/Data5/25econ30tc4/"+song_name+"_vocals.wav")
        writer1(audio_out)
        writer2=essentia.standard.MonoWriter(bitrate=32,filename="/home/pritish/pc_bss/datasets/MSD100/Data5/25econ30tc4//"+song_name+"_bass.wav")
        writer2(audio_out2)
        # write("/home/pritish/pc_bss/datasets/MSD100/Data5/outputs4/"+song_name+"_drums.wav",44100,audio_out)
        # write("/home/pritish/pc_bss/datasets/MSD100/Data5/outputs4/"+song_name+"_acc.wav",44100,audio_out2)
        bmag,mm=overlapadd(output2,nchunks)
        #import pdb;pdb.set_trace()
        audio_out=megatron.compute_inverse(bmag[:len(ph)],ph)
        audio_out=audio_out/audio_out.max()
        audio_out=essentia.array(audio_out)
        audio_out2= megatron.compute_inverse(mm[:len(ph)],ph) 
        audio_out2=audio_out2/audio_out2.max()
        audio_out2=essentia.array(audio_out2)
        writer3=essentia.standard.MonoWriter(bitrate=32,filename="/home/pritish/pc_bss/datasets/MSD100/Data5/25econ30tc4//"+song_name+"_drums.wav")
        writer3(audio_out)
        writer4=essentia.standard.MonoWriter(bitrate=32,filename="/home/pritish/pc_bss/datasets/MSD100/Data5/25econ30tc4//"+song_name+"_others.wav")
        writer4(audio_out2)
        writer1=None
        writer2=None
        writer3=None
        writer4=None
        # # write("/home/pritish/pc_bss/datasets/MSD100/Data5/outputs4/"+song_name+"_drumsa.wav",44100,audio_out)
        # write("/home/pritish/pc_bss/datasets/MSD100/Data5/outputs4/"+song_name+"_acca.wav",44100,audio_out2)

        # print("Done with file "+song_name+", took {:.3f}s".format(time.time()-start_time))


        # Then we print the results for this epoch:

    return [losser,loss2]    

def main():
    circle=np.arange(1,11)
    # circle=np.roll(circle,-1)
    train_dir='/home/pritish/pc_bss/datasets/MSD100/Data6/FFT/Train'

    valid_dir='/home/pritish/pc_bss/datasets/MSD100/Data6/FFT/Valid'

    test_dir='/home/pritish/pc_bss/datasets/MSD100/Data6/FFT/Test'
    
    for a in range(1):
        if not circle[0]==11:
            start_time=time.time()
            print("Now doing the "+str(circle[0])+"th round")
            # if not os.path.exists(train_dir):
            #     os.makedirs(train_dir)
            # if not os.path.exists(valid_dir):
            #     os.makedirs(valid_dir)
            # if not os.path.exists(test_dir):
            #     os.makedirs(test_dir)    
            # for b in range(8):
            #     src='/home/pritish/pc_bss/datasets/MSD100/Data5/FFT/d/'+str(circle[b])
            #     src_files = os.listdir(src)
            #     for file_name in src_files:
            #         full_file_name = os.path.join(src, file_name)
            #         if (os.path.isfile(full_file_name)):
            #             shutil.copy(full_file_name, train_dir)
            
            # src='/home/pritish/pc_bss/datasets/MSD100/Data5/FFT/d/'+str(circle[8])
            # src_files = os.listdir(src)
            # for file_name in src_files:
            #     full_file_name = os.path.join(src, file_name)
            #     if (os.path.isfile(full_file_name)):
            #         shutil.copy(full_file_name, valid_dir)

            # src='/home/pritish/pc_bss/datasets/MSD100/Data5/FFT/d/'+str(circle[9])
            # src_files = os.listdir(src)
            # f=file('/home/pritish/pc_bss/datasets/MSD100/Data5/Test_dirs/Test_file_i5_'+str(circle[0])+".file",'wb')
            # f.writelines(line+'\n' for line in src_files)
            # f.close()
            # for file_name in src_files:
            #     full_file_name = os.path.join(src, file_name)
            #     if (os.path.isfile(full_file_name)):
            #         shutil.copy(full_file_name, test_dir)        

            # print("copied files, took {:.3f}s".format(time.time()-start_time))
         
            # val_accs,test_acc,classifier=train_classifier(circle[0],train_dir,test_dir,valid_dir)

            # f = file("/home/pritish/pc_bss/datasets/MSD100/Data5/class_accs/val_acc_cqt_"+str(circle[0])+".data", 'wb')
            # cPickle.dump(val_accs,f,protocol=cPickle.HIGHEST_PROTOCOL)
            # f.close()
            # f = file("/home/pritish/pc_bss/datasets/MSD100/Data5/class_accs/test_acc_i4_"+str(circle[0])+".data", 'wb')
            # cPickle.dump(test_acc,f,protocol=cPickle.HIGHEST_PROTOCOL)
            # f.close()

            # shutil.rmtree(train_dir)
            # shutil.rmtree(valid_dir)

            # shutil.rmtree(test_dir)

            # if os.path.exists(train_dir):
            #     shutil.rmtree(train_dir)
            # if os.path.exists(valid_dir):
            #     shutil.rmtree(valid_dir)
            # if os.path.exists(test_dir):
            #     shutil.rmtree(test_dir)   

            # if not os.path.exists(train_dir):
            #     os.makedirs(train_dir)
            # # if not os.path.exists(valid_dir):
            # #     os.makedirs(valid_dir)    
            
            # for b in range(5):
            #     src='/home/pritish/pc_bss/datasets/MSD100/Data6/FFT/'+str(circle[b])
            #     src_files = os.listdir(src)
            #     for file_name in src_files:
            #         full_file_name = os.path.join(src, file_name)
            #         if (os.path.isfile(full_file_name)):
            #             shutil.copy(full_file_name, train_dir)

            # f=file('/home/pritish/pc_bss/datasets/MSD100/Data5/Test_dirs/Test_file_ffto_'+str(circle[0])+".file",'wb')
            # for b in range(5,10):
            #     src='/home/pritish/pc_bss/datasets/MSD100/Data6/FFT/'+str(circle[b])
            #     src_files = os.listdir(src)                
            #     f.writelines(line+'\n' for line in src_files)
            # f.close()
            # for file_name in src_files:
                # full_file_name = os.path.join(src, file_name)
                # if (os.path.isfile(full_file_name)):
                #     shutil.copy(full_file_name, valid_dir)

            print("Written files")
      
            input_var = T.tensor4('inputs')     

            classifier=build_cnn(input_var)

            # params=load_model("/home/pritish/pc_bss/datasets/MSD100/Data5/class_models/classifier-d"+str(circle[0])+".pkl")
    
            # lasagne.layers.set_all_param_values(classifier,params)

            train_errs,test_errs=train_auto(kaka=circle[0],train_dir=train_dir,valid_dir=valid_dir,network=classifier)        

            f = file("/home/pritish/pc_bss/datasets/MSD100/Data5/auto_errors/train_errors_fft_4"+str(circle[0])+".data", 'wb')
            cPickle.dump(train_errs,f,protocol=cPickle.HIGHEST_PROTOCOL)
            f.close()

            # f = file("/home/pritish/pc_bss/datasets/MSD100/Data5/auto_errors/test_errors_fft_30_conv_"+str(circle[0])+".data", 'wb')
            # cPickle.dump(test_errs,f,protocol=cPickle.HIGHEST_PROTOCOL)
            # f.close()
            
            # shutil.rmtree(train_dir)
            # shutil.rmtree(valid_dir)

            print("Round "+str(circle[0])+" took {:.3f}s".format(time.time()-start_time))
        circle=np.roll(circle,-1)
        
main()




