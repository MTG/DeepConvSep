import os,sys
import transform_general
from transform_general import transformFFT2,transformCQT,transformFFT
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
import theano.sandbox.rng_mrg

import lasagne
from lasagne.layers import ReshapeLayer,Layer

import dataset_general
from dataset_general import LargeDatasetPitch1,LargeDataset
from lasagne.init import Normal
from lasagne.regularization import regularize_layer_params_weighted, l2, l1
from lasagne.regularization import regularize_layer_params
import util

logging = climate.get_logger('trainer')

climate.enable_default_logging()


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
    #scaled_tanh = lasagne.nonlinearities.ScaledTanH(scale_in=1, scale_out=0.5)

    l_in_1 = lasagne.layers.InputLayer(shape=input_shape, input_var=input_var)

    l_conv11 = lasagne.layers.Conv2DLayer(l_in_1, num_filters=30, filter_size=(1,50),stride=(1,5), pad='valid', nonlinearity=None)
    l_conv11b= lasagne.layers.BiasLayer(l_conv11)

    l_pool11 = lasagne.layers.MaxPool2DLayer(l_conv11b, pool_size=(1, 2))

    l_conv21 = lasagne.layers.Conv2DLayer(l_pool11, num_filters=30, filter_size=(10,20),stride=(1,1), pad='valid', nonlinearity=None)
    l_conv21b= lasagne.layers.BiasLayer(l_conv21)

    l_rs1 = ReshapeLayer(l_conv21b, (batch_size,-1))


    l_conv12 = lasagne.layers.Conv2DLayer(l_in_1, num_filters=30, filter_size=(1,30),stride=(1,3), pad='valid', nonlinearity=None)
    l_conv12b= lasagne.layers.BiasLayer(l_conv12)

    l_pool12 = lasagne.layers.MaxPool2DLayer(l_conv12b, pool_size=(1, 4))

    l_conv22 = lasagne.layers.Conv2DLayer(l_pool12, num_filters=30, filter_size=(10,5),stride=(1,1), pad='valid', nonlinearity=None)
    l_conv22b= lasagne.layers.BiasLayer(l_conv22)

    l_rs2 = ReshapeLayer(l_conv22b, (batch_size,-1))

    l_merge1=lasagne.layers.ConcatLayer([l_rs1,l_rs2],axis=1)

    l_fc=lasagne.layers.DenseLayer(l_merge1,512)

   
    l_fc11=lasagne.layers.DenseLayer(l_fc,l_conv21.output_shape[1]*l_conv21.output_shape[2]*l_conv21.output_shape[3])
    l_reshape1 = lasagne.layers.ReshapeLayer(l_fc11,(batch_size,l_conv21.output_shape[1],l_conv21.output_shape[2], l_conv21.output_shape[3]))
    l_inverse11=lasagne.layers.InverseLayer(l_reshape1, l_conv21)
    l_inverse31=lasagne.layers.InverseLayer(l_inverse11, l_pool11)
    l_inverse41=lasagne.layers.InverseLayer(l_inverse31, l_conv11)

  
    l_fc12=lasagne.layers.DenseLayer(l_fc,l_conv22.output_shape[1]*l_conv22.output_shape[2]*l_conv22.output_shape[3])
    l_reshape2 = lasagne.layers.ReshapeLayer(l_fc12,(batch_size,l_conv22.output_shape[1],l_conv22.output_shape[2], l_conv22.output_shape[3]))
    l_inverse12=lasagne.layers.InverseLayer(l_reshape2, l_conv22)
    l_inverse32=lasagne.layers.InverseLayer(l_inverse12, l_pool12)
    l_inverse42=lasagne.layers.InverseLayer(l_inverse32, l_conv12)


    l_merge=lasagne.layers.ConcatLayer([l_inverse41,l_inverse42],axis=1)

    l_out = lasagne.layers.NonlinearityLayer(lasagne.layers.BiasLayer(l_merge), nonlinearity=lasagne.nonlinearities.rectify)
   
    return l_out


def build_ca1(input_var=None, batch_size=32,time_context=30,feat_size=1025):

    input_shape=(batch_size,1,time_context,feat_size)
    #scaled_tanh = lasagne.nonlinearities.ScaledTanH(scale_in=1, scale_out=0.5)

    l_in_1 = lasagne.layers.InputLayer(shape=input_shape, input_var=input_var)

    l_conv11 = lasagne.layers.Conv2DLayer(l_in_1, num_filters=30, filter_size=(1,50),stride=(1,5), pad='valid', nonlinearity=None)
    l_conv11b= lasagne.layers.BiasLayer(l_conv11)

    l_pool11 = lasagne.layers.MaxPool2DLayer(l_conv11b, pool_size=(1, 2))

    l_conv21 = lasagne.layers.Conv2DLayer(l_pool11, num_filters=30, filter_size=(10,20),stride=(1,1), pad='valid', nonlinearity=None)
    l_conv21b= lasagne.layers.BiasLayer(l_conv21)

    l_rs1 = ReshapeLayer(l_conv21b, (batch_size,-1))


    l_conv12 = lasagne.layers.Conv2DLayer(l_in_1, num_filters=30, filter_size=(1,50),stride=(1,5), pad='valid', nonlinearity=None)
    l_conv12b= lasagne.layers.BiasLayer(l_conv12)

    l_pool12 = lasagne.layers.MaxPool2DLayer(l_conv12b, pool_size=(1, 2))

    l_conv22 = lasagne.layers.Conv2DLayer(l_pool12, num_filters=30, filter_size=(20,1),stride=(1,1), pad='valid', nonlinearity=None)
    l_conv22b= lasagne.layers.BiasLayer(l_conv22)

    l_rs2 = ReshapeLayer(l_conv22b, (batch_size,-1))

    l_merge1=lasagne.layers.ConcatLayer([l_rs1,l_rs2],axis=1)

    l_fc=lasagne.layers.DenseLayer(l_merge1,512)

   
    l_fc11=lasagne.layers.DenseLayer(l_fc,l_conv21.output_shape[1]*l_conv21.output_shape[2]*l_conv21.output_shape[3])
    l_reshape1 = lasagne.layers.ReshapeLayer(l_fc11,(batch_size,l_conv21.output_shape[1],l_conv21.output_shape[2], l_conv21.output_shape[3]))
    l_inverse11=lasagne.layers.InverseLayer(l_reshape1, l_conv21)
    l_inverse31=lasagne.layers.InverseLayer(l_inverse11, l_pool11)
    l_inverse41=lasagne.layers.InverseLayer(l_inverse31, l_conv11)

  
    l_fc12=lasagne.layers.DenseLayer(l_fc,l_conv22.output_shape[1]*l_conv22.output_shape[2]*l_conv22.output_shape[3])
    l_reshape2 = lasagne.layers.ReshapeLayer(l_fc12,(batch_size,l_conv22.output_shape[1],l_conv22.output_shape[2], l_conv22.output_shape[3]))
    l_inverse12=lasagne.layers.InverseLayer(l_reshape2, l_conv22)
    l_inverse32=lasagne.layers.InverseLayer(l_inverse12, l_pool12)
    l_inverse42=lasagne.layers.InverseLayer(l_inverse32, l_conv12)


    l_merge=lasagne.layers.ConcatLayer([l_inverse41,l_inverse42],axis=1)

    l_out = lasagne.layers.NonlinearityLayer(lasagne.layers.BiasLayer(l_merge), nonlinearity=lasagne.nonlinearities.rectify)
   
    return l_out


def build_ca_msd(input_var=None, batch_size=32,time_context=30,feat_size=1025):

    input_shape=(batch_size,1,time_context,feat_size)
    #scaled_tanh = lasagne.nonlinearities.ScaledTanH(scale_in=1, scale_out=0.5)

    l_in_1 = lasagne.layers.InputLayer(shape=input_shape, input_var=input_var)

    l_conv11 = lasagne.layers.Conv2DLayer(l_in_1, num_filters=30, filter_size=(1,30),stride=(1,3), pad='valid', nonlinearity=None)
    l_conv11b= lasagne.layers.BiasLayer(l_conv11)

    l_pool11 = lasagne.layers.MaxPool2DLayer(l_conv11b, pool_size=(1, 4))

    l_conv21 = lasagne.layers.Conv2DLayer(l_pool11, num_filters=30, filter_size=(10,20),stride=(1,1), pad='valid', nonlinearity=None)
    l_conv21b= lasagne.layers.BiasLayer(l_conv21)

    l_rs1 = ReshapeLayer(l_conv21b, (batch_size,-1))


    l_conv12 = lasagne.layers.Conv2DLayer(l_in_1, num_filters=30, filter_size=(1,30),stride=(1,3), pad='valid', nonlinearity=None)
    l_conv12b= lasagne.layers.BiasLayer(l_conv12)

    l_pool12 = lasagne.layers.MaxPool2DLayer(l_conv12b, pool_size=(1, 4))

    l_conv22 = lasagne.layers.Conv2DLayer(l_pool12, num_filters=30, filter_size=(10,20),stride=(1,1), pad='valid', nonlinearity=None)
    l_conv22b= lasagne.layers.BiasLayer(l_conv22)

    l_rs2 = ReshapeLayer(l_conv22b, (batch_size,-1))

    l_merge1=lasagne.layers.ConcatLayer([l_rs1,l_rs2],axis=1)

    l_fc=lasagne.layers.DenseLayer(l_merge1,512)

   
    l_fc11=lasagne.layers.DenseLayer(l_fc,l_conv21.output_shape[1]*l_conv21.output_shape[2]*l_conv21.output_shape[3])
    l_reshape1 = lasagne.layers.ReshapeLayer(l_fc11,(batch_size,l_conv21.output_shape[1],l_conv21.output_shape[2], l_conv21.output_shape[3]))
    l_inverse11=lasagne.layers.InverseLayer(l_reshape1, l_conv21)
    l_inverse31=lasagne.layers.InverseLayer(l_inverse11, l_pool11)
    l_inverse41=lasagne.layers.InverseLayer(l_inverse31, l_conv11)

  
    l_fc12=lasagne.layers.DenseLayer(l_fc,l_conv22.output_shape[1]*l_conv22.output_shape[2]*l_conv22.output_shape[3])
    l_reshape2 = lasagne.layers.ReshapeLayer(l_fc12,(batch_size,l_conv22.output_shape[1],l_conv22.output_shape[2], l_conv22.output_shape[3]))
    l_inverse12=lasagne.layers.InverseLayer(l_reshape2, l_conv22)
    l_inverse32=lasagne.layers.InverseLayer(l_inverse12, l_pool12)
    l_inverse42=lasagne.layers.InverseLayer(l_inverse32, l_conv12)


    l_merge=lasagne.layers.ConcatLayer([l_inverse41,l_inverse42],axis=1)

    l_out = lasagne.layers.NonlinearityLayer(lasagne.layers.BiasLayer(l_merge), nonlinearity=lasagne.nonlinearities.rectify)
   
    return l_out




def train_auto(train,fun,transform,testdir,outdir,num_epochs=30,model="1.pkl",scale_factor=0.3,load=False,skip_train=False,skip_sep=False):
    logging.info("Building Autoencoder")
    input_var2 = T.tensor4('inputs')
    target_var2 = T.tensor4('targets')
    rand_num = T.tensor4('rand_num')

    eps=1e-8
    alpha=0.9
    beta_acc=0.005
    beta_voc=0.02

    # seed = 9
    # theano_rng = theano.sandbox.rng_mrg.MRG_RandomStreams(seed)
    
    
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

    # l1_penalty = regularize_layer_params(layer2, l1)
    
    loss=abs(vocals_error+acc_error-negative_error_voc)

    params1 = lasagne.layers.get_all_params(network2, trainable=True)

    updates = lasagne.updates.adadelta(loss, params1)

    train_fn = theano.function([input_var2,target_var2], loss, updates=updates,allow_input_downcast=True)

    train_fn1 = theano.function([input_var2,target_var2], [vocals_error,acc_error,negative_error_voc,negative_error_acc], allow_input_downcast=True)

    predict_function2=theano.function([input_var2],[vocals,acc],allow_input_downcast=True)
    predict_function3=theano.function([input_var2],[prediction2[:,0:1,:,:],prediction2[:,1:2,:,:]],allow_input_downcast=True)

    # val_fn=theano.function([input_var2], loss1,updates=val_updates,allow_input_downcast=True)

    losser=[]
    loss2=[]

    if not skip_train:

        logging.info("Training...")
        #import pdb;pdb.set_trace()
        for epoch in range(num_epochs):

            train_err = 0
            train_batches = 0
            vocals_err=0
            acc_err=0        
            beta_voc=0
            beta_acc=0
            start_time = time.time()
            for batch in range(train.iteration_size): #Have to change this line, train is not a generator, just a callabale
            #for batch in range(2):
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
        #compute transform
        for f in os.listdir(testdir):
            if f.endswith(".wav"):
                loader = essentia.standard.AudioLoader(filename=testdir+f)
                audioObj = loader()
                sampleRate=audioObj[1]
                if sampleRate != 44100:
                    print 'sample rate is not consistent'
                audio = (audioObj[0][:,0] + audioObj[0][:,1]) / 2
                mag,ph=transform.compute_file(audio,phase=True)
                #magv,phv=transform.compute_file(audioObj[0][:,1],phase=True)
                mag=scale_factor*mag.astype(np.float32)

                batches,nchunks = util.generate_overlapadd(mag,input_size=mag.shape[-1],time_context=train.time_context,overlap=train.overlap,batch_size=train.batch_size,sampleRate=44100)
                output=[]
                #output1=[]

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
                    #output1.append(predict_function3(batch))

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
                # magt=transform.compute_file(audio_out)
                # import pdb;pdb.set_trace()
                writer1=essentia.standard.MonoWriter(filename=outdir+f.replace(".wav","-voice.wav"))
                writer1(audio_out)
                writer2=essentia.standard.MonoWriter(filename=outdir+f.replace(".wav","-music.wav"))
                writer2(audio_out2)   
                writer2=None
                writer1=None   

    return losser  




if __name__ == "__main__": 
    if len(sys.argv)>-1:
        climate.add_arg('--db', help="the ikala dataset path")
        climate.add_arg('--model', help="the name of the model to test/save")
        climate.add_arg('--nepochs', help="number of epochs to train the net")
        climate.add_arg('--time_context', help="number of frames for the recurrent/lstm/conv net")
        climate.add_arg('--batch_size', help="batch size for training")
        climate.add_arg('--batch_memory', help="number of big batches to load into memory")
        climate.add_arg('--id_transform', help="id of the fft/cqt transform")
        climate.add_arg('--overlap', help="overlap time context for training")
        climate.add_arg('--nprocs', help="number of processor to parallelize file reading")
        climate.add_arg('--scale_factor', help="scale factor for the data")
        #import pdb;pdb.set_trace()
        kwargs = climate.parse_args()
        #import pdb;pdb.set_trace()
        #print 'args: ' + kwargs.get('input_pkl') + '   '+ kwargs.get('output_path')
        if kwargs.__getattribute__('db'):
            db = kwargs.__getattribute__('db')
        else:
            db='/home/marius/Documents/Database/iKala/'      
        if kwargs.__getattribute__('model'):
            model = kwargs.__getattribute__('model')
        else:
            model="fft_1024_1_split_filtmus"
        if kwargs.__getattribute__('id_transform'):
            id_transform = int(kwargs.__getattribute__('id_transform')) 
        else:
            id_transform = 1
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

    pitchhop=0.032*44100.0 #seconds to frames
    tt=[]
    tt.append(transformFFT(frameSize=2048, hopSize=512, sampleRate=44100, window=blackmanharris))
    tt.append(transformFFT(frameSize=1024, hopSize=512, sampleRate=44100, window=blackmanharris))
    tt.append(transformFFT2(frameSize=2048, hopSize=512, sampleRate=44100, window=blackmanharris))
    tt.append(transformFFT2(frameSize=1024, hopSize=512, sampleRate=44100, window=blackmanharris))
    # tt.append(transformCQT(frameSize=1024, hopSize=512, bins=513, sampleRate=44100, window=blackmanharris, tffmin=50, tffmax=14000, iscale = 'log'))
    # tt.append(transformCQT(frameSize=2048, hopSize=512, bins=1025, sampleRate=44100, window=blackmanharris, tffmin=50, tffmax=14000, iscale = 'log'))
    
    dirtt=[]
    dirtt.append(db +'transforms/t1/')
    dirtt.append(db +'transforms/t2/')
    dirtt.append(db +'transforms/t3/')
    dirtt.append(db +'transforms/t4/')

    ld1 = LargeDataset(path_transform_in=dirtt[id_transform], batch_size=batch_size, batch_memory=batch_memory, time_context=time_context, overlap=overlap, nprocs=nprocs,mult_factor_in=scale_factor,mult_factor_out=scale_factor)
    logging.info("  Maximum:\t\t{:.6f}".format(ld1.getMax()))
    logging.info("  Mean:\t\t{:.6f}".format(ld1.getMean()))
    logging.info("  Standard dev:\t\t{:.6f}".format(ld1.getStd()))
    # ld1 = LargeDatasetPitch1(path_transform_in=dinf1, batch_size=batch_size, batch_memory=200, time_context=time_context, overlap=10, nprocs=8,
    #     sampleRate=tt1.sampleRate,fmin=tt1.fmin, fmax=tt1.fmax,ttype=tt1.ttype,iscale=tt1.iscale, 
    #     nharmonics=20, interval=50, tuning_freq=440, pitched=True, save_mask=True, pitch_norm=127.)
    # model="fft1"

    # ld2 = LargeDatasetPitch1(path_transform_in=dinc2, batch_size=batch_size, batch_memory=10, time_context=time_context, overlap=10,
    #     sampleRate=tt2.sampleRate,fmin=tt2.fmin, fmax=tt2.fmax,ttype=tt2.ttype,iscale=tt2.iscale, 
    #     nharmonics=20, interval=50, tuning_freq=440, pitched=True, save_mask=True, pitch_norm=127.)

    if not os.path.exists(db+'output/'):
        os.makedirs(db+'output/')
    if not os.path.exists(db+'output/'+model):
        os.makedirs(db+'output/'+model)
    if not os.path.exists(db+'models/'):
        os.makedirs(db+'models/')

    train_errs=train_auto(train=ld1,fun=build_ca,transform=tt[id_transform],outdir=db+'output/'+model+"/",testdir=db+'Wavfile/',model=db+"models/"+"model_"+model+".pkl",num_epochs=nepochs,scale_factor=scale_factor)   #,load=True,skip_train=True     
    f = file(db+"models/"+"loss_"+model+".data", 'wb')
    cPickle.dump(train_errs,f,protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()

