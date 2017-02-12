import os,sys
import transform
from transform import transformFFT
from scipy.io.wavfile import read
import numpy as np
import re
from scipy.signal import blackmanharris as blackmanharris



db='/home/marius/Documents/Database/iKala/'
#db='/Volumes/Macintosh HD 2/Documents/Database/iKala/'
pitchhop=0.032*44100.0 #seconds to frames
tt1=transformFFT(frameSize=2048, hopSize=512, sampleRate=44100, window=blackmanharris)
tt2=transformFFT(frameSize=1024, hopSize=512, sampleRate=44100, window=blackmanharris)
# tt3=transformFFT2(frameSize=2048, hopSize=512, sampleRate=44100, window=blackmanharris)
# tt4=transformFFT2(frameSize=1024, hopSize=512, sampleRate=44100, window=blackmanharris)
# tt5=transformCQT(frameSize=2048, hopSize=512, bins=1025, sampleRate=44100, window=np.hanning, tffmin=50, tffmax=14000, iscale = 'log')
# tt6=transformCQT(frameSize=1024, hopSize=512, bins=513, sampleRate=44100, window=np.hanning, tffmin=50, tffmax=14000, iscale = 'log')
#ttt = transform1(frameSize=1024, hopSize=512,window = blackmanharris,ttype ='mqt')
# tw=transformWP(frameSize=1024, hopSize=512, sampleRate=44100, window=blackmanharris)
#compute transform
for f in os.listdir(db+"Wavfile"):
    if f.endswith(".wav"):
        loader = essentia.standard.AudioLoader(filename=db+"Wavfile/"+f)
        audioObj = loader()
        sampleRate=audioObj[1]
        if sampleRate != 44100:
            print 'sample rate is not consistent'
        audio = np.zeros((audioObj[0].shape[0],3))
        audio[:,0] = (audioObj[0][:,0] + audioObj[0][:,1]) / 2
        audio[:,1] = audioObj[0][:,1] #voice
        audio[:,2] = audioObj[0][:,0] #acc
        audioObj=None 
        lines = np.loadtxt(db+"PitchLabel/"+f.replace('wav','pv'), comments="#", delimiter="\n", unpack=False)
        if not os.path.exists(db+"transforms/t1/"):
            os.makedirs(db+"transforms/t1/")
        if not os.path.exists(db+"transforms/t2/"):
            os.makedirs(db+"transforms/t2/")
        if not os.path.exists(db+"transforms/t3/"):
            os.makedirs(db+"transforms/t3/")
        if not os.path.exists(db+"transforms/t4/"):
            os.makedirs(db+"transforms/t4/")
      
        # mag,ph=tt6.compute_file(audio[:,1],phase=True)
        # audio_out=tt6.compute_inverse(mag,ph)
        # magt=tt6.compute_file(audio_out)
        # magt = magt[:mag.shape[0]]
        # import pdb;pdb.set_trace()   

        # mag,ph=ttt.compute_file(audio[:,1],phase=True)
        # audio_out=ttt.compute_inverse(mag,ph)
        # magt=ttt.compute_file(audio_out)
        # magt = magt[:mag.shape[0]]
        # import pdb;pdb.set_trace()  

        tt1.compute_transform(audio,db+"transforms/t1/"+f.replace('.wav','.data'),pitch=lines[np.newaxis,np.newaxis,:],phase=False)
        tt2.compute_transform(audio,db+"transforms/t2/"+f.replace('.wav','.data'),pitch=lines[np.newaxis,np.newaxis,:],phase=False)
        # tt3.compute_transform(audio,db+"transforms/t3/"+f.replace('.wav','.data'),pitch=lines[np.newaxis,np.newaxis,:],phase=False)
        # tt4.compute_transform(audio,db+"transforms/t4/"+f.replace('.wav','.data'),pitch=lines[np.newaxis,np.newaxis,:],phase=False)
        # #import pdb;pdb.set_trace()
       
# import dataset_general
# from dataset_general import LargeDatasetPitch1
# dinf='/home/marius/Documents/Database/iKala/transforms/t1/'
# dinc='/home/marius/Documents/Database/iKala/transforms/t2/'

# ld1 = LargeDatasetPitch1(path_transform_in=dinf, batch_size=32, batch_memory=10, time_context=30, overlap=10,
#     sampleRate=tt1.sampleRate,fmin=tt1.fmin, fmax=tt1.fmax,ttype=tt1.ttype,iscale=tt1.iscale, 
#     nharmonics=20, interval=50, tuning_freq=440, pitched=True, save_mask=True, pitch_norm=127.)

# ld2 = LargeDatasetPitch1(path_transform_in=dinc, batch_size=32, batch_memory=10, time_context=30, overlap=10,
#     sampleRate=tt2.sampleRate,fmin=tt2.fmin, fmax=tt2.fmax,ttype=tt2.ttype,iscale=tt2.iscale, 
#     nharmonics=20, interval=50, tuning_freq=440, pitched=True, save_mask=True, pitch_norm=127.)
