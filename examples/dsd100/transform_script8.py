import os,sys
import transform
from transform import transformFFT2
import numpy as np
import re
import essentia
from essentia.standard import *


megatron=transformFFT2()

#directory = raw_input('Enter MSD directory for  transformation, remove space at end')
directory='/home/pritish/pc_bss/datasets/DSD100'
#directory='/Users/pc2752/Desktop/Thesis_Datasets/MSD100'
# typeofmix=raw_input('Enter v for vocals, b for bass or d for drums')
typeofmix='v'
mixture_directory=directory+'/Mixtures'
source_directory=directory+'/Sources'

#Others and drums are stereo. Vocals and Bass are mono.
opdir='/home/pritish/pc_bss/datasets/DSD100/Data'
subdirectories = os.listdir(mixture_directory+'/Dev')
print 'Analysing Dev'
    #cd /home/pritish/pc_bss/datasets/MSD100
    #/home/pritish/pc_bss/nueralnets/neuralnets/rwc_cello
cc=1
count=0
for x in subdirectories:
    print(x)
    if not x.startswith('.'):
        if count==10:
            count=0
            cc+=1
        if count==0:
            diri=opdir+"/"+str(cc)
            if not os.path.exists(diri):
                os.makedirs(diri)
                print("made directory"+diri)
        if count<10:
            count=count+1
            print str(count/50.0*100.0)+'% done'
            #mix_raw=np.average(read(mixture_directory+'/Dev/'+x+"/mixture.wav")[1],axis=1)
            loader1 = essentia.standard.MonoLoader(filename = mixture_directory+'/Dev/'+x+"/mixture.wav",sampleRate=44100)
            mix_raw=loader1()
            loader2 = essentia.standard.MonoLoader(filename = source_directory+'/Dev/'+x+"/vocals.wav",sampleRate=44100)
            vocals=loader2()
            loader3 = essentia.standard.MonoLoader(filename = source_directory+'/Dev/'+x+"/bass.wav",sampleRate=44100)
            bass=loader3()      
            loader4 = essentia.standard.MonoLoader(filename = source_directory+'/Dev/'+x+"/drums.wav",sampleRate=44100)
            drums=loader4()
            loader5 = essentia.standard.MonoLoader(filename = source_directory+'/Dev/'+x+"/other.wav",sampleRate=44100)
            others=loader5()    

            number_of_blocks=len(mix_raw)/(44100*30)          
            print "starting"        
            for i in range(number_of_blocks):
                b=np.ndarray(shape=(44100*30,5), dtype=float, order='F')
                b[:,0]=mix_raw[i*30*44100:(i+1)*30*44100] #Take samples of 30 secs, assuming sample rate is 44100, this might not be true for all cases and should be checked.
                b[:,1]=vocals[i*44100*30:(i+1)*30*44100]
                b[:,2]=bass[i*44100*30:(i+1)*44100*30]
                b[:,3]=drums[i*44100*30:(i+1)*44100*30]
                b[:,4]=others[i*44100*30:(i+1)*44100*30]
                #print directory+'/Data/'+x+'_'+str(i)+'.data'
                megatron.compute_transform(b,diri+"/"+x+"_"+str(i)+'.data',phase=False)
            
            one=mix_raw[(i+1)*30*44100:]
            b=np.ndarray(shape=(len(one),5), dtype=float, order='F')
            b[:,0]=one
            b[:,1]=vocals[(i+1)*30*44100:]
            b[:,2]=bass[44100*30*(i+1):]
            b[:,3]=drums[44100*30*(i+1):]
            b[:,4]=others[44100*30*(i+1):]
            megatron.compute_transform(b,diri+"/"+x+"_"+str(i+1)+'.data',phase=False)

subdirectories = os.listdir(mixture_directory+'/Test')
print 'Analysing Test'
    #cd /home/pritish/pc_bss/datasets/MSD100
    #/home/pritish/pc_bss/nueralnets/neuralnets/rwc_cello

for x in subdirectories:
    print(x)
    if not x.startswith('.'):
        if count==10:
            count=0
            cc+=1
        if count==0:
            diri=opdir+"/"+str(cc)
            if not os.path.exists(diri):
                os.makedirs(diri)
                print("made directory"+diri)
        if count<10:
            count=count+1
            print str(count/50.0*100.0)+'% done'
            #mix_raw=np.average(read(mixture_directory+'/Dev/'+x+"/mixture.wav")[1],axis=1)
            loader1 = essentia.standard.MonoLoader(filename = mixture_directory+'/Test/'+x+"/mixture.wav",sampleRate=44100)
            mix_raw=loader1()
            loader2 = essentia.standard.MonoLoader(filename = source_directory+'/Test/'+x+"/vocals.wav",sampleRate=44100)
            vocals=loader2()
            loader3 = essentia.standard.MonoLoader(filename = source_directory+'/Test/'+x+"/bass.wav",sampleRate=44100)
            bass=loader3()      
            loader4 = essentia.standard.MonoLoader(filename = source_directory+'/Test/'+x+"/drums.wav",sampleRate=44100)
            drums=loader4()
            loader5 = essentia.standard.MonoLoader(filename = source_directory+'/Test/'+x+"/other.wav",sampleRate=44100)
            others=loader5()    

            number_of_blocks=len(mix_raw)/(44100*30)          
            print "starting"        
            for i in range(number_of_blocks):
                b=np.ndarray(shape=(44100*30,5), dtype=float, order='F')
                b[:,0]=mix_raw[i*30*44100:(i+1)*30*44100] #Take samples of 30 secs, assuming sample rate is 44100, this might not be true for all cases and should be checked.
                b[:,1]=vocals[i*44100*30:(i+1)*30*44100]
                b[:,2]=bass[i*44100*30:(i+1)*44100*30]
                b[:,3]=drums[i*44100*30:(i+1)*44100*30]
                b[:,4]=others[i*44100*30:(i+1)*44100*30]
                #print directory+'/Data/'+x+'_'+str(i)+'.data'
                megatron.compute_transform(b,diri+"/"+x+"_"+str(i)+'.data',phase=False)
            
            one=mix_raw[(i+1)*30*44100:]
            b=np.ndarray(shape=(len(one),5), dtype=float, order='F')
            b[:,0]=one
            b[:,1]=vocals[(i+1)*30*44100:]
            b[:,2]=bass[44100*30*(i+1):]
            b[:,3]=drums[44100*30*(i+1):]
            b[:,4]=others[44100*30*(i+1):]
            megatron.compute_transform(b,diri+"/"+x+"_"+str(i+1)+'.data',phase=False)
