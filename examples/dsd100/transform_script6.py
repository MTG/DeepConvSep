import os,sys
import transform
from transform import transformFFT2
from scipy.io.wavfile import read
import numpy as np
import re
import essentia
from essentia.standard import *


megatron=transformFFT2(frameSize=256, hopSize=64)

#directory = raw_input('Enter MSD directory for  transformation, remove space at end')
directory='/home/pritish/pc_bss/datasets/MSD100'
#typeofmix=raw_input('Enter v for vocals, b for bass or d for drums')
mixture_directory=directory+'/Mixtures'
source_directory=directory+'/Sources'

#Others and drums are stereo. Vocals and Bass are mono.

subdirectories = os.listdir(mixture_directory+'/Dev')
print 'Analysing Dev'
count=0
for x in subdirectories:
	if not x.startswith('.'):
		count=count+1
		print str(count/50.0*100.0)+'% done'
		#mix_raw=np.average(read(mixture_directory+'/Dev/'+x+"/mixture.wav")[1],axis=1)
		#loader1 = essentia.standard.MonoLoader(filename = mixture_directory+'/Dev/'+x+"/mixture.wav",sampleRate=44100)
		#mix_raw=loader1()



#cd /home/pritish/pc_bss/datasets/MSD100/Data
#/home/pritish/pc_bss/nueralnets/neuralnets/rwc_cello
        
		# vocals=read(source_directory+'/Dev/'+x+"/vocals.wav")[1]
		# bass=read(source_directory+'/Dev/'+x+"/bass.wav")[1]
		# drums=read(source_directory+'/Dev/'+x+"/drums.wav")[1]
		# others=read(source_directory+'/Dev/'+x+"/other.wav")[1]
		#loader2 = essentia.standard.MonoLoader(filename = source_directory+'/Dev/'+x+"/vocals.wav",sampleRate=44100)
		#vocals=loader2()
		#loader3 = essentia.standard.MonoLoader(filename = source_directory+'/Dev/'+x+"/bass.wav",sampleRate=44100)
		#bass=loader3()		
		loader4 = essentia.standard.MonoLoader(filename = source_directory+'/Dev/'+x+"/drums.wav",sampleRate=44100)
		audio=loader4()
		#loader5 = essentia.standard.MonoLoader(filename = source_directory+'/Dev/'+x+"/other.wav",sampleRate=44100)
		#others=loader5()	
		# if isinstance(vocals[0],np.ndarray): #if stereo convert to mono
		# 	vocals=np.average(vocals,axis=1)
		# if isinstance(bass[0],np.ndarray):
		# 	bass=np.average(bass,axis=1)
		# if isinstance(others[0],np.ndarray):
		# 	others=np.average(others,axis=1)
		# if isinstance(drums[0],np.ndarray):
		# 	drums=np.average(drums,axis=1) 	 
		#print(mix_raw.max())
		oppy=[]
		for i in range(len(audio)/441):
			if abs(audio[441*i:441*(i+1)]).mean()>0.02:
				oppy=np.concatenate((oppy,audio[441*i:441*(i+1)]),axis=0)

		number_of_blocks=len(oppy)/(44100*30)
		last_block=len(oppy)%44100
		# if typeofmix=='v':
		# 	#others=(others+drums+bass)/3
		# 	#others=others/max(abs(others))    
		# 	target=vocals
		# elif typeofmix=='b':
		# 	#others=(others+drums+vocals)/3
		# 	#others=others/max(abs(others)) 
		# 	target=bass
		# elif typeofmix=='d':
		# 	#others=(others+bass+vocals)/3
		# 	#others=others/max(abs(others)) 
		#target=drums
		#others=mix_raw-target
		#print(target.max())
		#print(others.max())
		for i in range(number_of_blocks):
			b=np.ndarray(shape=(44100*30,2), dtype=float, order='F')
			b[:,0]=oppy[i*30*44100:(i+1)*30*44100] #Take samples of 30 secs, assuming sample rate is 44100, this might not be true for all cases and should be checked.
			b[:,1]=oppy[i*44100*30:(i+1)*30*44100]
			#b[:,2]=others[i*44100*30:(i+1)*44100*30]
			#print directory+'/Data/'+x+'_'+str(i)+'.data'
			megatron.compute_transform(b,directory+'/Data/FFT/Dev/d'+'/'+x+'_'+str(i)+'.data',phase=False)
		
		one=oppy[(i+1)*30*44100:]
		b=np.ndarray(shape=(len(one),2), dtype=float, order='F')
		b[:,0]=one
		b[:,1]=oppy[(i+1)*30*44100:]
		#b[:,2]=others[44100*30*(i+1):]
		megatron.compute_transform(b,directory+'/Data/FFT/Dev/d'+'/'+x+'_'+str(i+1)+'.data',phase=False)

