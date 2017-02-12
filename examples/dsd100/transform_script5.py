import os,sys
import transform
from transform import transformFFT2
from scipy.io.wavfile import read
import numpy as np
import re
import essentia
from essentia.standard import *


megatron=transformFFT2()

#directory = raw_input('Enter MSD directory for  transformation, remove space at end')
directory='/home/pritish/pc_bss/datasets/MSD100'
#directory='/Users/pc2752/Desktop/Thesis_Datasets/MSD100'
# typeofmix=raw_input('Enter v for vocals, b for bass or d for drums')
typeofmix='v'
mixture_directory=directory+'/Mixtures'
source_directory=directory+'/Sources'

#Others and drums are stereo. Vocals and Bass are mono.

subdirectories = os.listdir(mixture_directory+'/Test')
print 'Analysing Test'

count=0
for i in range(40,50):
	x=subdirectories[i]
	print(x)
	if not x.startswith('.'):
		if count<50:
			count=count+1
			print str(count/50.0*100.0)+'% done'
			#mix_raw=np.average(read(mixture_directory+'/Test/'+x+"/mixture.wav")[1],axis=1)
			loader1 = essentia.standard.MonoLoader(filename = mixture_directory+'/Test/'+x+"/mixture.wav",sampleRate=44100)
			mix_raw=loader1()
			number_of_blocks=len(mix_raw)/(44100*30)
			last_block=len(mix_raw)%44100


	#cd /home/pritish/pc_bss/datasets/MSD100
	#/home/pritish/pc_bss/nueralnets/neuralnets/rwc_cello
	        
			# vocals=read(source_directory+'/Test/'+x+"/vocals.wav")[1]
			# bass=read(source_directory+'/Test/'+x+"/bass.wav")[1]
			# drums=read(source_directory+'/Test/'+x+"/drums.wav")[1]
			# others=read(source_directory+'/Test/'+x+"/other.wav")[1]
			loader2 = essentia.standard.MonoLoader(filename = source_directory+'/Test/'+x+"/vocals.wav",sampleRate=44100)
			vocals=loader2()
			loader3 = essentia.standard.MonoLoader(filename = source_directory+'/Test/'+x+"/bass.wav",sampleRate=44100)
			bass=loader3()		
			loader4 = essentia.standard.MonoLoader(filename = source_directory+'/Test/'+x+"/drums.wav",sampleRate=44100)
			drums=loader4()
			loader5 = essentia.standard.MonoLoader(filename = source_directory+'/Test/'+x+"/other.wav",sampleRate=44100)
			others=loader5()	
			# if isinstance(vocals[0],np.ndarray): #if stereo convert to mono
			# 	vocals=np.average(vocals,axis=1)
			# if isinstance(bass[0],np.ndarray):
			# 	bass=np.average(bass,axis=1)
			# if isinstance(others[0],np.ndarray):
			# 	others=np.average(others,axis=1)
			# if isinstance(drums[0],np.ndarray):
			# 	drums=np.average(drums,axis=1) 	 
			#print(mix_raw.max())

			if typeofmix=='v':
				#others=(others+drums+bass)/3
				#others=others/max(abs(others))    
				target=vocals
			elif typeofmix=='b':
				#others=(others+drums+vocals)/3
				#others=others/max(abs(others)) 
				target=bass
			elif typeofmix=='d':
				#others=(others+bass+vocals)/3
				#others=others/max(abs(others)) 
				target=drums
			others=mix_raw-target
			#print(target.max())
			#print(others.max())
			target1=[]
			others1=[]
			mix=[]
			for i in range(len(target)/441):
				if abs(target[441*i:441*(i+1)]).mean()>0.02:
					target1=np.concatenate((target1,target[441*i:441*(i+1)]),axis=0)
					others1=np.concatenate((others1,others[441*i:441*(i+1)]),axis=0)
					mix=np.concatenate((mix,mix_raw[441*i:441*(i+1)]),axis=0)
			number_of_blocks=len(target1)/(44100*30)
			last_block=len(target1)%44100				
			print "starting"		
			for i in range(number_of_blocks):
				b=np.ndarray(shape=(44100*30,3), dtype=float, order='F')
				b[:,0]=mix[i*30*44100:(i+1)*30*44100] #Take samples of 30 secs, assuming sample rate is 44100, this might not be true for all cases and should be checked.
				b[:,1]=target1[i*44100*30:(i+1)*30*44100]
				b[:,2]=others1[i*44100*30:(i+1)*44100*30]
				#print directory+'/Data/'+x+'_'+str(i)+'.data'
				megatron.compute_transform(b,directory+'/Data4/FFT/Test/'+typeofmix+'/'+x+'_'+str(i)+'.data',phase=False)
			
			one=mix[(i+1)*30*44100:]
			b=np.ndarray(shape=(len(one),3), dtype=float, order='F')
			b[:,0]=one
			b[:,1]=target1[(i+1)*30*44100:]
			b[:,2]=others1[44100*30*(i+1):]
			megatron.compute_transform(b,directory+'/Data4/FFT/Test/'+typeofmix+'/'+x+'_'+str(i+1)+'.data',phase=False)

