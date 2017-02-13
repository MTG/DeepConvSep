import os

#path_in is the output folder where the separation was saved
path_in = '/home/marius/Documents/Database/DSD100/DSD6/'
sources = ['_bass','_drums','_vocals','_others']
sourcesd = ['bass','drums','vocals','other']

filelist = [f.split('_bass',1)[0] for f in os.listdir(path_in) if f.endswith('_bass.wav') ]

for f in filelist:
    if not os.path.exists(os.path.join(path_in,f)):
        os.makedirs(os.path.join(path_in,f))
        for s in range(len(sources)):
            os.rename(os.path.join(path_in,f+sources[s]+'.wav'),os.path.join(path_in,f,sourcesd[s]+'.wav'))

