import sys,os, getopt
import numpy as np
import scipy
from scipy.signal import blackmanharris as blackmanharris
from scipy import io
from scipy.io import wavfile
try:
    import cPickle as pickle
except:
    import pickle
import theano
import theano.tensor as T
import theano.sandbox.rng_mrg
import lasagne


def load_model(filename):
    f=file(filename,'rb')
    params=pickle.load(f)
    f.close()
    return params


def compute_file(audio, phase=False, frameSize=1024, hopSize=512, sampleRate=44100, window=np.hanning):
    win = window(frameSize)
    X = stft_norm(audio, window=win, hopsize=float(hopSize), nfft=float(frameSize), fs=float(sampleRate))
    mag = np.abs(X)  
    mag = mag  / np.sqrt(frameSize) 
    if phase:
        ph = np.angle(X)
        return mag,ph
    else:
        return mag


def compute_inverse(mag, phase, frameSize=1024, hopSize=512, sampleRate=44100, window=np.hanning):
    win = window(frameSize)
    mag = mag  * np.sqrt(frameSize) 
    Xback = mag * np.exp(1j*phase)
    data = istft_norm(Xback, window=win, analysisWindow=win, hopsize=float(hopSize), nfft=float(frameSize))
    return data                


def sinebell(lengthWindow):
    window = np.sin((np.pi*(np.arange(lengthWindow)))/(1.0*lengthWindow))
    return window


def stft_norm(data, window=sinebell(2048),
         hopsize=512.0, nfft=2048.0, fs=44100.0):

    lengthWindow = window.size   
    lengthData = data.size    
  
    numberFrames = int(np.ceil(lengthData / np.double(hopsize)) + 2)
    newLengthData = int((numberFrames-1) * hopsize + lengthWindow)
    
    data = np.concatenate((np.zeros(int(lengthWindow/2.0)), data))
    
    data = np.concatenate((data, np.zeros(newLengthData - data.size)))
    
    numberFrequencies = int(nfft / 2 + 1)
    
    STFT = np.zeros([numberFrequencies, numberFrames], dtype=complex)
    
    for n in np.arange(numberFrames):
        beginFrame = int(n*hopsize)
        endFrame = beginFrame+lengthWindow
        frameToProcess = window*data[beginFrame:endFrame]
        STFT[:,n] = np.fft.rfft(frameToProcess, np.int32(nfft));

    F = np.arange(numberFrequencies)/np.double(nfft)*fs
    N = np.arange(numberFrames)*hopsize/np.double(fs)
    
    return STFT.T


def istft_norm(X, window=sinebell(2048),
          analysisWindow=None,
          hopsize=512.0, nfft=2048.0):
  
    X=X.T
    if analysisWindow is None:
        analysisWindow = window
    
    lengthWindow = np.array(window.size)
    numberFrequencies, numberFrames = X.shape
    lengthData = int(hopsize*(numberFrames-1) + lengthWindow)
    
    normalisationSeq = np.zeros(lengthData)
    
    data = np.zeros(lengthData)
    
    for n in np.arange(numberFrames):
        beginFrame = int(n * hopsize)
        endFrame = beginFrame + lengthWindow
        frameTMP = np.fft.irfft(X[:,n], np.int32(nfft))
        frameTMP = frameTMP[:lengthWindow]
        normalisationSeq[beginFrame:endFrame] = (
            normalisationSeq[beginFrame:endFrame] +
            window * analysisWindow)
        data[beginFrame:endFrame] = (
            data[beginFrame:endFrame] + window * frameTMP)
    
    data = data[int(lengthWindow/2.0):]
    normalisationSeq = normalisationSeq[int(lengthWindow/2.0):]
    normalisationSeq[normalisationSeq==0] = 1.
  
    data = data / normalisationSeq
    
    return data


def generate_overlapadd(allmix,input_size=513,time_context=30, overlap=10,batch_size=32,sampleRate=44100):
    window = np.linspace(0., 1.0, num=overlap)
    window = np.concatenate((window,window[::-1]))
    window = np.repeat(np.expand_dims(window, axis=1),input_size,axis=1)
  
    if input_size == allmix.shape[-1]:
        
        i=0
        start=0  
        while (start + time_context) < allmix.shape[0]:
            i = i + 1
            start = start - overlap + time_context 
        fbatch = np.empty([int(np.ceil(float(i)/batch_size)),batch_size,1,time_context,input_size])

        i=0
        start=0  
 
        while (start + time_context) < allmix.shape[0]:
            fbatch[int(i/batch_size),int(i%batch_size),:,:,:]=allmix[start:start+time_context,:]
            i = i + 1 
            start = start - overlap + time_context 
    return fbatch,i



def overlapadd_multi(fbatch,obatch,nchunks,overlap=10):

    input_size=fbatch.shape[-1]
    time_context=fbatch.shape[-2]
    batch_size=fbatch.shape[2]
    nsources = fbatch.shape[1]
    #print time_context
    #print batch_size

    #window = np.sin((np.pi*(np.arange(2*overlap+1)))/(2.0*overlap))
    window = np.linspace(0., 1.0, num=overlap)
    window = np.concatenate((window,window[::-1]))
    #time_context = net.network.find('hid2', 'hh').size
    # input_size = net.layers[0].size  #input_size is the number of spectral bins in the fft
    window = np.repeat(np.expand_dims(window, axis=1),input_size,axis=1)
    sep = np.zeros((nsources, nchunks*(time_context-overlap)+time_context, input_size)) #allocate for output of prediction 
    for s in range(nsources):
        i=0
        start=0 
        while i < nchunks:
            # import pdb;pdb.set_trace()
            fbatch1=fbatch[:,s,:,:,:]
            source= fbatch1[int(i/batch_size),int(i%batch_size),0,:,:]
            if start==0:
                sep[s,0:time_context] = source
            else:
                sep[s,start+overlap:start+time_context] = source[overlap:time_context]
                sep[s,start:start+overlap] = window[overlap:]*sep[s,start:start+overlap] + window[:overlap]*source[:overlap]
            i = i + 1 #index for each block
            start = start - overlap + time_context #starting point for each block
    return sep


def build_ca(input_var=None, batch_size=32,time_context=30,feat_size=513):
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

    input_shape=(batch_size,1,time_context,feat_size)

    #input layer
    l_in_1 = lasagne.layers.InputLayer(shape=input_shape, input_var=input_var)

    #vertical convolution layer
    l_conv1 = lasagne.layers.Conv2DLayer(l_in_1, num_filters=50, filter_size=(1,feat_size),stride=(1,1), pad='valid', nonlinearity=None)
    l_conv1b= lasagne.layers.BiasLayer(l_conv1)

    #horizontal convolution layer
    l_conv2 = lasagne.layers.Conv2DLayer(l_conv1b, num_filters=50, filter_size=(int(time_context/2),1),stride=(1,1), pad='valid', nonlinearity=None)
    l_conv2b= lasagne.layers.BiasLayer(l_conv2)

    #bottlneck layer
    l_fc=lasagne.layers.DenseLayer(l_conv2b,128)

    #build output for source1
    l_fc11=lasagne.layers.DenseLayer(l_fc,l_conv2.output_shape[1]*l_conv2.output_shape[2]*l_conv2.output_shape[3])
    l_reshape1 = lasagne.layers.ReshapeLayer(l_fc11,(batch_size,l_conv2.output_shape[1],l_conv2.output_shape[2], l_conv2.output_shape[3]))
    l_inverse11=lasagne.layers.InverseLayer(l_reshape1, l_conv2)
    l_inverse41=lasagne.layers.InverseLayer(l_inverse11, l_conv1)

    #build output for source2
    l_fc12=lasagne.layers.DenseLayer(l_fc,l_conv2.output_shape[1]*l_conv2.output_shape[2]*l_conv2.output_shape[3])
    l_reshape2 = lasagne.layers.ReshapeLayer(l_fc12,(batch_size,l_conv2.output_shape[1],l_conv2.output_shape[2], l_conv2.output_shape[3]))
    l_inverse12=lasagne.layers.InverseLayer(l_reshape2, l_conv2)
    l_inverse42=lasagne.layers.InverseLayer(l_inverse12, l_conv1)

    #build output for source3
    l_fc13=lasagne.layers.DenseLayer(l_fc,l_conv2.output_shape[1]*l_conv2.output_shape[2]*l_conv2.output_shape[3])
    l_reshape3 = lasagne.layers.ReshapeLayer(l_fc13,(batch_size,l_conv2.output_shape[1],l_conv2.output_shape[2], l_conv2.output_shape[3]))
    l_inverse13=lasagne.layers.InverseLayer(l_reshape3, l_conv2)
    l_inverse43=lasagne.layers.InverseLayer(l_inverse13, l_conv1)

    #build output for source4
    l_fc14=lasagne.layers.DenseLayer(l_fc,l_conv2.output_shape[1]*l_conv2.output_shape[2]*l_conv2.output_shape[3])
    l_reshape4 = lasagne.layers.ReshapeLayer(l_fc12,(batch_size,l_conv2.output_shape[1],l_conv2.output_shape[2], l_conv2.output_shape[3]))
    l_inverse14=lasagne.layers.InverseLayer(l_reshape4, l_conv2)
    l_inverse44=lasagne.layers.InverseLayer(l_inverse14, l_conv1)

    #build final output 
    l_merge=lasagne.layers.ConcatLayer([l_inverse41,l_inverse42,l_inverse43,l_inverse44],axis=1)
    l_out = lasagne.layers.NonlinearityLayer(lasagne.layers.BiasLayer(l_merge), nonlinearity=lasagne.nonlinearities.rectify)
   
    return l_out


def train_auto(filein,outdir,model,scale_factor=0.3,time_context = 30,overlap = 20,batch_size=32,input_size=513):
    input_var2 = T.tensor4('inputs')
    target_var2 = T.tensor4('targets')
    rand_num = T.tensor4('rand_num')
    source = ['vocals','bass','drums','other']

    eps=1e-8
    network2 = build_ca(input_var2,batch_size,time_context,input_size)    
  
    #print("Loading model...")
    params=load_model(model)
    lasagne.layers.set_all_param_values(network2,params)

    prediction2 = lasagne.layers.get_output(network2, deterministic=True)
    rand_num = np.random.uniform(size=(batch_size,1,time_context,input_size))
    network2=None
    params=None
    rand_num = np.random.uniform(size=(batch_size,1,time_context,input_size))

    voc=prediction2[:,0:1,:,:]+eps*rand_num
    bas=prediction2[:,1:2,:,:]+eps*rand_num
    dru=prediction2[:,2:3,:,:]+eps*rand_num
    oth=prediction2[:,3:4,:,:]+eps*rand_num

    mask1=voc/(voc+bas+dru+oth)
    mask2=bas/(voc+bas+dru+oth)
    mask3=dru/(voc+bas+dru+oth)
    mask4=oth/(voc+bas+dru+oth)

    vocals=mask1*input_var2
    bass=mask2*input_var2
    drums=mask3*input_var2
    others=mask4*input_var2  

    predict_function2=theano.function([input_var2],[vocals,bass,drums,others],allow_input_downcast=True) 
 
    sampleRate, audioObj = scipy.io.wavfile.read(filein)  
    
    try:
        maxv = np.finfo(audioObj.dtype).max
    except:
        maxv = np.iinfo(audioObj.dtype).max

    audioObj = audioObj.astype('float') / maxv
      
    if sampleRate == 44100:     
        if (len(audioObj.shape))>1 and (audioObj.shape[1]>1):
            audioObj[:,0] = (audioObj[:,0] + audioObj[:,1]) / 2
            audioObj = audioObj[:,0]
   
        mag,ph=compute_file(audioObj,phase=True)     
        mag=scale_factor*mag.astype(np.float32)

        batches,nchunks = generate_overlapadd(mag,input_size=mag.shape[-1],time_context=time_context,overlap=overlap,batch_size=batch_size,sampleRate=44100)
        output=[]
 
        batch_no=1
        for batch in batches:
            batch_no+=1           
            output.append(predict_function2(batch))

        output=np.array(output)
        mm = overlapadd_multi(output,batches,nchunks,overlap=overlap)

        for i in range(mm.shape[0]):
            audio_out=compute_inverse(mm[i,:len(ph)]/scale_factor,ph)
            if len(audio_out)>len(audioObj):
                audio_out=audio_out[:len(audioObj)]
            maxn = np.iinfo(np.int16).max  
            path, filename = os.path.split(filein)  
            scipy.io.wavfile.write(filename=os.path.join(outdir,filename.replace(".wav","_"+source[i]+".wav")), rate=sampleRate, data=(audio_out*maxn).astype('int16'))
            audio_out=None 
        audioObj = None
    else:
        print "Sample rate is not 44100"


def main(argv):  
    try:
       opts, args = getopt.getopt(argv,"hi:o:m:",["ifile=","odir=","--mfile"])
    except getopt.GetoptError:
       print 'python separate_dsd.py -i <inputfile> -o <outputdir> -m <path_to_model.pkl>'
       sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
          print 'python separate_dsd.py -i <inputfile> -o <outputdir> -m <path_to_model.pkl>'
          sys.exit()
        elif opt in ("-i", "--ifile"):
          inputfile = arg
        elif opt in ("-o", "--odir"):
          outdir = arg
        elif opt in ("-m", "--mfile"):
          model = arg
    train_auto(inputfile,outdir,model,0.3,30,25,32,513)   

if __name__ == "__main__":
    main(sys.argv[1:])