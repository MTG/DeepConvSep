# DeepConvSep
Deep Convolutional Neural Networks for Musical Source Separation 

This repository contains routines for data generation and preprocessing, useful in training neural networks with large datasets that do not fit into memory. Additionally, it contains python code to train convolutional neural networks for music source separation and matlab code to evaluate the quality of separation. 

# Data generation
Compute the features for a given set of audio signals extending the "Transform" class in transform.py

For instance the TransformFFT class helps computing the STFT of an audio signal and saves the magnitude spectrogram as a binary file.   

Examples

    ### 1. Computing the STFT of a matrix of signals \"audio\" and writing the STFT data in \"path\" (except the phase)
    tt1=transformFFT(frameSize=2048, hopSize=512, sampleRate=44100)
    tt1.compute_transform(audio,out_path=path, phase=False)

    ### 2. Computing the STFT of a single signal \"audio\" and returning the magnitude and phase
    tt1=transformFFT(frameSize=2048, hopSize=512, sampleRate=44100)
    mag,ph = tt1.compute_file(audio,phase=True)

    ### 3. Computing the inverse STFT using the magnitude and phase and returning the audio data
    #we use the tt1 from 2.
    audio = tt1.compute_inverse(mag,phase)


# Data preprocessing
Load features which have been computed with transform.py, and yield batches necessary for training neural networks. These classes are useful when the data does not fit into memory, and the batches can be loaded in chunks.

Example   
    
    ### Load binary training data from the out_path folder
    train = LargeDataset(path_transform_in=out_path, batch_size=32, batch_memory=200, time_context=30, overlap=20, nprocs=7)
    

# References
More details on the separation method can be found in the following article:

P. Chandna, M. Miron, J. Janer, and E. Gomez,
“Monoaural audio source separation using deep convolutional neural networks” 
International Conference on Latent Variable Analysis and Signal Separation, 2017.
<a href="http://mtg.upf.edu/node/3680">PDF</a>

#Dependencies
climate, numpy, scipy, cPickle, theano, lasagne

The dependencies can be installed with pip:

    pip install numpy scipy pickle cPickle climate theano 
    pip install https://github.com/Lasagne/Lasagne/archive/master.zip

#Separating Professionally Produced Music
<a href="http://www.sisec17.audiolabs-erlangen.de">SiSEC MUS</a> using <a href="https://sisec.inria.fr/home/2016-professionally-produced-music-recordings/">DSD100</a> dataset

#iKala - Singing voice separation
<a href="http://www.music-ir.org/mirex/wiki/2016:Singing_Voice_Separation_Results">iKala (2nd place MIREX Singing voice separation 2016) </a>


#Running examples

For iKala:

    python -m examples.ikala.compute_features
    python -m examples.ikala.trainCNN

For SiSEC MUS using DSD100 dataset

#Evaluation 

The metrics are computed with bsseval images v3.0, as described <a href="http://bass-db.gforge.inria.fr/bss_eval/">here</a>. 

The evaluation scripts can be found in the subfolder "evaluation".
The subfolder "script_cluster" contains scripts to run the evaluation script in parallel on a HPC cluster system.

For iKala, you need to run the script evaluate_SS_iKala.m for each of the 252 files in the dataset.
The script takes as parameters the id of the file, the path to the dataset, and the method of separation, which needs to be a directory containing the separation results, stored in 'output' folder.

    for id=1:252
        evaluate_SS_iKala(id,'/homedtic/mmiron/data/iKala/','fft_2048');
    end

For SiSEC-MUS/DSD100, change in the python file "prepare_files_dsd100pc.py" the variable "path_in" to the path on your hard-drive where you stored the separation. Then, use the scripts at the <a href="https://github.com/faroit/dsd100mat">web-page</a>.

#License

    Copyright (c) 2014-2017 Marius Miron <miron.marius at gmail.com>, Pritish Chandna 

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.