# DeepConvSep
Deep Convolutional Neural Networks for Musical Source Separation

This repository contains classes for data generation and preprocessing and feature computation, useful in training neural networks with large datasets that do not fit into memory. Additionally, you can find classes to query samples of instrument sounds from <a href="https://staff.aist.go.jp/m.goto/RWC-MDB/">RWC instrument sound dataset</a>.

In the 'examples' folder you can find use cases for the classes above for the case of music source separation. We provide code for feature computation (STFT) and for training convolutional neural networks for music source separation: singing voice source separation with the dataset iKala dataset, for voice, bass, drums separation with DSD100 dataset, for bassoon, clarinet, saxophone, violin with <a href="http://music.cs.northwestern.edu/data/Bach10.html">Bach10 dataset</a>. The later is a good example for training a neural network with instrument samples from the RWC instrument sound database <a href="https://staff.aist.go.jp/m.goto/RWC-MDB/">RWC instrument sound dataset</a>, when the original score is available.

In the 'evaluation' folder you can find matlab code to evaluate the quality of separation, based on <a href="http://bass-db.gforge.inria.fr/bss_eval/">BSS eval</a>.

For training neural networks we use <a href="http://lasagne.readthedocs.io/">Lasagne</a> and <a href="http://deeplearning.net/software/theano/">Theano</a>.

We provide code for separation using already trained models for different tasks.

Separate music into vocals, bass, drums, accompaniment in examples/dsd100/separate_dsd.py :

    python separate_dsd.py -i <inputfile> -o <outputdir> -m <path_to_model.pkl>

where :
- \<inputfile\> is the wav file to separate
- \<outputdir\> is the output directory where to write the separation
- \<path_to_model.pkl\> is the local path to the .pkl file you can download from <a href="https://drive.google.com/open?id=0B-Th_dYuM4nOb281azdKc2tWbFk">this address</a>

Singing voice source separation in examples/ikala/separate_ikala.py :

    python separate_ikala.py -i <inputfile> -o <outputdir> -m <path_to_model.pkl>

where :
- \<inputfile\> is the wav file to separate
- \<outputdir\> is the output directory where to write the separation
- \<path_to_model.pkl\> is the local path to the .pkl file you can download from <a href="https://drive.google.com/open?id=0B-Th_dYuM4nOYlRxQTl3eDBxQTg">this address</a>

Separate Bach chorales from the Bach10 dataset into bassoon, clarinet, saxophone, violin in examples/bach10/separate_bach10.py :

    python separate_bach10.py -i <inputfile> -o <outputdir> -m <path_to_model.pkl>

where :
- \<inputfile\> is the wav file to separate
- \<outputdir\> is the output directory where to write the separation
- \<path_to_model.pkl\> is the local path to the .pkl file you can download from <a href="https://drive.google.com/open?id=0B-Th_dYuM4nOa3ZMSmhwRkwzaGM">this address</a>

Score-informed separation of Bach chorales from the Bach10 dataset into bassoon, clarinet, saxophone, violin in examples/bach10_scoreinformed/separate_bach10.py:

python separate_bach10.py -i <inputfile> -o <outputdir> -m <path_to_model.pkl>

where :
- \<inputfile\> is the wav file to separate
- \<outputdir\> is the output directory where to write the separation
- \<path_to_model.pkl\> is the local path to the .pkl file you can download from <a href="https://zenodo.org/record/1009144">zenodo</a>

The folder with the \<inputfile\> must contain the scores: 'bassoon_b.txt','clarinet_b.txt','saxophone_b.txt','violin_b.txt'. The score file as a note on each line with the format: note_onset_time,note_offset_time,note_name .


# Feature computation
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

# Audio sample querying using RWC database
The <a href="https://staff.aist.go.jp/m.goto/RWC-MDB/">RWC instrument sound dataset</a> contains samples played by various musicians in various styles and dynamics, comprising different instruments.
You can obtain a sample for a given midi note, instrument, style, dynamics and musician(1,2,3) by using the classes in 'rwc.py'.

Example

    ### construct lists for the desired dynamics,styles,musician and instrument codes
    allowed_styles = ['NO']
    allowed_dynamics = ['F','M','P']
    allowed_case = [1,2,3]
    instrument_nums=[30,31,27,15] #bassoon,clarinet,saxophone,violin
    instruments = []
    for ins in range(len(instrument_nums)):
        #for each instrument construct an Instrument object
        instruments.append(rwc.Instrument(rwc_path,instrument_nums[ins],allowed_styles,allowed_case,allowed_dynamics))

    #then, for a given instrument 'i' and midi note 'm', dynamics 'd', style 's', musician 'n'
    note = self.instruments[i].getNote(melNotes[m],d,s,n)
    #get the audio vector for the note
    audio = note.getAudio()

# Data generation
Bach10 experiments offer examples of data generation (or augmentation). Starting from the score or from existing pieces, we can augment the existing data or generate new data with some desired factors.
For instance if you have four factors time_shifts,intensity_shifts,style_shifts,timbre_shifts, you can generate the possible combinations between them for a set of pieces and instruments(sources).

    #create the product of these factors
    cc=[(time_shifts[i], intensity_shifts[j], style_shifts[l], timbre_shifts[k]) for i in xrange(len(time_shifts)) for j in xrange(len(intensity_shifts)) for l in xrange(len(style_shifts)) for k in xrange(len(timbre_shifts))]

    #create combinations for each of the instruments (sources)
    if len(cc)<len(sources):
        combo1 = list(it.product(cc,repeat=len(sources)))
        combo = []
        for i in range(len(combo1)):
          c = np.array(combo1[i])
          #if (all(x == c[0,0] for x in c[:,0]) or all(x == c[0,1] for x in c[:,1])) \
          if (len(intensity_shifts)==1 and not(all(x == c[0,0] for x in c[:,0]))) \
            or (len(time_shifts)==1 and not(all(x == c[0,1] for x in c[:,1]))):
              combo.append(c)
        combo = np.array(combo)
    else:
        combo = np.array(list(it.permutations(cc,len(sources))))
    if len(combo)==0:
        combo = np.array([[[time_shifts[0],intensity_shifts[0],style_shifts[0],timbre_shifts[0]] for s in sources]])

    #if there are too many combination, you can just randomly sample
    if sample_size<len(combo):
        sampled_combo = combo[np.random.choice(len(combo),size=sample_size, replace=False)]
    else:
        sampled_combo = combo

# References
More details on the separation method can be found in the following article:

P. Chandna, M. Miron, J. Janer, and E. Gomez,
“Monoaural audio source separation using deep convolutional neural networks”
International Conference on Latent Variable Analysis and Signal Separation, 2017.
<a href="http://mtg.upf.edu/node/3680">PDF</a>

M. Miron, J. Janer, and E. Gomez,
"Generating data to train convolutional neural networks for low latency classical music source separation"
Sound and Music Computing Conference 2017

M. Miron, J. Janer, and E. Gomez,
"Monaural score-informed source separation for classical music using convolutional neural networks"
ISMIR Conference 2017


# Dependencies
python 2.7

climate, numpy, scipy, cPickle, theano, lasagne

The dependencies can be installed with pip:

    pip install numpy scipy pickle cPickle climate theano
    pip install https://github.com/Lasagne/Lasagne/archive/master.zip

# Separating classical music mixtures with Bach10 dataset
We separate bassoon,clarinet,saxophone,violing using <a href="http://music.cs.northwestern.edu/data/Bach10.html">Bach10 dataset</a>, which comprises 10 Bach chorales. Our approach consists in synthesing the original scores considering different timbres, dynamics, playing styles, and local timing deviations to train a more robust model for classical music separation.

We have three experiments:

-Oracle: train with the original pieces (obviously overfitting, hence this is the "Oracle");

-Sibelius: train with the pieces sythesized with Sibelius software;

-RWC: train with the pieces synthesized using the samples in <a href="https://staff.aist.go.jp/m.goto/RWC-MDB/">RWC instrument sound dataset</a>.

The code for feature computation and training the network can be found in "examples/bach10" folder.

# Score-informed separation of classical music mixtures with Bach10 dataset
We separate bassoon,clarinet,saxophone,violing using <a href="http://music.cs.northwestern.edu/data/Bach10.html">Bach10 dataset</a>, which comprises 10 Bach chorales and the associated score.

We generate training data with the approach mentioned above using the RWC database. Consequently, we train with the pieces synthesized using the samples in <a href="https://staff.aist.go.jp/m.goto/RWC-MDB/">RWC instrument sound dataset</a>.

The score is given in .txt files containing the name of the of the instrument and an additional suffix, e.g. 'bassoon_g.txt'. The format for a note in the text file is: onset, offset, midinotename , as the following example: 6.1600,6.7000,F4# .

The code for feature computation and training the network can be found in "examples/bach10_sourceseparation" folder.


# Separating Professionally Produced Music
We separate voice, bass, drums and accompaniment using DSD100 dataset comprising professionally produced music. For more details about the challenge, please refer to <a href="http://www.sisec17.audiolabs-erlangen.de">SiSEC MUS</a> challenge and <a href="https://sisec.inria.fr/home/2016-professionally-produced-music-recordings/">DSD100</a> dataset.

The code for feature computation and training the network can be found in "examples/dsd100" folder.

# iKala - Singing voice separation
We separate voice and accompaniment using the iKala dataset. For more details about the challenge, please refer to <a href="http://www.music-ir.org/mirex/wiki/2016:Singing_Voice_Separation_Results">MIREX Singing voice separation 2016</a> and <a href="http://mac.citi.sinica.edu.tw/ikala/">iKala</a> dataset.

The code for feature computation and training the network can be found in "examples/ikala" folder.

# Training models

For Bach10 dataset :

    #train with the original dataset
    python -m examples.bach10.compute_features_bach10 --db '/path/to/Bach10/'
    #train with the the synthetic dataset generated with Sibelius
    python -m examples.bach10.compute_features_bach10sibelius --db '/path/to/Bach10Sibelius/'
    #train with the rwc dataset
    python -m examples.bach10.compute_features_bach10rwc --db '/path/to/Bach10Sibelius/' --rwc '/path/to/rwc/'
    ### Replace gpu0 with cpu,gpu,cuda,gpu0 etc. depending on your system configuration
    THEANO_FLAGS=mode=FAST_RUN,device=gpu0,floatX=float32,lib.cnmem=0.95 python -m examples.bach10.trainCNNrwc --db '/path/to/Bach10/' --dbs '/path/to/Bach10Sibelius/' --output '/output/path/'
    THEANO_FLAGS=mode=FAST_RUN,device=gpu0,floatX=float32,lib.cnmem=0.95 python -m examples.bach10.trainCNNSibelius --db '/path/to/Bach10/' --dbs '/path/to/Bach10Sibelius/' --output '/output/path/'

For iKala :

    python -m examples.ikala.compute_features --db '/path/to/iKala/'
    ### Replace gpu0 with cpu,gpu,cuda,gpu0 etc. depending on your system configuration
    THEANO_FLAGS=mode=FAST_RUN,device=gpu0,floatX=float32,lib.cnmem=0.95 python -m examples.ikala.trainCNN --db '/path/to/iKala/'

For SiSEC MUS using DSD100 dataset :

    python -m examples.dsd100.compute_features --db '/path/to/DSD100/'
    ### Replace gpu0 with cpu,gpu,cuda,gpu0 etc. depending on your system configuration
    THEANO_FLAGS=mode=FAST_RUN,device=gpu0,floatX=float32,lib.cnmem=0.95 python -m examples.dsd100.trainCNN --db '/path/to/DSD100/'


# Evaluation

The metrics are computed with bsseval images v3.0, as described <a href="http://bass-db.gforge.inria.fr/bss_eval/">here</a>.

The evaluation scripts can be found in the subfolder "evaluation".
The subfolder "script_cluster" contains scripts to run the evaluation script in parallel on a HPC cluster system.

For Bach10, you need to run the script Bach10_eval_only.m for each method in the 'base_estimates_directory' folder and for the 10 pieces. To evaluate the separation of the <a href="https://zenodo.org/record/321361#.WNFhKt-i7J8">Bach10 Sibeliust dataset</a>, use the 'Bach10_eval_only_original.m' script. Be careful not to mix the estimation directories for the two datasets.

For iKala, you need to run the script evaluate_SS_iKala.m for each of the 252 files in the dataset.
The script takes as parameters the id of the file, the path to the dataset, and the method of separation, which needs to be a directory containing the separation results, stored in 'output' folder.

    for id=1:252
        evaluate_SS_iKala(id,'/homedtic/mmiron/data/iKala/','fft_1024');
    end

For SiSEC-MUS/DSD100, use the scripts at the <a href="https://github.com/faroit/dsd100mat">web-page</a>.

If you have access to a HPC cluster, you can use the .sh scripts in the script_cluster folder which call the corresponding .m files.

# Research reproducibility
For DSD100 and iKAla, the framework was tested as a part of a public evaluation campaign and the results were published online (see the sections above).

For Bach10, we provide the synthetic <a href="https://zenodo.org/record/321361#.WNFhKt-i7J8">Bach10 Sibeliust dataset</a> and the <a href="https://zenodo.org/record/344499#.WNFjMN-i7J8">Bach10 Separation SMC2017 dataset</a> containing the separation for each method as .wav files and the evaluation results as .mat files.

If you want to compute the features and re-train the models, check the 'examples/bach10' folder and the instructions above. Alternatively, you can <a href="https://drive.google.com/open?id=0B-Th_dYuM4nOa3ZMSmhwRkwzaGM">download</a> an already trained model and perform separation with 'separate_bach10.py'.

If you want to evaluate the methods in <a href="https://zenodo.org/record/344499#.WNFjMN-i7J8">Bach10 Separation SMC2017 dataset</a>, then you can use the scripts in evaluation directory, which we explained above in the 'Evaluation' section.

If you want to replicate the plots in the SMC2017 paper, you need to have installed 'pandas' and 'seaborn' (pip install pandas seaborn) and then run the script in the plots subfolder:

    bach10_smc_stats.py --db 'path-to-results-dir'

Where 'path-to-results-dir' is the path to the folder where you have stored the results for each method (e.g. if you downloaded the Bach10 Separation SMC2017, it would be the 'results' subfolder).

# Acknowledgments
The TITANX used for this research was donated by the NVIDIA Corporation.

# License

    Copyright (c) 2014-2017
    Marius Miron <miron.marius at gmail dot com>,
    Pritish Chandna <pc2752 at gmail dot com>,
    Gerard Erruz, and Hector Martel
    Music Technology Group, Universitat Pompeu Fabra, Barcelona <mtg.upf.edu>

    This program is free software: you can redistribute it and/or modify
    it under the terms of the Affero GPL license published by
    the Free Software Foundation, either version 3 of the License, or (at your
    option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    Affero GPL license for more details.

    You should have received a copy of the Affero GPL license
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
