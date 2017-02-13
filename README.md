# DeepConvSep
Deep Convolutional Neural Networks for Musical Source Separation 

More details in the following article:

P. Chandna, M. Miron, J. Janer, and E. Gomez,
\“Monoaural audio source separation using deep convolutional neural networks,\” 
International Conference on Latent Variable Analysis and Signal Separation, 2017.
http://mtg.upf.edu/node/3680

#Separating Professionally Produced Music
SiSEC MUS <a href="http://www.sisec17.audiolabs-erlangen.de">SiSEC MUS</a> using <a href="https://sisec.inria.fr/home/2016-professionally-produced-music-recordings/">DSD100</a> dataset

#iKala - Singing voice separation
<a href="http://www.music-ir.org/mirex/wiki/2016:Singing_Voice_Separation_Results">Ikala (2nd place MIREX Singing voice separation 2016) </a>


#Dependencies
climate, numpy, scipy, cPickle, theano, lasagne

The dependencies can be installed with pip:
pip install numpy scipy pickle cPickle climate theano 
pip install https://github.com/Lasagne/Lasagne/archive/master.zip

#Running examples

For iKala

    python -m examples.ikala.compute_features
    python -m examples.ikala.trainCNN

For SiSEC MUS using DSD100 dataset

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