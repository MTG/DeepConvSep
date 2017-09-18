import pickle
import os
import sys
import itertools
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
import pandas as pd
import numpy as np
from scipy import stats
from scipy import io
import climate

if __name__=='__main__':
    if len(sys.argv)>-1:
        climate.add_arg('--db', help="the path to the results directory")

        kwargs = climate.parse_args()
        if kwargs.__getattribute__('db'):
            db = kwargs.__getattribute__('db')
        else:
            db='/Volumes/Macintosh HD 2/Documents/Database/Bach10/results_paper/'

        methods = []
        for d in sorted(os.listdir(db)):
            if not os.path.isfile(os.path.join(db, d)):
                methods.append(d)

        sns.set()
        sns.set_context("notebook", font_scale=1.4)
        sns.set_palette(sns.cubehelix_palette(8, start=.5, rot=-.75))

        mixSDR=[[],[],[],[],[]]
        for i in range(len(methods)):
            k=0
            for f in sorted(os.listdir(os.path.join(db, methods[i]))):
                if f.endswith(".mat"):
                    if os.path.isfile(os.path.join(db, methods[i],f)):
                        mat = io.loadmat(os.path.join(db, methods[i],f))
                        import pdb;pdb.set_trace()
                        for j in range(4):
                            mixSDR[0].append(mat['results'][0][0][j+1][0][0][0][0][0])
                            mixSDR[1].append(0)

                            mixSDR[0].append(mat['results'][0][0][j+1][0][0][1][0][0])
                            mixSDR[1].append(1)

                            mixSDR[0].append(mat['results'][0][0][j+1][0][0][2][0][0])
                            mixSDR[1].append(2)

                            for kk in range(3):
                                mixSDR[2].append(i)
                                mixSDR[3].append(j)
                                mixSDR[4].append(k)
                            k=k+1
                    else:
                        print 'mat file could not be found: '+f


        mix = np.array(mixSDR).T
        df1 = pd.DataFrame(mix,columns=['dB','measure','approach','source','piece'])
        df1['measure']=df1['measure'].replace([0,1,2],['SDR','SIR','SAR']).astype(str)
        df1['source']=df1['source'].replace([0,1,2,3],['bassoon','clarinet','saxophone','violin']).astype(str)
        df1['approach']=df1['approach'].replace(list(range(len(methods))),methods).astype(str)

        ax = sns.barplot(data=df1,x='measure',y='dB',hue='approach')
        sns.plt.show()

        df3=df1[df1['measure']=='SDR']
        ax = sns.barplot(data=df3,x='source',y='dB',hue='approach')
        sns.plt.show()
