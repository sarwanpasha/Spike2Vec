#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import time
from sklearn.svm import SVC
#import RandomBinningFeatures
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import RidgeClassifier
from sklearn import metrics
import scipy
import matplotlib.pyplot as plt 
#%matplotlib inline 
import csv

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.decomposition import TruncatedSVD
import random
# import seaborn as sns
import os.path as path
import os
# import matplotlib
# import matplotlib.font_manager
# import matplotlib.pyplot as plt # graphs plotting
# import Bio
from Bio import SeqIO # some BioPython that will come in handy
#matplotlib inline

from itertools import cycle

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
from sklearn.metrics import roc_auc_score


# from matplotlib import rc
# # for Arial typefont
# matplotlib.rcParams['font.family'] = 'Arial'

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn import svm

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics

import pandas as pd
import sklearn
from sklearn import preprocessing
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler  
from sklearn.neural_network import MLPClassifier 
from sklearn.metrics import classification_report, confusion_matrix 

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

from pandas import DataFrame

from sklearn.model_selection import KFold 
from sklearn.model_selection import RepeatedKFold

from sklearn.metrics import confusion_matrix

from numpy import mean
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

from yellowbrick.cluster import KElbowVisualizer
from sklearn.model_selection import ShuffleSplit # or StratifiedShuffleSplit


## for Palatino and other serif fonts use:
# rc('font',**{'family':'serif','serif':['Palatino']})
# matplotlib.rcParams['mathtext.fontset'] = 'cm'

## for LaTeX typefont
# matplotlib.rcParams['mathtext.fontset'] = 'stix'
# matplotlib.rcParams['font.family'] = 'STIXGeneral'

## for another LaTeX typefont
# rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})

# rc('text', usetex = True)

print("Packages Loaded!!!")



# In[2]:


def fun_freq_vec_read(path_tmp_fun):

    frequency_vector_read = []

    with open(path_tmp_fun) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            tmp = row
            frequency_vector_read.append(tmp)
    return frequency_vector_read

def freq_int_convert(freq_vec_in_fun):
    frequency_vector_read_int = []

    for u in range(len(freq_vec_in_fun)):
        aa = freq_vec_in_fun[u]
        test_list = []
        for i in range(0, len(aa)):
            test_list.append(int(aa[i]))
            
     
        frequency_vector_read_int.append(test_list)
    return frequency_vector_read_int


final_freq_vec = []

#print("Reading Dataset")
read_path_1 = "/alina-data1/sarwan/IEEE_BigData/Dataset/t_sne_plot_some_variants_sequences.csv"    
frequency_vector_read_1 = fun_freq_vec_read(read_path_1) 

print("Dataset Reading Done!!!")


# In[3]:


for ind_lst in range(len(frequency_vector_read_1)):
    final_freq_vec.append(frequency_vector_read_1[ind_lst])

#final_freq_vec.append(frequency_vector_read_1)
print("Final Dataset Length ==>>",str(len(final_freq_vec)))

#print("Frequency Vector Data Reading Done with length ==>>",len(final_freq_vec))


print("Starting int conversion!!!")


frequency_vector_read_final = freq_int_convert(final_freq_vec)
  
print("Frequency Vector integer conversion Done")


# In[ ]:


from sklearn.manifold import TSNE
import matplotlib.pyplot as plt # graphs plotting
import matplotlib.cm as cm
import seaborn as sns

print("Starting t-SNE")

X_embedded = TSNE(n_components = 2, perplexity = 30, random_state = 1).fit_transform(frequency_vector_read_final)

print("Writting File!!!")

write_path_11 = "/alina-data1/sarwan/IEEE_BigData/Dataset/t_sne_plot_2_dim.csv"

with open(write_path_11, 'w', newline='') as file:
    writer = csv.writer(file)
    for i in range(0,len(kmean_clust_labels)):
        ccv = str(X_embedded)
        writer.writerow(X_embedded)




