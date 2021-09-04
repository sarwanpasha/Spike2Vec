#!/usr/bin/env python
# coding: utf-8

# In[7]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.decomposition import TruncatedSVD
import random
import seaborn as sns
import os.path as path
import os
import matplotlib
import matplotlib.font_manager
import matplotlib.pyplot as plt # graphs plotting
from Bio import SeqIO # some BioPython that will come in handy
#matplotlib inline
import numpy
import csv 

from matplotlib import rc

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


from itertools import cycle

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
from sklearn.metrics import roc_auc_score

import statistics

from sklearn.cluster import KMeans

from sklearn.datasets import load_digits
from sklearn.decomposition import KernelPCA

import math
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import confusion_matrix

import itertools
from itertools import product

print("Checking!!!")
print("Packages imported")


read_path = "/alina-data1/sarwan/IEEE_BigData/Dataset/filled_final_only_protein_sequences_data.csv"



prot_seq = []
# host_names_ne = []

with open(read_path) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        tmp = row
#         host_names_new.append(', '.join(row))
        prot_seq.append(tmp)
  
print("Reading sequences file successful!!")


print("Total sequences = ",len(prot_seq))

# We create a class fasta_sequence so that we would be able to use the sequence data easily 

class fasta_sequence:
    def __init__(self, sequence, type_of_encoding = "onehot"):
        
        # we read the input data
        
        self.sequence = sequence


        def encoding(sequence, type_of_encoding):

            # define universe of possible input values
#             alphabet = 'ABCDEFGHIJKLMNPQRSTUVWXYZ-'
            alphabet = 'ACDEFGHIKLMNPQRSTVWXY-'
            # define a mapping of chars to integers
            char_to_int = dict((c, i) for i, c in enumerate(alphabet))


            # integer encoding
            integer_encoded = [char_to_int[char] for char in sequence]

            # one-hot encoding
            onehot_encoded = list()
            for value in integer_encoded:
                letter = [0 for _ in range(len(alphabet)-1)]
                if value != len(alphabet)-1:
                    letter[value] = 1
                onehot_encoded.append(letter)
            flat_list = [item for sublist in onehot_encoded for item in sublist]

            if type_of_encoding == "onehot":
                return flat_list
            else:
                return integer_encoded
            
        #  we use the encoding function to create a new attribute for the sequence -- its encoding        
        self.encoded = encoding(sequence, type_of_encoding)

def EncodeAndTarget(list_of_sequences):
    # encoding the sequences
    list_of_encoded_sequences = [entry.encoded for entry in list_of_sequences]
    return list_of_encoded_sequences         

#ind_range = [0,100000]

sequences_tmp = []
cnt_tmp = 0
cnt_tmp2 = 0
for ind_tmp in range(2500000,len(prot_seq)):
#    if(cnt_tmp==100000):
    sequences_tmp.append(prot_seq[ind_tmp])
    cnt_tmp = cnt_tmp+1
    cnt_tmp2 = cnt_tmp2 + 1
        
sequences = []
for i in range(0, len(sequences_tmp)):
    current_sequence = fasta_sequence(sequences_tmp[i])
    sequences.append(current_sequence)

encoded_sequences = EncodeAndTarget(sequences)

print("Writting Frequency Vectors ==>",cnt_tmp2)   
write_path_11 = "/alina-data1/sarwan/IEEE_BigData/Dataset/baseline_one_hot_vec_" + str(len(prot_seq)) + ".csv"

with open(write_path_11, 'w', newline='') as file:
    writer = csv.writer(file)
    for ij in range(0,len(encoded_sequences)):
        ccv = encoded_sequences[ij]
        writer.writerow(ccv)



print("Writting One-hot Vectors DONE!!!!") 

