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




## for Palatino and other serif fonts use:
# rc('font',**{'family':'serif','serif':['Palatino']})
# matplotlib.rcParams['mathtext.fontset'] = 'cm'

## for LaTeX typefont
# matplotlib.rcParams['mathtext.fontset'] = 'stix'
# matplotlib.rcParams['font.family'] = 'STIXGeneral'

## for another LaTeX typefont
# rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})

# rc('text', usetex = True)

print("New Packages Loaded!!!")


# # Read Frequency Vector

# In[ ]:



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
arr_all_datasets = [100000,200000,300000,400000,500000,600000,700000,800000,900000,1000000,1100000,1200000,1300000,1400000,1500000,1600000,1700000,1800000,1900000,2000000,2100000,2200000,2300000,2400000,2500000,2519386]
#arr_all_datasets = [100000]

for lp_tmp in arr_all_datasets:
    print("Reading Dataset ==>>",str(lp_tmp))
    read_path_1 = "/alina-data1/sarwan/IEEE_BigData/Dataset/baseline_one_hot_vec_" + str(lp_tmp) + ".csv"    
    frequency_vector_read_1 = fun_freq_vec_read(read_path_1) 
    
    for ind_lst in range(len(frequency_vector_read_1)):
        final_freq_vec.append(frequency_vector_read_1[ind_lst])
    
    #final_freq_vec.append(frequency_vector_read_1)
    print("Final Dataset Length ==>>",str(len(final_freq_vec)))

print("One Hot Encoding Vector Data Reading Done with length ==>>",len(final_freq_vec))






frequency_vector_read_final = freq_int_convert(final_freq_vec)
  
print("One Hot Encoding Vector integer conversion Done")



read_path = "/alina-data1/sarwan/IEEE_BigData/Dataset/final_other_attributes_only.csv"

variant_orig = []

with open(read_path) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        tmp = row
        variant_orig.append(tmp[1])
        
print("Attributed data Reading Done")


# In[14]:


unique_varaints = list(np.unique(variant_orig))


# In[18]:


int_variants = []
for ind_unique in range(len(frequency_vector_read_final)):
    variant_tmp = variant_orig[ind_unique]
    ind_tmp = unique_varaints.index(variant_tmp)
    int_variants.append(ind_tmp)
    
    
print("Attribute data preprocessing Done")

#final_int_variants = int_variants[0:100000]

# # Train-Test Split

X = np.array(frequency_vector_read_final)
y =  np.array(int_variants)
y_orig =  np.array(variant_orig)

print("Now doing Train-Test Split")

sss = ShuffleSplit(n_splits=1, test_size=0.9)
sss.get_n_splits(X, y)
train_index, test_index = next(sss.split(X, y)) 

X_train, X_test = X[train_index], X[test_index]
y_train, y_test = y[train_index], y[test_index]
y_orig_train, y_orig_test = y_orig[train_index], y_orig[test_index]


print("Train Test split Done---")
print("Random Fourier Features Starts here!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

start_time = time.time()
rbf_feature = RBFSampler(gamma=1, n_components=500)
rbf_feature.fit(X_train)
X_features_train = rbf_feature.transform(X_train)
X_features_final = rbf_feature.transform(X_test)

#X_features_final = rbf_feature.transform(X)

print("Random Fourier Features Done---")



from sklearn.cluster import KMeans

# number_of_clusters = [5,8,10,12,14,16,18,20]

number_of_clusters = [22]

for clust_ind in range(len(number_of_clusters)):
    print("Number of Clusters = ",number_of_clusters[clust_ind])
    clust_num = number_of_clusters[clust_ind]
    
    kmeans = KMeans(n_clusters=clust_num, random_state=0).fit(X_features_final)
    kmean_clust_labels = kmeans.labels_


    write_path_11 = "/alina-data1/sarwan/IEEE_BigData/Dataset/one_hot_freq_vec_hard_kmeans_clustering_k_" + str(clust_num) + ".csv"

    with open(write_path_11, 'w', newline='') as file:
        writer = csv.writer(file)
        for i in range(0,len(kmean_clust_labels)):
            ccv = str(kmean_clust_labels[i])
            writer.writerow([ccv])
            
    write_path_11 = "/alina-data1/sarwan/IEEE_BigData/Dataset/one_hot_attributes_freq_vec_hard_kmeans_clustering_k_" + str(clust_num) + ".csv"
    with open(write_path_11, 'w', newline='') as file:
        writer = csv.writer(file)
        for i in range(0,len(y_orig_test)):
            ccv = str(y_orig_test[i])
            writer.writerow([ccv])

print("All Processing Done!!!")
