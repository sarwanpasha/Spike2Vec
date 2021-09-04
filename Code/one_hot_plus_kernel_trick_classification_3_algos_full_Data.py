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

print("Packages Loaded!!!")


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
for ind_unique in range(len(variant_orig)):
    variant_tmp = variant_orig[ind_unique]
    ind_tmp = unique_varaints.index(variant_tmp)
    int_variants.append(ind_tmp)
    
print("Attribute data preprocessing Done")

#final_int_variants = int_variants[0:100000]

# # Train-Test Split

freq_vec_reduced = []
int_variant_reduced = []
name_variant_reduced = []

for ind_reduced in range(len(frequency_vector_read_final)):
    if variant_orig[ind_reduced]=="B.1.1.7" or variant_orig[ind_reduced]=="B.1.617.2" or variant_orig[ind_reduced]=="AY.4" or variant_orig[ind_reduced]=="B.1.2" or variant_orig[ind_reduced]=="B.1" or variant_orig[ind_reduced]=="B.1.177"  or variant_orig[ind_reduced]=="P.1" or variant_orig[ind_reduced]=="B.1.1" or variant_orig[ind_reduced]=="B.1.429"  or variant_orig[ind_reduced]=="AY.12" or variant_orig[ind_reduced]=="B.1.160" or variant_orig[ind_reduced]=="B.1.526" or variant_orig[ind_reduced]=="B.1.1.519" or variant_orig[ind_reduced]=="B.1.351" or variant_orig[ind_reduced]=="B.1.1.214"  or variant_orig[ind_reduced]=="B.1.427" or variant_orig[ind_reduced]=="B.1.221" or variant_orig[ind_reduced]=="B.1.258" or variant_orig[ind_reduced]=="B.1.177.21" or variant_orig[ind_reduced]=="D.2" or variant_orig[ind_reduced]=="B.1.243"  or variant_orig[ind_reduced]=="R.1":
        freq_vec_reduced.append(frequency_vector_read_final[ind_reduced])
        int_variant_reduced.append(int_variants[ind_reduced])
        name_variant_reduced.append(variant_orig[ind_reduced])


idx = pd.Index(name_variant_reduced) # creates an index which allows counting the entries easily
print('Variant Distribution --- : \n', len(idx),"entries in total")
aq = idx.value_counts()
print(aq)

print("Total Sequences after reducing data ==>>",len(freq_vec_reduced))


X = np.array(freq_vec_reduced)
y =  np.array(int_variant_reduced)


from sklearn.model_selection import ShuffleSplit # or StratifiedShuffleSplit
sss = ShuffleSplit(n_splits=1, test_size=0.9)
sss.get_n_splits(X, y)
train_index, test_index = next(sss.split(X, y)) 

X_train, X_test = X[train_index], X[test_index]
y_train, y_test = y[train_index], y[test_index]

print("Train-Test Split Done")

# Classification Functions !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# In[4]
def roc_auc_score_multiclass(actual_class, pred_class, average = "macro"):

    unique_class = set(actual_class)
    roc_auc_dict = {}
    for per_class in unique_class:
        #creating a list of all the classes except the current class 
        other_class = [x for x in unique_class if x != per_class]

        #marking the current class as 1 and all other classes as 0
        new_actual_class = [0 if x in other_class else 1 for x in actual_class]
        new_pred_class = [0 if x in other_class else 1 for x in pred_class]

        #using the sklearn metrics method to calculate the roc_auc_score
        roc_auc = roc_auc_score(new_actual_class, new_pred_class, average = average)
        roc_auc_dict[per_class] = roc_auc
    
    
    check = pd.DataFrame(roc_auc_dict.items())
    return mean(check)

def svm_fun_kernel(X_train,y_train,X_test,y_test,kernel_mat):

#     clf = svm.SVC()
    clf = svm.SVC(kernel=kernel_mat)
    
    #Train the model using the training sets
    clf.fit(kernel_mat, y_train)

    #Predict the response for test dataset
    y_pred = clf.predict(X_test)
    
    svm_acc = metrics.accuracy_score(y_test, y_pred)
#     print("SVM Accuracy:",svm_acc)
    
    svm_prec = metrics.precision_score(y_test, y_pred,average='weighted')
#     print("SVM Precision:",svm_prec)
    
    svm_recall = metrics.recall_score(y_test, y_pred,average='weighted')
#     print("SVM Recall:",svm_recall)

    svm_f1_weighted = metrics.f1_score(y_test, y_pred,average='weighted')
#     print("SVM F1 Weighted:",svm_f1_weighted)
    
    svm_f1_macro = metrics.f1_score(y_test, y_pred,average='macro')
#     print("SVM F1 macro:",svm_f1_macro)
    
    svm_f1_micro = metrics.f1_score(y_test, y_pred,average='micro')
#     print("SVM F1 micro:",svm_f1_micro)
    
    confuse = confusion_matrix(y_test, y_pred)
#    print("Confusion Matrix SVM : \n", confuse)
#    print("SVM Kernel Class Wise Accuracy : ",confuse.diagonal()/confuse.sum(axis=1))
    ######################## Compute ROC curve and ROC area for each class ################
    y_prob = y_pred
    macro_roc_auc_ovo = roc_auc_score_multiclass(y_test, y_prob, average='macro')
#    print(macro_roc_auc_ovo[1])
    check = [svm_acc,svm_prec,svm_recall,svm_f1_weighted,svm_f1_macro,svm_f1_micro,macro_roc_auc_ovo[1]]
    return(check)
    
# In[5]
##########################  SVM Classifier  ################################
def svm_fun(X_train,y_train,X_test,y_test):
    #Create a svm Classifier
    clf = svm.SVC(kernel='linear') # Linear Kernel

    #Train the model using the training sets
    clf.fit(X_train, y_train)

    #Predict the response for test dataset
    y_pred = clf.predict(X_test)
    
    svm_acc = metrics.accuracy_score(y_test, y_pred)
#     print("SVM Accuracy:",svm_acc)
    
    svm_prec = metrics.precision_score(y_test, y_pred,average='weighted')
#     print("SVM Precision:",svm_prec)
    
    svm_recall = metrics.recall_score(y_test, y_pred,average='weighted')
#     print("SVM Recall:",svm_recall)

    svm_f1_weighted = metrics.f1_score(y_test, y_pred,average='weighted')
#     print("SVM F1 Weighted:",svm_f1_weighted)
    
    svm_f1_macro = metrics.f1_score(y_test, y_pred,average='macro')
#     print("SVM F1 macro:",svm_f1_macro)
    
    svm_f1_micro = metrics.f1_score(y_test, y_pred,average='micro')
#     print("SVM F1 micro:",svm_f1_micro)
    
    confuse = confusion_matrix(y_test, y_pred)
#    print("Confusion Matrix SVM : \n", confuse)
#    print("SVM Class Wise Accuracy : ",confuse.diagonal()/confuse.sum(axis=1))
    ######################## Compute ROC curve and ROC area for each class ################
    y_prob = y_pred
    macro_roc_auc_ovo = roc_auc_score_multiclass(y_test, y_prob, average='macro')
#    print(macro_roc_auc_ovo[1])
    check = [svm_acc,svm_prec,svm_recall,svm_f1_weighted,svm_f1_macro,svm_f1_micro,macro_roc_auc_ovo[1]]
    return(check)
    


# In[5]
##########################  NB Classifier  ################################
def gaus_nb_fun(X_train,y_train,X_test,y_test):
    gnb = GaussianNB()
    y_pred = gnb.fit(X_train, y_train).predict(X_test)


    NB_acc = metrics.accuracy_score(y_test, y_pred)
#     print("Gaussian NB Accuracy:",NB_acc)

    NB_prec = metrics.precision_score(y_test, y_pred,average='weighted')
#     print("Gaussian NB Precision:",NB_prec)
    
    NB_recall = metrics.recall_score(y_test, y_pred,average='weighted')
#     print("Gaussian NB Recall:",NB_recall)
    
    NB_f1_weighted = metrics.f1_score(y_test, y_pred,average='weighted')
#     print("Gaussian NB F1 weighted:",NB_f1_weighted)
    
    NB_f1_macro = metrics.f1_score(y_test, y_pred,average='macro')
#     print("Gaussian NB F1 macro:",NB_f1_macro)
    
    NB_f1_micro = metrics.f1_score(y_test, y_pred,average='micro')
#     print("Gaussian NB F1 micro:",NB_f1_micro)
    
    confuse = confusion_matrix(y_test, y_pred)
#    print("Confusion Matrix NB : \n", confuse)
#    print("NB Class Wise Accuracy : ",confuse.diagonal()/confuse.sum(axis=1))
    ######################## Compute ROC curve and ROC area for each class ################
    y_prob = y_pred
    macro_roc_auc_ovo = roc_auc_score_multiclass(y_test, y_prob, average='macro')
    check = [NB_acc,NB_prec,NB_recall,NB_f1_weighted,NB_f1_macro,NB_f1_micro,macro_roc_auc_ovo[1]]
    return(check)

# In[5]
##########################  MLP Classifier  ################################
def mlp_fun(X_train,y_train,X_test,y_test):
    # Feature scaling
    scaler = StandardScaler()  
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)  
    X_test_2 = scaler.transform(X_test)


    # Finally for the MLP- Multilayer Perceptron
    mlp = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=1000)  
    mlp.fit(X_train, y_train)


    y_pred = mlp.predict(X_test_2)
    
    MLP_acc = metrics.accuracy_score(y_test, y_pred)
#     print("MLP Accuracy:",MLP_acc)
    
    MLP_prec = metrics.precision_score(y_test, y_pred,average='weighted')
#     print("MLP Precision:",MLP_prec)
    
    MLP_recall = metrics.recall_score(y_test, y_pred,average='weighted')
#     print("MLP Recall:",MLP_recall)
    
    MLP_f1_weighted = metrics.f1_score(y_test, y_pred,average='weighted')
#     print("MLP F1:",MLP_f1_weighted)
    
    MLP_f1_macro = metrics.f1_score(y_test, y_pred,average='macro')
#     print("MLP F1:",MLP_f1_macro)
    
    MLP_f1_micro = metrics.f1_score(y_test, y_pred,average='micro')
#     print("MLP F1:",MLP_f1_micro)
    
    confuse = confusion_matrix(y_test, y_pred)
#    print("Confusion Matrix MLP : \n", confuse)
#    print("MLP Class Wise Accuracy : ",confuse.diagonal()/confuse.sum(axis=1))
    ######################## Compute ROC curve and ROC area for each class ################
    y_prob = y_pred
    macro_roc_auc_ovo = roc_auc_score_multiclass(y_test, y_prob, average='macro')
    
    check = [MLP_acc,MLP_prec,MLP_recall,MLP_f1_weighted,MLP_f1_macro,MLP_f1_micro,macro_roc_auc_ovo[1]]
    return(check)

# In[5]
##########################  knn Classifier  ################################
def knn_fun(X_train,y_train,X_test,y_test):
    classifier = KNeighborsClassifier(n_neighbors=5)
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)

    knn_acc = metrics.accuracy_score(y_test, y_pred)
#     print("Knn Accuracy:",knn_acc)
    
    knn_prec = metrics.precision_score(y_test, y_pred,average='weighted')
#     print("Knn Precision:",knn_prec)
    
    knn_recall = metrics.recall_score(y_test, y_pred,average='weighted')
#     print("Knn Recall:",knn_recall)
    
    knn_f1_weighted = metrics.f1_score(y_test, y_pred,average='weighted')
#     print("Knn F1 weighted:",knn_f1_weighted)
    
    knn_f1_macro = metrics.f1_score(y_test, y_pred,average='macro')
#     print("Knn F1 macro:",knn_f1_macro)
    
    knn_f1_micro = metrics.f1_score(y_test, y_pred,average='micro')
#     print("Knn F1 micro:",knn_f1_micro)
    
    confuse = confusion_matrix(y_test, y_pred)
#   print("Confusion Matrix KNN : \n", confuse)
#    print("KNN Class Wise Accuracy : ",confuse.diagonal()/confuse.sum(axis=1))
    ######################## Compute ROC curve and ROC area for each class ################
    y_prob = y_pred
    macro_roc_auc_ovo = roc_auc_score_multiclass(y_test, y_prob, average='macro')
    
    check = [knn_acc,knn_prec,knn_recall,knn_f1_weighted,knn_f1_macro,knn_f1_micro,macro_roc_auc_ovo[1]]
    return(check)

# In[5]
##########################  Random Forest Classifier  ################################
def rf_fun(X_train,y_train,X_test,y_test):
    # Import the model we are using
    from sklearn.ensemble import RandomForestClassifier
    # Instantiate model with 1000 decision trees
    rf = RandomForestClassifier(n_estimators = 100)
    # Train the model on training data
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    fr_acc = metrics.accuracy_score(y_test, y_pred)
#     print("Random Forest Accuracy:",fr_acc)
    
    fr_prec = metrics.precision_score(y_test, y_pred,average='weighted')
#     print("Random Forest Precision:",fr_prec)
    
    fr_recall = metrics.recall_score(y_test, y_pred,average='weighted')
#     print("Random Forest Recall:",fr_recall)
    
    fr_f1_weighted = metrics.f1_score(y_test, y_pred,average='weighted')
#     print("Random Forest F1 weighted:",fr_f1_weighted)
    
    fr_f1_macro = metrics.f1_score(y_test, y_pred,average='macro')
#     print("Random Forest F1 macro:",fr_f1_macro)
    
    fr_f1_micro = metrics.f1_score(y_test, y_pred,average='micro')
#     print("Random Forest F1 micro:",fr_f1_micro)
    
    confuse = confusion_matrix(y_test, y_pred)
#    print("Confusion Matrix RF : \n", confuse)
#    print("RF Class Wise Accuracy : ",confuse.diagonal()/confuse.sum(axis=1))
    ######################## Compute ROC curve and ROC area for each class ################
    y_prob = y_pred
    macro_roc_auc_ovo = roc_auc_score_multiclass(y_test, y_prob, average='macro')
    
    check = [fr_acc,fr_prec,fr_recall,fr_f1_weighted,fr_f1_macro,fr_f1_micro,macro_roc_auc_ovo[1]]
    return(check)

# In[5]
    ##########################  Logistic Regression Classifier  ################################
def lr_fun(X_train,y_train,X_test,y_test):

    model = LogisticRegression(solver='liblinear', random_state=0)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    LR_acc = metrics.accuracy_score(y_test, y_pred)
#     print("Logistic Regression Accuracy:",LR_acc)
    
    LR_prec = metrics.precision_score(y_test, y_pred,average='weighted')
#     print("Logistic Regression Precision:",LR_prec)
    
    LR_recall = metrics.recall_score(y_test, y_pred,average='weighted')
#     print("Logistic Regression Recall:",LR_recall)
    
    LR_f1_weighted = metrics.f1_score(y_test, y_pred,average='weighted')
#     print("Logistic Regression F1 weighted:",LR_f1_weighted)
    
    LR_f1_macro = metrics.f1_score(y_test, y_pred,average='macro')
#     print("Logistic Regression F1 macro:",LR_f1_macro)
    
    LR_f1_micro = metrics.f1_score(y_test, y_pred,average='micro')
#     print("Logistic Regression F1 micro:",LR_f1_micro)
    
    confuse = confusion_matrix(y_test, y_pred)
#    print("Confusion Matrix LR : \n", confuse)
#    print("LR Class Wise Accuracy : ",confuse.diagonal()/confuse.sum(axis=1))
    ######################## Compute ROC curve and ROC area for each class ################
    y_prob = y_pred
    macro_roc_auc_ovo = roc_auc_score_multiclass(y_test, y_prob, average='macro')
    
    check = [LR_acc,LR_prec,LR_recall,LR_f1_weighted,LR_f1_macro,LR_f1_micro,macro_roc_auc_ovo[1]]
    return(check)


def fun_decision_tree(X_train,y_train,X_test,y_test):
    from sklearn import tree
    
    clf = tree.DecisionTreeClassifier()    
    clf = clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    dt_acc = metrics.accuracy_score(y_test, y_pred)
#     print("Logistic Regression Accuracy:",LR_acc)
    
    dt_prec = metrics.precision_score(y_test, y_pred,average='weighted')
#     print("Logistic Regression Precision:",LR_prec)
    
    dt_recall = metrics.recall_score(y_test, y_pred,average='weighted')
#     print("Logistic Regression Recall:",LR_recall)
    
    dt_f1_weighted = metrics.f1_score(y_test, y_pred,average='weighted')
#     print("Logistic Regression F1 weighted:",LR_f1_weighted)
    
    dt_f1_macro = metrics.f1_score(y_test, y_pred,average='macro')
#     print("Logistic Regression F1 macro:",LR_f1_macro)
    
    dt_f1_micro = metrics.f1_score(y_test, y_pred,average='micro')
#     print("Logistic Regression F1 micro:",LR_f1_micro)
    
    confuse = confusion_matrix(y_test, y_pred)
#    print("Confusion Matrix DT : \n", confuse)
#    print("DT Class Wise Accuracy : ",confuse.diagonal()/confuse.sum(axis=1))
    ######################## Compute ROC curve and ROC area for each class ################
    y_prob = y_pred
    macro_roc_auc_ovo = roc_auc_score_multiclass(y_test, y_prob, average='macro')
    
    check = [dt_acc,dt_prec,dt_recall,dt_f1_weighted,dt_f1_macro,dt_f1_micro,macro_roc_auc_ovo[1]]
    return(check)

def fun_ridge_classifier(X_train,y_train,X_test,y_test):
    
    clf = RidgeClassifier(alpha=0.0001, solver='lsqr')
    clf.fit(X_train, y_train) 
    y_pred = clf.predict(X_test)
    
    RC_acc = metrics.accuracy_score(y_test, y_pred)
#     print("Logistic Regression Accuracy:",LR_acc)
    
    RC_prec = metrics.precision_score(y_test, y_pred,average='weighted')
#     print("Logistic Regression Precision:",LR_prec)
    
    RC_recall = metrics.recall_score(y_test, y_pred,average='weighted')
#     print("Logistic Regression Recall:",LR_recall)
    
    RC_f1_weighted = metrics.f1_score(y_test, y_pred,average='weighted')
#     print("Logistic Regression F1 weighted:",LR_f1_weighted)
    
    RC_f1_macro = metrics.f1_score(y_test, y_pred,average='macro')
#     print("Logistic Regression F1 macro:",LR_f1_macro)
    
    RC_f1_micro = metrics.f1_score(y_test, y_pred,average='micro')
#     print("Logistic Regression F1 micro:",LR_f1_micro)
    
    confuse = confusion_matrix(y_test, y_pred)
#    print("Confusion Matrix Ridge Classifier : \n", confuse)
#    print("RC Class Wise Accuracy : ",confuse.diagonal()/confuse.sum(axis=1))
    ######################## Compute ROC curve and ROC area for each class ################
    y_prob = y_pred
    macro_roc_auc_ovo = roc_auc_score_multiclass(y_test, y_prob, average='macro')
    
    check = [RC_acc,RC_prec,RC_recall,RC_f1_weighted,RC_f1_macro,RC_f1_micro,macro_roc_auc_ovo[1]]
    return(check)


def classification_err(y_test, y_pred):
    return (0.5 - np.dot(np.sign(y_test), y_pred)/len(y_test)/2)
    
print("X_train rows = ",len(X_train),"X_train columns = ",len(X_train[0]))
print("X_test rows = ",len(X_test),"X_test columns = ",len(X_test[0]))

print("Random Fourier Features Starts here!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

start_time = time.time()
rbf_feature = RBFSampler(gamma=1, n_components=500)
rbf_feature.fit(X_train)
X_features_train = rbf_feature.transform(X_train)
X_features_test = rbf_feature.transform(X_test)


gauu_nb_table = []
lr_table = []
rc_table = []

start_time = time.time()
gauu_nb_return = gaus_nb_fun(X_features_train,y_train,X_features_test,y_test)
end_time = time.time() - start_time
print("Naive Bayes Total Time in seconds =>",end_time)


start_time = time.time()
lr_return = lr_fun(X_features_train,y_train,X_features_test,y_test)
end_time = time.time() - start_time
print("LR Total Time in seconds =>",end_time)


start_time = time.time()
rc_return = fun_ridge_classifier(X_features_train,y_train,X_features_test,y_test)
end_time = time.time() - start_time
print("RC Total Time in seconds =>",end_time)


gauu_nb_table.append(gauu_nb_return)
lr_table.append(lr_return)
rc_table.append(rc_return)
     
gauu_nb_table_final = DataFrame(gauu_nb_table, columns=["Accuracy","Precision","Recall",
                                                "F1 (weighted)","F1 (Macro)","F1 (Micro)","ROC AUC"])
lr_table_final = DataFrame(lr_table, columns=["Accuracy","Precision","Recall",
                                                "F1 (weighted)","F1 (Macro)","F1 (Micro)","ROC AUC"])
rc_table_final = DataFrame(rc_table, columns=["Accuracy","Precision","Recall",
                                                "F1 (weighted)","F1 (Macro)","F1 (Micro)","ROC AUC"])
                                                
final_mean_mat = []

final_mean_mat.append(np.transpose((list(gauu_nb_table_final.mean()))))
final_mean_mat.append(np.transpose((list(lr_table_final.mean()))))
final_mean_mat.append(np.transpose((list(rc_table_final.mean()))))

final_avg_mat = DataFrame(final_mean_mat,columns=["Accuracy","Precision","Recall",
                                                "F1 (weighted)","F1 (Macro)","F1 (Micro)","ROC AUC"], 
                          index=["NB","LR","RC"])

print(final_avg_mat)                                                

print("Random Binning Features Functions starts here!!!")

import numpy as np
import time
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import Ridge
from sklearn import metrics
import scipy
from sklearn.linear_model import RidgeClassifier 
# coding: utf-8


# In[ ]:


# https://github.com/matejbalog/mondrian-kernel/blob/master/random_binning.py
class RandomBinning(object):
    def __init__(self, D, M):
        """ Sets up a random binning object for the isotropic Laplacian kernel in D dimensions.
         A random binning object is a 3-tuple (widths, shifts, keys) where
         - widths is a list of D reals, specifying bin widths in each input dimension
         - shifts is a list of D reals, specifying bin shifts
         - keys is a dictionary int -> int giving sequential numbers to non-empty bins
        """

        self.widths = [np.array([np.random.gamma(shape=2, scale=1.0) for _ in range(D)]) for _ in range (M)]
        self.shifts = [np.array([np.random.uniform(low=0.0, high=width) for width in widths]) for widths in self.widths]
        self.keys = {}
        self.C = 0
        self.M = M
        self.D = D

    def get_features(self, X, M=None, expand=True):
        """ Returns unnormalized Random binning features for the provided datapoints X (one datapoint in each row).
        :param X: Matrix of dimensions NxD, containing N datapoints (one in each row).
        :param expand: Specifies whether new features should be created if a datapoint lies in a bin
         that has been empty so far. (True for training, False for testing.)
        :return: Sparse binary matrix of dimensions NxC, where C is the number of generated features.
        Each row is the feature expansion of one datapoint and contains at most M ones.
        """
        N = np.shape(X)[0]

        if M is None:
            M = self.M
        assert M <= self.M

        # stacking experiment
        X_stack = np.tile(X, self.M)
        shifts_stack = np.concatenate(self.shifts)
        widths_stack = np.concatenate(self.widths)
        X_coordinates = np.ceil((X_stack - shifts_stack) / widths_stack).astype(int)

        # compute indices
        row_indices = []
        col_indices = []
        X_coordinates.flags.writeable = False
        feature_from_repetition = []
        for m in range(M):
            X_coords = X_coordinates[:, (self.D*m):(self.D*(m+1))]
            X_coords.flags.writeable = False
            for n, coordinates in enumerate(X_coords):
                coordinates.flags.writeable = False
                #h = hash(coordinates.data)
                h = tuple(coordinates.tolist())
                if (m, h) in self.keys:
                    row_indices.append(n)
                    col_indices.append(self.keys[(m, h)])
                elif expand:
                    row_indices.append(n)
                    col_indices.append(self.C)
                    self.keys[(m, h)] = self.C
                    feature_from_repetition.append(m)
                    self.C += 1

        # construct features
        values = [1]*len(row_indices)
        Z = scipy.sparse.coo_matrix((values, (row_indices, col_indices)), shape=(N, self.C))
        return Z.tocsr(), np.array(feature_from_repetition)


    def random_binning_features(X, R_max):
        D = X.shape[1]
        rb = RandomBinning(D, R_max)
        return rb.get_features(X)


    def evaluate_random_binning(X, y, X_test, y_test, M, task):
        # construct random binning features
        start_time = time.time()
        rb = RandomBinning(X.shape[1], M)
        Z, _ = rb.get_features(X) / np.sqrt(M)
        Z_test, _ = rb.get_features(X_test, expand=False) / np.sqrt(M)
        if(task=='classification'):
            clf = RidgeClassifier(solver='auto', alpha=0.0001)
            clf.fit(Z, y) 
            y_pred = clf.predict(Z_test)
            error_test = (0.5 - np.dot(np.sign(y_test), y_pred)/len(y_test)/2)*100
            print("--- %s seconds ---" % (time.time() - start_time))
            print("C = %d; error_test = %.2f" % (np.shape(Z)[1], error_test)+'%')
        elif(task=='regression'):
            clf = Ridge(alpha=0.01, solver='lsqr',random_state=42)
            clf.fit(Z, y) 
            y_pred = clf.predict(Z_test)
            error_test = np.linalg.norm((y_test-y_pred))/np.linalg.norm(y_test)*100
            print("--- %s seconds ---" % (time.time() - start_time))
            print("C = %d; error_test = %.2f" % (np.shape(Z)[1], error_test)+'%')
        else:
            error_test = 'error!'
            print('No such a task, please check the task name!')
        return error_test

#err = RandomBinning.evaluate_random_binning(X_train, y_train, X_test, y_test, 30, 'classification')

print("All Processing Done!!!")
