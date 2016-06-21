# -*- coding: utf-8 -*-
"""
Created on Sun Jun 19 12:36:32 2016

@author: 笑峰
"""

# coding:utf-8 
import numpy as np 
import pandas as p 
import time as time 
import csv 
from sklearn.decomposition import IncrementalPCA
 
 
 
# http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k 
# 64 classes (0-9, A-Z, a-z) 0-
# 7705 characters obtained from natural images 
# 3410 hand drawn characters using a tablet PC 
# 62992 synthesised characters from computer fonts   we use this data 
 
 
csv_test_path = 'csvdata/CSV_alphabets_test/csvall_alphabets_test.csv' 
csv_train_path = 'csvdata/CSV_alphabets_train/csvall_alphabets_train.csv' 
learning_model_path = 'pca900decomp_alphabets_learning_model' 
 
 
 
dim=64
pca=IncrementalPCA(dim)
 
 
 
print("Now Loading Data") 
start_time = time.clock() 
train_data = np.array(p.read_csv(filepath_or_buffer=csv_train_path, header=None, sep=',', index_col=0))[:, :] 
train_label = np.ravel( 
    np.array(p.read_csv(filepath_or_buffer=csv_train_path, header=None, sep=',', usecols=[0]))[:, :]) 
 
 
# print(train_data) 
# print(train_label) 
 
 
end_time = time.clock() 
print("Loading Complete \nTime =", end_time - start_time) 
# Time = 
 
 
 
 
# Data decomposition 
print("Now Decompositing Data") 
start_time = time.clock() 
 
 
#from sklearn.decomposition import TruncatedSVD 
 

#decomp = TruncatedSVD(n_components=1000,n_iter=5) 
#decomp.fit(train_data)  
train_data = pca.fit_transform(train_data)
 
 
end_time = time.clock() 
print("Decompositing Complete \nTime =", end_time - start_time) 
# Time = 
print(train_data) 
 
 
 

# Saving decomposed data as csv 
csv_decomp_train_path = 'csv_pca900decomp_alphabets_train.csv' 
 
 
with open( csv_decomp_train_path, 'w') as f: 
                writer = csv.writer(f, lineterminator='\n')  
                writer.writerow(train_data)    
                #writer.writerow('\n')    
 
 
 
########## Learning ################################### 
 
 
from sklearn.externals import joblib 
from sklearn import svm 
 
 
clf = svm.LinearSVC() 
 
 
## Saving data 
# joblib.dump(clf,learning_model_path) 
 
 
## Loading data 
# clf = joblib.load(learning_model_path) 
# print("Now Loading...") 
 
 
start_time = time.clock() 
print("Now Learning...") 
clf.fit(train_data, train_label)  # learn from the data 
end_time = time.clock() 
print("Learning Complete \nTime =", end_time - start_time) 
# Time = 
 
 
# Saving data 
joblib.dump(clf, learning_model_path) 

########### Testing #################################### 

 
print("Loading Test Data...") 
test_data = np.array(p.read_csv(filepath_or_buffer=csv_test_path, header=None, sep=',', index_col=0))[:, :] 
test_label = np.ravel(np.array(p.read_csv(filepath_or_buffer=csv_test_path, header=None, sep=',', usecols=[0]))[:, :]) 
 
 
# Data decomposition 
print("Now Decompositing Data") 
start_time = time.clock() 
 
 
#from sklearn.decomposition import TruncatedSVD 
 
 
#decomp = TruncatedSVD(n_components=1000,n_iter=5) 
#decomp.fit(test_data) 
test_data = pca.fit_transform(test_data)
 
 
end_time = time.clock() 
print("Decompositing Complete \nTime =", end_time - start_time) 
# Time = 
print(test_data) 
 
 
# Saving decomposed data as csv 
csv_decomp_test_path = 'csv_pca900decomp_alphabets_test.csv' 
 
 
with open( csv_decomp_test_path, 'w') as f: 
                writer = csv.writer(f, lineterminator='\n')  
                writer.writerow(test_data)    
                #writer.writerow('\n')   
 
 
 
 
 
 
print("Calculating Score...") 
predict = clf.predict(test_data) 
 
 
from sklearn.metrics import accuracy_score 
print(accuracy_score(test_label, predict)) 
 
 
from sklearn.metrics import classification_report 
print(classification_report(test_label, predict)) 
 
 
from sklearn import metrics 
print ( metrics.confusion_matrix(test_label, predict) ) 

