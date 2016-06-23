# -*- coding: utf-8 -*-
"""
Created on Sun Jun 19 12:36:32 2016

@author:武本
個別に次元圧縮した62個のｃｓｖを作る
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
 
 
csv_train_path = 'C:\\Users\\mech-user\\Documents\\AlphabetsRecognition\\CSV_train'
csv_test_path = 'C:\\Users\\mech-user\\Documents\\AlphabetsRecognition\\CSV_test'
learning_model_path = 'C:\\Users\\mech-user\\Documents\\AlphabetsRecognition\\Learning_Models\\pca900decomp_SVC_64_all_learning_model' 
 
 
dim=64
pca=IncrementalPCA(n_components=dim)
 
 
 
print("Now Loading Data") 

start_time1 = time.clock()

for i in range(0,1):
    if i < 10:
        if i == 9:
            csv_train_path2 = csv_train_path + '\\train_digits\\' + 'train0' + str(i+1) + '.csv'
        else:
            csv_train_path2 = csv_train_path + '\\train_digits\\' + 'train00' + str(i+1) + '.csv'
            
    else:
        csv_train_path2 = csv_train_path + '\\train_alphabets\\' + 'train0' + str(i+1) + '.csv'

    # Making CSVs
    print("Making Decomped CSV",i+1,"...")
    start_time2 = time.clock() 
    train_data = np.array(p.read_csv(filepath_or_buffer=csv_train_path2, header=None, sep=',', index_col=0))[:, :] 
    #train_label = np.ravel(np.array(p.read_csv(filepath_or_buffer=csv_train_path2, header=None, sep=',', usecols=[0]))[:, :]) 
     
     
    # print(train_data) 
    # print(train_label) 
     
     
    #end_time = time.clock() 
    #print("Loading Complete \nTime =", end_time - start_time) 
    # Time =         
     
    # Data decomposition 
    #print("Now Decompositing Data") 
    #start_time = time.clock() 
     
     
    #from sklearn.decomposition import TruncatedSVD 
     
    
    #decomp = TruncatedSVD(n_components=1000,n_iter=5) 
    #decomp.fit(train_data)  
    train_data = pca.fit_transform(train_data)
     
     
    end_time2 = time.clock() 
    print("Decompositing",i+1," Complete : Time =", end_time2 - start_time2) 
    # Time = 7.5
    #print(train_data) 
     
     
     
    
    # Saving decomposed data as csv 
    csv_decomp_train_path = 'C:\\Users\\mech-user\\Documents\\AlphabetsRecognition\\CSV_train\\Decomp\\pca900decomp_revised2_train' 
    
    if i < 9:
        csv_decomp_train_path = csv_decomp_train_path + '00' + str(i+1) + '.csv'
    else:
        csv_decomp_train_path = csv_decomp_train_path + '0' + str(i+1) + '.csv'
     
#    train_data.to_csv(csv_decomp_train_path, index=None)     
     
    with open( csv_decomp_train_path, 'w', newline='') as f: 
                    writer = csv.writer(f, delimiter='\n', lineterminator='\n')  
                    writer.writerow(train_data)    
                    #writer.writerow('\n')    
     
end_time1 = time.clock() 
print("Decompositing All Complete : Time =", end_time1 - start_time1)  
#Time = 500    
 


########### Testing #################################### 

 
print("Now Loading Test Data") 

start_time1 = time.clock()

for i in range(0,1):
    if i < 10:
        if i == 9:
            csv_test_path2 = csv_test_path + '\\test_digits\\' + 'test0' + str(i+1) + '.csv'
        else:
            csv_test_path2 = csv_test_path + '\\test_digits\\' + 'test00' + str(i+1) + '.csv'
            
    else:
        csv_test_path2 = csv_test_path + '\\test_alphabets\\' + 'test0' + str(i+1) + '.csv'

    # Making CSVs
    print("Making Decomped CSV",i+1,"...\n")
    start_time2 = time.clock() 
    test_data = np.array(p.read_csv(filepath_or_buffer=csv_test_path2, header=None, sep=',', index_col=0))[:, :] 
    #test_label = np.ravel(np.array(p.read_csv(filepath_or_buffer=csv_test_path2, header=None, sep=',', usecols=[0]))[:, :]) 
     
     
    # print(test_data) 
    # print(test_label) 
     
     
    #end_time = time.clock() 
    #print("Loading Complete \nTime =", end_time - start_time) 
    # Time =         
     
    # Data decomposition 
    #print("Now Decompositing Data") 
    #start_time = time.clock() 
     
     
    #from sklearn.decomposition import TruncatedSVD 
     
    
    #decomp = TruncatedSVD(n_components=1000,n_iter=5) 
    #decomp.fit(test_data)  
    test_data = pca.fit_transform(test_data)
     
     
    end_time2 = time.clock() 
    print("Decompositing",i+1," Complete : Time =", end_time2 - start_time2) 
    # Time = 1.3
    #print(test_data) 
     
     
     
    
    # Saving decomposed data as csv 
    csv_decomp_test_path = 'C:\\Users\\mech-user\\Documents\\AlphabetsRecognition\\CSV_test\\Decomp\\pca900decomp_revised_test' 
    
    if i < 9:
        csv_decomp_test_path = csv_decomp_test_path + '00' + str(i+1) + '.csv'
    else:
        csv_decomp_test_path = csv_decomp_test_path + '0' + str(i+1) + '.csv'
     
    with open( csv_decomp_test_path, 'w') as f: 
                    writer = csv.writer(f, delimiter='\n', lineterminator='\n')   
                    writer.writerow(test_data)    
                    #writer.writerow('\n')    
     
end_time1 = time.clock() 
print("Decompositing All Complete : Time =", end_time1 - start_time1)
#Time = 94 
 
 
 
 
 
 
 
 
 
 
 
########### Learning ################################### 
# 
# 
#from sklearn.externals import joblib 
#from sklearn import svm 
# 
# 
#clf = svm.LinearSVC() 
# 
# 
### Saving data 
## joblib.dump(clf,learning_model_path) 
# 
# 
### Loading data 
## clf = joblib.load(learning_model_path) 
## print("Now Loading...") 
# 
# 
#start_time = time.clock() 
#print("Now Learning...") 
#clf.fit(train_data, train_label)  # learn from the data 
#end_time = time.clock() 
#print("Learning Complete \nTime =", end_time - start_time) 
## Time = 
# 
# 
## Saving data 
#joblib.dump(clf, learning_model_path) 







#
# 
#print("Calculating Score...") 
#predict = clf.predict(test_data) 
# 
# 
#from sklearn.metrics import accuracy_score 
#print(accuracy_score(test_label, predict)) 
# 
# 
#from sklearn.metrics import classification_report 
#print(classification_report(test_label, predict)) 
# 
# 
#from sklearn import metrics 
#print ( metrics.confusion_matrix(test_label, predict) ) 
#
