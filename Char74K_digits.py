# coding:utf-8
import numpy as np
import pandas as p
import time as time

# http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k
# 64 classes (0-9, A-Z, a-z)
# 7705 characters obtained from natural images
# 3410 hand drawn characters using a tablet PC
# 62992 synthesised characters from computer fonts   we use this data

csv_test_path = '/home/mech-user/Alphabet_Recognition/AlphabetsRecognition/Char74K/CSV_test/test_alphabets/test_alphabets.csv'
csv_train_path = '/home/mech-user/Alphabet_Recognition/AlphabetsRecognition/Char74K/CSV_train/train_alphabets/train_alphabets.csv'
learning_model_path = '/home/mech-user/Alphabet_Recognition/AlphabetsRecognition/alphabet_model_NuSVC'


########## Loading ###################################
print("Now Loading Data")
start_time = time.clock()
train_data = np.array(p.read_csv(filepath_or_buffer=csv_train_path, header=None, sep=',', index_col=0))[:, :]
train_label = np.ravel(np.array(p.read_csv(filepath_or_buffer=csv_train_path, header=None, sep=',', usecols=[0]))[:, :])
# print(train_data)
# print(train_label)
end_time = time.clock()
print("Loading Complete \nTime =", end_time - start_time)
# Time =




########## Learning ###################################

from sklearn.externals import joblib
from sklearn.svm import NuSVC

# Settiong Classifier
clf = NuSVC()

## Saving data
# joblib.dump(clf,learning_model_path)

## Loading data
# clf = joblib.load(learning_model_path)
# print("Now Loading Model...")

start_time = time.clock()
print("Now Learning...")
clf.fit(train_data, train_label)  
end_time = time.clock()
print("Learning Complete \nTime =", end_time - start_time)
# Time = 7276.782202

# Saving data
joblib.dump(clf, learning_model_path)




########### Testing ####################################

print("Making Testing Data...")
test_data = np.array(p.read_csv(filepath_or_buffer=csv_test_path, header=None, sep=',', index_col=0))[:, :]
test_label = np.ravel(np.array(p.read_csv(filepath_or_buffer=csv_test_path, header=None, sep=',', usecols=[0]))[:, :])

print("Calculating Score...")
predict = clf.predict(test_data)

from sklearn.metrics import accuracy_score
print(accuracy_score(test_label, predict))

from sklearn.metrics import classification_report
print(classification_report(test_label, predict))

from sklearn import metrics
print ( metrics.confusion_matrix(test_label, predict) )


########### Results ####################################

# SVM

# alphabets
# 0.811771561772
#             precision    recall  f1-score   support
#          A       0.98      0.87      0.92        99
#          B       0.73      0.96      0.83        99
#          C       0.71      0.72      0.71        99
#          D       0.90      0.95      0.93        99
#          E       0.84      0.88      0.86        99
#          F       0.91      0.95      0.93        99
#          G       0.98      0.95      0.96        99
#          H       0.80      0.89      0.84        99
#          I       0.60      0.18      0.28        99
#          J       0.93      0.81      0.86        99
#          K       0.96      0.90      0.93        99
#          L       0.95      0.92      0.93        99
#          M       0.77      0.87      0.82        99
#          N       0.87      0.88      0.87        99
#          O       0.63      0.74      0.68        99
#          P       0.95      0.82      0.88        99
#          Q       0.93      0.78      0.85        99
#          R       0.90      0.97      0.93        99
#          S       0.72      0.78      0.75        99
#          T       0.88      0.92      0.90        99
#          U       0.74      0.79      0.76        99
#          V       0.76      0.47      0.58        99
#          W       0.74      0.66      0.70        99
#          X       0.71      0.73      0.72        99
#          Y       0.99      0.80      0.88        99
#          Z       0.67      0.77      0.72        99

#          a       0.85      0.83      0.84        99
#          b       0.93      0.93      0.93        99
#          c       0.71      0.71      0.71        99
#          d       0.87      0.86      0.86        99
#          e       0.82      0.85      0.83        99
#          f       0.90      0.90      0.90        99
#          g       0.86      0.70      0.77        99
#          h       0.87      0.87      0.87        99
#          i       0.88      0.92      0.90        99
#          j       0.97      0.86      0.91        99
#          k       0.94      0.88      0.91        99
#          l       0.39      0.95      0.55        99
#          m       0.96      0.94      0.95        99
#          n       0.89      0.92      0.91        99
#          o       0.66      0.64      0.65        99
#          p       0.87      0.92      0.89        99
#          q       0.94      0.93      0.93        99
#          r       0.88      0.85      0.87        99
#          s       0.82      0.64      0.72        99
#          t       0.98      0.86      0.91        99
#          u       0.76      0.85      0.80        99
#          v       0.59      0.70      0.64        99
#          w       0.75      0.77      0.76        99
#          x       0.73      0.63      0.67        99
#          y       0.84      0.82      0.83        99
#          z       0.76      0.60      0.67        99





# digits
##0.971717171717

##             precision    recall  f1-score   support
##          0       0.99      0.93      0.96        99
##          1       1.00      0.97      0.98        99
##          2       0.98      0.98      0.98        99
##          3       0.99      0.96      0.97        99
##          4       1.00      1.00      1.00        99
##          5       0.92      0.99      0.96        99
##          6       0.99      0.98      0.98        99
##          7       1.00      0.99      0.99        99
##          8       0.89      0.97      0.93        99
##          9       0.97      0.95      0.96        99

##avg / total       0.97      0.97      0.97       990
