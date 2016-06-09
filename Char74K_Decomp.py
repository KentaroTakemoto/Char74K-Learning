# coding:utf-8
import numpy as np
import pandas as p
import time as time
import csv


# http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k
# 64 classes (0-9, A-Z, a-z)
# 7705 characters obtained from natural images
# 3410 hand drawn characters using a tablet PC
# 62992 synthesised characters from computer fonts   we use this data

csv_test_path = '/home/mech-user/Alphabet_Recognition/AlphabetsRecognition/Char74K/CSV_test/test_alphabets.csv'
csv_train_path = '/home/mech-user/Alphabet_Recognition/AlphabetsRecognition/Char74K/CSV_train/train_alphabets.csv'
learning_model_path = '/home/mech-user/Alphabet_Recognition/AlphabetsRecognition/alphabet_model'




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

from sklearn.decomposition import TruncatedSVD

decomp = TruncatedSVD(n_components=1000,n_iter=5)
decomp.fit(train_data) 
train_data = decomp.transform(train_data)

end_time = time.clock()
print("Decompositing Complete \nTime =", end_time - start_time)
# Time =
print(train_data)


# Saving decomposed data as csv
csv_decomp_train_path = '/home/mech-user/Alphabet_Recognition/AlphabetsRecognition/Char74K/CSV_train/train_decomp/train_decomp1000.csv'

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

from sklearn.decomposition import TruncatedSVD

decomp = TruncatedSVD(n_components=1000,n_iter=5)
decomp.fit(test_data)
test_data = decomp.transform(test_data)

end_time = time.clock()
print("Decompositing Complete \nTime =", end_time - start_time)
# Time =
print(test_data)

# Saving decomposed data as csv
csv_decomp_test_path = '/home/mech-user/Alphabet_Recognition/AlphabetsRecognition/Char74K/CSV_test/test_decomp/test_decomp1000.csv'

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



# # Decomp 1000
# Loading Complete
# Time = 254.276317
# Decompositing Complete
# Time = 5131.0992
# Learning Complete
# Time = 1111.5657549999996
# Decompositing Complete
# Time = 666.2742910000006
# [[  2.93731326e+04   5.13242727e+03   1.06393448e+03 ...,   5.54499138e-01
#    -2.89055795e-01   6.10910455e-01]
#  [  2.80703184e+04   5.19591941e+03  -1.87858527e+03 ...,   1.60324481e-01
#     7.36669039e-01   1.09203467e+00]
#  [  2.94014606e+04   3.45419476e+03   8.20173420e+02 ...,  -2.16738593e-01
#     8.23282648e-01   7.58075205e-01]
#  ...,
#  [  2.72105476e+04   1.47139779e+03  -1.64335986e+03 ...,   2.25652893e-02
#    -4.54676152e-01   6.52803351e-01]
#  [  2.60323089e+04   8.05754109e+02  -3.07140282e+03 ...,  -2.28755292e-01
#    -3.68096689e-01  -1.74621740e+00]
#  [  2.72105476e+04   1.47139779e+03  -1.64335986e+03 ...,   2.25652893e-02
#    -4.54676152e-01   6.52803351e-01]]
# Calculating Score...
# 0.0270007770008
#              precision    recall  f1-score   support
#
#           A       0.00      0.00      0.00        99
#           B       0.09      0.05      0.07        99
#           C       0.04      0.01      0.02        99
#           D       0.00      0.00      0.00        99
#           E       0.08      0.13      0.10        99
#           F       0.26      0.32      0.29        99
#           G       0.01      0.01      0.01        99
#           H       0.00      0.00      0.00        99
#           I       0.00      0.00      0.00        99
#           J       0.00      0.00      0.00        99
#           K       0.11      0.05      0.07        99
#           L       0.38      0.24      0.30        99
#           M       0.10      0.03      0.05        99
#           N       0.00      0.00      0.00        99
#           O       0.01      0.02      0.01        99
#           P       0.09      0.03      0.05        99
#           Q       0.11      0.08      0.09        99
#           R       0.00      0.00      0.00        99
#           S       0.00      0.00      0.00        99
#           T       0.00      0.00      0.00        99
#           U       0.01      0.02      0.01        99
#           V       0.00      0.00      0.00        99
#           W       0.08      0.06      0.07        99
#           X       0.00      0.00      0.00        99
#           Y       0.00      0.00      0.00        99
#           Z       0.00      0.00      0.00        99

#           a       0.00      0.00      0.00        99
#           b       0.00      0.00      0.00        99
#           c       0.00      0.00      0.00        99
#           d       0.00      0.00      0.00        99
#           e       0.00      0.00      0.00        99
#           f       0.00      0.00      0.00        99
#           g       0.00      0.00      0.00        99
#           h       0.09      0.06      0.07        99
#           i       0.00      0.00      0.00        99
#           j       0.00      0.00      0.00        99
#           k       0.03      0.01      0.01        99
#           l       0.00      0.00      0.00        99
#           m       0.00      0.00      0.00        99
#           n       0.00      0.00      0.00        99
#           o       0.00      0.00      0.00        99
#           p       0.11      0.09      0.10        99
#           q       0.13      0.07      0.09        99
#           r       0.06      0.10      0.08        99
#           s       0.00      0.00      0.00        99
#           t       0.00      0.00      0.00        99
#           u       0.00      0.00      0.00        99
#           v       0.00      0.00      0.00        99
#           w       0.02      0.01      0.01        99
#           x       0.00      0.00      0.00        99
#           y       0.00      0.00      0.00        99
#           z       0.00      0.00      0.00        99
#
# avg / total       0.03      0.03      0.03      5148
#




# Decomp 50
# Loading Complete
# Time = 203.62258599999998
# Now Decompositing Data
# Decompositing Complete
# Time = 359.39450899999997
# Now Learning...
# Learning Complete
# Time = 493.374194
# Loading Test Data...
# Now Decompositing Data
# Decompositing Complete
# Time = 56.27740899999981
# 0.0143745143745
# /home/mech-user/anaconda3/lib/python3.5/site-packages/sklearn/metrics/classification.py:1074: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples.
#   'precision', 'predicted', average, warn_for)
#              precision    recall  f1-score   support
#
#           A       0.60      0.03      0.06        99
#           B       0.00      0.00      0.00        99
#           C       0.00      0.00      0.00        99
#           D       0.00      0.00      0.00        99
#           E       0.00      0.00      0.00        99
#           F       0.06      0.03      0.04        99
#           G       0.00      0.00      0.00        99
#           H       0.00      0.00      0.00        99
#           I       0.00      0.00      0.00        99
#           J       0.00      0.00      0.00        99
#           K       0.07      0.09      0.08        99
#           L       0.00      0.00      0.00        99
#           M       0.00      0.00      0.00        99
#           N       0.00      0.00      0.00        99
#           O       0.00      0.00      0.00        99
#           P       0.27      0.08      0.12        99
#           Q       0.09      0.02      0.03        99
#           R       0.12      0.07      0.09        99
#           S       0.00      0.00      0.00        99
#           T       0.00      0.00      0.00        99
#           U       0.00      0.00      0.00        99
#           V       0.00      0.00      0.00        99
#           W       0.00      0.00      0.00        99
#           X       0.00      0.00      0.00        99
#           Y       0.00      0.00      0.00        99
#           Z       0.00      0.00      0.00        99
#           a       0.00      0.00      0.00        99
#           b       0.00      0.00      0.00        99
#           c       0.00      0.00      0.00        99
#           d       0.00      0.00      0.00        99
#           e       0.01      0.15      0.01        99
#           f       0.00      0.00      0.00        99
#           g       0.00      0.00      0.00        99
#           h       0.00      0.00      0.00        99
#           i       0.00      0.00      0.00        99
#           j       0.00      0.00      0.00        99
#           k       0.00      0.00      0.00        99
#           l       0.00      0.00      0.00        99
#           m       0.02      0.01      0.01        99
#           n       0.00      0.00      0.00        99
#           o       0.00      0.00      0.00        99
#           p       0.21      0.13      0.16        99
#           q       0.12      0.06      0.08        99
#           r       0.00      0.00      0.00        99
#           s       0.00      0.00      0.00        99
#           t       0.00      0.00      0.00        99
#           u       0.00      0.00      0.00        99
#           v       0.05      0.02      0.03        99
#           w       0.50      0.05      0.09        99
#           x       0.00      0.00      0.00        99
#           y       0.00      0.00      0.00        99
#           z       0.00      0.00      0.00        99
#
# avg / total       0.04      0.01      0.02      5148
#
# [[3 2 0 ..., 0 0 0]
#  [0 0 0 ..., 0 0 0]
#  [0 0 0 ..., 0 0 0]
#  ...,
#  [0 0 0 ..., 0 0 0]
#  [0 0 0 ..., 0 0 0]
#  [0 0 0 ..., 0 0 0]]








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
