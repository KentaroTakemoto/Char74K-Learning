# coding:utf-8
import numpy as np
import pandas as p
import time as time

from sklearn.neural_network import BernoulliRBM

# http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k
# 64 classes (0-9, A-Z, a-z)
# 7705 characters obtained from natural images
# 3410 hand drawn characters using a tablet PC
# 62992 synthesised characters from computer fonts   we use this data

csv_test_path = '/home/mech-user/Alphabet_Recognition/AlphabetsRecognition/Char74K/CSV_test/test_alphabets/test_alphabets.csv'
csv_train_path = '/home/mech-user/Alphabet_Recognition/AlphabetsRecognition/Char74K/CSV_train/train_alphabets/train_alphabets.csv'
learning_model_path = '/home/mech-user/Alphabet_Recognition/AlphabetsRecognition/alphabet_model_Neural_BernoulliRBM'


print("Now Loading Data")
start_time = time.clock()
train_data = np.array(p.read_csv(filepath_or_buffer=csv_train_path, header=None, sep=',', index_col=0))[:, :]
train_label = np.ravel(np.array(p.read_csv(filepath_or_buffer=csv_train_path, header=None, sep=',', usecols=[0]))[:, :])

# print(train_data)
# print(train_label)
end_time = time.clock()
print("Loading Complete \nTime =", end_time - start_time)
# Time =




from scipy.ndimage import convolve
from sklearn import linear_model, datasets, metrics
from sklearn.pipeline import Pipeline


###############################################################################
# Setting up

def nudge_dataset(X, Y):
    """
    This produces a dataset 5 times bigger than the original one,
    by moving the 8x8 images in X around by 1px to left, right, down, up
    """
    direction_vectors = [
        [[0, 1, 0],
         [0, 0, 0],
         [0, 0, 0]],

        [[0, 0, 0],
         [1, 0, 0],
         [0, 0, 0]],

        [[0, 0, 0],
         [0, 0, 1],
         [0, 0, 0]],

        [[0, 0, 0],
         [0, 0, 0],
         [0, 1, 0]]]

    shift = lambda x, w: convolve(x.reshape((8, 8)), mode='constant',
                                  weights=w).ravel()
    X = np.concatenate([X] +
                       [np.apply_along_axis(shift, 1, X, vector)
                        for vector in direction_vectors])
    Y = np.concatenate([Y for _ in range(5)], axis=0)
    return X, Y









########## Learning ###################################

from sklearn.externals import joblib
# from sklearn.neural_network import MLPClassifier

logistic = linear_model.LogisticRegression()
rbm = BernoulliRBM(random_state=0, verbose=True)
classifier = Pipeline(steps=[('rbm', rbm), ('logistic', logistic)])

## Saving data
# joblib.dump(clf,learning_model_path)

# Loading data
# clf = joblib.load(learning_model_path)
# print("Now Loading...")

start_time = time.clock()
print("Now Learning...")
###############################################################################
# Training

# Hyper-parameters. These were set by cross-validation,
# using a GridSearchCV. Here we are not performing cross-validation to
# save time.
rbm.learning_rate = 0.06
rbm.n_iter = 20
# More components tend to give better prediction performance, but larger
# fitting time
rbm.n_components = 100
logistic.C = 6000.0

# Training RBM-Logistic Pipeline
classifier.fit(train_data, train_label)

# Training Logistic regression
logistic_classifier = linear_model.LogisticRegression(C=100.0)
logistic_classifier.fit(train_data, train_label)

end_time = time.clock()
print("Learning Complete \nTime =", end_time - start_time)
# Time = 7276.782202

# Saving data
joblib.dump(rbm, learning_model_path)




########### Testing ####################################

print("Making Testing Data...")
test_data = np.array(p.read_csv(filepath_or_buffer=csv_test_path, header=None, sep=',', index_col=0))[:, :]
test_label = np.ravel(np.array(p.read_csv(filepath_or_buffer=csv_test_path, header=None, sep=',', usecols=[0]))[:, :])

print("Calculating Score...")

###############################################################################
# Evaluation


print("Logistic regression using RBM features:\n%s\n" % (
    metrics.classification_report(
        test_label,
        classifier.predict(test_data))))

print("Logistic regression using raw pixel features:\n%s\n" % (
    metrics.classification_report(
        test_label,
        logistic_classifier.predict(test_data))))



#
# from sklearn.metrics import accuracy_score
# print(accuracy_score(test_label, predict))
#
# from sklearn.metrics import classification_report
# print(classification_report(test_label, predict))
#
# from sklearn import metrics
# print ( metrics.confusion_matrix(test_label, predict) )
#











# # Neural
#
# Now Loading Data
# Loading Complete
# Time = 216.816466
# Now Learning...
# [BernoulliRBM] Iteration 1, pseudo-likelihood = -7985102781.41, time = 316.44s
# [BernoulliRBM] Iteration 2, pseudo-likelihood = -15970605425.42, time = 437.10s
# [BernoulliRBM] Iteration 3, pseudo-likelihood = -23956108097.34, time = 336.35s
# [BernoulliRBM] Iteration 4, pseudo-likelihood = -31941610747.06, time = 334.23s
# [BernoulliRBM] Iteration 5, pseudo-likelihood = -39927113441.26, time = 328.40s
# [BernoulliRBM] Iteration 6, pseudo-likelihood = -47912616023.35, time = 331.11s
# [BernoulliRBM] Iteration 7, pseudo-likelihood = -55898118707.56, time = 331.69s
# [BernoulliRBM] Iteration 8, pseudo-likelihood = -63883621394.26, time = 327.87s
# [BernoulliRBM] Iteration 9, pseudo-likelihood = -71869123992.47, time = 330.43s
# [BernoulliRBM] Iteration 10, pseudo-likelihood = -79854626671.29, time = 328.03s
# [BernoulliRBM] Iteration 11, pseudo-likelihood = -87840129309.34, time = 330.72s
# [BernoulliRBM] Iteration 12, pseudo-likelihood = -95825631976.12, time = 329.17s
# [BernoulliRBM] Iteration 13, pseudo-likelihood = -103811134715.28, time = 333.43s
# [BernoulliRBM] Iteration 14, pseudo-likelihood = -111796637355.67, time = 329.51s
# [BernoulliRBM] Iteration 15, pseudo-likelihood = -119782140098.32, time = 331.03s
# [BernoulliRBM] Iteration 16, pseudo-likelihood = -127767642711.60, time = 329.75s
# [BernoulliRBM] Iteration 17, pseudo-likelihood = -135753145311.81, time = 326.70s
# [BernoulliRBM] Iteration 18, pseudo-likelihood = -143738647981.58, time = 327.57s
# [BernoulliRBM] Iteration 19, pseudo-likelihood = -151724150701.49, time = 328.24s
# [BernoulliRBM] Iteration 20, pseudo-likelihood = -159709653203.19, time = 331.01s
#
# Learning Complete
# Time = 113972.60193199999
# Making Testing Data...
# Calculating Score...
# /home/mech-user/anaconda3/lib/python3.5/site-packages/sklearn/metrics/classification.py:1074: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples.
#   'precision', 'predicted', average, warn_for)
# Logistic regression using RBM features:
#              precision    recall  f1-score   support
#
#           A       0.00      0.00      0.00        99
#           B       0.00      0.00      0.00        99
#           C       0.00      0.00      0.00        99
#           D       0.00      0.00      0.00        99
#           E       0.00      0.00      0.00        99
#           F       0.00      0.00      0.00        99
#           G       0.00      0.00      0.00        99
#           H       0.00      0.00      0.00        99
#           I       0.00      0.00      0.00        99
#           J       0.00      0.00      0.00        99
#           K       0.00      0.00      0.00        99
#           L       0.00      0.00      0.00        99
#           M       0.00      0.00      0.00        99
#           N       0.00      0.00      0.00        99
#           O       0.00      0.00      0.00        99
#           P       0.00      0.00      0.00        99
#           Q       0.00      0.00      0.00        99
#           R       0.00      0.00      0.00        99
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
#           e       0.00      0.00      0.00        99
#           f       0.00      0.00      0.00        99
#           g       0.00      0.00      0.00        99
#           h       0.00      0.00      0.00        99
#           i       0.00      0.00      0.00        99
#           j       0.00      0.00      0.00        99
#           k       0.00      0.00      0.00        99
#           l       0.00      0.00      0.00        99
#           m       0.00      0.00      0.00        99
#           n       0.00      0.00      0.00        99
#           o       0.00      0.00      0.00        99
#           p       0.00      0.00      0.00        99
#           q       0.00      0.00      0.00        99
#           r       0.00      0.00      0.00        99
#           s       0.00      0.00      0.00        99
#           t       0.00      0.00      0.00        99
#           u       0.00      0.00      0.00        99
#           v       0.00      0.00      0.00        99
#           w       0.00      0.00      0.00        99
#           x       0.00      0.00      0.00        99
#           y       0.00      0.00      0.00        99
#           z       0.02      1.00      0.04        99
#
# avg / total       0.00      0.02      0.00      5148
#
#
# Logistic regression using raw pixel features:
#              precision    recall  f1-score   support
#
#           A       0.94      0.86      0.90        99
#           B       0.89      0.93      0.91        99
#           C       0.69      0.70      0.69        99
#           D       0.91      0.96      0.94        99
#           E       0.85      0.89      0.87        99
#           F       0.91      0.95      0.93        99
#           G       0.98      0.94      0.96        99
#           H       0.82      0.90      0.86        99
#           I       0.56      0.48      0.52        99
#           J       0.93      0.82      0.87        99
#           K       0.96      0.90      0.93        99
#           L       0.95      0.95      0.95        99
#           M       0.79      0.90      0.84        99
#           N       0.84      0.92      0.88        99
#           O       0.66      0.69      0.67        99
#           P       0.93      0.81      0.86        99
#           Q       0.94      0.77      0.84        99
#           R       0.87      0.97      0.92        99
#           S       0.76      0.79      0.78        99
#           T       0.88      0.91      0.90        99
#           U       0.72      0.79      0.75        99
#           V       0.65      0.71      0.68        99
#           W       0.75      0.66      0.70        99
#           X       0.76      0.74      0.75        99
#           Y       0.99      0.78      0.87        99
#           Z       0.68      0.76      0.71        99
#           a       0.90      0.83      0.86        99
#           b       0.94      0.96      0.95        99
#           c       0.67      0.71      0.69        99
#           d       0.90      0.88      0.89        99
#           e       0.83      0.82      0.82        99
#           f       0.88      0.89      0.88        99
#           g       0.86      0.77      0.81        99
#           h       0.81      0.88      0.84        99
#           i       0.88      0.93      0.91        99
#           j       0.97      0.87      0.91        99
#           k       0.96      0.88      0.92        99
#           l       0.54      0.86      0.66        99
#           m       0.99      0.94      0.96        99
#           n       0.90      0.92      0.91        99
#           o       0.62      0.68      0.65        99
#           p       0.86      0.90      0.88        99
#           q       0.94      0.92      0.93        99
#           r       0.88      0.85      0.86        99
#           s       0.78      0.69      0.73        99
#           t       0.96      0.89      0.92        99
#           u       0.74      0.82      0.78        99
#           v       0.62      0.66      0.64        99
#           w       0.74      0.75      0.74        99
#           x       0.74      0.69      0.71        99
#           y       0.87      0.80      0.83        99
#           z       0.74      0.62      0.67        99
#
# avg / total       0.83      0.82      0.82      5148





# SVM

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


