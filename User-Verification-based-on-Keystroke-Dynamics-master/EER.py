# -*- coding: utf-8 -*-

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np


def evaluateEER(user_scores, imposter_scores):
    labels = [0]*len(user_scores) + [1]*len(imposter_scores)
    #print(len(user_scores))
    fpr, tpr, thresholds = roc_curve(labels, user_scores + imposter_scores)
    auc_ = auc(fpr, tpr)

    if len(user_scores) == 10200:
        x=[0,1]
        y=[1,0]
        plt.figure(figsize=(5, 5), dpi=100)
        plt.plot(fpr, tpr, linestyle='-', label='manhatten_Scaled_Filtered (auc = %0.3f)' % auc_)
        plt.plot(y,x)
        plt.xlabel('False Positive Rate -->')
        plt.ylabel('True Positive Rate -->')

        plt.legend()

        plt.show()


    missrates = 1 - tpr
    farates = fpr

    dists = missrates - farates
    idx1 = np.argmin(dists[dists >= 0])
    idx2 = np.argmax(dists[dists < 0])
    x = [missrates[idx1], farates[idx1]]
    y = [missrates[idx2], farates[idx2]]
    #print(np.shape(x))
    a = ( x[0] - x[1] ) / ( y[1] - x[1] - y[0] + x[0] )
    eer = x[0] + a * ( y[0] - x[0] )
    return eer
        