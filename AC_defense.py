"""
Activation Clustering (AC) defense:
X0: clean features from TC
X1: attack features
Author: Zhen Xiang
Date: 6/17/2020
"""

from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

if not os.path.isdir('features'):
    print('Extract features first!')
    sys.exit(0)

# Parameters
NC = 10
dim = 2
thres = 0.4055

# Load in features for backdoor training images
X1 = np.load('./features/feature_attack.npy')
TC = np.load('./features/attack_TC.npy')

# AC detection
detection_flag = False
score_max = 0
t_est = NC
decomp_t = None
kmeans_t = None
for c in range(NC):
    if c != TC:
        X = np.load('./features/feature_{}.npy'.format(str(c)))
        # PCA
        X = X - np.mean(X, axis=0)
        decomp = PCA(n_components=dim,
                     whiten=True,
                     )
        decomp.fit(X)
        X = decomp.transform(X)
        kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
        score = silhouette_score(X, kmeans.labels_)
        if score > thres:
            print('Score of class {}: {}'.format(c, score))
            detection_flag = True
        if score > score_max:
            t_est = c
            score_max = score
            decomp_t = decomp
            kmeans_t = kmeans
    else:
        X0 = np.load('./features/feature_{}.npy'.format(str(c)))
        X = np.concatenate((X0, X1))
        X = X - np.mean(X, axis=0)
        decomp = PCA(n_components=dim,
                     whiten=True,
                     )
        decomp.fit(X)
        X = decomp.transform(X)
        kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
        score = silhouette_score(X, kmeans.labels_)
        if score > thres:
            print('Score of target class {}: {}'.format(c, score))
            detection_flag = True
        if score > score_max:
            t_est = c
            score_max = score
            decomp_t = decomp
            kmeans_t = kmeans

if not os.path.isdir('AC_results'):
    os.mkdir('AC_results')

if detection_flag is True:
    print('Attack detected!')
else:
    print('Attack not detected -- a failure!')
    sys.exit()

if t_est == TC:
    print('Target class correctly inferred!')
else:
    print('Target class incorrectly inferred -- a failure!')
    sys.exit()

# Training set cleansing
TP_count = 0
FP_count = 0
attack_total = X1.shape[0]
X0 = np.load('./features/feature_{}.npy'.format(str(TC)))
ind = np.load('./features/ind_{}.npy'.format(str(TC)))
clean_total = X0.shape[0]
X = np.concatenate((X0, X1))
X = decomp_t.transform(X)

# Remove the component with smaller mass
count_10 = np.sum(kmeans_t.labels_[:len(X0)])             # clean TC samples labeled to cluster 1
count_00 = len(kmeans_t.labels_[:len(X0)]) - count_10     # clean TC samples labeled to cluster 0
count_11 = np.sum(kmeans_t.labels_[len(X0):])             # backdoor samples labeled to cluster 1
count_01 = len(kmeans_t.labels_[len(X0):]) - count_11     # backdoor samples labeled to cluster 0
if count_10 + count_11 > count_00 + count_01:
    TP_count = count_01
    FP_count = count_00
    remove_ind_attack = [i for i, label in enumerate(kmeans_t.labels_[len(X0):]) if label == 0]
    ind_class = [i for i, label in enumerate(kmeans_t.labels_[:len(X0)]) if label == 0]
    remove_ind = ind[ind_class]
else:
    TP_count = count_11
    FP_count = count_10
    remove_ind_attack = [i for i, label in enumerate(kmeans_t.labels_[len(X0):]) if label == 1]
    ind_class = [i for i, label in enumerate(kmeans_t.labels_[:len(X0)]) if label == 1]
    remove_ind = ind[ind_class]

TPR = TP_count / attack_total
FPR = FP_count / clean_total

print('TPR: {}; FPR: {}'.format(TPR, FPR))

np.save('./AC_results/remove_ind', remove_ind)
np.save('./AC_results/remove_ind_attack', remove_ind_attack)
