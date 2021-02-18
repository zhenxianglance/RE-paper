"""
Spectral Signature (SS) defense:
X0: clean features from TC
X1: attack features
Author: Zhen Xiang
Date: 6/15/2020
"""

from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

if not os.path.isdir('features'):
    print('Extract features first!')
    sys.exit(0)

# Parameter
eps = 750/5500  # 500 backdoor training images, hence removing 750 images

# Load in features for backdoor training images
X1 = np.load('./features/feature_attack.npy')
TC = np.load('./features/attack_TC.npy')

# Load in features for the target class clean training images
X0 = np.load('./features/feature_{}.npy'.format(str(TC)))
ind = np.load('./features/ind_{}.npy'.format(str(TC)))

X = np.concatenate((X0, X1))

# PCA
X = X - np.mean(X, axis=0)
decomp = PCA(n_components=2,
             whiten=True,
             )
decomp.fit(X)
X0 = decomp.transform(X0)
X1 = decomp.transform(X1)
X = decomp.transform(X)

# Visualize the internal layer activations projected onto the first two principal component
plt.scatter(X0[:, 0], X0[:, 1], alpha=0.5, marker='o', label='clean')
plt.scatter(X1[:, 0], X1[:, 1], alpha=0.5, marker='s', label='backdoor')
plt.axis('off')
plt.legend()
plt.savefig('2D_separation.png')

# SS defense
X = np.abs(X[:, 0] - np.mean(X[:, 0]))
rank = np.argsort(X)
ind_class = rank[round((1-eps)*len(X)):]
remove_ind_attack = ind_class[ind_class >= len(X0)] - len(X0)
ind_class = ind_class[ind_class < len(X0)]
remove_ind = ind[ind_class]

TPR = len(remove_ind_attack) / len(X1)
FPR = len(remove_ind) / len(X0)

print('TPR: {}; FPR: {}'.format(TPR, FPR))

if not os.path.isdir('SS_results'):
    os.mkdir('SS_results')
np.save('./SS_results/remove_ind', remove_ind)
np.save('./SS_results/remove_ind_attack', remove_ind_attack)
