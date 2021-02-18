## Supplementary code for "Reverse Engineering Imperceptible Backdoor Attacks on Deep Neural Networks for Detection and Training Set Cleansing"

# Introduction

The current version of this repository contains only the re-implementation of "Spectral Signature" and "Activation Clustering".
Please contact the authors (zhen.xiang.lance@gmail.com) for the code of our method.

### References

Spectral Signature: 
B. Tran, J. Li, and A. Madry, “Spectral signatures in
backdoor attacks,” in Proc. NIPS, 2018.

Activation Clustering: B. Chen, W. Carvalho, N. Baracaldo, H. Ludwig, B. Edwards, T. Lee, I. Molloy, and B. Srivastava, “Detecting
Backdoor Attacks on Deep Neural Networks by Activation Clustering,” http://arxiv.org/abs/1811.03728, Nov 2018.
### Preparation

Pytorch 1.6.0

CUDA V10.1.243

### Usage

Create an attack (1SC attack with pattern B in our paper) by

    python attack_crafting.py

The backdoor pattern can be visualized by

    python true_pert_plot.py

Train a DNN on the possibly poisoned training set by
    
    python train_models_contam.py

SS and AC both requires extracting internal layer features, which can be done by

    python get_features.py

SS defense:

    python SS_defense.py

AC defense:

    python AC_defense.py
