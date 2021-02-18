"""
This code extract features for all samples from the possibly poisoned training set.
Author: Zhen Xiang
Date: 1/27/2020
"""

from __future__ import absolute_import
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import numpy as np

from src.resnet import ResNet18

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Hook():
    '''
    For Now we assume the input[0] to last linear layer is a 1*d tensor
    the layerOutput is a list of those tensor value in numpy array
    '''
    def __init__(self, module):
            self.hook = module.register_forward_hook(self.hook_fn)
            self.layerOutput = None

    def hook_fn(self, module, input, output):
        feature = input[0].cpu().numpy()
        if self.layerOutput is None:
            self.layerOutput = feature
        else:
            self.layerOutput = np.append(hooker.layerOutput, feature, axis=0)
        pass

    def close(self):
        self.hook.remove()


def getLayerOutput(ds, model, hook, ic=None, batch_size=128):
    ''' Get the layer outputs
    Args:
        ds (torch.tensor): dataset of data
        model (torch.module):
        hook (Hook): self-defined hook class
        ind_correct (np.array): record the indices of samples correctly classified
        outs (None/np.array): record  nn models' ouput (num_samples, class_nums)
            if none, no recording
    Returns: None
    '''
    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=1)
    model.eval()
    correct = 0
    tot = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dl): 
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)

            correct += predicted.eq(targets).sum().item()
            tot += targets.size(0)

            if ic is not None:
                ic = torch.cat((ic, predicted.eq(targets).nonzero().squeeze()))

    hook.close()
    print('acc: {}/{} = {:.2f}'.format(correct, tot, correct/tot))
    return ic


transform_train = transforms.Compose([
    transforms.ToTensor()
])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    
# Load in backdoor training images
if not os.path.isdir('attacks'):
    print('Attack images not found, please craft attack images first!')
    sys.exit(0)
train_attacks = torch.load('./attacks/train_attacks')
train_images_attacks = train_attacks['image']
train_labels_attacks = train_attacks['label']
attackset = torch.utils.data.TensorDataset(train_images_attacks, train_labels_attacks)

# Delete training images used for backdoor training)
ind_train = torch.load('./attacks/ind_train')
trainset.data = np.delete(trainset.data, ind_train, axis=0)
trainset.targets = np.delete(trainset.targets, ind_train, axis=0)

model = ResNet18()
model.load_state_dict(torch.load('./contam/model_contam.pth'))
model = model.to(device)

targetlayer = model._modules['linear']
layer_name = 'linear'
if not os.path.isdir('features'):
    os.mkdir('features')
np.save('./features/layer_name', layer_name)

# Extract features for clean training images
hooker = Hook(targetlayer)
ind_correct = torch.tensor([], dtype=torch.long).to(device)
ind_correct = getLayerOutput(trainset, model, hooker, ind_correct, batch_size=32)
feature_clean = hooker.layerOutput
# Categorize features based on labels
NC = np.max(trainset.targets)+1
for c in range(NC):
    ind = [i for i, label in enumerate(trainset.targets) if label == c]
    np.save('./features/feature_{}'.format(str(c)), feature_clean[ind, :])
    np.save('./features/ind_{}'.format(str(c)), ind)

# Extract features for backdoor training images
hooker = Hook(targetlayer)
getLayerOutput(attackset, model, hooker, None, batch_size=32)
feature_attack = hooker.layerOutput
np.save('./features/feature_attack', feature_attack)
np.save('./features/attack_TC', train_labels_attacks[0])
