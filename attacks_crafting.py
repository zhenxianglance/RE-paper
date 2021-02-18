"""
This program crafts backdoor images.
Author: Zhen Xiang
Date: 2/26/2019
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
import matplotlib.pyplot as plt
from PIL import Image
import PIL
import random
import numpy as np

from src.utils import pattern_craft, add_backdoor

parser = argparse.ArgumentParser(description='PyTorch Backdoor Attack Crafting')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

random.seed()

# Attack parameters
SC = 1
TC = 7
NUM_OF_ATTACKS = 500
PATTERN_TYPE = 'static' # chess_board, cross, 4pixel
PERT_SIZE = 3/255

# Load raw data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

pert = pattern_craft(trainset.__getitem__(0)[0].size(), PATTERN_TYPE, PERT_SIZE)

# Crafting training backdoor images
ind_train = [i for i, label in enumerate(trainset.targets) if label==SC]
ind_train = np.random.choice(ind_train, NUM_OF_ATTACKS, False)
train_images_attacks = None
train_labels_attacks = None
for i in ind_train:
    if train_images_attacks is not None:
        train_images_attacks = torch.cat([train_images_attacks, add_backdoor(trainset.__getitem__(i)[0], pert).unsqueeze(0)], dim=0)
        train_labels_attacks = torch.cat([train_labels_attacks, torch.tensor([TC], dtype=torch.long)], dim=0)
    else:
        train_images_attacks = add_backdoor(trainset.__getitem__(i)[0], pert).unsqueeze(0)
        train_labels_attacks = torch.tensor([TC], dtype=torch.long)

# Crafting test backdoor images
ind_test = [i for i, label in enumerate(testset.targets) if label==SC]
test_images_attacks = None
test_labels_attacks = None
for i in ind_test:
    if test_images_attacks is not None:
        test_images_attacks = torch.cat([test_images_attacks, add_backdoor(testset.__getitem__(i)[0], pert).unsqueeze(0)], dim=0)
        test_labels_attacks = torch.cat([test_labels_attacks, torch.tensor([TC], dtype=torch.long)], dim=0)
    else:
        test_images_attacks = add_backdoor(testset.__getitem__(i)[0], pert).unsqueeze(0)
        test_labels_attacks = torch.tensor([TC], dtype=torch.long)

# Create attack dir and save attack images
if not os.path.isdir('attacks'):
    os.mkdir('attacks')
train_attacks = {'image': train_images_attacks, 'label': train_labels_attacks}
test_attacks = {'image': test_images_attacks, 'label': test_labels_attacks}
torch.save(train_attacks, './attacks/train_attacks')
torch.save(test_attacks, './attacks/test_attacks')
torch.save(ind_train, './attacks/ind_train')
torch.save(PATTERN_TYPE, './attacks/pattern_type')
torch.save(pert, './attacks/pert')
