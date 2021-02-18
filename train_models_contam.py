"""
This program trains backdoored ResNet for CIFAR-10.
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
import numpy as np

from src.resnet import ResNet18

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training (with backdoor)')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.ToTensor()
])

transform_test = transforms.Compose([
    transforms.ToTensor()
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

# Load in attack data
if not os.path.isdir('attacks'):
    print ('Attack images not found, please craft attack images first!')
    sys.exit(0)
train_attacks = torch.load('./attacks/train_attacks')
train_images_attacks = train_attacks['image']
train_labels_attacks = train_attacks['label']
test_attacks = torch.load('./attacks/test_attacks')
test_images_attacks = test_attacks['image']
test_labels_attacks = test_attacks['label']
testset_attacks = torch.utils.data.TensorDataset(test_images_attacks, test_labels_attacks)

# Poison the training set
image_dtype = trainset.data.dtype
train_images_attacks = np.rint(np.transpose(train_images_attacks.numpy()*255, [0, 2, 3, 1])).astype(image_dtype)
trainset.data = np.concatenate((trainset.data, train_images_attacks))
trainset.targets = np.concatenate((trainset.targets, train_labels_attacks))
ind_train = torch.load('./attacks/ind_train')
trainset.data = np.delete(trainset.data, ind_train, axis=0)
trainset.targets = np.delete(trainset.targets, ind_train, axis=0)

# Load in the datasets
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
attackloader = torch.utils.data.DataLoader(testset_attacks, batch_size=100, shuffle=False, num_workers=2)
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
net = ResNet18()
net = net.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)


def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    acc = 100. * correct / total
    print('Train ACC: %.3f' % acc)

    return net


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    acc = 100. * correct / total
    print('Test ACC: %.3f' % acc)


def test_attack(epoch):
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(attackloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    acc = 100. * correct / total
    print('Attack success rate: %.3f' % acc)


for epoch in range(start_epoch, start_epoch+100):
    model_contam = train(epoch)
    test(epoch)
    test_attack(epoch)

    # Save model
    if not os.path.isdir('contam'):
        os.mkdir('contam')
    torch.save(model_contam.state_dict(), './contam/model_contam.pth')
