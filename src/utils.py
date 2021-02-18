import os
import sys
import time
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np
import matplotlib.pyplot as plt
import random

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def pattern_craft(im_size, pattern_type, pert_size):
    if pattern_type == 'chess_board':
        pert = torch.zeros(im_size)
        for i in range(im_size[1]):
            for j in range(im_size[2]):
                if (i+j)%2 == 0:
                    pert[:, i, j] = torch.ones(im_size[0])
        pert *= pert_size
    elif pattern_type == 'static':
        pert = torch.zeros(im_size)
        for i in range(im_size[1]):
            for j in range(im_size[2]):
                if (i%2==0) and (j%2==0):
                    pert[:, i, j] = torch.ones(im_size[0])
        pert *= pert_size

    return pert


def add_backdoor(image, pert):
    image += pert
    image *= 255
    image = image.round()
    image /= 255
    image.clamp(0, 1)
    
    return image
