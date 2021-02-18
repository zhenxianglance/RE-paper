"""
Author: Zhen Xiang
Date: 4/13/2020
"""

from __future__ import absolute_import
from __future__ import print_function

import torch
import matplotlib.pyplot as plt
import numpy as np

pert = torch.load('./attacks/pert')
pert = pert.numpy()
pert = np.transpose(pert, [1, 2, 0])
plt.axis('off')
plt.imshow(pert+0.5)
plt.show()
