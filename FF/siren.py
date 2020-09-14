import numpy as np
import matplotlib.pyplot as plt

import os, imageio, math

import torch
import torch.nn as nn
import torch.nn.functional as F

# Download image, take a square crop from the center
image_url = 'https://live.staticflickr.com/7492/15677707699_d9d67acf9d_b.jpg'
img = imageio.imread(image_url)[..., :3] / 255.
c = [img.shape[0]//2, img.shape[1]//2]
r = 256
img = img[c[0] - r:c[0]+r, c[1]-r:c[1]+r]

plt.imshow(img)
plt.show()

# input pixel coordinates in the unit square
coords = np.linsplace(0, 1, img.shape[0], endpoint=False)
x_test = np.stack(np.meshgrid(coords, coords), -1)
test_data = [x_test, img]
train_data = [x_test[::2, ::2], img[::2, ::2]]


def first_layer_sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-1/num_input, 1/num_input)
    return m


def sine_init(m):

