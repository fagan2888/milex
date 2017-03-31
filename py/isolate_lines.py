#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imread
from scipy.ndimage import gaussian_filter as gf
from scipy.ndimage import median_filter as mf

# -- read the file and convert to grayscale
fname = os.path.join("..", "data", "2017_01_17-10_47_40.png")
img   = imread(fname)
imgL  = img.mean(-1)

# -- create a patch on which to test
rr   = (0, 117)
cc   = (365, 505)
pat  = img[rr[0]:rr[1], cc[0]:cc[1]]
patL = imgL[rr[0]:rr[1], cc[0]:cc[1]]

# -- high pass filter
hpfL = patL - gf(patL, 1)

# -- ...and again
hpfLf = hpfL - gf(hpfL, 1)

# -- median filter to clean up
hpfLfm = mf(hpfLf, 1)

# -- plot (and watch out for edge effects on the top row...)
fig, ax = plt.subplots(1, 2, figsize=(8, 3.5))
fig.subplots_adjust(0.02, 0.05, 0.98, 0.95, 0.05)
ax[0].axis("off")
ax[1].axis("off")
im0 = ax[0].imshow(pat[2:]) 
im1 = ax[1].imshow(hpfLfm[2:], cmap="viridis", clim=[0.5, 5]) 
fig.canvas.draw()
fig.savefig(os.path.join("..", "output", "filtered_example.png"), clobber=True)
