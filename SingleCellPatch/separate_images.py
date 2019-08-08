#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 15:37:12 2019

@author: zqwu
"""

import tifffile
import cv2
import os
import numpy as np
import h5py

sites = set('_'.join(n.split('_')[:2]) for n in os.listdir('./Raw'))

for site in sites:
  if not os.path.exists('./Separated/%s' % site):
    os.mkdir('./Separated/%s' % site)
  print("Processing site %s" % site)  
  if not os.path.exists('./Separated/%s/bf_0_0.png' % site):
    mats = tifffile.imread('./Raw/%s_Brightfield_computed.tif' % site)
    image_range = (np.percentile(mats[:5], 0.5), np.percentile(mats[:5], 99.5))
    mats_adjusted = (mats - image_range[0])/(image_range[1] - image_range[0])
    mats_adjusted = np.clip(mats_adjusted, 0., 1.)
    for i in range(mats.shape[0]):
      for j in range(mats.shape[1]):
        mat = mats_adjusted[i, j]
        cv2.imwrite('./Separated/%s/bf_%d_%d.png' % (site, i, j), mat*255)
    del mats
    del mats_adjusted
  
  if not os.path.exists('./Separated/%s/orient_0_0.png' % site):
    mats = tifffile.imread('./Raw/%s_Orientation.tif' % site)
    image_range = (mats[:5].min(), mats[:5].max())
    mats_adjusted = (mats - image_range[0])/(image_range[1] - image_range[0])
    mats_adjusted = np.clip(mats_adjusted, 0., 1.)
    for i in range(mats.shape[0]):
      for j in range(mats.shape[1]):
        mat = mats_adjusted[i, j]
        cv2.imwrite('./Separated/%s/orient_%d_%d.png' % (site, i, j), mat*255)
    del mats
    del mats_adjusted
  
  if not os.path.exists('./Separated/%s/polarization_0_0.png' % site):
    mats = tifffile.imread('./Raw/%s_Polarization.tif' % site)
    image_range = (np.percentile(mats[:5], 0.5), np.percentile(mats[:5], 99.5))
    mats_adjusted = (mats - image_range[0])/(image_range[1] - image_range[0])
    mats_adjusted = np.clip(mats_adjusted, 0., 1.)
    for i in range(mats.shape[0]):
      for j in range(mats.shape[1]):
        mat = mats_adjusted[i, j]
        cv2.imwrite('./Separated/%s/polarization_%d_%d.png' % (site, i, j), mat*255)
    del mats
    del mats_adjusted
  
  if not os.path.exists('./Separated/%s/retardance_0_0.png' % site):
    mats = tifffile.imread('./Raw/%s_Retardance.tif' % site)
    image_range = (np.percentile(mats[:5], 0.5), np.percentile(mats[:5], 99.5))
    mats_adjusted = (mats - image_range[0])/(image_range[1] - image_range[0])
    mats_adjusted = np.clip(mats_adjusted, 0., 1.)
    for i in range(mats.shape[0]):
      for j in range(mats.shape[1]):
        mat = mats_adjusted[i, j]
        cv2.imwrite('./Separated/%s/retardance_%d_%d.png' % (site, i, j), mat*255)
    del mats
    del mats_adjusted

  if not os.path.exists('./Combined/%s.h5' % site):
    all_files = os.listdir('./Separated/%s' % site)
    ids = sorted(list(map(int, set(name.split('_')[1] for name in all_files))))
    
    time_series = []
    for i in ids:
      files = sorted([name for name in all_files if name.split('_')[1] == str(i)])
      mats = [cv2.imread('./Separated/%s/%s' % (site, name))[:, :, 0] for name in files]
      mats = np.stack(mats, 2)
      mats = np.expand_dims(mats, 2) # 2048 * 2048 * 1 * 20
      time_series.append(mats)
  
    with h5py.File('./Combined/%s.h5' % site, 'w') as f:
      for i, mats in enumerate(time_series):
        f.create_dataset('time_%02d' % i, data=mats, dtype=np.uint8)


