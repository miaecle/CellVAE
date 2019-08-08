#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 15:33:52 2019

@author: michaelwu
"""

import cv2
import h5py
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

dat = h5py.File('../Combined/D4-Site_0.h5', 'r')
f1 = dat['time_00']
f2 = dat['time_01']

frame1 = cv2.resize(f1[:, :, 0, 4:5], (512, 512))
frame2 = cv2.resize(f2[:, :, 0, 4:5], (512, 512))
#frame2 = np.concatenate([frame2[5:], frame2[:5]], 0)

#cv2.imwrite("transition_f1.png", frame1)
#cv2.imwrite("transition_f2.png", frame2)
flow = cv2.calcOpticalFlowFarneback(frame1, #prev
                                    frame2, #next
                                    None, #flow initialization
                                    0.5, #pyr_scale
                                    3, #levels
                                    4, #winsize
                                    3, #iterations
                                    5, #poly_n
                                    1.2, #poly_sigma
                                    0) #flags

mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
hsv = np.zeros((512, 512, 3), dtype='uint8')
hsv[...,1] = 255
hsv[...,0] = ang*180/np.pi/2
hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
cv2.imwrite('transition.png', bgr)  

#annotations = h5py.File('../Combined/B5-Site_7_Annotations.h5', 'r')['exported_data'][0, :, :, 0, 0]
#annotated_pixels = np.array(list(zip(*np.where(annotations == 2))))
#clustering = DBSCAN(eps=10, min_samples=240).fit(annotated_pixels)
#annotated_labels = clustering.labels_
#annotated_centers = []
#for i in np.unique(np.unique(annotated_labels)):
#  if i >= 0:
#    annotated_centers.append(np.mean(annotated_pixels[np.where(annotated_labels == i)], 0))
#p1 = np.stack(annotated_centers, 0).reshape((-1, 1, 2)).astype('float32')/4

p1 = pickle.load(open('../Data/StaticPatches/D4-Site_0/cell_positions.pkl', 'rb'))
p1 = np.stack([pair[1] for pair in p1[0]], 0).reshape((-1, 1, 2)).astype('float32')/4

out_mat = np.stack([frame1] * 3, 2)
for p in p1:
  x_start = max(0, int(p[0, 0] - 2))
  x_end = min(512, int(p[0, 0] + 2))
  y_start = max(0, int(p[0, 1] - 2))
  y_end = min(512, int(p[0, 1] + 2))
  out_mat[x_start:x_end, y_start:y_end] = np.array([255, 0, 0]).reshape((1, 1, 3))
cv2.imwrite('transition1.png', out_mat)


p2 = [(p[0][0] + flow[int(p[0][0]), int(p[0][1])][1], p[0][1] + flow[int(p[0][0]), int(p[0][1])][0]) for p in p1]
p2 = np.reshape(np.array(p2), (-1, 1, 2))
out_mat = np.stack([frame2] * 3, 2)
for p in p2:
  if p[0, 0] > 0 and p[0, 0] < 511 and p[0, 1] > 0 and p[0, 1] < 511:
    x_start = max(0, int(p[0, 0] - 2))
    x_end = min(512, int(p[0, 0] + 2))
    y_start = max(0, int(p[0, 1] - 2))
    y_end = min(512, int(p[0, 1] + 2))
    out_mat[x_start:x_end, y_start:y_end] = np.array([255, 0, 0]).reshape((1, 1, 3))
cv2.imwrite('transition2.png', out_mat)