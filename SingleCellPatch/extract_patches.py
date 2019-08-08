#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 15:43:41 2019

@author: zqwu
"""

import cv2
import numpy as np
import h5py
import os
from sklearn.cluster import DBSCAN
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
from copy import deepcopy
from scipy.signal import convolve2d
import pickle

size1 = 5
filter1 = np.zeros((size1, size1), dtype=int)
for i in range(size1):
  for j in range(size1):
    if np.sqrt((i-size1//2)**2 + (j-size1//2)**2) <= size1//2:
      filter1[i, j] = 1

size2 = 21
filter2 = np.zeros((size2, size2), dtype=int)
for i in range(size2):
  for j in range(size2):
    if np.sqrt((i-size2//2)**2 + (j-size2//2)**2) < size2//2:
      filter2[i, j] = 1

def select_window(mat, window, padding=0.):
  # Select the submatrix from mat according to window, negative boundaries allowed (padded with -1)
  if window[0][0] < 0:
    output_mat = np.concatenate([padding * np.ones_like(mat[window[0][0]:]), mat[:window[0][1]]], 0)
  elif window[0][1] > 2048:
    output_mat = np.concatenate([mat[window[0][0]:], padding * np.ones_like(mat[:(window[0][1] - 2048)])], 0)
  else:
    output_mat = mat[window[0][0]:window[0][1]]

  if window[1][0] < 0:
    output_mat = np.concatenate([padding * np.ones_like(output_mat[:, window[1][0]:]), output_mat[:, :window[1][1]]], 1)
  elif window[1][1] > 2048:
    output_mat = np.concatenate([output_mat[:, window[1][0]:], padding * np.ones_like(output_mat[:, :(window[1][1] - 2048)])], 1)
  else:
    output_mat = output_mat[:, window[1][0]:window[1][1]]
  return output_mat

def within_range(r, pos):
  if pos[0] >= r[0][1] or pos[0] < r[0][0]:
    return False
  if pos[1] >= r[1][1] or pos[1] < r[1][0]:
    return False
  return True

def remove_close_points(masking_points, target_points):
  dist = np.abs(np.array(masking_points).reshape((-1, 1, 2)) - np.array(target_points).reshape((1, -1, 2))).sum(2).min(1)
  return [p for i, p in masking_points if dist[i] > 5]

def generate_mask(positions, positions_labels, cell_id, window, window_segmentation):
  remove_mask = np.zeros((256, 256), dtype=int)
  target_mask = np.zeros((256, 256), dtype=int)

  for i, p in enumerate(positions):
    if not within_range(window, p):
      continue
    if positions_labels[i] != cell_id and positions_labels[i] >= 0:
      remove_mask[p[0] - window[0][0], p[1] - window[1][0]] = 1
    if positions_labels[i] == cell_id:
      target_mask[p[0] - window[0][0], p[1] - window[1][0]] = 1
  remove_mask[np.where(window_segmentation == 2)] = 1

  remove_mask = np.sign(convolve2d(remove_mask, filter1, mode='same'))
  target_mask = np.sign(convolve2d(target_mask, filter2, mode='same'))
  remove_mask = ((remove_mask - target_mask) > 0) * 1

  remove_mask[np.where(window_segmentation == -1)] = 1
  return remove_mask.reshape((256, 256, 1)), target_mask.reshape((256, 256, 1))

def instance_clustering(cell_segmentation, ct_thr=500, instance_map=True, map_path=None):
  cell_positions = []

  positions_mg = np.array(list(zip(*np.where(cell_segmentation == 1))))
  if len(positions_mg) < 1000:
    return [], np.zeros((0, 2), dtype=int), np.zeros((0,), dtype=int)

  clustering = DBSCAN(eps=10, min_samples=250).fit(positions_mg)
  positions_mg_labels = clustering.labels_

  cell_ids, point_cts = np.unique(positions_mg_labels, return_counts=True)
  for cell_id, ct in zip(cell_ids, point_cts):
    if cell_id < 0:
      continue
    if ct <= ct_thr:
      continue
    points = positions_mg[np.where(positions_mg_labels == cell_id)[0]]
    mean_pos = np.mean(points, 0).astype(int)
    window = [(mean_pos[0]-128, mean_pos[0]+128), (mean_pos[1]-128, mean_pos[1]+128)]
    outliers = [p for p in points if not within_range(window, p)]
    if len(outliers) > len(points) * 0.05:
      continue
    cell_positions.append((cell_id, mean_pos))

  if instance_map and map_path is not None:
    segmented = np.zeros_like(cell_segmentation) - 1
    for cell_id, mean_pos in cell_positions:
      points = positions_mg[np.where(positions_mg_labels == cell_id)[0]]
      for p in points:
        segmented[p[0], p[1]] = cell_id%10

    plt.clf()
    cmap = matplotlib.cm.get_cmap('tab10')
    cmap.set_under(color='k')
    plt.imshow(segmented, cmap=cmap, vmin=-0.001, vmax=10.001)

    font = {'color': 'white', 'size': 4}
    for cell_id, mean_pos in cell_positions:
      plt.text(mean_pos[1], mean_pos[0], str(cell_id), fontdict=font)
    plt.axis('off')
    plt.savefig(map_path, dpi=300)

  return cell_positions, positions_mg, positions_mg_labels


if __name__ == '__main__':
  sites = set(f[:9] for f in os.listdir('../Combined'))
  for site in sites:
    print("On site %s" % site)
    image_stack = h5py.File('../Combined/%s.h5' % site,'r+')
    image_stack = np.stack([image_stack[k] for k in sorted(image_stack.keys())], 0) #txyzc
    segmentation_stack = h5py.File('../Combined/%s_NNProbabilities.h5' % site,'r+')['exported_data']

    if not os.path.exists('../Data/StaticPatches/%s' % site):
      os.mkdir('../Data/StaticPatches/%s' % site)

    cell_positions = {}
    cell_pixel_assignments = {}
    for t_point in range(image_stack.shape[0]):
      print("\tClustering time %d" % t_point)
      cell_segmentation = np.argmax(segmentation_stack[t_point, :, :, 0], 2)
      instance_map_path = '../Data/StaticPatches/%s/segmentation_%d.png' % (site, t_point)
      res = instance_clustering(cell_segmentation, instance_map=True, map_path=instance_map_path)

      cell_positions[t_point] = res[0]
      cell_pixel_assignments[t_point] = res[1:]


    with open('../Data/StaticPatches/%s/cell_positions.pkl' % site, 'wb') as f:
      pickle.dump(cell_positions, f)
    with open('../Data/StaticPatches/%s/cell_pixel_assignments.pkl' % site, 'wb') as f:
      pickle.dump(cell_pixel_assignments, f)


    ### Generate time-independent static patches ###
    for t_point in range(image_stack.shape[0]):
      print("\tWriting time %d" % t_point)
      raw_image = image_stack[t_point, :, :, 0]
      cell_segmentation = np.argmax(segmentation_stack[t_point, :, :, 0], 2)
      positions_mg, positions_mg_labels = cell_pixel_assignments[t_point]
      

      valid_cells = cell_positions[t_point]

      background_pool = raw_image[np.where(segmentation_stack[t_point, :, :, 0, 0] > 0.9)]

      for cell_id, cell_position in valid_cells:
        background_filling = np.random.choice(np.arange(background_pool.shape[0]), size=(256, 256))
        background_filling = np.take(background_pool, background_filling, 0)

        window = [(cell_position[0]-128, cell_position[0]+128),
                  (cell_position[1]-128, cell_position[1]+128)]
        window_segmentation = select_window(cell_segmentation, window, padding=-1)
        remove_mask, target_mask = generate_mask(positions_mg, positions_mg_labels, cell_id, window, window_segmentation)

        output_mat = select_window(raw_image, window, padding=0)
        masked_output_mat = output_mat * (1 - remove_mask) + background_filling * remove_mask

        cv2.imwrite('../Data/StaticPatches/%s/%d_%d_retardance.png' % (site, t_point, cell_id), output_mat[:, :, -2])
        cv2.imwrite('../Data/StaticPatches/%s/%d_%d_retardance_masked.png' % (site, t_point, cell_id), masked_output_mat[:, :, -2])
        with h5py.File('../Data/StaticPatches/%s/%d_%d.h5' % (site, t_point, cell_id), 'w') as f:
          f.create_dataset("mat", data=output_mat)
          f.create_dataset("masked_mat", data=masked_output_mat)
