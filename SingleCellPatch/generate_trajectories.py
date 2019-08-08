# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 14:22:51 2019

@author: michael.wu
"""
import numpy as np
import scipy
import os
import ot
import tifffile
import pickle
import h5py
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from extract_patches import within_range, generate_mask, select_window

def frame_matching(f1, f2, dist_cutoff=50):
  f1 = np.array(f1).reshape((-1, 2))
  f2 = np.array(f2).reshape((-1, 2))
  cost_mat = np.ones((len(f1)+len(f2), len(f1)+len(f2))) * (dist_cutoff ** 2 * 10)
  dist_mat = cdist(f1, f2) ** 2
  dist_mat[np.where(dist_mat>=(dist_cutoff**2))] = (dist_cutoff ** 2 * 10)
  cost_mat[:len(f1), :len(f2)] = dist_mat
  for i in range(len(f1)):
    cost_mat[i, i+len(f2)] = 1.05 * (dist_cutoff ** 2)
  for j in range(len(f2)):
    cost_mat[len(f1)+j, j] = 1.05 * (dist_cutoff ** 2)
  cost_mat[len(f1):, len(f2):] = np.transpose(dist_mat)
  links = linear_sum_assignment(cost_mat)
  pairs = []
  for pair in zip(*links):
    if pair[0] < len(f1) and pair[1] < len(f2):
      pairs.append(pair)
  return pairs

def frame_matching_ot(f1, f2, int1, int2, dist_cutoff=50):

  dist_mat = cdist(f1, f2) ** 2
  dist_mat[np.where(dist_mat>=(dist_cutoff**2))] = (dist_cutoff ** 2 * 10)

  int1 = np.array(int1)/np.sum(int1)
  int2 = np.array(int2)/np.sum(int2)
  ot_mat = ot.sinkhorn(int1, int2, dist_mat, 10.)

  return ot_mat

def trajectory_connection(trajectories,
                          trajectories_positions,
                          intensities_dict,
                          dist_cutoff=100,
                          only_gap=True):
  starts = [min(t.keys()) for t in trajectories_positions]
  ends = [max(t.keys()) for t in trajectories_positions]

  cost_mat_d1 = len(trajectories_positions)
  if not only_gap:
    cost_mat_d2 = sum(len(t) for t in trajectories_positions)
  cost_mat_d3 = len(trajectories_positions)

  # Upper left: gap connection
  cost_mat_upper_left = np.ones((cost_mat_d1, cost_mat_d1)) * (dist_cutoff ** 2 * 10)
  positions_x = [trajectories_positions[i][end] for i, end in enumerate(ends)]
  positions_y = [trajectories_positions[j][start] for j, start in enumerate(starts)]
  dist_mat = cdist(positions_x, positions_y) ** 2

  mask_mat = ((np.array(starts).reshape((1, -1)) - np.array(ends).reshape((-1, 1))) == 2)*1 + \
             ((np.array(starts).reshape((1, -1)) - np.array(ends).reshape((-1, 1))) == 3)*4 # Allow gap up to 3 time intervals
  mask_mat[np.where(dist_mat>=(dist_cutoff**2))] = 0
  cost_mat_upper_left = mask_mat * dist_mat + (1 - np.sign(mask_mat)) * cost_mat_upper_left

  if not only_gap:
    # Upper middle: merge
    cost_mat_upper_middle = np.ones((cost_mat_d1, cost_mat_d2)) * (dist_cutoff ** 2 * 10)
    positions_x = [trajectories_positions[i][end] for i, end in enumerate(ends)]
    positions_y = [[t[t_point] for t_point in   sorted(t.keys())] for t in trajectories_positions]
    positions_y = np.concatenate(positions_y)
    dist_mat = cdist(positions_x, positions_y) ** 2

    strength_x = np.array([intensities_dict[end][trajectories[i][end]] for i, end in enumerate(ends)]).reshape((-1, 1)).astype(float)
    strength_y = [[intensities_dict[t_point][t[t_point]] for t_point in sorted(t.keys())] for t in trajectories]
    strength_y = np.concatenate(strength_y).reshape((1, -1)).astype(float)
    strength_y_tminus1 = [[np.max(strength_y)*2] + [intensities_dict[t_point][t[t_point]] for t_point in sorted(t.keys())[:-1]] for t in trajectories]
    strength_y_tminus1 = np.concatenate(strength_y_tminus1).reshape((1, -1)).astype(float)
    ratio_mat = np.clip(strength_y/(strength_x + strength_y_tminus1), 0.3, 3.)
    ratio_mat = (ratio_mat > 1.) * ratio_mat + (ratio_mat <= 1.) * (ratio_mat ** -2)

    join_points = np.concatenate([sorted(t.keys()) for t in trajectories])
    mask_mat = ((np.array(join_points).reshape((1, -1)) - np.array(ends).reshape((-1, 1))) == 1)*1
    mask_mat[np.where((dist_mat * ratio_mat)>=(dist_cutoff**2))] = 0
    cost_mat_upper_middle = mask_mat * dist_mat * ratio_mat + (1 - np.sign(mask_mat)) * cost_mat_upper_middle

    # Middle left: split
    cost_mat_middle_left = np.ones((cost_mat_d2, cost_mat_d1)) * (dist_cutoff ** 2 * 10)
    positions_x = [[t[t_point] for t_point in sorted(t.keys())] for t in trajectories_positions]
    positions_x = np.concatenate(positions_x)
    positions_y = [trajectories_positions[j][start] for j, start in enumerate(starts)]
    dist_mat = cdist(positions_x, positions_y) ** 2

    strength_x = [[intensities_dict[t_point][t[t_point]] for t_point in sorted(t.keys())] for t in trajectories]
    strength_x = np.concatenate(strength_x).reshape((-1, 1)).astype(float)
    strength_x_tplus1 = [[intensities_dict[t_point][t[t_point]] for t_point in sorted(t.keys())[1:]] + [0.] for t in trajectories]
    strength_x_tplus1 = np.concatenate(strength_x_tplus1).reshape((-1, 1)).astype(float)
    strength_y = np.array([intensities_dict[start][trajectories[j][start]] for j, start in enumerate(starts)]).reshape((1, -1)).astype(float)
    ratio_mat = np.clip(strength_x/(strength_x_tplus1 + strength_y), 0.3, 3.)
    ratio_mat = (ratio_mat > 1.) * ratio_mat + (ratio_mat <= 1.) * (ratio_mat ** -2)

    join_points = np.concatenate([sorted(t.keys()) for t in trajectories])
    mask_mat = ((np.array(starts).reshape((1, -1)) - np.array(join_points).reshape((-1, 1))) == 1)*1
    mask_mat[np.where((dist_mat * ratio_mat)>=(dist_cutoff**2))] = 0
    cost_mat_middle_left = mask_mat * dist_mat * ratio_mat + (1 - np.sign(mask_mat)) * cost_mat_middle_left

    # Middle right: split alternative
    cost_mat_middle_right = np.ones((cost_mat_d2, cost_mat_d3)) * (dist_cutoff ** 2 * 10)
    begin_ind = 0
    for i, t in enumerate(trajectories_positions):
      aver_movement = np.array([t[t_point] for t_point in sorted(t.keys())])
      aver_movement = ((aver_movement[1:] - aver_movement[:-1])**2).sum(1).mean()

      t_inds = trajectories[i]
      strength_x = [intensities_dict[t_point][t_inds[t_point]] for t_point in sorted(t_inds.keys())]
      strength_x_tplus1 = strength_x[1:] + [strength_x[-1]]
      ratio = np.array(strength_x).astype(float)/np.array(strength_x_tplus1).astype(float)
      ratio = (ratio > 1) * ratio + (ratio <= 1) * (ratio ** -2)

      filling = np.clip(aver_movement * ratio, 0, (dist_cutoff ** 2 * 10))
      cost_mat_middle_right[begin_ind:(begin_ind + len(t)), i] = filling
      begin_ind += len(t)

    # Lower middle: merge alternative
    cost_mat_lower_middle = np.ones((cost_mat_d3, cost_mat_d2)) * (dist_cutoff ** 2 * 10)
    begin_ind = 0
    for i, t in enumerate(trajectories_positions):
      aver_movement = np.array([t[t_point] for t_point in sorted(t.keys())])
      aver_movement = ((aver_movement[1:] - aver_movement[:-1])**2).sum(1).mean()

      t_inds = trajectories[i]
      strength_x = [intensities_dict[t_point][t_inds[t_point]] for t_point in sorted(t_inds.keys())]
      strength_x_tminus1 = [strength_x[0]] + strength_x[:-1]

      ratio = np.array(strength_x).astype(float)/np.array(strength_x_tminus1).astype(float)
      ratio = (ratio > 1) * ratio + (ratio <= 1) * (ratio ** -2)

      filling = np.clip(aver_movement * ratio, 0, (dist_cutoff ** 2 * 10))
      cost_mat_lower_middle[i, begin_ind:(begin_ind + len(t))] = filling
      begin_ind += len(t)

    cost_mat_middle_middle = np.ones((cost_mat_d2, cost_mat_d2)) * (dist_cutoff ** 2 * 10)

  # Upper right: normal termination
  cost_mat_upper_right = np.ones((cost_mat_d1, cost_mat_d3)) * (dist_cutoff ** 2 * 10)
  if only_gap:
    diagonal_assignment = cost_mat_upper_left[np.where(cost_mat_upper_left < np.max(cost_mat_upper_left))]
  else:
    diagonal_assignment = np.concatenate([cost_mat_upper_left[np.where(cost_mat_upper_left < np.max(cost_mat_upper_left))],
                                          cost_mat_upper_middle[np.where(cost_mat_upper_middle < np.max(cost_mat_upper_middle))]])
  diagonal_assignment = np.percentile(diagonal_assignment, 90)
  np.fill_diagonal(cost_mat_upper_right, diagonal_assignment)

  # Lower left: normal initialization
  cost_mat_lower_left = np.ones((cost_mat_d3, cost_mat_d1)) * (dist_cutoff ** 2 * 10)
  if only_gap:
    diagonal_assignment = cost_mat_upper_left[np.where(cost_mat_upper_left < np.max(cost_mat_upper_left))]
  else:
    diagonal_assignment = np.concatenate([cost_mat_upper_left[np.where(cost_mat_upper_left < np.max(cost_mat_upper_left))],
                                          cost_mat_middle_left[np.where(cost_mat_middle_left < np.max(cost_mat_middle_left))]])
  diagonal_assignment = np.percentile(diagonal_assignment, 90)
  np.fill_diagonal(cost_mat_lower_left, diagonal_assignment)

  # Lower right: transpose of upper left
  cost_mat_lower_right = np.transpose(cost_mat_upper_left)


  if only_gap:
    cost_mat = np.concatenate(
      [np.concatenate([cost_mat_upper_left, cost_mat_upper_right], 1),
       np.concatenate([cost_mat_lower_left, cost_mat_lower_right], 1)], 0)
    links = linear_sum_assignment(cost_mat)

    connection_maps = {}
    for pair in zip(*links):
      if pair[0] < cost_mat_d1 and pair[1] < cost_mat_d1:
        assert pair[1] > pair[0]
        connection_maps[pair[0]] = pair[1]

    connected = []
    involved = set()
    for i in range(len(trajectories)):
      if i in involved:
        continue
      else:
        con = [i]
        involved.add(i)
        start = i
        while i in connection_maps:
          con.append(connection_maps[i])
          involved.add(connection_maps[i])
          i = connection_maps[i]
        connected.append(con)
    new_trajectories = []
    for con in connected:
      t = trajectories[con[0]]
      for c in con[1:]:
        t.update(trajectories[c])
      new_trajectories.append(t)
    return new_trajectories
  else:
    cost_mat = np.concatenate(
        [np.concatenate([cost_mat_upper_left, cost_mat_upper_middle, cost_mat_upper_right], 1),
         np.concatenate([cost_mat_middle_left, cost_mat_middle_middle, cost_mat_middle_right], 1),
         np.concatenate([cost_mat_lower_left, cost_mat_lower_middle, cost_mat_lower_right], 1)], 0)
    # TODO: finish off the merge/split process

if __name__ == '__main__':
  sites = set(f[:9] for f in os.listdir('../Combined'))
  for site in sites:
    print("On site %s" % site)
    cell_positions = pickle.load(open('../Data/StaticPatches/%s/cell_positions.pkl' % site, 'rb'))
    cell_pixel_assignments = pickle.load(open('../Data/StaticPatches/%s/cell_pixel_assignments.pkl' % site, 'rb'))
    t_points = sorted(cell_positions.keys())
    assert np.allclose(np.array(t_points)[1:] - 1, np.array(t_points)[:-1])

    # Mapping to centroid positions
    cell_positions_dict = {k: dict(cell_positions[k]) for k in cell_positions}

    # Mapping to size of segmentation
    intensities_dict = {}
    for t_point in t_points:
      intensities_d = dict(zip(*np.unique(cell_pixel_assignments[t_point][1], return_counts=True)))
      intensities_d = {p[0]: intensities_d[p[0]] for p in cell_positions[t_point]}
      intensities_dict[t_point] = intensities_d

    matchings = {}
    for t_point in t_points[:-1]:
      f1 = [p[1] for p in cell_positions[t_point]]
      ids1 = [p[0] for p in cell_positions[t_point]]
      int1 = [intensities_dict[t_point][ids] for ids in ids1]
      f2 = [p[1] for p in cell_positions[t_point+1]]
      ids2 = [p[0] for p in cell_positions[t_point+1]]
      int2 = [intensities_dict[t_point+1][ids] for ids in ids2]
      pairs = frame_matching(f1, f2, dist_cutoff=100)
      matchings[t_point] = [(ids1[p1], ids2[p2]) for p1, p2 in pairs]

    trajectories = []
    for t_point in t_points[:-1]:
      pairs = matchings[t_point]
      for pair in pairs:
        for t in trajectories:
          if t_point in t and t[t_point] == pair[0]:
            t[t_point+1] = pair[1]
            break
        else:
          trajectories.append({t_point:pair[0], t_point+1:pair[1]})

    trajectories_positions = [{t_point: cell_positions_dict[t_point][t[t_point]] for t_point in t} for t in trajectories]
    trajectories = trajectory_connection(trajectories, trajectories_positions, intensities_dict, dist_cutoff=100., only_gap=True)
    trajectories = [t for t in trajectories if len(t) > 10]
    trajectories_positions = [{t_point: cell_positions_dict[t_point][t[t_point]] for t_point in t} for t in trajectories]
    
    ### Generate segmentation stacks for trajectories
    if not os.path.exists('../Data/DynamicPatches/%s' % site):
      os.mkdir('../Data/DynamicPatches/%s' % site)
    
    with open('../Data/DynamicPatches/%s/all_traj.pkl' % site, 'wb') as f:
      pickle.dump(trajectories, f)

    for i, t in enumerate(trajectories):
      mat_traj = np.zeros((len(t), 2048, 2048), dtype=bool)
      for j, t_point in enumerate(sorted(t.keys())):
        cell_id = t[t_point]
        positions = cell_pixel_assignments[t_point][0][np.where(cell_pixel_assignments[t_point][1] == cell_id)[0]]
        mat_traj[j, positions[:, 0], positions[:, 1]] = True
      tifffile.imwrite('../Data/DynamicPatches/%s/Trajectory%d.tif' % (site, i), mat_traj, dtype=bool)

    image_stack = h5py.File('../Combined/%s.h5' % site,'r+')
    image_stack = np.stack([image_stack[k] for k in sorted(image_stack.keys())], 0) #txyzc
    segmentation_stack = h5py.File('../Combined/%s_NNProbabilities.h5' % site,'r+')['exported_data']
    for i, t in enumerate(trajectories):
      print("\t Writing trajectory %d/%d" % (i, len(trajectories)))
      t_positions = trajectories_positions[i]
      # Moving window
      f_n = '../Data/DynamicPatches/%s/Trajectory%d_moving.h5' % (site, i)
      if not os.path.exists(f_n):
        with h5py.File(f_n, 'w') as f:
          for _t_point in sorted(t.keys()):
            f2 = h5py.File('../Data/StaticPatches/%s/%d_%d.h5' % (site, _t_point, t[_t_point]), 'r')
            f.create_dataset("mat_%d" % _t_point, data=np.array(f2["mat"]))
            f.create_dataset("masked_mat_%d" % _t_point, data=np.array(f2["masked_mat"]))
            f.create_dataset("position_%d" % _t_point, data=t_positions[_t_point])
      
      positions = np.stack([t_positions[t_point] for t_point in sorted(t_positions.keys())], 0)
      center = np.median(positions, 0)
      window = [(int(center[0]) - 128,
                 int(center[0]) + 128),
                (int(center[1]) - 128,
                 int(center[1]) + 128)]
      for t_point in sorted(t_positions.keys()):
        if not within_range(window, t_positions[t_point]):
          break
      else:
        f_n = '../Data/DynamicPatches/%s/Trajectory%d_fixed.h5' % (site, i)
        if not os.path.exists(f_n):
          with h5py.File(f_n, 'w') as f:
            for _t_point in sorted(t.keys()):
              f.create_dataset("position_%d" % _t_point, data=t_positions[_t_point])
              img = image_stack[_t_point, :, :, 0]
              output_mat = select_window(img, window, padding=-1)
              f.create_dataset("mat_%d" % _t_point, data=output_mat)

              ### Masking ###
              cell_segmentation = np.argmax(segmentation_stack[_t_point, :, :, 0], 2)
              window_segmentation = select_window(cell_segmentation, window, padding=-1)

              background_pool = img[np.where(segmentation_stack[t_point, :, :, 0, 0] > 0.9)]
              background_filling = np.random.choice(np.arange(background_pool.shape[0]), size=(256, 256))
              background_filling = np.take(background_pool, background_filling, 0)
              remove_mask, target_mask = generate_mask(cell_pixel_assignments[_t_point][0],
                                                       cell_pixel_assignments[_t_point][1],
                                                       t[_t_point],
                                                       window,
                                                       window_segmentation)
              masked_output_mat = output_mat * (1 - remove_mask) + background_filling * remove_mask
              f.create_dataset("masked_mat_%d" % _t_point, data=masked_output_mat)
              ################

