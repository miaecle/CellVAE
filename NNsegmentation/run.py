#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 18:10:01 2019

@author: zqwu
"""
import tensorflow as tf
import numpy as np
import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
import pickle
from data import generate_patches, generate_ordered_patches, predict_whole_map
from models import Segment
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score

# input_params = {
#     'label_input': 'annotation',
#     'x_size': 256,
#     'y_size': 256,
#     'label_value_threshold': 0.5,
#     'rotate': True,
#     'mirror': True,
#     'seed': 123,
#     'time_slices': 5}
#
# train_patches = generate_ordered_patches(input_file, label_file, **input_params)
# test_patches = generate_patches(input_file, label_file, n_patches=200, **input_params)

train_patches = pickle.load(open('../Data/NNSegment/Annotation_patches_8slices_large.pkl', 'rb'))
test_patches = train_patches[-50:]
train_patches = train_patches[:-50]

model_path = './temp_save/'
if not os.path.exists(model_path):
  os.mkdir(model_path)
model = Segment(input_shape=(256, 256, 20),
                unet_feat=32,
                fc_layers=[64, 32],
                n_classes=3,
                model_path=model_path)

for epoch in range(25):
  inds = np.random.choice(np.arange(len(train_patches)), (200,), replace=False)
  train_subset = [train_patches[i] for i in inds]
  model.fit(train_subset,
            label_input='annotation',
            n_epochs=4,
            valid_patches=test_patches,
            valid_label_input='annotation')

model.save(model.model_path + '/final.h5')

sites = ['D4-Site_0', 'D4-Site_1', 'D4-Site_2', 'D4-Site_3', 'D5-Site_0', 'D5-Site_1', 'D5-Site_3', 'D5-Site_4']
for site in sites:
  predict_whole_map('../Combined/%s.h5' % site, model, n_classes=3, batch_size=8, n_supp=5)
