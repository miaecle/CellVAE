import os
import numpy as np
import h5py
from classification_models.resnet import ResNet50, preprocess_input
from keras.models import Model
import cv2

DATA_ROOT = '../Data/StaticPatches'

def read_file_path(root):
  files = []
  for dir_name, dirs, fs in os.walk(root):
    for f in fs:
      if f.endswith('.h5'):
        files.append(os.path.join(dir_name, f))
  return files

def initiate_model():
  model = ResNet50((224, 224, 3), weights='imagenet')
  target_layer = [l for l in model.layers if l.name == 'pool1'][0]
  hidden_extractor = Model(model.input, target_layer.output)
  hidden_extractor.compile(loss='mean_squared_error', optimizer='sgd')
  return hidden_extractor

def preprocess(f_n, cs=[0, 4, 15, 19]):
  dat = h5py.File(f_n, 'r')['masked_mat']
  if cs is None:
    cs = np.arange(dat.shape[2])
  stacks = []
  for c in cs:
    patch_c = cv2.resize(np.array(dat[:, :, c]).astype(float), (224, 224))
    stacks.append(np.stack([patch_c] * 3, 2))
  return np.stack(stacks, 0)

if __name__ == '__main__':
  fs = read_file_path(DATA_ROOT)
  extractor = initiate_model()
  for f_n in fs:
    print("Processing %s" % f_n)
    x = preprocess_input(preprocess(f_n))
    y = extractor.predict(x)
    f_n_out = f_n.replace("StaticPatches", "EncodedStaticPatches")
    np.save(f_n_out, y)
