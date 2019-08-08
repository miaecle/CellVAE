import os
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import cv2

DATA_ROOT = '../Data/EncodedStaticPatches'

files = []
for dir_name, dirs, fs in os.walk(DATA_ROOT):
  for f in fs:
    if f.startswith('imagenet_') and f.endswith('.npy'):
      files.append(os.path.join(dir_name, f))

dats = np.stack([np.load(f) for f in files], 0)
dats = dats.reshape((len(files), -1))

pca = PCA(n_components=0.7)
dats_ = pca.fit_transform(dats)

tsne = TSNE()
dats__ = tsne.fit_transform(dats_)
#plt.clf()
#plt.scatter(dats_[:, 0], dats_[:, 1], s=0.1)
#np.random.seed(123)
#random_samples = np.random.choice(np.arange(len(files)), (10,), replace=False)
#font = {'family': 'serif',
#        'color': 'red',
#        'weight': 'normal',
#        'size': 12}
#for i, ind in enumerate(random_samples):
#  original_image = files[ind].replace('Encoded', '').replace('imagenet_', '').replace('.npy', '_retardance_masked.png')
#  plt.text(dats_[ind, 0], dats_[ind, 1], str(i), fontdict=font)
#  os.system('cp %s %d.png' % (original_image, i))
#plt.savefig('PCA.png', dpi=300)



site = 'D4-Site_3'
for i in range(20, 40):
  t = trajectories[i]
  
  #names = ['../Data/EncodedStaticPatches/%s/imagenet_%d_%d.npy' % (site, k, t[k]) for k in sorted(t.keys())]
  names = ['../Data/StaticPatches/%s/%d_%d.h5' % (site, k, t[k]) for k in sorted(t.keys())]
  names2 = ['../Data/StaticPatches/%s/%d_%d_retardance_masked.png' % (site, k, t[k]) for k in sorted(t.keys())]
  tiff_traj = np.stack([cv2.imread(n)[:, :, 0] for n in names2], 0)
  tifffile.imwrite("sample_traj_%d.tiff" % i, tiff_traj)
  
  inds = [files.index(n) for n in names]
  
  t_dats_ = dats_[np.array(inds)]
  plt.clf()
  plt.scatter(dats_[:, 0], dats_[:, 1], s=0.1)
  plt.plot(t_dats_[:, 0], t_dats_[:, 1], c='r')
  plt.savefig("PCA_with_traj_%d.png" % i, dpi=300)
  
  
plt.clf()  
plt.scatter(dats_[:, 0], dats_[:, 1], s=0.1)
t = trajectories[3]  
names = ['../Data/StaticPatches/%s/%d_%d.h5' % (site, k, t[k]) for k in sorted(t.keys())]
names2 = ['../Data/StaticPatches/%s/%d_%d_retardance_masked.png' % (site, k, t[k]) for k in sorted(t.keys())]
tiff_traj = np.stack([cv2.imread(n)[:, :, 0] for n in names2], 0)
  
inds = [files.index(n) for n in names]
t_dats_ = dats_[np.array(inds)]

plt.plot(t_dats_[:, 0], t_dats_[:, 1], c='r', linewidth=1.)
t = trajectories[13]  
names = ['../Data/StaticPatches/%s/%d_%d.h5' % (site, k, t[k]) for k in sorted(t.keys())]
names2 = ['../Data/StaticPatches/%s/%d_%d_retardance_masked.png' % (site, k, t[k]) for k in sorted(t.keys())]
tiff_traj = np.stack([cv2.imread(n)[:, :, 0] for n in names2], 0)
  
inds = [files.index(n) for n in names]
t_dats_ = dats_[np.array(inds)]

plt.plot(t_dats_[:, 0], t_dats_[:, 1], c='y', linewidth=1.)
t = trajectories[19]  
names = ['../Data/StaticPatches/%s/%d_%d.h5' % (site, k, t[k]) for k in sorted(t.keys())]
names2 = ['../Data/StaticPatches/%s/%d_%d_retardance_masked.png' % (site, k, t[k]) for k in sorted(t.keys())]
tiff_traj = np.stack([cv2.imread(n)[:, :, 0] for n in names2], 0)
  
inds = [files.index(n) for n in names]
t_dats_ = dats_[np.array(inds)]

plt.plot(t_dats_[:, 0], t_dats_[:, 1], c='m', linewidth=1.)
plt.savefig("PCA_with_traj_z.png", dpi=300)


