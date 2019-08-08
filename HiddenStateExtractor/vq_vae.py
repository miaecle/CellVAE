import os
import h5py
import cv2
import numpy as np
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from naive_imagenet import DATA_ROOT, read_file_path

CHANNEL_VAR = np.array([0.07, 0.07, 0.09])

class VectorQuantizer(nn.Module):
  def __init__(self, embedding_dim=128, num_embeddings=128, commitment_cost=0.25, gpu=True):
    super(VectorQuantizer, self).__init__()
    self.embedding_dim = embedding_dim
    self.num_embeddings = num_embeddings
    self.commitment_cost = commitment_cost
    self.gpu = gpu
    self.w = nn.Embedding(num_embeddings, embedding_dim)

  def forward(self, inputs):

    # inputs: Batch * Num_hidden(=embedding_dim) * H * W
    distances = t.sum((inputs.unsqueeze(1) - self.w.weight.reshape((1, self.num_embeddings, self.embedding_dim, 1, 1)))**2, 2)

    # Decoder input
    encoding_indices = t.argmax(-distances, 1)
    quantized = self.w(encoding_indices).transpose(2, 3).transpose(1, 2)
    assert quantized.shape == inputs.shape
    output_quantized = inputs + (quantized - inputs).detach()

    # Commitment loss
    e_latent_loss = F.mse_loss(quantized.detach(), inputs)
    q_latent_loss = F.mse_loss(quantized, inputs.detach())
    loss = q_latent_loss + self.commitment_cost * e_latent_loss

    # Perplexity (used to monitor)
    # TODO: better deal with the gpu case here
    encoding_onehot = t.zeros(encoding_indices.flatten().shape[0], self.num_embeddings)
    if self.gpu:
      encoding_onehot = encoding_onehot.cuda()
    encoding_onehot.scatter_(1, encoding_indices.flatten().unsqueeze(1), 1)
    avg_probs = t.mean(encoding_onehot, 0)
    perplexity = t.exp(-t.sum(avg_probs*t.log(avg_probs + 1e-10)))

    return output_quantized, loss, perplexity

  @property
  def embeddings(self):
    return self.w.weight

  def encode_inputs(self, inputs):
    # inputs: Batch * Num_hidden(=embedding_dim) * H * W
    distances = t.sum((inputs.unsqueeze(1) - self.w.weight.reshape((1, self.num_embeddings, self.embedding_dim, 1, 1)))**2, 2)
    # Decoder input
    encoding_indices = t.argmax(-distances, 1)
    return encoding_indices

class ResidualBlock(nn.Module):
  def __init__(self,
               num_hiddens=128,
               num_residual_hiddens=512,
               num_residual_layers=2):
    super(ResidualBlock, self).__init__()
    self.num_hiddens = num_hiddens
    self.num_residual_layers = num_residual_layers
    self.num_residual_hiddens = num_residual_hiddens

    self.layers = []
    for _ in range(self.num_residual_layers):
      self.layers.append(nn.Sequential(
          nn.ReLU(),
          nn.Conv2d(self.num_hiddens, self.num_residual_hiddens, 3, padding=1),
          nn.BatchNorm2d(self.num_residual_hiddens),
          nn.ReLU(),
          nn.Conv2d(self.num_residual_hiddens, self.num_hiddens, 1),
          nn.BatchNorm2d(self.num_hiddens)))
    self.layers = nn.ModuleList(self.layers)

  def forward(self, x):
    output = x
    for i in range(self.num_residual_layers):
      output = output + self.layers[i](output)
    return output

class VQ_VAE(nn.Module):
  def __init__(self,
               num_inputs=3,
               num_hiddens=128,
               num_residual_hiddens=64,
               num_residual_layers=2,
               num_embeddings=128,
               commitment_cost=0.25,
               channel_var=CHANNEL_VAR,
               **kwargs):
    super(VQ_VAE, self).__init__(**kwargs)
    self.num_inputs = num_inputs
    self.num_hiddens = num_hiddens
    self.num_residual_layers = num_residual_layers
    self.num_residual_hiddens = num_residual_hiddens
    self.num_embeddings = num_embeddings
    self.commitment_cost = commitment_cost
    self.channel_var = nn.Parameter(t.from_numpy(channel_var).float().reshape((1, 3, 1, 1)), requires_grad=False)
    self.enc = nn.Sequential(
        nn.Conv2d(self.num_inputs, self.num_hiddens//2, 1),
        nn.Conv2d(self.num_hiddens//2, self.num_hiddens//2, 4, stride=2, padding=1),
        nn.BatchNorm2d(self.num_hiddens//2),
        nn.ReLU(),
        nn.Conv2d(self.num_hiddens//2, self.num_hiddens, 4, stride=2, padding=1),
        nn.BatchNorm2d(self.num_hiddens),
        nn.ReLU(),
        nn.Conv2d(self.num_hiddens, self.num_hiddens, 3, padding=1),
        nn.BatchNorm2d(self.num_hiddens),
        ResidualBlock(self.num_hiddens, self.num_residual_hiddens, self.num_residual_layers))
    self.vq = VectorQuantizer(self.num_hiddens, self.num_embeddings, commitment_cost=self.commitment_cost)
    self.dec = nn.Sequential(
        nn.Conv2d(self.num_hiddens, self.num_hiddens, 3, padding=1),
        ResidualBlock(self.num_hiddens, self.num_residual_hiddens, self.num_residual_layers),
        nn.ConvTranspose2d(self.num_hiddens, self.num_hiddens//2, 4, stride=2, padding=1),
        nn.BatchNorm2d(self.num_hiddens//2),
        nn.ReLU(),
        nn.ConvTranspose2d(self.num_hiddens//2, self.num_hiddens//4, 4, stride=2, padding=1),
        nn.BatchNorm2d(self.num_hiddens//4),
        nn.ReLU(),
        nn.Conv2d(self.num_hiddens//4, self.num_inputs, 1))

  def forward(self, inputs):
    # inputs: Batch * num_inputs(channel) * H * W, each channel from 0 to 1
    z_before = self.enc(inputs)
    z_after, c_loss, perplexity = self.vq(z_before)
    decoded = self.dec(z_after)
    recon_loss = t.mean(F.mse_loss(decoded, inputs, reduce=False)/self.channel_var)
    total_loss = recon_loss + c_loss
    return decoded, \
           {'recon_loss': recon_loss,
            'commitment_loss': c_loss,
            'total_loss': total_loss,
            'perplexity': perplexity}

def train(model, dataset, n_epochs=10, lr=0.001, batch_size=16, gpu=True):
  optimizer = t.optim.Adam(model.parameters(), lr=lr, betas=(.9, .999))
  model.zero_grad()

  data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
  for epoch in range(n_epochs):
    recon_loss = []
    perplexities = []
    print('start epoch %d' % epoch)
    for batch in data_loader:
      batch = batch[0]
      if gpu:
        batch = batch.cuda()

      _, loss_dict = model(batch)
      loss_dict['total_loss'].backward()
      optimizer.step()
      model.zero_grad()

      recon_loss.append(loss_dict['recon_loss'])
      perplexities.append(loss_dict['perplexity'])
    print('epoch %d recon loss: %f perplexity: %f' % \
        (epoch, sum(recon_loss).item()/len(recon_loss), sum(perplexities).item()/len(perplexities)))
  return model

def prepare_dataset(fs, cs=[2, 4, 17], input_shape=(128, 128)):
  tensors = []
  for f_n in fs:
    dat = h5py.File(f_n, 'r')['masked_mat']
    if cs is None:
      cs = np.arange(dat.shape[2])
    stacks = []
    for c in cs:
      stacks.append(cv2.resize(np.array(dat[:, :, c]).astype(float), input_shape) / 255.)
    tensors.append(t.from_numpy(np.stack(stacks, 0)).float())
  dataset = TensorDataset(t.stack(tensors, 0))
  return dataset


if __name__ == '__main__':
  cs = [2, 4, 17]
  input_shape = (128, 128)
  gpu = True

  fs = read_file_path(DATA_ROOT)
  #dataset = prepare_dataset(fs, cs=cs, input_shape=input_shape)
  dataset = t.load('../Data/StaticPatchesAll.pt')
  model = VQ_VAE(num_inputs=3,
                 num_hiddens=8,
                 num_residual_hiddens=64,
                 num_residual_layers=2,
                 num_embeddings=32,
                 commitment_cost=0.25,)
  # t.save(model.state_dict(), 'save.pt')
  # if gpu:
  #   model = model.cuda()
  # model = train(model, dataset, n_epochs=100, lr=0.001, batch_size=128, gpu=gpu)
  # t.save(model.state_dict(), 'save.pt')
  sd = t.load('save.pt')
  model.load_state_dict(sd)
  
  
  z_befores = []
  z_afters = []
  for i in range(len(fs)):
    sample = dataset[i:(i+1)][0].cuda()
    z_b = model.enc(sample)
    z_a = model.vq(z_b)[0]
    z_befores.append(z_b.cpu().data.numpy())
    z_afters.append(z_a.cpu().data.numpy())
  # used_indices = []
  # for i in range(500):
  #   sample = dataset[i:(i+1)][0].cuda()
  #   z_before = model.enc(sample)
  #   indices = model.vq.encode_inputs(z_before)
  #   used_indices.append(np.unique(indices.cpu().data.numpy()))
