import argparse
import os
import itertools
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import sys
sys.path.append("../pytorch_lightning/")
import pytorch_lightning as pl

from networks import Encoder, Generator, Discriminator
from hyperspherical_uniform import HypersphericalUniform
from von_mises_fisher import VonMisesFisher

class VAECycleGAN(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()

        # output path
        self.output_path = "outputs/"

        # self.logger.log_hyperparams(hparams)  # log hyperparameters
        self.hparams = hparams

        self.files = ["../data.csv"]
        self.n = len(self.files)

        # load data
        self.data = [torch.from_numpy(np.genfromtxt(file, delimiter=',').transpose()[1:, 1:]).float() for file in self.files]

        self.datasets = [torch.utils.data.TensorDataset(data) for data in self.data]

        self.test_size = 60

        self.train_dataset, self.test_dataset = zip(*(
            torch.utils.data.random_split(dataset, (len(dataset) - self.test_size, self.test_size))
            for dataset in self.datasets))

        input_dim = 3000

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # distance matrix
        print("computing distance matrices")
        self.distance_matrix_train = self.distance_matrix(torch.stack(list(self.UnTensorDataset(self.train_dataset[0]))))
        if not self.hparams.correlation_distance_loss:
            self.distance_total_train = self.distance_matrix_train.sum(1)
            # self.distance_matrix_train_norm = self.distance_matrix_train / self.distance_total_train.unsqueeze(1)
            self.distance_matrix_train_norm = self.distance_matrix_train / self.distance_total_train.sum()
        print("done")

        # define VAEs
        self.E = [Encoder(input_dim, self.hparams.latent_dim, self.hparams.hypersphere).to(device) for _ in range(self.n)]  # hparams available for activation and dropout

        self.G = [Generator(self.hparams.latent_dim, input_dim).to(device) for _ in range(self.n)]

        # share weights
        if self.hparams.share_weights:
            for E in self.E[1:]:
                E.s1 = self.E[0].s1
                E.s2m = self.E[0].s2m
                E.s2v = self.E[0].s2v
            for G in self.G[1:]:
                G.s1 = self.G[0].s1
                G.s2 = self.G[0].s2

        # define discriminators
        if self.hparams.separate_D:
            self.D = [[Discriminator(input_dim).to(device) if i != j else None
                       for j in range(self.n)] for i in range(self.n)]
        else:
            self.D = [[Discriminator(input_dim).to(device) for _ in range(self.n)]] * self.n

        # named modules to make pytorch lightning happy
        self.E0 = self.E[0]
        self.G0 = self.G[0]

        # hyperspherical distribution
        self.p_z = HypersphericalUniform(self.hparams.latent_dim - 1, device=device) \
            if self.hparams.hypersphere else None

        # cache
        self.prev_g_loss = None
        self.current_z = self.forward(torch.stack(list(self.UnTensorDataset(self.train_dataset[0]))).to(device), first=True).z_a.detach()

    def distance_matrix(self, x, y=None):
        if y is None:
            y = x
        n = x.size(0)
        m = y.size(0)
        x = x.unsqueeze(1).expand(n, m, -1)
        y = y.unsqueeze(0).expand(n, m, -1)
        if self.hparams.correlation_distance_loss:
            x = x - x.mean(2).unsqueeze(2)
            y = y - y.mean(2).unsqueeze(2)
            return (x * y).sum(2) / torch.sqrt((x * x).sum(2) * (y * y).sum(2))
        else:
            return torch.sqrt(torch.sum((x - y) ** 2, dim=2))

    def reparameterize(self, m, v):  # sample from mean and log variance
        if self.hparams.hypersphere:
            m = m / m.norm(dim=-1, keepdim=True)
            var = F.softplus(v) + 1
            q_z = VonMisesFisher(m, var)
            z = q_z.rsample()
            return z
        else:
            s = torch.exp(0.5 * v)
            # s = F.softplus(v)
            e = torch.randn_like(s)
            return m + e * s

    class UnTensorDataset(torch.utils.data.Dataset):  # wrapper to get rid of TensorDataset stuff
        def __init__(self, data):
            self.data = data
            self.length = len(data)  # compute length

        def __len__(self):
            return self.length

        def __getitem__(self, index):
            (item,) = self.data[index]  # TensorDataset returns tuples
            return item

    class NumberedDataset(torch.utils.data.Dataset):  # wrapper to return the index of a dataset
        def __init__(self, data):
            self.data = data
            self.length = len(data)  # compute length

        def __len__(self):
            return self.length

        def __getitem__(self, index):
            (item,) = self.data[index]  # TensorDataset returns tuples
            return index, item

    def train_dataloader(self):
        # dl = [torch.utils.data.DataLoader(dataset, batch_size=self.hparams.batch_size, shuffle=True)
        #       for dataset in self.train_dataset]
        # mixed_datasets = [self.DataZip(dl[i], dl[j], domain=(i, j)) for i, j in itertools.combinations(range(self.n), 2)]
        # dataset = torch.utils.data.ConcatDataset(mixed_datasets)
        # return torch.utils.data.DataLoader(dataset, batch_size=None, sampler=torch.utils.data.RandomSampler(dataset))
        dataset = self.NumberedDataset(self.train_dataset[0])
        return torch.utils.data.DataLoader(dataset, batch_size=self.hparams.batch_size,
                                           sampler=torch.utils.data.RandomSampler(dataset))

    def val_dataloader(self):
        # dl = [torch.utils.data.DataLoader(dataset, batch_size=self.hparams.batch_size, shuffle=True)
        #       for dataset in self.test_dataset]
        # mixed_datasets = [self.DataZip(dl[i], dl[j], domain=(i, j)) for i, j in itertools.combinations(range(self.n), 2)]
        # dataset = torch.utils.data.ConcatDataset(mixed_datasets)
        # return torch.utils.data.DataLoader(dataset, batch_size=None, sampler=torch.utils.data.RandomSampler(dataset))
        dataset = self.NumberedDataset(self.test_dataset[0])
        return torch.utils.data.DataLoader(dataset, batch_size=self.hparams.batch_size,
                                           sampler=torch.utils.data.RandomSampler(dataset))

    def test_dataloader(self):
        # mixed_datasets = [self.DataZip([(self.data[i],)], [(self.data[j],)], domain=(i, j))
        #                   for i, j in itertools.combinations(range(self.n), 2)]
        # dataset = torch.utils.data.ConcatDataset(mixed_datasets)
        # return [torch.utils.data.DataLoader(dataset, batch_size=None)]
        data = self.data[0]
        return [torch.utils.data.DataLoader(self.UnTensorDataset([(data,)]), batch_size=None)]

    def configure_optimizers(self):
        lr = self.hparams.lr
        #b1 = self.hparams.b1
        #b2 = self.hparams.b2
        #l2 = self.hparams.l2

        # optimizers for generator and discriminator
        opt_G = torch.optim.Adam(itertools.chain(*(E.parameters() for E in self.E), *(G.parameters() for G in self.G)),
                                 lr=lr) #betas=(b1, b2)#, weight_decay=l2)
        # if self.hparams.separate_D:
        #     opt_D = torch.optim.Adam(itertools.chain(*(D.parameters() for L in self.D for D in L if D is not None)),
        #                              lr=lr, betas=(b1, b2), weight_decay=l2)
        # else:
        #     opt_D = torch.optim.Adam(itertools.chain(*(D.parameters() for D in self.D[0])),
        #                              lr=lr, betas=(b1, b2), weight_decay=l2)

        # sched_G = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_G, patience=8)
        # sched_D = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_D, patience=8)

        # return [opt_G, opt_D], [sched_G, sched_D]
        return [opt_G]  # , [sched_G]

    class Namespace(dict):
        def __init__(self, **kwargs):
            super().__init__(kwargs)
            self.__dict__.update(kwargs)

    def forward(self, x_a, first=False):
        # encoding
        z_a_m, z_a_v = self.E[0](x_a)
        z_a = self.reparameterize(z_a_m, z_a_v)

        # reconstruction
        x_aa = self.G[0](z_a)

        # distances
        if not first:
            dist = self.distance_matrix(z_a, self.current_z)
        else:
            dist = None

        outputs = {'x_a': x_a, 'z_a_m': z_a_m, 'z_a_v': z_a_v, 'z_a': z_a, 'x_aa': x_aa, 'dist': dist}

        return self.Namespace(**outputs)

    def gen_loss(self, x_a, z_a_m, z_a_v, z_a, x_aa, dist, index=None, **kwargs):
        # normal samples for MMD calculation
        samples = torch.randn(256, self.hparams.latent_dim).to(x_a.device) if self.hparams.w_vae_m != 0 else 0

        # VAE loss
        loss_vae_a_x = self.VAE_loss(x_aa, x_a)
        loss_vae_a_m = self.MMD_loss(z_a_m, samples) if self.hparams.w_vae_m != 0 else 0
        loss_vae_a_k = self.KLD_loss(z_a_m, z_a_v) if self.hparams.w_vae_k != 0 else 0

        # distance loss
        if index is not None:
            if self.hparams.correlation_distance_loss:
                target = self.distance_matrix_train[index].to(x_a.device)
                error = (dist - target) ** 2
                loss_dist = torch.sum(error.sum(1))
            else:
                # target = self.distance_matrix_train_norm[index].to(dist.device) * dist.sum(1).unsqueeze(1).detach()
                target = self.distance_matrix_train_norm[index].to(x_a.device) * self.distance_matrix(self.current_z.detach()).sum()
                error = (dist - target) ** 2
                loss_dist = torch.sum(error.sum(1) - error.gather(1, index.unsqueeze(1)).squeeze(1))  # subtract self error

        # loss coefficients
        w_vae_x = self.hparams.w_vae_x
        w_vae_m = self.hparams.w_vae_m
        w_vae_k = self.hparams.w_vae_k
        w_vae_d = self.hparams.w_vae_d

        # loss components
        loss_vae = loss_vae_a_x
        loss_mmd = loss_vae_a_m
        loss_kld = loss_vae_a_k

        loss =  (loss_vae + w_vae_k * 1.0 * loss_kld ) / x_a.shape[0]
        #/ x_a.shape[0]

        #loss = w_vae_x * loss_vae + \
               #w_vae_m * loss_mmd + \
               #w_vae_k * 0.1 * loss_kld

        if index is not None:
            loss += loss_dist * w_vae_d

        outputs = {
            'loss_vae': loss_vae,
            'loss_mmd': loss_mmd,
            'loss_kld': loss_kld,
            'loss': loss
        }

        if index is not None:
            outputs['loss_dist'] = loss_dist

        return self.Namespace(**outputs)

    def training_step(self, batch, batch_idx):
        index, x_a = batch

        outputs = self.forward(x_a)

        # generator loss
        g = self.gen_loss(**outputs, index=index)

        self.prev_g_loss = g.loss

        # update geometry
        # if batch_idx % 10 == 9:
        self.current_z = self.forward(torch.stack(list(self.UnTensorDataset(self.train_dataset[0]))).to(x_a.device), first=True).z_a.detach()

        log = {
            'train_vae_loss': g.loss_vae,
            'train_mmd_loss': g.loss_mmd,
            'train_kld_loss': g.loss_kld,
            'train_dist_loss': g.loss_dist,
            'train_total_loss': g.loss
        }
        output = {
            'loss': g.loss,
            'progress_bar': {'g_loss': g.loss},
            'log': log
        }
        return output

    def on_before_zero_grad(self, optimizer):
        # print(self.E[0].l1.weight.grad.mean() +
        #       self.E[0].l1.bias.grad.mean() +
        #       self.E[0].l2.weight.grad.mean() +
        #       self.E[0].l2.bias.grad.mean() +
        #       self.E[0].s1.weight.grad.mean() +
        #       self.E[0].s1.bias.grad.mean() +
        #       self.E[0].s2m.weight.grad.mean() +
        #       self.E[0].s2m.bias.grad.mean() +
        #       self.E[0].s2v.weight.grad.mean() +
        #       self.E[0].s2v.bias.grad.mean())
        pass  # TODO: log histograms of gradients

    def validation_step(self, batch, batch_idx):
        index, x_a = batch

        outputs = self.forward(x_a)

        # generator loss
        g = self.gen_loss(**outputs)

        self.prev_g_loss = g.loss

        log = {
            'val_vae_loss': g.loss_vae,
            'val_mmd_loss': g.loss_mmd,
            'val_kld_loss': g.loss_kld,
            'val_total_loss': g.loss
        }
        return log

    def validation_end(self, outputs):  # TODO: GAN accuracy
        mean_output = {key: sum(output[key] for output in outputs) / len(outputs) for key in outputs[0]}
        return {
            'progress_bar': {'g_loss': mean_output['val_total_loss']},
            'log': mean_output
        }

    def test_step(self, batch, batch_idx):
        x_a = batch

        y = self.forward(x_a)

        del self.distance_matrix_train
        if not self.hparams.correlation_distance_loss:
            del self.distance_total_train
            del self.distance_matrix_train_norm

        print()
        print("computing distance matrices")
        data_dist = self.distance_matrix(x_a.cpu())
        z_dist = self.distance_matrix(y.z_a.cpu())
        print("done")

        print("computing reconstruction correlation")
        x_a_n = x_a - x_a.mean(1).unsqueeze(1)
        x_aa_n = y.x_aa - y.x_aa.mean(1).unsqueeze(1)
        recon_corr = (x_a_n * x_aa_n).sum(1) / torch.sqrt((x_a_n * x_a_n).sum(1) * (x_aa_n * x_aa_n).sum(1))
        print("done")

        files = {
            'orginal': x_a,
            'a_mean': y.z_a_m,
            'a_logvar': y.z_a_v,
            'observed_counts_logNorm_5pop_continuous_A_depth-1000_alpha-0.005_Z': y.z_a,
            'observed_counts_logNorm_5pop_continuous_A_depth-1000_alpha-0.005_recreated_from_vae': y.x_aa,
            'distance_data': data_dist,
            'distance_z': z_dist,
            'reconstruction_correlation': recon_corr
        }
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
        for s, v in files.items():
            np.savetxt(self.output_path + s + '.csv', v.detach().cpu().numpy(),fmt="%2.7g", delimiter=",")

        log = {
            '(placeholder; ignore)': 0
        }
        return log

    def test_end(self, outputs):
        output = outputs[0]
        return {
            'log': output
        }

    def compute_kernel(self, x, y):
        x_size = x.size(0)
        y_size = y.size(0)
        dim = x.size(1)
        x = x.unsqueeze(1)  # (x_size, 1, dim)
        y = y.unsqueeze(0)  # (1, y_size, dim)
        tiled_x = x.expand(x_size, y_size, dim)
        tiled_y = y.expand(x_size, y_size, dim)
        kernel_input = (tiled_x - tiled_y).pow(2).mean(2) / float(dim)
        return torch.exp(-kernel_input)  # (x_size, y_size)

    def VAE_loss(self, x1, x):  # VAE loss
        if self.hparams.bce_vae_loss:
            return F.binary_cross_entropy(x1, x)
        else:
            return F.mse_loss(x1, x)

    def MMD_loss(self, x, y):
        x_kernel = self.compute_kernel(x, x)
        y_kernel = self.compute_kernel(y, y)
        xy_kernel = self.compute_kernel(x, y)
        mmd = x_kernel.mean() + y_kernel.mean() - 2 * xy_kernel.mean()
        return mmd

    def KLD_loss(self, m, v):
        if self.hparams.hypersphere:
            m = m / m.norm(dim=-1, keepdim=True)
            var = F.softplus(v) + 1
            q_z = VonMisesFisher(m, var)
            return torch.distributions.kl.kl_divergence(q_z, self.p_z).mean()
        else:
            return -0.5 * torch.mean(1 + v - m.pow(2) - v.exp())
            #torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

    def pearson_correlation(self, x, y):
        normx = x - torch.mean(x)
        normy = y - torch.mean(y)

        return torch.mean(torch.sum(normx * normy, dim=1) / (
                torch.sqrt(torch.sum(normx ** 2, dim=1)) * torch.sqrt(torch.sum(normy ** 2, dim=1))))
