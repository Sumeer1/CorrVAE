import argparse
import os
import shutil
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import numpy as np
import random
from euclidean_vae1 import VAECycleGAN

name = "euclidean_run1"

hparams = argparse.Namespace(share_weights=True,
                             latent_dim=32,
                             batch_size=10,
                             lr=0.001, #b1=0.9, b2=0.999, l2=0,
                             gan=True,
                             correlation_distance_loss=True,
                             bce_vae_loss=False,
                             separate_D=True,
                             hypersphere=False,
                             w_vae_x=1.0, w_vae_m=0, w_vae_k=0.1, w_vae_d= 0.001,
                             w_cc_x=1, w_cc_m=0, w_cc_k=1,
                             w_gan=1)
# MMD should not be enabled with hypersphere!

if os.path.exists("tensorboard/" + name):
    shutil.rmtree("tensorboard/" + name)
logger = TensorBoardLogger("tensorboard", name=name)

early_stop_callback = pl.callbacks.EarlyStopping(
                                                 monitor='g_loss',
                                                 min_delta=0.0,
                                                 patience=12,
                                                 verbose=False,
                                                 mode='min'
                                                 )

gpus = 1 if torch.cuda.is_available() else None

model = VAECycleGAN(hparams)

model.output_path = name + "/outputs/"

trainer = pl.Trainer(logger=logger,
                     early_stop_callback=early_stop_callback,
                     min_epochs=200,
                     max_epochs=400,
                     reload_dataloaders_every_epoch=True,
                     gpus=gpus)

trainer.fit(model)
if not os.path.exists(name + "/outputs"):
    os.makedirs(name + "/outputs")
trainer.save_checkpoint(name + "/model.pth")
trainer.test(model)
