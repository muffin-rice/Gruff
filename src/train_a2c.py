import os
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.plugins import DDPPlugin

from a2c.models import BridgeActorCritic
from a2c.dataset import A2CDataset

NUM_ACTIONS = 38
BATCH_SIZE = 128
GAMMA = 0.9
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 40
LEARNING_RATE = 1e-4

def main(num_episodes : int, pretrained_model : str): 
    
    logs_dir = 'logs'
    
    pl.utilities.seed.seed_everything(seed=0, workers=torch.cuda.is_available())
    
    model = BridgeActorCritic(pretrained_model, inner_dim=1024)
    
    model_ckpt = ModelCheckpoint(every_n_epochs=100,
                                 save_top_k=-1,
                                 filename='bridge-a2c-{epoch}')
    
    logger = TensorBoardLogger(logs_dir, 
                               name='a2c')
    
    trainer = pl.Trainer(max_epochs=num_episodes,
                         gpus=-1,
                         callbacks=[model_ckpt],
                         strategy=DDPPlugin(find_unused_parameters=False),
                         logger=logger,
                         num_sanity_val_steps=0)
    
    try:
        loader = DataLoader(A2CDataset())
        trainer.fit(model=model,
                    train_dataloaders=loader)
    except:
        trainer.save_checkpoint('crash.ckpt')

    print('Complete')


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Supervised training for Bridge.')

    parser.add_argument(
        '-e', '--episodes', default=1000, help='num. of epochs', type=int
    )

    parser.add_argument(
        '-pt', '--pretrained', help='pretrained model location', type=str
    )
    
    args = parser.parse_args()

    main(args.episodes, args.pretrained)
