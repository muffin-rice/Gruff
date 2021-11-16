import argparse

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from BridgeNetwork import BridgeSupervised
from BridgeDataModule import BridgeDataModule

data_dir = '../data'
logs_dir = '../logs'


def train(batch_size, epochs):
    
    pl.utilities.seed.seed_everything(seed=0, workers=torch.cuda.is_available())
    
    model = BridgeSupervised()
    
    # checkpointing
    model_ckpt = ModelCheckpoint(every_n_epochs=1,
                                 save_top_k=-1,
                                 filename='bridge-supervised-{epoch}')
    
    logger = TensorBoardLogger(logs_dir, 
                               name='supervised')
    
    trainer = pl.Trainer(max_epochs=epochs,
                         gpus=[1],
                         callbacks=[model_ckpt],
                         strategy='ddp',
                         logger=logger,
                         num_sanity_val_steps=0,
                         progress_bar_refresh_rate=None)
    
    trainer.fit(model=model,
                datamodule=BridgeDataModule(data_dir=data_dir, batch_size=batch_size))
    
    
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Supervised training for Bridge.')
    
    parser.add_argument(
        '-b', '--batchsize', default=32, help='batchsize', type=int
    )

    parser.add_argument(
        '-e', '--epochs', default=10, help='num. of epochs', type=int
    )
    
    args = parser.parse_args()

    train(args.batchsize, args.epochs)
    
    


