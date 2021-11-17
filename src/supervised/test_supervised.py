import argparse

import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import pyspiel

from BridgeNetwork import BridgeSupervised
from BridgeDataModule import BridgeDataModule

MIN_ACTION = 52

def test(batch_size, epochs, data_dir, ckpt_path):
    
    pl.utilities.seed.seed_everything(seed=0, workers=torch.cuda.is_available())
    
    model = BridgeSupervised()
    
    model.load_from_checkpoint(ckpt_path)
    
    datamodule = BridgeDataModule(data_dir=data_dir, batch_size=1)
    datamodule.setup()
    
    val = datamodule.val_dataloader()
    
    GAME = pyspiel.load_game('bridge(use_double_dummy_result=true)')

    with torch.no_grad():
        for batch in val:
            obs, label, traj = batch['observation'], batch['labels'], batch['traj']
            traj = torch.tensor(traj).tolist()
            
            out = model(obs)
            
            state = GAME.new_initial_state()
            action_index = np.random.randint(52, len(traj))
            for action in traj[:action_index]:
                state.apply_action(action)
            
            print('-'*64)
            print()
            try:
                state.apply_action(out.item() + MIN_ACTION)
                print('Pyspiel State:')
                print(state)
                
            except:
                print('Invalid action attempted by agent')
                
            print()
            print('-'*64)
        
    print('done')
        
    
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Supervised training for Bridge.')
    
    parser.add_argument(
        '--datadir', default='../data', help='path to data files'
    )

    parser.add_argument(
        '--ckpt', default='', help='path to log files (checkpoint stored here)'
    )

    parser.add_argument(
        '-b', '--batchsize', default=32, help='batchsize', type=int
    )

    parser.add_argument(
        '-e', '--epochs', default=10, help='num. of epochs', type=int
    )
    
    args = parser.parse_args()

    test(args.batchsize, args.epochs, args.datadir, args.ckpt)
    
    


