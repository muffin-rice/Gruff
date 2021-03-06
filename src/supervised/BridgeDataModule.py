import pytorch_lightning as pl
from typing import Optional
from torch.utils.data import random_split, DataLoader, Dataset
import torch.nn.functional as F
import numpy as np
import pyspiel
import torch

GAME = pyspiel.load_game('bridge(use_double_dummy_result=true)')

NUM_ACTIONS = 38
MIN_ACTION = 52
NUM_CARDS = 52
NUM_PLAYERS = 4
TOP_K_ACTIONS = 5  # How many alternative actions to display

def _no_play_trajectory(line: str):
    """Returns the deal and bidding actions only given a text trajectory."""
    actions = [int(x) for x in line.split(' ')]
    # Usually a trajectory is NUM_CARDS chance events for the deal, plus one
    # action for every bid of the auction, plus NUM_CARDS actions for the play
    # phase. Exceptionally, if all NUM_PLAYERS players Pass, there is no play
    # phase and the trajectory is just of length NUM_CARDS + NUM_PLAYERS.
    if len(actions) == NUM_CARDS + NUM_PLAYERS:
        return tuple(actions)
    else:
        return tuple(actions[:-NUM_CARDS])

class BridgeDataset(Dataset):
    def __init__(self, filename : str):
        self.filename = filename
        self.all_trajectories = [_no_play_trajectory(line) for line in open(self.filename)]

    def __len__(self):
        return len(self.all_trajectories)

    def __getitem__(self, idx):
        traj = np.zeros(90, dtype = np.int32)
        idtraj = self.all_trajectories[idx]
        traj[:len(idtraj)] = idtraj

        state = GAME.new_initial_state()
        action_index = np.random.randint(52, len(self.all_trajectories[idx]))
        for action in traj[:action_index]:
            state.apply_action(action)

        obs = torch.tensor(state.observation_tensor())
        labels = torch.tensor(traj[action_index], dtype = torch.long) - MIN_ACTION 
        traj = torch.tensor(traj)

        return {'observation' : obs, 'labels' : labels, 'traj': traj}


class BridgeDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "./", batch_size : int = 32, prefetch: int = 4, workers: int = 0):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.prefetch = prefetch
        self.workers = workers if torch.cuda.is_available() else 0
        self.bridge_paths = {}

    def setup(self, stage: Optional[str] = None):
        for mode in ['train', 'test', 'valid']:
            self.bridge_paths[mode] = f'{self.data_dir}/{mode}.txt'

    def train_dataloader(self):
        return DataLoader(BridgeDataset(self.bridge_paths['train']),
                          shuffle=True,
                          prefetch_factor=self.prefetch,
                          batch_size=self.batch_size,
                          num_workers=self.workers)

    def val_dataloader(self):
        return DataLoader(BridgeDataset(self.bridge_paths['valid']),
                          prefetch_factor=self.prefetch,
                          batch_size=self.batch_size,
                          num_workers=self.workers)

    def test_dataloader(self):
        return DataLoader(BridgeDataset(self.bridge_paths['test']),
                          prefetch_factor=self.prefetch,
                          batch_size=self.batch_size,
                          num_workers=self.workers)

    def teardown(self, stage: Optional[str] = None) -> None:
        pass
