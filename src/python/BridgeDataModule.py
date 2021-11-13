import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader, Dataset
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
        self.all_trajectories = [_no_play_trajectory(line) for line in open(self.data_dir)]

    def __len__(self):
        return len(self.all_trajectories)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        obs = np.zeros([len(idx)] + GAME.observation_tensor_shape(), np.float32)
        labels = np.zeros(len(idx), dtype=np.int32)

        for i in range(len(idx)):
            traj = self.all_trajectories[idx[i]]
            state = GAME.new_initial_state()
            action_index = np.random.randint(52, len(self.all_trajectories[i]))
            for action in traj[:action_index]:
                state.apply_action(action)

            obs[i] = state.observation_tensor()
            labels[i] = self.traj[action_index]

        return {'observation' : obs, 'labels' : labels}


class BridgeDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "./", batch_size : int = 32):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.bridge_data = {}

    def setup(self, stage: Optional[str] = None):
        for mode in ['train', 'test', 'valid']:
            self.bridge_data[mode] = BridgeDataset(f'{self.data_dir}_{mode}')

    def train_dataloader(self):
        return DataLoader(self.bridge_data['train'], batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.bridge_data['valid'], batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.bridge_data['test'], batch_size=self.batch_size)

    def teardown(self, stage: Optional[str] = None) -> None:
        pass