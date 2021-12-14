from collections import namedtuple, deque

import torch
from torch.utils.data.dataset import IterableDataset
    
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def get(self):
        return self.memory

    def clear(self):
        self.memory.clear()

    def __len__(self):
        return len(self.memory)

class A2CDataset(IterableDataset):
    def __init__(self):
        pass
        
    def __iter__(self):
        return iter([0])