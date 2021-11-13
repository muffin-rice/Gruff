import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader, Dataset
import torch

class RFC(pl.LightningModule):
    def __init__(self, dim):
        super().__init__()

        # RFCs should connect
        self.fc1 = torch.nn.Linear(dim, dim)
        self.fc2 = torch.nn.Linear(dim, dim)

    def forward(self, x):
        l1 = torch.relu(self.fc1(x))
        l2 = torch.relu(self.fc2(l1))

        return l2 + x


class RFCBackbone(pl.LightningModule):
    def __init__(self, input_dim = 571, inner_dim = 256, num_blocks = 2):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_dim, inner_dim)
        self.rfc1 = [RFC(inner_dim) for i in range(2)]
        self.efc = torch.nn.Linear(inner_dim, input_dim)

        self.fc2 = torch.nn.Linear(input_dim * 2, inner_dim)
        self.rfc2 = [RFC(inner_dim) for i in range(2)]

    def forward(self, x):
        x1 = self.fc1(x)
        for rfc in self.rfc1:
            x1 = rfc(x1)

        x1 = self.efc(x1)

        x_modified = torch.cat(x, x1)

        x2 = self.fc2(x_modified)
        for rfc in self.rfc2:
            x2 = rfc(x2)

        return x2 #return the latent dimension, final predict values come later


class BridgeSupervised(pl.LightningModule):
    def __init__(self, input_dim = 571, inner_dim = 256, num_blocks=2):
        super().__init__()
        self.inner_dim = inner_dim
        self.backbone = RFCBackbone(input_dim, inner_dim, num_blocks)
        self.out = torch.nn.Linear(inner_dim, 1)

    def forward(self, x):
        x = self.backbone(x)
