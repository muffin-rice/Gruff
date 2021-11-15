import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader, Dataset
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

NUM_ACTIONS = 38 #89 - 52 + 1
MIN_ACTION = 52

class RFC(pl.LightningModule):
    def __init__(self, dim):
        super().__init__()

        # RFCs should connect
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)

    def forward(self, x):
        l1 = F.relu(self.fc1(x))
        l2 = F.relu(self.fc2(l1))

        return l2 + x


class RFCBackbone(pl.LightningModule):
    def __init__(self, input_dim = 571, inner_dim = 256, num_blocks = 2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, inner_dim)
        self.rfc1 = [RFC(inner_dim) for i in range(2)]
        self.efc = nn.Linear(inner_dim, input_dim)

        self.fc2 = nn.Linear(2*input_dim, inner_dim)
        self.rfc2 = [RFC(inner_dim) for i in range(2)]

    def forward(self, x):
        x1 = self.fc1(x)
        for rfc in self.rfc1:
            x1 = rfc(x1)

        x1 = self.efc(x1)

        x_modified = torch.cat([x, x1], dim=1)

        x2 = self.fc2(x_modified)
        for rfc in self.rfc2:
            x2 = rfc(x2)

        return x2 #return the latent dimension, final predict values come later


class BridgeSupervised(pl.LightningModule):
    def __init__(self, input_dim = 571, inner_dim = 256, num_blocks=2):
        super().__init__()
        self.inner_dim = inner_dim
        self.backbone = RFCBackbone(input_dim, inner_dim, num_blocks)
        self.out = nn.Linear(inner_dim, NUM_ACTIONS)

        self.loss_fn = nn.CrossEntropyLoss()
        self.metrics_fn = lambda yhat,y: {'acc' : (yhat == y).float().mean()}
        
        self.save_hyperparameters()

    def forward(self, x):
        '''
        Outputs one-hot encoding of action.
        Use .argmax(dim=1) + MIN_ACTION to get index per minibatch.
        '''
        x = self.backbone(x)
        x = F.softmax(self.out(x), dim=1)
        return x + MIN_ACTION
    
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-4)

    def training_step(self, batch, batch_idx):
        x, y = batch['observation'], batch['labels']
        yhat = self.forward(x)

        loss = self.loss_fn(yhat, y)
        metrics = self.metrics_fn(yhat.argmax(dim=1), y)

        return {'loss' : loss, **metrics}

    def validation_step(self, batch, batch_idx):
        x, y = batch['observation'], batch['labels']
        yhat = self.forward(x)

        loss = self.loss_fn(yhat, y)
        metrics = self.metrics_fn(yhat.detach().argmax(dim=1), y.detach())

        return {'loss' : loss, **metrics}