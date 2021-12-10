import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader, Dataset
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

NUM_ACTIONS = 38 #89 - 52 + 1
MIN_ACTION = 52

class BaseModel(nn.Module): 
    #base model in samples/bridge_supervised_learning
    def __init__(self, dim, num_fc = 4):
        super().__init__()

        # self.dim = dim 
        self.fc1 = nn.Linear(dim, 1024)
        self.fcs = nn.Sequential(*[nn.Linear(1024, 1024) for i in range(num_fc-1)])
            
        
    def forward(self, x): 
        x = self.fc1(x) 
        for layer in self.fcs: 
            x = F.relu(x) 
            x = layer(x) 

        x = F.relu(x)
        return x


class RFC(nn.Module):
    def __init__(self, dim):
        super().__init__()

        # RFCs should connect
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)

    def forward(self, x):
        identity = x
        x1 = self.fc1(x)
        x1 = F.relu(x1)
        x1 = F.relu(self.fc2(x1))

        return x1 + identity

class RFCBackbone(pl.LightningModule):
    def __init__(self, input_dim = 571, inner_dim = 256, num_blocks = 2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, inner_dim)
        self.rfc1 = nn.Sequential(*[RFC(inner_dim) for i in range(2)])
        self.efc = nn.Linear(inner_dim, input_dim)

        self.fc2 = nn.Linear(2*input_dim, inner_dim)
        self.rfc2 = nn.Sequential(*[RFC(inner_dim) for i in range(2)])

    def forward(self, x):
        x1 = self.fc1(x)
        x1 = self.rfc1(x1)

        x1 = self.efc(x1)

        x_modified = torch.cat([x, x1], dim=1)

        x2 = self.fc2(x_modified)
        x2 = self.rfc2(x2)

        return x2 #return the latent dimension, final predict values come later


class BridgeBase(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.base = BaseModel()
        self.out = nn.Linear(1024, NUM_ACTIONS)
        
        self.loss_fn = nn.CrossEntropyLoss()
        self.metrics_fn = lambda yhat,y: {'acc' : (yhat == y).float().mean()}
        
        self.save_hyperparameters()
        
    
    def forward_half(self,x):
        '''Outputs probabilities'''
        x = self.base(x)
        x = self.out(x)
        return x
    
    def forward(self, x):
        '''
        Outputs one-hot encoding of action.
        Use .argmax(dim=1) + MIN_ACTION to get index per minibatch.
        '''
        x = F.softmax(self.forward_half(x))
        return x.argmax(dim=1) + MIN_ACTION

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-4)

    def training_step(self, batch, batch_idx):
        x, y = batch['observation'], batch['labels']
        yhat = self.forward_half(x)

        loss = self.loss_fn(yhat, y)
        metrics = self.metrics_fn(yhat.argmax(dim=1), y)

        return {'loss' : loss, **metrics}

    def validation_step(self, batch, batch_idx):
        x, y = batch['observation'], batch['labels'] - MIN_ACTION
        yhat = self.forward_half(x)

        loss = self.loss_fn(yhat, y)
        metrics = self.metrics_fn(yhat.detach().argmax(dim=1), y.detach())

        return {'loss' : loss, **metrics}

class BridgeSupervised(pl.LightningModule):
    def __init__(self, input_dim = 571, inner_dim = 256, num_blocks=2):
        super().__init__()
        self.inner_dim = inner_dim
        self.backbone = RFCBackbone(input_dim, inner_dim, num_blocks)
        self.head = BaseModel(inner_dim)
        self.out = nn.Linear(1024, NUM_ACTIONS)

        self.loss_fn = nn.CrossEntropyLoss()
        self.metrics_fn = lambda yhat,y: {'acc' : (yhat == y).float().mean()}
        
        self.save_hyperparameters()

    def forward_half(self, x): 
        '''outputs probabilities'''
        x = self.backbone(x)
        x = self.head(x)
        x = self.out(x)
        return x

    def forward(self, x):
        '''
        Outputs one-hot encoding of action.
        Use .argmax(dim=1) + MIN_ACTION to get index per minibatch.
        '''
        x = F.softmax(self.forward_half(x))
        return x.argmax(dim=1) + MIN_ACTION
    
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-4)

    def training_step(self, batch, batch_idx):
        x, y = batch['observation'], batch['labels']
        yhat = self.forward_half(x)

        loss = self.loss_fn(yhat, y)
        metrics = self.metrics_fn(yhat.argmax(dim=1), y)

        return {'loss' : loss, **metrics}

    def validation_step(self, batch, batch_idx):
        x, y = batch['observation'], batch['labels'] - MIN_ACTION
        yhat = self.forward_half(x)

        loss = self.loss_fn(yhat, y)
        metrics = self.metrics_fn(yhat.detach().argmax(dim=1), y.detach())

        return {'loss' : loss, **metrics}
