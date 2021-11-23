import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader, Dataset
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

NUM_ACTIONS = 38 #89 - 52 + 1
MIN_ACTION = 52

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


class BridgeCritic(BridgeSupervised): # learns the Value of a given state (discounted total reward)
    def __init__(self, input_dim = 571, inner_dim = 256, num_blocks=2):
        super().__init__(input_dim, inner_dim, num_blocks)
        self.critic_out = nn.Linear(NUM_ACTIONS, 1)
        self.loss_fn = nn.MSELoss()
        self.metrics_fn = lambda yhat,y: {'r2' :1 - ((y - yhat)^2).sum()/((y - y.mean())^2).sum() }

    def forward(self, x):
        '''
        Outputs single value
        '''
        x = self.backbone(x)
        x = self.out(x)
        x = self.critic_out(x)
        return x
    
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-4)

    def training_step(self, batch, batch_idx):
        x, y = batch['observation'], batch['labels']
        yhat = self.forward(x)

        loss = self.loss_fn(yhat, y)
        metrics = self.metrics_fn(yhat, y)

        return {'loss' : loss, **metrics}

    def validation_step(self, batch, batch_idx):
        x, y = batch['observation'], batch['labels']
        yhat = self.forward(x)

        loss = self.loss_fn(yhat, y)
        metrics = self.metrics_fn(yhat.detach(), y.detach())

        return {'loss' : loss, **metrics}

class BridgeActor(BridgeSupervised): # learns the optimal policy fn ( optimal f(action, state) = probability(action|state) )
    def __init__(self, input_dim = 571, inner_dim = 256, num_blocks=2):
        super().__init__(input_dim, inner_dim, num_blocks)
        self.loss_fn = nn.MSELoss()
    
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-4)

    def training_step(self, batch, batch_idx):
        x, y = batch['observation'], batch['labels']
        yhat = self.forward(x)

        loss = self.loss_fn(yhat, y)
        metrics = self.metrics_fn(yhat, y)

        return {'loss' : loss, **metrics}

    def validation_step(self, batch, batch_idx):
        x, action, advantage = batch['observation'], batch['action'], batch['advantage']
        yhat = self.forward(x)



        loss = self.loss_fn(yhat, y)
        metrics = self.metrics_fn(yhat.detach(), y.detach())

        return {'loss' : loss, **metrics}


class BridgeActorCritic(pl.LightningModule):
    def __init__(self, input_dim = 571, inner_dim = 256, num_blocks=2):
        super().__init__()
        self.actor = BridgeActor(input_dim, inner_dim, num_blocks)
        self.critic = BridgeCritic(input_dim, inner_dim, num_blocks)
    
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-4)
 
    def forward(self, x):
        '''
        Outputs value, policy_distribution
        '''
        value = self.critic.forward(x)
        policy_dist = self.actor.forward(x)
        return value, policy_dist