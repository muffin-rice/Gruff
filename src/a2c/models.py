import numpy as np
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import pytorch_lightning as pl

from supervised.BridgeNetwork import *
from a2c.env import Agent, BridgeEnv

NUM_ACTIONS = 38
BATCH_SIZE = 128
GAMMA = 0.9
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 40
LEARNING_RATE = 1e-4

class BridgeActorCritic(pl.LightningModule):
    def __init__(self, file_dir, input_dim = 571, inner_dim = 256, num_blocks=2,
                 trailing_avg_size=100):
        super().__init__()
        
        self.actor = BridgeActor(file_dir, input_dim, inner_dim, num_blocks)
        self.critic = BridgeCritic(file_dir, input_dim, inner_dim, num_blocks)
        
        self.env = BridgeEnv()
        self.agent = Agent()
        
        self.trailing_avg_reward = deque(maxlen=trailing_avg_size)
    
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-4)
 
    def forward(self, x):
        '''
        Outputs value, policy_distribution
        '''
        value = self.critic.forward(x)
        policy_dist = self.actor.forward(x)
        return value, policy_dist
    
    def training_step(self, batch, batch_idx):
        
        rewards = []
        values = []
        log_probs = []
        
        episode_len = 0
        while True:
            episode_len += 1
            value, dist, new_state, reward, done, metadata = self.agent.play_step(self, self.device)
            
            action = metadata['action']
            log_prob = torch.max(dist[action].log(), torch.tensor(1e-8))
            entropy = -(dist.mean() * dist.log()).sum()
            
            print(log_prob)

            rewards.append(reward)
            values.append(value)
            log_probs.append(log_prob)
            
            if done:
                Qval = self.critic.forward(new_state)
                self.trailing_avg_reward.append(reward)
                
                self.log('eps_reward', float(reward), 
                         on_step=False,
                         on_epoch=True, 
                         rank_zero_only=True,
                         sync_dist=torch.cuda.device_count() > 1)
                
                self.log('avg_reward', np.mean(self.trailing_avg_reward),
                         on_step=False,
                         on_epoch=True, 
                         rank_zero_only=True,
                         sync_dist=torch.cuda.device_count() > 1)
                
                self.log('eps_len', float(episode_len),
                         on_step=False,
                         on_epoch=True, 
                         rank_zero_only=True,
                         sync_dist=torch.cuda.device_count() > 1)
                break
            
        self.agent.env.reset()
        
        rewards = torch.tensor(rewards).float()
        values = torch.cat(values)
        log_probs = torch.stack(log_probs)
        
        # lightning-specific for creating tensors
        rewards = rewards.type_as(rewards)
        values = values.type_as(values)
        # log_probs should already be on the correct device
        
        # Update the target network, copying all weights and biases in DQN
        Qvals = torch.zeros_like(values)
        for t in range(len(rewards)-1,-1,-1):
            Qval = rewards[t]/100 + GAMMA * Qval
            Qvals[t] = Qval
            
        advantage = Qvals - values
        actor_loss =  (-log_probs * advantage).mean()
        critic_loss = (0.5 * advantage * advantage).mean()
        print('aloss', actor_loss)
        print('closs', critic_loss)
        ac_loss = actor_loss + critic_loss
        
        self.log('loss', ac_loss,
                 on_step=False,
                 on_epoch=True, 
                 rank_zero_only=True,
                 sync_dist=torch.cuda.device_count() > 1)

        return { 'loss': ac_loss }
        
            
class BridgeCritic(BridgeBase): # learns the Value of a given state (discounted total reward)
    def __init__(self, file_dir, input_dim = 571, inner_dim = 256, num_blocks=2):
        super().__init__(input_dim, inner_dim, num_blocks)
        self.load_state_dict(torch.load(file_dir)['state_dict'])
        self.critic_out = nn.Linear(NUM_ACTIONS, 1)
        self.loss_fn = nn.MSELoss()
        self.metrics_fn = lambda yhat,y: {'r2' :1 - ((y - yhat)**2).sum()/((y - y.mean())**2).sum() }

    def forward(self, x):
        '''
        Outputs single value
        '''
        x = self.forward_half(x)
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

class BridgeActor(BridgeBase): # learns the optimal policy fn ( optimal f(action, state) = probability(action|state) )
# class BridgeActor(BridgeBase): # learns the optimal policy fn ( optimal f(action, state) = probability(action|state) )
    def __init__(self, file_dir, input_dim = 571, inner_dim = 256, num_blocks=2):
        super().__init__(input_dim, inner_dim, num_blocks)
        self.load_state_dict(torch.load(file_dir)['state_dict'])
        self.loss_fn = nn.MSELoss()
    
    def forward(self, x): 
        half = F.softmax(self.forward_half(x), dim=1)
        return half

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