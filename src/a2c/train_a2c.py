import pyspiel
import numpy as np
import gym, gym.spaces
import math
import random
import numpy as np
from collections import namedtuple, deque
import argparse
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import pytorch_lightning as pl

from BridgeNetwork import *

NUM_ACTIONS = 38
BATCH_SIZE = 128
GAMMA = 0.9
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 40
LEARNING_RATE = 1e-4

device = torch.device("cpu")

#class BridgeCritic(BridgeSupervised): # learns the Value of a given state (discounted total reward)
class BridgeCritic(BridgeBase): # learns the Value of a given state (discounted total reward)
    def __init__(self, file_dir, input_dim = 571, inner_dim = 256, num_blocks=2):
        #super().__init__(input_dim, inner_dim, num_blocks)
        super().__init__()
        self.load_state_dict(torch.load(file_dir)['state_dict'])
        self.critic_out = nn.Linear(NUM_ACTIONS, 1)
        self.loss_fn = nn.MSELoss()
        self.metrics_fn = lambda yhat,y: {'r2' :1 - ((y - yhat)^2).sum()/((y - y.mean())^2).sum() }

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

#class BridgeActor(BridgeSupervised): # learns the optimal policy fn ( optimal f(action, state) = probability(action|state) )
class BridgeActor(BridgeBase): # learns the optimal policy fn ( optimal f(action, state) = probability(action|state) )
    def __init__(self, file_dir, input_dim = 571, inner_dim = 256, num_blocks=2):
        #super().__init__(input_dim, inner_dim, num_blocks)
        super().__init__()
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


class BridgeActorCritic(pl.LightningModule):
    def __init__(self, file_dir, input_dim = 571, inner_dim = 256, num_blocks=2):
        super().__init__()
        self.actor = BridgeActor(file_dir, input_dim, inner_dim, num_blocks)
        self.critic = BridgeCritic(file_dir, input_dim, inner_dim, num_blocks)
    
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-4)
 
    def forward(self, x):
        '''
        Outputs value, policy_distribution
        '''
        value = self.critic.forward(x)
        policy_dist = self.actor.forward(x)
        return value, policy_dist

GAME = pyspiel.load_game('bridge(use_double_dummy_result=true)')

class BridgeEnv(gym.Env):
    """Custom Environment that follows gym interface"""

    def __init__(self):
        super(BridgeEnv, self).__init__()    # Define action and observation space
        self.action_space = gym.spaces.Box(low=0, high=1, shape=(38,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(571,), dtype=np.float32)
        self.reset()

    def reset(self):
        self.state = self.generate_random_game()
        return np.array(self.state.observation_tensor())

    def step(self, action):
        action = self.pick_action(action)

        self.state.apply_action(action+52)
        
        if self.state.current_phase() == 3:
            return self.calculate_terminal_reward(action)
        
        # random opposing team
        self.state.apply_action(random.choice(self.state.legal_actions()))

        if self.state.current_phase() == 3:
            return self.calculate_terminal_reward(action)

        return self.calculate_default_reward(action)

    def calculate_default_reward(self, action):
        obs = np.array(self.state.observation_tensor())
        reward = 0
        done = False
        return obs, reward, done, {"action": action}

    def calculate_terminal_reward(self, action):
        obs = np.zeros(571)
        reward = self.state.score_by_contract()[self.state.contract_index()]
        if self.state.current_player() in {1,3}:
            reward = -reward
        done = True
        return obs, reward, done, {"action": action}

    def pick_action(self, action_vector):
        action_vector = self.softmax(action_vector)
        legal_action_mask = np.array(self.state.legal_actions_mask())[52:52+self.action_space.shape[0]]
        masked_action_vector = action_vector*legal_action_mask / sum(action_vector*legal_action_mask)
        action = np.random.choice(self.action_space.shape[0], p=masked_action_vector)

        if action + 52 not in self.state.legal_actions():
            print(action+52, self.state.legal_actions())
            print(action_vector[:6])
            print(legal_action_mask[:6])
            print((action_vector*legal_action_mask)[:6])
            print(masked_action_vector[:6])

        return action

    def softmax(self, x):
        y = np.exp(x - np.max(x))
        f_x = y / np.sum(y)
        return f_x

    def generate_random_game(self): 
        state = GAME.new_initial_state()
        # deal all 52 cards randomly
        for i in np.random.choice(52, size=(52,), replace=False):
            state.apply_action(i)
        return state

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

def main(num_episodes : int, pretrained_model : str): 
    env = BridgeEnv()

    # Get number of actions from gym action space
    n_actions = env.action_space.shape[0]
    n_observations = sum(env.observation_space.shape)

    actor_critic = BridgeActorCritic(pretrained_model).to(device)

    optimizer = optim.Adam(actor_critic.parameters(), lr = LEARNING_RATE)
    memory = ReplayMemory(10000)

    steps_done = 0

    trailing_avg_reward = deque()
    trailing_avg_size = 100

    for i_episode in range(num_episodes):
        
        log_probs = []
        values = []
        rewards = []


        # Initialize the environment and state
        state = env.reset()
        state = torch.from_numpy(state).to(device).float().unsqueeze(0)
        for t in count():
            # TODO
            while True: 
                value, policy_dist = actor_critic.forward(state)
                if torch.any(torch.isnan(policy_dist)): 
                    #print(state)
                    state = env.reset() 
                    state = torch.from_numpy(state).to(device).float().unsqueeze(0)
                    continue
                break

            value = value.detach().numpy()[0,0]
            dist = policy_dist.detach().squeeze().numpy()

            new_state, reward, done, metadata = env.step(dist)
            new_state = torch.from_numpy(new_state).to(device).float().unsqueeze(0)

            action = metadata["action"]
            log_prob = torch.log(policy_dist.squeeze(0)[action])
            entropy = -np.sum(np.mean(dist) * np.log(dist))

            rewards.append(reward)
            values.append(value)
            log_probs.append(log_prob)

            state = new_state
            
            if done:

                Qval, _ = actor_critic.forward(new_state)

                trailing_avg_reward.append(reward)
                if len(trailing_avg_reward) > trailing_avg_size:
                    trailing_avg_reward.popleft()

                
                print(f"episode #{i_episode}, episode reward: {reward}, avg_reward: {round(np.mean(trailing_avg_reward),2)}, episode length: {t+1}")


                #print(env.state)
                break
        # Update the target network, copying all weights and biases in DQN
        Qvals = np.zeros_like(values)
        for t in reversed(range(len(rewards))):
            Qval = rewards[t]/100 + GAMMA * Qval
            Qvals[t] = Qval

        values = torch.FloatTensor(values) # values calculated by Critic
        Qvals = torch.FloatTensor(Qvals) # real values (calculated by sum of episode reward * discount factor)
        log_probs = torch.stack(log_probs) # log probability of each move in the episode
        
        advantage = Qvals - values
        actor_loss =  (-log_probs * advantage).mean()
        critic_loss = 0.5 * advantage.pow(2).mean()
        ac_loss = actor_loss + critic_loss
        print(ac_loss)

        optimizer.zero_grad()
        ac_loss.backward()
        optimizer.step()

    print('Complete')


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Supervised training for Bridge.')

    parser.add_argument(
        '-e', '--episodes', default=1000, help='num. of epochs', type=int
    )

    parser.add_argument(
        '-pt', '--pretrained', default='logs/bridge-supervised-epoch=7.ckpt', help='pretrained model location', type=str
    )
    
    args = parser.parse_args()

    main(args.episodes, args.pretrained)
