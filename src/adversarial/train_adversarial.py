import pyspiel
import numpy as np
import gym, gym.spaces
import math
import random
import numpy as np
from collections import namedtuple, deque
from itertools import count
import argparse
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

from supervised.BridgeNetwork import *

GAME = pyspiel.load_game('bridge(use_double_dummy_result=true)')

BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 40

ADVERSARY_RANDOMNESS_DECAY = 0.90
ADVERSARY_RANDOMNESS = 1.0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Adversary(ABC):
    def getAction(observation):
        raise NotImplemented

class RandomAdversary(Adversary):
    def __init__(self, possible_actions):
        self.possible_actions = possible_actions
    def get_action(self, observation):
        return np.random.rand(self.possible_actions)

class PolicyNetAdversary(Adversary):
    def __init__(self, policy_net):
        self.policy_net = policy_net
    
    def get_action(self, observation):
        with torch.no_grad():
            self.policy_net.eval()
            observation = torch.from_numpy(observation).to(device).float().unsqueeze(0)
            return self.policy_net(observation).cpu().numpy()


class WeightedRandomSelectedAdversary(Adversary):
    def __init__(self, adversaries, weights = None) -> None:
        self.adversaries = adversaries
        self.weights = weights

        if self.weights == None:
            self.weights = np.full(len(self.adversaries), 1/len(self.adversaries))  
    
    def get_action(self, observation):
        return random.choices(self.adversaries, weights=self.weights, k=1)[0].get_action(observation)
        

class AdversarialBridgeEnv(gym.Env):
    """Custom Environment that follows gym interface"""

    def __init__(self, adversary, adversary_plays_first = False):
        super(AdversarialBridgeEnv, self).__init__()    # Define action and observation space
        self.action_space = gym.spaces.Box(low=0, high=1, shape=(38,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(571,), dtype=np.float32)
        self.adversary = adversary
        self.reset(adversary_plays_first = adversary_plays_first)

    def reset(self, adversary = None, adversary_plays_first = False):
        self.state = self.generate_random_game()
        self.adversary = adversary if adversary != None else self.adversary
        return np.array(self.state.observation_tensor())

    def step(self, action_dist):
        action = self.pick_action(action_dist)
        self.state.apply_action(action+52)
        
        if self.state.current_phase() == 3:
            return self.calculate_terminal_reward(action)
        
        # opposing team action
        self.adversary_step()

        if self.state.current_phase() == 3:
            return self.calculate_terminal_reward(action)

        return self.calculate_default_reward(action)

    def adversary_step(self):
        self.state.apply_action(
            self.pick_action(
                self.adversary.get_action(np.array(self.state.observation_tensor()))
            ) + 52
        )

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
        masked_action_vector = action_vector*legal_action_mask
        action = np.argmax(masked_action_vector)

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

    def __len__(self):
        return len(self.memory)

def select_action(state, env):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            policy_net.eval()
            return policy_net(state).cpu()
    else:
        choice = random.choice(env.state.legal_actions())-52
        return torch.tensor(env.action_space.sample().reshape((1,-1)), device=device, dtype=torch.long)

def main(num_episodes : int, pretrained_model : str):
    envs = [AdversarialBridgeEnv(RandomAdversary(38))]

    # Get number of actions from gym action space
    n_actions = env.action_space.shape[0]
    n_observations = sum(env.observation_space.shape)

    policy_net = BridgeBase().to(device)
    policy_net.load_state_dict(torch.load(pretrained_model)['state_dict'])
    target_net = BridgeBase().to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.RMSprop(policy_net.parameters())
    memory = ReplayMemory(10000)


    steps_done = 0

    trailing_avg_reward = deque()
    trailing_avg_size = 100

    def optimize_model():
        if len(memory) < BATCH_SIZE:
            return
        transitions = memory.sample(BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
                                                    
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = policy_net(state_batch).gather(1, action_batch.type(torch.int64).to(device).unsqueeze(1)).squeeze()

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(BATCH_SIZE, device=device)
        target_net.train()
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values)

        # Optimize the model
        optimizer.zero_grad()
        loss.backward()
        for param in policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        optimizer.step()

        return loss

    num_episodes = 10000
    for i_episode in range(num_episodes):
        if i_episode % 100 == 0:
            adversary_randomness = ADVERSARY_RANDOMNESS * ADVERSARY_RANDOMNESS_DECAY ** (i_episode // 100)
            envs.append(
                AdversarialBridgeEnv(
                    WeightedRandomSelectedAdversary((
                        RandomAdversary(38), 
                        PolicyNetAdversary(target_net)), [adversary_randomness, 1-adversary_randomness])))

        env = random.choice(envs)
        # Initialize the environment and state
        obs = env.reset()
        obs = torch.from_numpy(obs).to(device).float().unsqueeze(0)
        for t in count():
            # Select and perform an action
            action = select_action(obs, env)
            new_obs, reward, done, metadata = env.step(np.array(action.cpu()))
            new_obs = torch.from_numpy(new_obs).to(device).float().unsqueeze(0)
            reward = torch.tensor([reward], device=device)

            # Store the transition in memory
            memory.push(obs, torch.Tensor(np.array([metadata["action"]], dtype=np.int64)), new_obs, reward)

            # Move to the next state
            obs = new_obs

            # Perform one step of the optimization (on the policy network)
            loss = optimize_model()
            if done:
                
                trailing_avg_reward.append(reward[0].cpu().numpy())
                if len(trailing_avg_reward) > trailing_avg_size:
                    trailing_avg_reward.popleft()

                print(f"episode #{i_episode}, episode reward: {reward[0]}, avg_reward: {round(np.mean(trailing_avg_reward),2)}, episode length: {t+1}, loss: {loss}")
                # print(env.state)
                break
        # Update the target network, copying all weights and biases in DQN
        if i_episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

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