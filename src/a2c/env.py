import pyspiel
import numpy as np
import gym, gym.spaces
import random

import torch
import torch.nn.functional as F

class Agent:
    def __init__(self):
        self.env = BridgeEnv()
        self.state = self.env.reset()
        self.fail_thresh = 10
        
    def play_step(self, net, device):
        
        if type(self.state) is not torch.Tensor:
            state = torch.from_numpy(self.state).to(device).float().unsqueeze(0)
        else:
            state = self.state
        
        fails = 0
        while True:
            value, policy_dist = net.forward(state)
            if torch.any(torch.isnan(policy_dist)):
                fails += 1
                if fails > self.fail_thresh: raise ValueError
                print('Manually resetting env...')
                state = self.env.reset() 
                state = torch.from_numpy(state).to(device).float().unsqueeze(0)
                continue
            break
        
        dist = policy_dist.squeeze()
        
        new_state, reward, done, metadata = self.env.step(dist.detach())
        new_state = torch.from_numpy(new_state).to(device).float().unsqueeze(0)
        
        self.state = new_state
        
        return value, dist, new_state, reward, done, metadata
    
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
        action_vector = F.softmax(action_vector, dim=0)
        legal_action_mask = torch.tensor(self.state.legal_actions_mask())[52:52+self.action_space.shape[0]].to(action_vector.device)
        masked_action_vector = action_vector*legal_action_mask / sum(action_vector*legal_action_mask)
        masked_action_vector = masked_action_vector.cpu().numpy()
        action = np.random.choice(self.action_space.shape[0], p=masked_action_vector)

        if action + 52 not in self.state.legal_actions():
            print(action+52, self.state.legal_actions())
            print(action_vector[:6])
            print(legal_action_mask[:6])
            print((action_vector*legal_action_mask)[:6])
            print(masked_action_vector[:6])

        return action

    def generate_random_game(self): 
        state = GAME.new_initial_state()
        # deal all 52 cards randomly
        for i in np.random.choice(52, size=(52,), replace=False):
            state.apply_action(i)
        return state