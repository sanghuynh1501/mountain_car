import gym
import math
import time
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):

    def __init__(self, n_features, hidden_units, n_actions):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(n_features, hidden_units)
        torch.nn.init.xavier_uniform_(self.fc1.weight, gain=1.0)
        # torch.nn.init.xavier_uniform_(self.fc1.bias, gain=1.0)
        self.fc2 = nn.Linear(hidden_units, n_actions)
        torch.nn.init.xavier_uniform_(self.fc2.weight, gain=1.0)
        # torch.nn.init.xavier_uniform_(self.fc2.bias, gain=1.0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)


env = gym.make('MountainCar-v0')
env = env.unwrapped

steps_done = 0
hidden_units = 10

GAMMA = 0.999
EPS_END = 0.05
EPS_START = 0.9
EPS_DECAY = 200
BATCH_SIZE = 128
TARGET_UPDATE = 10

n_actions = env.action_space.n
n_features = env.observation_space.shape[0]

policy_net = DQN(n_features, 10, n_actions).to(device)
target_net = DQN(n_features, 10, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(1000)


def optimize_model():
    print("len(memory) ", len(memory))
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
                                            batch.next_state)), device=device, dtype=torch.uint8)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                       if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


def obs2tensor(obs):
    obs = np.expand_dims(obs, axis=0)
    return torch.from_numpy(obs).float().to(device)


num_episodes = 10
for i_episode in range(num_episodes):
    obs = obs2tensor(env.reset())
    print("i_episode ", i_episode)
    while True:
        env.render()
        # Select and perform an action
        action = select_action(obs)
        obs_, reward, done, _ = env.step(action.item())
        reward = abs(obs_[0] + 0.5)
        obs_ = obs2tensor(obs_)
        reward = torch.tensor([reward], device=device)
        if done:
            obs_ = None
        memory.push(obs, action, obs_, reward)
        obs = obs_
        # Perform one step of the optimization (on the target network)
        optimize_model()
        if done:
            break

    # Update the target network, copying all weights and biases in DQN
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())
