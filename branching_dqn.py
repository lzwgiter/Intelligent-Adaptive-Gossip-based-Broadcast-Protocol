#!/usr/bin/env python
# coding=utf-8
'''
@Author: lzwgiter & JK211
@Email: float311@163.com & jerryren2884@gmail.com
@Date: 2022-07-1 11:09:49
@LastEditor: JK211
LastEditTime: 2022-08-01 11:10:11
@Discription: This .py is the gossip environment of NS3 inherit from gym.Env
@Environment: python 3.8
'''
import copy
import sys
import time

from gym.spaces import flatdim
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

from GossipEnvironment import GossipEnvironment
from model import BranchingQNetwork
from utils import ExperienceReplayMemory, AgentConfig
import utils


class BranchingDQN(nn.Module):

    def __init__(self, obs, ac, n_1, n_2, config):

        super().__init__()

        self.q = BranchingQNetwork(obs, ac, n_1, n_2)
        self.target = BranchingQNetwork(obs, ac, n_1, n_2)

        self.target.load_state_dict(self.q.state_dict())

        self.target_net_update_freq = config.target_net_update_freq
        self.update_counter = 0

    def get_action(self, x, mask):
        with torch.no_grad():
            out = self.q(x, mask)
            action_pb = torch.argmax(out[0], dim=1)
            action_nei = torch.argmax(out[1], dim=1)
        return action_pb.numpy(), action_nei.numpy()

    def update_policy(self, adam, memory, params):

        b_states, b_actions, b_adj, b_next_adj, b_rewards, b_next_states, b_masks = memory.sample(params.batch_size)

        states = torch.tensor(b_states).float()
        adjs = torch.tensor(b_adj)
        next_adjs = torch.tensor(b_next_adj)
        pb_index = []
        nei_index = []
        tmp_1 = []
        tmp_2 = []
        for each_batch in b_actions:
            for each_uav_actions in each_batch:
                tmp_1.append(each_uav_actions[0])
                tmp_2.append(each_uav_actions[1])
            pb_index.append(tmp_1.copy())
            nei_index.append(tmp_2.copy())
            tmp_1.clear()
            tmp_2.clear()

        actions_pb_index = torch.tensor(pb_index)
        actions_pb_index = actions_pb_index.reshape(actions_pb_index.shape[0], actions_pb_index.shape[1], 1)

        actions_nei_index = torch.tensor(nei_index)
        actions_nei_index = actions_nei_index.reshape(actions_nei_index.shape[0], actions_nei_index.shape[1], 1)

        rewards = torch.tensor(b_rewards).float()

        next_states = torch.tensor(b_next_states).float()
        masks = torch.tensor(b_masks).float().reshape(-1, 1)


        q_1, q_2 = self.q(states, adjs)
        current_q_value_1 = q_1.gather(2, actions_pb_index).squeeze(-1)
        current_q_value_2 = q_2.gather(2, actions_nei_index).squeeze(-1)

        with torch.no_grad():
            argmax_1 = torch.argmax(self.q(next_states, next_adjs)[0], dim=2)
            argmax_2 = torch.argmax(self.q(next_states, next_adjs)[1], dim=2)

            argmax_1 = argmax_1.reshape(argmax_1.shape[0], argmax_1.shape[1], 1)
            argmax_2 = argmax_2.reshape(argmax_2.shape[0], argmax_2.shape[1], 1)

            max_next_q_1_val = self.target(next_states, next_adjs)[0].gather(2, argmax_1).squeeze(-1)
            max_next_q_2_val = self.target(next_states, next_adjs)[1].gather(2, argmax_2).squeeze(-1)

        expected_q_1_vals = rewards + max_next_q_1_val * 0.95 * masks
        expected_q_2_vals = rewards + max_next_q_2_val * 0.95 * masks
        loss = (expected_q_1_vals - current_q_value_1).pow(2).mean()+(expected_q_2_vals-current_q_value_2).pow(2).mean()

        adam.zero_grad()
        loss.backward()

        for p in self.q.parameters():
            p.grad.data.clamp_(-1., 1.)
        adam.step()

        self.update_counter += 1
        if self.update_counter % self.target_net_update_freq == 0:
            self.update_counter = 0
            self.target.load_state_dict(self.q.state_dict())


bins = 2
env = GossipEnvironment(sys.argv)

config = AgentConfig()
memory = ExperienceReplayMemory(config.memory_size)
agent = BranchingDQN(flatdim(env.observation_space), 2, 11, pow(2, 5), config)
adam = optim.Adam(agent.q.parameters(), lr=config.lr)


recap = []
total_frame = 0
p_bar = tqdm(total=config.max_frames)
start_time = time.time()
for frame in range(config.max_frames):
    s, adj, snap = env.reset()


    if frame > 100:
        config.epsilon -= 0.001
        if config.epsilon < 0.1:
            config.epsilon= 0.1
    ep_reward = 0.
    actions = []
    last_actions=[]
    for step in range(config.max_step):


        last_actions=copy.deepcopy(actions)
        actions = []
        if np.random.rand() > config.epsilon :
            raw_actions = agent.get_action(s, adj)
            for i in range(env.cmd.numNodes):
                actions.append([raw_actions[0][i].item(), raw_actions[1][i].item()])
            print("model actions is {}".format(actions))
        else:
            for i in range(env.cmd.numNodes):
                action_1 = np.random.randint(0, 11, size=1)
                action_2 = np.random.randint(0, pow(2, 5), size=1)
                actions.append([action_1.item(), action_2.item()])

        if step == 0:
            ns, next_adj, r, done, snap = env.step(copy.deepcopy(actions), is_reset=False, snap_bitgrahp=snap,
                                        is_first=True)
        else:
            ns, next_adj, r, done, snap = env.step(copy.deepcopy(actions), is_reset=False, snap_bitgrahp=snap,
                                                   is_first=False)
            for i in range(len(r)):
                r[i]+=(last_actions[i][0]-actions[i][0])*0.1

        ep_reward += np.sum(r)

        if done:

            recap.append(ep_reward)
            p_bar.set_description('Rew: {:.3f}'.format(ep_reward))
            ep_reward = 0.
            break

        memory.push((s.numpy().tolist(), actions, adj.numpy().tolist(), next_adj.numpy().tolist(), r,
                     ns.numpy().tolist(), 0. if done else 1.))

        s = ns
        adj = next_adj


    p_bar.update(1)

    if frame > config.learning_starts:
        agent.update_policy(adam, memory, config)

    if frame % 100 == 0:
        utils.save(agent, recap)

p_bar.close()
