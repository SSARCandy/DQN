import gym
import random
import numpy as np


class ReplayMemory:
    def __init__(self, max_size, batch_size, frame_history=4):
        super(ReplayMemory, self).__init__()
        self.max_size = max_size
        self.batch_size = batch_size
        self.frame_history = frame_history
        self.state_memory = None
        self.action_memory = None
        self.reward_memory = None
        self.pos = 0
        self.size = 0

    def store(self, transition):
        s, a, r, s_ = transition
        s, a, r = np.array([s]), np.array(a), np.array(r)
        if self.size == 0:
            self.state_memory = np.zeros((self.max_size, s.shape[0], s.shape[1]))
            self.action_memory = np.zeros(self.max_size)
            self.reward_memory = np.zeros(self.max_size)

        self.state_memory[self.pos] = s
        self.action_memory[self.pos] = a
        self.reward_memory[self.pos] = r
    
        self.pos = (self.pos + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self):
        assert self.can_sample
        candidate_idxs = random.sample(list(range(self.frame_history, self.size)), self.batch_size)
        batch_s = np.vstack([self.state_memory[idx - self.frame_history:idx]] for idx in candidate_idxs)#.reshape(self.batch_size, -1, 4, 4)
        # print(batch_s.shape)
        batch_a = self.action_memory[candidate_idxs]
        batch_r = self.reward_memory[candidate_idxs]
        batch_s_ = np.vstack([self.state_memory[idx + 1 - self.frame_history:idx + 1]] for idx in candidate_idxs)#.reshape(self.batch_size, -1, 4, 4)
        return (batch_s, batch_a, batch_r, batch_s_)

    def encode_last_state(self):
        p = self.pos - 1
        if p >= self.frame_history:
            encoded = [self.state_memory[p - self.frame_history:p]]
        elif p > 0:
            encoded = [self.state_memory[-(self.frame_history - p):]] + [self.state_memory[:p]] 
        else:
            encoded = [self.state_memory[-self.frame_history:]]
        # print(encoded)
        return encoded

    @property
    def can_sample(self):
        return self.size > self.batch_size + self.frame_history
