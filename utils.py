import gym
import random
import numpy as np


class ReplayMemory:
    def __init__(self, size, batch_size, frame_history=4):
        super(ReplayMemory, self).__init__()
        self.size = size
        self.batch_size = batch_size
        self.frame_history = frame_history
        self.memory = []
        self.pos = 0

    def store(self, transition):
        if len(self.memory) < self.size:
            self.memory.append(None)
        self.memory[self.pos] = transition
        self.pos = (self.pos + 1) % self.size
        # print(transition)

    def sample(self):
        return random.sample(self.memory, self.batch_size)

    def __len__(self):
        return len(self.memory)

    @property
    def can_sample(self):
        return len(self.memory) > self.batch_size