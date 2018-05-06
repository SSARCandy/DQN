import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import hyper_params as params


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQN(nn.Module):
    def __init__(self, in_channel, action):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_channel, 128),
            nn.ReLU(),
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Linear(512, action)
        ).to(device)
        self.target_net = nn.Sequential(
            nn.Linear(in_channel, 128),
            nn.ReLU(),
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Linear(512, action)            
        ).to(device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=params.LR)
        self.update = 0

    def forward(self, x):
        x = self.net(x)
        return x

    def learn(self, batches):
        state_batch, action_batch, reward_batch, next_state_batch = batches
        q_value = self.net(state_batch).gather(1, action_batch)
        next_q_value = self.target_net(next_state_batch).max(1)[0].detach().view(-1, 1)
        expected_q_value = reward_batch + (next_q_value * params.GAMMA)

        loss = F.mse_loss(q_value, expected_q_value)
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        self.update += 1
        if self.update % params.TARGET_UPDATE_INTERVAL == 0:
            self.target_net.load_state_dict(self.net.state_dict())
        
        return loss
        
