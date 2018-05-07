import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import hyper_params as params


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQN(nn.Module):
    def __init__(self, in_channel, actions):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_channel, 128), # 32*4*1*4
            nn.ReLU(),
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Linear(512, actions)
        ).to(device)
        self.target_net = nn.Sequential(
            nn.Linear(in_channel, 128),
            nn.ReLU(),
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Linear(512, actions)
        ).to(device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=params.LR)
        self.update = 0

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.net(x)

    def learn(self, batches):
        state_batch, action_batch, reward_batch, next_state_batch = batches
        state_batch = state_batch.view(state_batch.size(0), -1)
        next_state_batch = state_batch.view(next_state_batch.size(0), -1)

        q_value = self.net(state_batch).gather(1, action_batch.view(-1, 1))
        # print(q_value.shape)
        # q_value = q_value.gather(1, action_batch.view(-1, 1))
        # assert False
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
        


            # nn.Conv2d(in_channel, 32, kernel_size=8, stride=4), # 32*4*84*84
            # nn.ReLU(),
            # nn.Conv2d(32, 64, kernel_size=4, stride=2), # 32*32*20*20
            # nn.ReLU(),
            # nn.Conv2d(64, 64, kernel_size=3, stride=1), # 32*64*9*9
            # nn.ReLU(),
            # nn.Linear(7*7*64, 512), # 32*64*7*7
            # nn.ReLU(),
            # nn.Linear(512, actions)