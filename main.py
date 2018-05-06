import gym
import random
import torch
import numpy as np
import hyper_params as params
from itertools import count
from gym import wrappers

from model import DQN
from utils import ReplayMemory

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def env_next_step(env, model, replay_memory, state):
    sample = random.random()

    if sample < params.EPS:
        action = env.action_space.sample()
    else:
        state_tensor = torch.FloatTensor([state]).to(device)
        action = model(state_tensor).max(1)[1].item()

    next_state, reward, done, _ = env.step(action)

    x, x_dot, theta, theta_dot = next_state
    r1 = (env.unwrapped.x_threshold - abs(x)) / env.unwrapped.x_threshold - 0.8
    r2 = (env.unwrapped.theta_threshold_radians - abs(theta)) / env.unwrapped.theta_threshold_radians - 0.5
    reward = r1 + r2

    env.render()

    if done:
        next_state = env.reset()
    else:
        replay_memory.store((list(state), action, reward, list(next_state)))
    
    return next_state, done


def optimize_dqn(model, replay_memory):
    transitions = replay_memory.sample()
    batch = list(zip(*transitions))
    state_batch = torch.FloatTensor(np.asarray(batch[0])).to(device)
    action_batch = torch.LongTensor(np.asarray(batch[1])).view(-1, 1).to(device)
    reward_batch = torch.FloatTensor(np.asarray(batch[2])).view(-1, 1).to(device)
    next_state_batch = torch.FloatTensor(np.asarray(batch[3])).to(device)

    loss = model.learn((state_batch, action_batch, reward_batch, next_state_batch))
    return loss


if __name__ == '__main__':
    best_score = 0
    
    env = gym.make('CartPole-v0')
    env = wrappers.Monitor(env, 'expt_dir', force=True, video_callable=lambda episode_id: False)

    # env = env.unwrapped
    model = DQN(env.observation_space.shape[0], env.action_space.n)
    replay_memory = ReplayMemory(params.MEMORY_SIZE, params.BATCH_SIZE)

    state = env.reset()
    print(state)
    for t in count():
        next_state, done = env_next_step(env, model, replay_memory, state)        
        state = next_state

        if replay_memory.can_sample:
            loss = optimize_dqn(model, replay_memory)
            
        ep_rewards = env.get_episode_rewards()
        if len(ep_rewards) > 0:
            if t % params.LOG_INTERVAL == 0:
                best_score = max(ep_rewards[-1], best_score)
                print('last ep: {}, best ep: {}, Loss: {:5f},'.format(
                    ep_rewards[-1], 
                    best_score,
                    loss[0].item()
                ))
