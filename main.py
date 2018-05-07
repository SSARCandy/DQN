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

def env_next_step(env, model, replay_memory):
    sample = random.random()

    if sample < params.EPS or not replay_memory.can_sample:
        action = env.action_space.sample()
    else:
        state_ = replay_memory.encode_last_state()
        state_tensor = torch.FloatTensor([state_]).to(device)
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
    
    return next_state, reward, done


def optimize_dqn(model, replay_memory):
    if not replay_memory.can_sample:
        return None
    
    batch_s, batch_a, batch_r, batch_s_ = replay_memory.sample()
    state_batch = torch.FloatTensor(batch_s).to(device)
    action_batch = torch.LongTensor(batch_a).view(-1, 1).to(device)
    reward_batch = torch.FloatTensor(batch_r).view(-1, 1).to(device)
    next_state_batch = torch.FloatTensor(batch_s_).to(device)

    loss = model.learn((state_batch, action_batch, reward_batch, next_state_batch))
    return loss


if __name__ == '__main__':
    best_score = 0
    
    # env = gym.make('BreakoutNoFrameskip-v4')
    env = gym.make('CartPole-v0')
    env = wrappers.Monitor(env, 'expt_dir', force=True, video_callable=lambda episode_id: False)

    # env = env.unwrapped
    w, = env.observation_space.shape
    in_channels = w*params.FRAME_HISTORY_LEN
    model = DQN(in_channels, env.action_space.n)
    replay_memory = ReplayMemory(params.MEMORY_SIZE, params.BATCH_SIZE, frame_history=params.FRAME_HISTORY_LEN)

    state = env.reset()
    # print(state)
    for t in count():
        state, reward, done = env_next_step(env, model, replay_memory)        

        loss = optimize_dqn(model, replay_memory)
            
        ep_rewards = env.get_episode_rewards()
        if len(ep_rewards) > 0 and t % params.LOG_INTERVAL == 0:
            best_score = max(ep_rewards[-1], best_score)
            print('ep: {}, best ep: {}, last ep: {}, Loss: {:5f},'.format(
                len(ep_rewards),
                best_score,
                ep_rewards[-1], 
                loss[0].item()
            ))
