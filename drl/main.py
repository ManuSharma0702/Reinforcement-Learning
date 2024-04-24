import sys
import gym
import envs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from DDPG import DDPGagent
from utils import *
import pudb
from tqdm import tqdm
import torch

from custom_gym import *

data = pd.read_csv('C:/Users/USER/Reinforcement Learning/drl/hvac_data.csv')

x = data.dropna(axis=0, how='any')
m = x.columns
p = x[x[m[3]] == max(x[m[3]])].index.values
len(p)
max(x[m[3]])
x = x.drop(p)
def normalize(arr, t_min, t_max):
    norm_arr = []
    diff = t_max - t_min
    diff_arr = max(arr) - min(arr) 
    for i in arr:
        temp = (((i - min(arr))*diff)/diff_arr) + t_min
        norm_arr.append(temp)
    return norm_arr

env = gym.make('CustomEnv-v0')
env.pass_df(x)
agent = DDPGagent(env)
noise = OUNoise(env.action_space)
batch_size = 128
rewards = []
avg_rewards = []
efficiency = [] 
tmp = []
temps = []
avg_temps = []
comf = []
avg_comf = []
for episode in tqdm(range(50)):
    state = env.reset()
    efficiency.append(env.efficiency)
    comfort_levels = []
    temps_list = []
    noise.reset()
    episode_reward = []
    
    for step in range(48):
        action = agent.get_action(state)
      
        action = noise.get_action(action, step)
    
        new_state, reward, done, _ = env.step(action) 
        agent.memory.push(state, action, reward, new_state, done)
        
        if len(agent.memory) > batch_size:
            agent.update(batch_size)        
        
        state = new_state
        episode_reward.append(reward)

        comfort_levels.append(env.comfort)
        temps_list.append(env.temp)
        if step == 49:
            sys.stdout.write("episode: {}, reward: {}, average_reward: {} \n".format(
                episode, np.round(episode_reward, decimals=2), np.mean(rewards[-10:])
            ))
            break
    comf.append(np.mean(comfort_levels))
    avg_comf.append(np.mean(comf[-10:]))
    temps.append(np.mean(temps_list))
    avg_temps.append(np.mean(temps[-10:]))
    rewards.append(np.mean(episode_reward))
    avg_rewards.append(np.mean(rewards[-10:]))
  
    # plt.plot(rewards)
    f1 = plt.figure()
    plt.plot(avg_rewards)
    
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Average Reward per Episode')
    plt.savefig("rewards.png")
    plt.close(f1)

    import os

    f2 = plt.figure()

    tmp.append(int(comfort_levels[-1]))

    plt.plot(tmp)
    plt.xlabel('Episode')
    plt.ylabel('comfort')
    plt.savefig("comfort.png")
    plt.close(f2)

    f3 = plt.figure()
    plt.plot(avg_temps)
    plt.xlabel('Episode')
    plt.ylabel('temperatures')
    plt.savefig("temperatures.png")
    plt.close(f3)

    avg_comf = [int(x) for x in avg_comf]
    for i in avg_comf:
        if i < -10 or i > 10:
            ind = avg_comf.index(i)
            avg_comf.pop(ind)
    f4 = plt.figure()
    # print(avg_comf)
    plt.plot(avg_comf)
    plt.xlabel('Episode')
    plt.ylabel('Comfort')
    plt.savefig("Comfort.png")
    plt.close(f4)
    

    # Ensure that the "models" directory exists
    models_dir = "models"
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    torch.save(agent.get_model().state_dict(), "models/"+str(episode)+".pth")
# Plot rewards and comfort levels


# Plot efficiency
# plt.figure(figsize=(10, 5))
# efficiency = np.array([x[0] for x in efficiency if x is not None])

# # Clip the efficiency values to be within the range [0, 1]
# efficiency = np.clip(efficiency, 0, 1 - np.finfo(efficiency.dtype).eps)
# 
     

# print(max(efficiency), max(comfort_levels))