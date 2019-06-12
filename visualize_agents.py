from unityagents import UnityEnvironment
from pathlib import Path
import numpy as np
import torch
from models.QNet import QNet
from DQNAgent import Agent

# create environment
env = UnityEnvironment(file_name="Banana.app")
# create brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]
# obtain environment
env_info = env.reset(train_mode=True)[brain_name]
nA = brain.vector_action_space_size
state = env_info.vector_observations[0]
nS = len(state)

agent = Agent(nS=nS,
              nA=nA,
              learning_rate=0.0005,
              gamma=0.99,
              epsilon_start=1.0,
              epsilon_end=0.1,
              epsilon_decay_frames=100000,
              target_qnet_update_rate=5,
              tau=0.001,
              replay_buffer_length=500000,
              batch_size=256,
              replay_start_size=1000)


start_epoch = 0
NUM_EPOCHS = 3000

checkpoint_path = "checkpoints/DDQN_e100k_batchnorm.tar" # path to the checkpoint file

# load checkpoint if available
checkpoint_file = Path(checkpoint_path)
if checkpoint_file.is_file():
    agent.load_checkpoint(checkpoint_path, mode='eval')
agent.epsilon = 0

env_info = env.reset(train_mode=False)[brain_name] # train_mode=False makes the env run at normal speed
state = env_info.vector_observations[0]

ret = 0
while True:
    action = agent.choose_action(state)
    env_info = env.step(action)[brain_name]
    reward = env_info.rewards[0]
    ret += reward
    new_state = env_info.vector_observations[0]
    done = env_info.local_done[0]
    agent.step(state, action, reward, new_state, done)
    state = new_state
    if done:
        print("Total Reward: {:2.0f}\tEpsilon: {:5.2f}".format(ret, agent.epsilon))
        break

env.close()
