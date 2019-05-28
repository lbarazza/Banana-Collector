from unityagents import UnityEnvironment
import numpy as np
import torch
from models.QNet import QNet
from DQNAgent import Agent

env = UnityEnvironment(file_name="Banana.app")
brain_name = env.brain_names[0]
brain = env.brains[brain_name]
env_info = env.reset(train_mode=True)[brain_name]
nA = brain.vector_action_space_size
state = env_info.vector_observations[0]
nS = len(state)

agent = Agent(nS=nS,
              nA=nA,
              gamma=0.99,
              epsilon_start=1.0,
              epsilon_end=0.1,
              epsilon_decay_frames=800000,
              target_qnet_update_rate=1000,
              replay_buffer_length=1000,
              batch_size=20)

NUM_EPOCHS = 3000

for epoch in range(NUM_EPOCHS):
    env_info = env.reset(train_mode=True)[brain_name] #train_mode=False makes the env run at 'normal speed'
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
            print("Epoch n. ", epoch, "   Total Reward: ", ret, "   Epsilon: ", agent.epsilon)
            break

env.close()
