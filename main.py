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
              epsilon_decay_frames=1000,
              target_qnet_update_rate=4,
              replay_buffer_length=10,
              batch_size=3)

env_info = env.reset(train_mode=False)[brain_name] #train_mode=False makes the env run at 'normal speed'
state = env_info.vector_observations[0]

print(state)
print()

t = QNet(nS, nA)
a = torch.argmax(t(Agent.preprocess_state(state)), dim=1)
print("a: ", a)
print("a.item()", a.item())

while True:
    action = agent.choose_action(state)
    env_info = env.step(action)[brain_name]
    reward = env_info.rewards[0]
    new_state = env_info.vector_observations[0]
    done = env_info.local_done[0]
    agent.step(state, action, reward, new_state, done)
    state = new_state
    if done:
        break

env.close()
