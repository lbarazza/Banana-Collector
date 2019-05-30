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
              learning_rate=0.0005,
              gamma=0.99,
              epsilon_start=1.0,
              epsilon_end=0.1,
              epsilon_decay_frames=400000, # 800000 - epsilon will get to 0.1 around 1600 episodes
              target_qnet_update_rate=1000,
              replay_buffer_length=100000,
              batch_size=64,
              replay_start_size=1000)

NUM_EPOCHS = 3000

rets = []
for epoch in range(NUM_EPOCHS):
    env_info = env.reset(train_mode=True)[brain_name]#(not (epoch%100==0)))[brain_name] #train_mode=False makes the env run at 'normal speed'
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
            rets.append(ret)
            if len(rets) > 100:
                avg_ret = sum(rets[-100:])/100.0
            else:
                avg_ret = sum(rets)/float(len(rets))
            print("Epoch n. {:5d}\tTotal Reward: {:2.0f}\tAverage Ret.: {:6.2f}\tEpsilon: {:5.2f}\tn. steps: {:9d}".format(epoch, ret, avg_ret, agent.epsilon, agent.steps))
            if epoch%250==0:
                torch.save(agent.qnet.state_dict(), "weights/NatureQNet.pt")
            break
# https://en.wikipedia.org/wiki/File:Temp-sunspot-co2.svg
env.close()
