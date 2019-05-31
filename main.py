from unityagents import UnityEnvironment
import matplotlib.pyplot as plt
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
              # making epsilon decay in 4000000 frames will make epsilon get to 0.1 in around 800 episodes
              epsilon_decay_frames=400000,
              target_qnet_update_rate=1000,
              replay_buffer_length=100000,
              batch_size=64,
              replay_start_size=1000)


start_epoch = 0
NUM_EPOCHS = 3000

checkpoint_path = "weights/NatureQNet.tar"

# load checkpoint if available
checkpoint_file = Path(checkpoint_path)
if checkpoint_file.is_file():
    start_epoch = agent.load_checkpoint(checkpoint_path, mode='train')

rets = []
for epoch in range(start_epoch, NUM_EPOCHS):
    env_info = env.reset(train_mode=True)[brain_name] # train_mode=False makes the env run at 'normal speed'
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
                #avg_ret = sum(rets[-100:])/100.0
                avg_ret = np.mean(rets[-100:])
            else:
                #avg_ret = sum(rets)/float(len(rets))
                avg_ret = np.mean(rets)
            print("Epoch n. {:5d}\tTotal Reward: {:2.0f}\tAverage Ret.: {:6.2f}\tEpsilon: {:5.2f}\tn. steps: {:9d}".format(epoch, ret, avg_ret, agent.epsilon, agent.steps))
            break
    if epoch%25==0:
        # save checkpoint
        agent.save_checkpoint(epoch, checkpoint_path)
        # plot agent's progress over time
        plt.plot(range(epoch-start_epoch+1), rets)
        plt.pause(0.05)

env.close()
plt.show()
