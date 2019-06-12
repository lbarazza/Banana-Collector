from unityagents import UnityEnvironment
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import torch
from models.QNet import QNet
from DQNAgent import Agent

# create environment
env = UnityEnvironment(file_name="Banana.app", seed=42) #seed = n
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
              # making epsilon decay in 1000000 frames will make epsilon get to 0.1 in around 200 episodes
              epsilon_decay_frames=100000,
              epsilon_decay_rate=0.99,
              target_qnet_update_rate=5,
              tau=0.001,
              replay_buffer_length=500000,
              batch_size=256,
              replay_start_size=1000,
              alpha=0,
              e=0)


start_epoch = 0
MAX_EPOCHS = 3000

checkpoint_path = "checkpoints/agent1.tar" # path to the checkopint file

# load checkpoint if available
checkpoint_file = Path(checkpoint_path)
if checkpoint_file.is_file():
    start_epoch = agent.load_checkpoint(checkpoint_path, mode='train')

# setup plot
plt.xlabel("n. Episodes")
plt.ylabel("Score")

rets = []
avg_rets = []
epoch = start_epoch
solved = False
while not solved or (epoch == MAX_EPOCHS):
    epoch+=1

    # reset environment
    env_info = env.reset(train_mode=True)[brain_name] # train_mode=True makes the env run faster to train faster
    state = env_info.vector_observations[0]

    ret = 0
    while True:
        # choose action
        action = agent.choose_action(state)
        # act and observe results
        env_info = env.step(action)[brain_name]
        reward = env_info.rewards[0]
        ret += reward
        new_state = env_info.vector_observations[0]
        done = env_info.local_done[0]
        # make the agent process the experiences
        agent.step(state, action, reward, new_state, done)
        state = new_state
        if done:
            rets.append(ret)
            # calculate mean
            if len(rets) > 100:
                avg_ret = np.mean(rets[-100:])
                avg_rets.append(avg_ret)
            else:
                avg_ret = np.mean(rets)
            if avg_ret >= 13:
                solved = True
                print("----> SOLVED <----")
            print("Epoch n. {:5d}\tTotal Reward: {:2.0f}\tAverage Ret.: {:6.2f}\tEpsilon: {:5.2f}\tn. steps: {:9d}".format(epoch, ret, avg_ret, agent.epsilon, agent.steps))
            break

    # save a checkpoint and update plot every 50 episodes or when done
    if (epoch%50==0) or solved:
        # save checkpoint
        agent.save_checkpoint(epoch, checkpoint_path)
        # plot agent's progress over time
        plt.plot(range(1, epoch-start_epoch+1), rets, color='dodgerblue')
        plt.plot(range(101, epoch-start_epoch+1), avg_rets, color='blue')
        plt.axhline(y=13, color='red')
        plt.pause(0.01)

env.close()
plt.show()
