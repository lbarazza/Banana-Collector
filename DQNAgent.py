from collections import deque
import random
import numpy as np
import torch
import torch.nn.functional as F
from models.QNet import QNet

class Agent:
    def __init__(self, nS, nA, gamma, epsilon_start, epsilon_end, epsilon_decay_frames, target_qnet_update_rate, replay_buffer_length, batch_size):

        #initialize env info
        self.nS = nS
        self.nA = nA

        #initialize hyperparamters
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_rate = (epsilon_start - epsilon_end)/epsilon_decay_frames
        self.target_qnet_update_rate = target_qnet_update_rate
        self.REPLAY_BUFFER_LENGTH = replay_buffer_length
        self.BATCH_SIZE = batch_size

        #keep track of number of steps so that you can update target_qnet every target_qnet_update_rate
        self.steps = 0

        #initialize Deep Q Network and target network
        self.qnet = QNet(self.nS, self.nA)
        self.target_qnet = self.qnet
        self.optimizer = torch.optim.Adam(self.qnet.parameters(), lr=0.001)

        #initialize replay buffer for experiences replay
        self.replay_buffer = deque(maxlen=self.REPLAY_BUFFER_LENGTH)

    # choose action based on epsilon-greedy policy if greedy=False, otherwise follow greedy policy
    def choose_action(self, state, greedy=False):
        if (random.random() > self.epsilon) or greedy:
            action = torch.argmax(self.qnet(Agent.preprocess_state(state)), dim=1).item()
        else:
            action = np.random.randint(self.nA)
        return action
        #p = 1-self.epsilon if not greedy else 1
        #return np.random.choice([action, np.random.randint(self.nA)], p=[p, 1-p])

    #
    def step(self, state, action, reward, new_state, done):
        #update replay_buffer with new experiences
        self.replay_buffer.append((state, action, reward, new_state, done))

        #do experience replay if there are enough experiences
        if len(self.replay_buffer) == self.REPLAY_BUFFER_LENGTH:
            self.experience_replay()

        #check whether to update target_qnet
        if self.steps % self.target_qnet_update_rate == 0:
            #update target_qnet parameters with qnet parameters
            self.target_qnet = self.qnet

        self.steps += 1
        # decrease epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon-self.epsilon_decay_rate)


    # do one step of experience replay
    def experience_replay(self):
        #experiences_batch = np.random.choice(range(0, self.replay_buffer_length), self.BATCH_SIZE)
        #list(self.replay_buffer)
        experiences_batch = random.sample(list(self.replay_buffer), self.BATCH_SIZE)
        states, actions, rewards, new_states, dones = Agent.preprocess_experiences(experiences_batch)
        self.learn(states, actions, rewards, new_states, dones)

    # turn state into a tensor, add batch dimension to feed it into the model and cast it to a
    # FloatTensor as that is the default type for weights and biases in the nn Module (better than
    # casting entire model to double as double operations are slow on GPUs)
    @staticmethod
    def preprocess_state(state):
        return torch.unsqueeze(torch.tensor(state).float(), 0)

    # extract all states, actions, rewards and new states from tuple and return a separate tensor for each
    @staticmethod
    def preprocess_experiences(experiences):
        states = torch.tensor(np.vstack([i[0] for i in experiences])).float()
        # we are going to use the actions variable as index to the gather function in learn(),
        # by default PyTorch uses the Long datatype to refer to indeces to allow inexing of very large
        # datasets (the int datatype would only allow for elements up to 4GB)
        actions = torch.tensor(np.vstack([i[1] for i in experiences])).long()
        rewards = torch.tensor(np.vstack([i[2] for i in experiences])).float()
        new_states = torch.tensor(np.vstack([i[3] for i in experiences])).float()
        dones = torch.tensor(np.vstack([i[4] for i in experiences]).astype(np.uint8)).float()
        return states, actions, rewards, new_states, dones

    # train the agent over a batch of experiences
    def learn(self, states, actions, rewards, new_states, dones):
        self.optimizer.zero_grad()
        td_targets = rewards + self.gamma*(torch.argmax(self.target_qnet(new_states), dim=1, keepdim=True)).float() * (1-dones)
        y = torch.gather(self.qnet(states), dim=1, index=actions)
        loss = F.mse_loss(y, td_targets)
        loss.backward()
        self.optimizer.step()
