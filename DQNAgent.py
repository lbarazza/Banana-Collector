from collections import deque
import random
import numpy as np
import torch
import torch.nn.functional as F
from models.QNet import QNet

class Agent:
    def __init__(self, nS, nA, learning_rate, gamma, epsilon_start, epsilon_end, epsilon_decay_frames, target_qnet_update_rate, replay_buffer_length, batch_size, replay_start_size):

        #initialize env info
        self.nS = nS
        self.nA = nA

        #initialize hyperparamters
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_rate = (epsilon_start - epsilon_end)/epsilon_decay_frames
        self.target_qnet_update_rate = target_qnet_update_rate
        self.REPLAY_BUFFER_LENGTH = replay_buffer_length
        self.BATCH_SIZE = batch_size
        self.replay_start_size = replay_start_size

        #keep track of number of n_qnet_updates so that you can update target_qnet every target_qnet_update_rate
        self.steps = 0

        #initialize Deep Q Network and target network
        self.qnet = QNet(self.nS, self.nA)
        self.target_qnet = self.qnet
        self.optimizer = torch.optim.Adam(self.qnet.parameters(), lr=learning_rate)

        #initialize replay buffer for experiences replay
        self.replay_buffer = deque(maxlen=self.REPLAY_BUFFER_LENGTH)

    # choose action based on epsilon-greedy policy if greedy=False, otherwise follow greedy policy
    def choose_action(self, state, greedy=False):
        if (random.random() > self.epsilon) or greedy:
            with torch.no_grad():
                action = torch.argmax(self.qnet(Agent.preprocess_state(state)), dim=1).item()
        else:
            action = np.random.randint(self.nA)
        return action

    # make the agent
    def step(self, state, action, reward, new_state, done):
        #update replay_buffer with new experiences
        self.replay_buffer.append((state, action, reward, new_state, done))

        #do experience replay if there are enough experiences
        if len(self.replay_buffer) >= self.replay_start_size:
            #print("Do experience replay")
            self.experience_replay()

        #check whether to update target_qnet
        if self.steps % self.target_qnet_update_rate == 0:
            #update target_qnet parameters with qnet parameters
            for target_qnet_param, qnet_param in zip(self.target_qnet.parameters(), self.qnet.parameters()):
                target_qnet_param.data.copy_(qnet_param.data)

        self.steps += 1

        # decrease epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon-self.epsilon_decay_rate)


    # do one step of experience replay
    def experience_replay(self):
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
        # by default PyTorch uses the Long datatype to refer to indeces to allow indexing of very large
        # datasets (the int datatype would only allow for elements up to 4GB)
        actions = torch.tensor(np.vstack([i[1] for i in experiences])).long()
        rewards = torch.tensor(np.vstack([i[2] for i in experiences])).float()
        new_states = torch.tensor(np.vstack([i[3] for i in experiences])).float()
        dones = torch.tensor(np.vstack([i[4] for i in experiences]).astype(np.uint8)).float()
        return states, actions, rewards, new_states, dones

    # train the agent over a batch of experiences
    def learn(self, states, actions, rewards, new_states, dones):
        #print("Learning...")
        self.optimizer.zero_grad() #shouldn't it be max instead of argmax?
        with torch.no_grad():
            td_targets = rewards + self.gamma*(torch.max(self.target_qnet(new_states), dim=1, keepdim=True)[0]).float() * (1-dones) #
        y = torch.gather(self.qnet(states), dim=1, index=actions)
        loss = F.mse_loss(y, td_targets)
        loss.backward()
        self.optimizer.step()

    # save a checkpoint of the agent
    def save_checkpoint(self, epoch, checkpoint_path):
        torch.save({
                    'epoch': epoch,
                    'epsilon': self.epsilon,
                    'qnet_state_dict': self.qnet.state_dict(),
                    'target_qnet_state_dict': self.target_qnet.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
        }, checkpoint_path)

    # load a checkpoint of the agent
    def load_checkpoint(self, checkpoint_path, mode):
        checkpoint = torch.load(checkpoint_path)
        self.epsilon = checkpoint['epsilon']
        self.qnet.load_state_dict(checkpoint['qnet_state_dict'])
        self.target_qnet.load_state_dict(checkpoint['target_qnet_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # set agent into either training or evaluation mode
        if mode == 'train':
            self.qnet.train()
            self.target_qnet.train()
        elif mode == 'eval':
            self.qnet.eval()
            self.target_qnet.eval()

        # return number of epochs elapsed so far
        return checkpoint['epoch']
