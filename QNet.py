import torch
import torch.nn as nn
import torch.nn.functional as F

class QNet(nn.Module):
    def __init__(self, nS, nA): #nS: state space, nA: action space
        super(QNet, self).__init__()

        self.input_layer = nn.Linear(nS, 64)
        self.h_layer1 = nn.Linear(64, 64)
        self.h_layer2 = nn.Linear(64, 64)
        self.h_layer3 = nn.Linear(64, 64)
        self.output_layer = nn.Linear(64, nA)

    def forward(self, x):

        x = F.relu(self.input_layer(x))
        x = F.relu(self.h_layer1(x))
        #x = F.relu(self.h_layer2(x))
        #x = F.relu(self.h_layer3(x))
        x = self.output_layer(x)

        return x
