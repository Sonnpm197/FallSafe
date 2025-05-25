import torch
import torch.nn as nn
import math

class NN_Model(nn.Module):
    def __init__(self):
        super(NN_Model, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(51, 128),     # Input layer: 51 → 128
            nn.ReLU(),
            nn.Linear(128, 64),     # Hidden layer: 128 → 64
            nn.ReLU(),
            nn.Linear(64, 1),       # Output layer: 64 → 1
            nn.Sigmoid()            # Because it's binary classification
        )

    def forward(self, x):
        return self.classifier(x)

class NN_Model_NO_CONF(nn.Module):
    def __init__(self):
        super(NN_Model_NO_CONF, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(34, 128),     # Input layer: 34 → 128
            nn.ReLU(),
            nn.Linear(128, 64),     # Hidden layer: 128 → 64
            nn.ReLU(),
            nn.Linear(64, 1),       # Output layer: 64 → 1
            nn.Sigmoid()            # Because it's binary classification
        )

    def forward(self, x):
        return self.classifier(x)

import torch.nn as nn

# testing the complex model
class NN_Model_NO_CONF_2(nn.Module):
    def __init__(self):
        super(NN_Model_NO_CONF_2, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(34, 512),     # Input layer: 34 → 512
            nn.ReLU(),
            nn.Linear(512, 256),    # Hidden layer: 512 → 256
            nn.ReLU(),
            nn.Linear(256, 128),    # Hidden layer: 256 → 128
            nn.ReLU(),
            nn.Linear(128, 64),     # Hidden layer: 128 → 64
            nn.ReLU(),
            nn.Linear(64, 32),      # Hidden layer: 64 → 32
            nn.ReLU(),
            nn.Linear(32, 1),       # Output layer: 32 → 1
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.classifier(x)
