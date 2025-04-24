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