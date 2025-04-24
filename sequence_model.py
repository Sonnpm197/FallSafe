import torch
import torch.nn as nn

class LSTM_Model(nn.Module):
    def __init__(self, input_dim=51, hidden_dim=64, num_layers=2, num_classes=2):
        super(LSTM_Model, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)  # hn shape: (num_layers, batch, hidden_dim)
        out = self.fc(hn[-1])      # use last layer's hidden state
        return out
