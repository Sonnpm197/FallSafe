import torch
import torch.nn as nn
import math

class CNN_Model(nn.Module):
    def __init__(self):
        super(CNN_Model, self).__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool = nn.AdaptiveMaxPool1d(1)
        
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))  # (B, 64, 51)
        x = torch.relu(self.bn2(self.conv2(x)))  # (B, 128, 51)
        x = self.pool(x)                         # (B, 128, 1)
        x = x.view(x.size(0), -1)                # (B, 128)
        x = torch.relu(self.fc1(x))              # (B, 64)
        x = self.fc2(x)                          # (B, 1)
        return self.sigmoid(x)

class LSTM_Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=3, dropout=0.5):
        super(LSTM_Model, self).__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_size, batch_first=True, num_layers=1)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, batch_first=True, num_layers=1)
        self.fc = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: [batch_size, seq_len=1, input_size=51]
        x, _ = self.lstm1(x)  # output: [batch, seq_len, 16]
        x, _ = self.lstm2(x)  # output: [batch, seq_len, 16]
        x = x[:, -1, :]       # take the output from the last time step
        x = self.fc(x)
        return self.sigmoid(x)

class BiLSTM_Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=3, dropout=0.5):
        super(BiLSTM_Model, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, 
                             batch_first=True, bidirectional=True, dropout=dropout)
        self.fc1 = nn.Linear(hidden_size * 2, 64)  # *2 for bidirectional
        self.fc2 = nn.Linear(64, 1)  # Final output layer
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x, _ = self.lstm(x)
        x = x[:, -1, :]  # Get the output of the last time step
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)  # Sigmoid for binary classification
        return x


def rule_fall_detection(flatten_pose, xmin, ymin, xmax, ymax):
    left_shoulder_x = flatten_pose[15]
    left_shoulder_y = flatten_pose[16]
    right_shoulder_y = flatten_pose[19]

    left_body_x = flatten_pose[33]
    left_body_y = flatten_pose[34]
    right_body_y = flatten_pose[37]

    left_foot_y = flatten_pose[46]
    right_foot_y = flatten_pose[49]

    # len_factor is the Euclidean distance between the left shoulder and left body keypoints.
    # This gives a rough estimate of the body length (i.e., the distance between the shoulder and the torso),
    # which will be used to help determine whether the body is upright or has fallen.
    len_factor = math.sqrt(((left_shoulder_y - left_body_y) ** 2 + (left_shoulder_x - left_body_x) ** 2))

    dx = int(xmax) - int(xmin)
    dy = int(ymax) - int(ymin)
    difference = dy - dx

    # Fall Detection Logic:
    # The function checks three main conditions to detect if a fall has occurred:
    # Shoulder and foot position: If the shoulder is higher than the foot position minus len_factor,
    # and if the body's position is above the foot (considering the length), it suggests an upright position.
    # Right side condition: Similar to the left side, if the right shoulder and right body are in proper position
    # relative to the foot, it suggests the person is standing.
    # Bounding box difference: If the difference between the bounding box height (dy) and width (dx) is negative, indicating a fall.
    # If any of these conditions are met, the function returns True, indicating a fall,
    # and returns the bounding box (xmin, ymin, xmax, ymax) to highlight the fallen person.
    if left_shoulder_y > left_foot_y - len_factor and left_body_y > left_foot_y - (len_factor / 2) \
            and left_shoulder_y > left_body_y - (len_factor / 2) or (
            right_shoulder_y > right_foot_y - len_factor and right_body_y > right_foot_y - (len_factor / 2) \
            and right_shoulder_y > right_body_y - (len_factor / 2)) \
            or difference < 0:
        return True, (xmin, ymin, xmax, ymax)

    return False, None

# each frame could have multiple people = poses
def Rule_Model(frame):
    height, width = frame.orig_img.shape[:2]
    for pose in frame:
        xmin, ymin, xmax, ymax = pose.boxes.xyxy.squeeze(0).tolist()

        flatten_pose = pose.keypoints.data.squeeze(0).flatten().tolist()
        for i in range(0, 51, 3):
            flatten_pose[i] /= width  # normalize x
            flatten_pose[i + 1] /= height  # normalize y
        prediction, bbox = rule_fall_detection(flatten_pose, xmin, ymin, xmax, ymax)

        if prediction:
            return prediction, bbox
    return False, None
