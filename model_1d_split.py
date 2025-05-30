import torch.nn as nn
import torch.nn.functional as F


class Simple1DCNN(nn.Module):
    def __init__(self, num_classes):
        super(Simple1DCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=2, out_channels=16, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm1d(16)

        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm1d(32)
        
        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm1d(64)

        self.fc = nn.Linear(64 * 512, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x