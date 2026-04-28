import matplotlib
import torch
import torch.nn as nn
import torch.nn.functional as F

class CancerCNN(nn.Module):
    def __init__(self):
        super(CancerCNN, self).__init__()

        # Input: 3 x 128 x 128 image
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=16,
            kernel_size=3,
            padding=1
        )

        self.conv2 = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=3,
            padding=1
        )

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # After two 2x2 pool layers:
        # 128 x 128 -> 64 x 64 -> 32 x 32
        # Final feature map: 32 channels x 32 x 32
        self.fc1 = nn.Linear(32 * 32 * 32, 64)
        self.fc2 = nn.Linear(64, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 3x128x128 -> 16x64x64
        x = self.pool(F.relu(self.conv2(x)))  # 16x64x64 -> 32x32x32

        x = torch.flatten(x, 1)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x