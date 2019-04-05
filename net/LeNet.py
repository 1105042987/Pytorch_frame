import torch
import torch.nn as nn

class LeNet(nn.Module):

    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Sequential(  # input_size=(3*32*32)
            nn.Conv2d(in_channels=3, out_channels=6,
                      kernel_size=5, stride=1, padding=0),  # 6*28*28
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 6*14*14
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 12, 5, 1, 0),  # 12*10*10
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 12*5*5
        )

        self.fc1 = nn.Sequential(
            nn.Linear(300, 120),
            nn.ReLU()
        )

        self.fc2 = nn.Sequential(
            nn.Linear(120, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = out.view(out.shape[0], -1)
        out = self.fc1(out)
        out = self.fc2(out)
        return out