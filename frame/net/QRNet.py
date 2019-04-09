import torch.nn as nn
import torch 

# conbine with mobile net and lenet
class QRNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.layer1 = nn.Sequential(  # input_size=(3*8*8)
            nn.Conv2d(in_channels=3, out_channels=6, groups=3,
                      kernel_size=3, stride=1, padding=1),  # 6*8*8
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 6*4*4
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(6, 1, 1, 1, 0),
            nn.Conv2d(1, 12, 3, 1, 1),  # input_size=(6*4*4)，output_size=12*4*4
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 12*2*2
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(12, 12, kernal_size=2, stride=1,
                      padding=0, groups=12),  # 12*1*1
            nn.ReLU()
        )

        self.layer4 = nn.Sequential(
            nn.Linear(12, 3),
            nn.Softmax()
        )

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        # out = out.view(out.shape[0], -1)
        out = self.layer4(out)
        return out
