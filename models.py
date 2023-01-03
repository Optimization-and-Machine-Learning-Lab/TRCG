import torch
import torch.nn as nn
import torch.nn.functional as F


class SmallNet(nn.Module):
    def __init__(self):
        super(SmallNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 1, 3, 1)
        self.conv2 = nn.Conv2d(1, 2, 3, 1)
        self.conv3 = nn.Conv2d(2, 4, 3, 1)
        self.conv4 = nn.Conv2d(4, 10, 3, 1)

    def forward(self, x):
        x = self.conv1(x)                
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv4(x)
        x = F.max_pool2d(x, 2)

        output = F.log_softmax(x, dim=1).squeeze(2).squeeze(2)
        return output
