import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append("game/")


class Net(nn.Module):
    # 定义一系列常数，其中，epsilon为每周期随机输出一个动作的概率
    ACTIONS = 2  # 有效输出动作的个数

    def __init__(self):
        super(Net, self).__init__()
        # 第一层卷积，从4通道到32通道，窗口大小8，跳跃间隔4，填空白2
        self.conv1 = nn.Conv2d(4, 32, 8, 4, padding=2)
        # Pooling层，窗口2*2
        self.pool = nn.MaxPool2d(2, 2)
        # 第二层卷积，从32通道到64通道，窗口大小4，跳跃间隔2，填空白1
        self.conv2 = nn.Conv2d(32, 64, 4, 2, padding=1)
        # 第二个Pooling层，窗口2＊2，空白1
        # self.pool2 = nn.MaxPool2d(2, 2, padding = 1)
        # 第三层卷积层，输入输出通道都是64，填空白为1
        self.conv3 = nn.Conv2d(64, 64, 3, 1, padding=1)

        # 最后有两层全链接层
        self.fc_sz = 1600
        self.fc1 = nn.Linear(self.fc_sz, 256)
        self.fc2 = nn.Linear(256, Net.ACTIONS)

    def forward(self, x):
        # 输入为一个batch的数据，每一个为前后相连的4张图像，每个图像为80*80的大小
        # x的尺寸为：batch_size, 4, 80, 80
        x = self.conv1(x)
        # x的尺寸为：batch_size, 32, 20, 20
        x = F.relu(x)
        x = self.pool(x)
        # x的尺寸为：batch_size, 32, 10, 10
        x = F.relu(self.conv2(x))
        # x的尺寸为：batch_size, 64, 5, 5
        # x = self.pool2(x)
        x = F.relu(self.conv3(x))
        # x的尺寸为：batch_size, 64, 5, 5
        # x = self.pool2(x)
        # 将x设为1600维的向量, batch_size, 1600
        x = x.view(-1, self.fc_sz)
        x = F.relu(self.fc1(x))
        readout = self.fc2(x)
        return readout, x

    def init(self):
        # 初始化所有的网络权重
        self.conv1.weight.data = torch.abs(0.01 * torch.randn(self.conv1.weight.size()))
        self.conv2.weight.data = torch.abs(0.01 * torch.randn(self.conv2.weight.size()))
        self.conv3.weight.data = torch.abs(0.01 * torch.randn(self.conv3.weight.size()))
        self.fc1.weight.data = torch.abs(0.01 * torch.randn(self.fc1.weight.size()))
        self.fc2.weight.data = torch.abs(0.01 * torch.randn(self.fc2.weight.size()))
        self.conv1.bias.data = torch.ones(self.conv1.bias.size()) * 0.01
        self.conv2.bias.data = torch.ones(self.conv2.bias.size()) * 0.01
        self.conv3.bias.data = torch.ones(self.conv3.bias.size()) * 0.01
        self.fc1.bias.data = torch.ones(self.fc1.bias.size()) * 0.01
        self.fc2.bias.data = torch.ones(self.fc2.bias.size()) * 0.01
