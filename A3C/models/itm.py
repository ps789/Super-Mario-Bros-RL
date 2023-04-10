import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def normalized_columns_initializer(weights, std=1.0):
    out = torch.randn(weights.size())
    out *= std / torch.sqrt(out.pow(2).sum(1, keepdim=True))
    return out


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = np.prod(weight_shape[1:4])
        fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)


class ITM(torch.nn.Module):
    def __init__(self, num_inputs, num_actions):
        super(ITM, self).__init__()
        self.downsample = nn.AvgPool2d(2,2) #downsample to 42*42
        self.conv1 = nn.Conv2d(num_inputs, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)

        lin1 = nn.Linear(num_actions + 288, 256)
        lin2 = nn.Linear(256, 1)

        self.apply(weights_init)
        lin1.weight.data = normalized_columns_initializer(
            lin1.weight.data, 0.01)
        lin1.bias.data.fill_(0)
        lin2.weight.data = normalized_columns_initializer(
            lin2.weight.data, 0.01)
        lin2.bias.data.fill_(0)

        self.time_estimator = nn.Sequential(lin1, nn.ReLU(), lin2)

        self.train()

    def forward(self, inputs):
        st, at = inputs

        xt = self.downsample(st)
        xt = F.elu(self.conv1(xt))
        xt = F.elu(self.conv2(xt))
        xt = F.elu(self.conv3(xt))
        xt = F.elu(self.conv4(xt))

        xt = xt.view(-1, 288)
        features = torch.cat((xt, at),1)

        return self.time_estimator(features)
