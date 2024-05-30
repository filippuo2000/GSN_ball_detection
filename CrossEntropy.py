import torch.nn as nn
import torch

class CrossEntropy(nn.Module):
    def __init__(self, epsilon=1e-10, size = (360, 640)):
        super().__init__()
        self.epsilon = epsilon
        self.size = size[0] * size [1]

    def forward(self, output, target):
        output = torch.clamp(output, self.epsilon, 1-self.epsilon)
        loss = -torch.sum(target * torch.log(output))
        loss = loss / (self.size * output.size(0))
        return loss