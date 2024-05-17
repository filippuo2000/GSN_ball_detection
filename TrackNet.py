import torch
import torch.nn as nn
import torch.nn.functional as F


class TrackNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(128)

        self.conv5 = nn.Conv2d(128, 256, 3, 1, 1)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, 3, 1, 1)
        self.bn6 = nn.BatchNorm2d(256)
        self.conv7 = nn.Conv2d(256, 256, 3, 1, 1)
        self.bn7 = nn.BatchNorm2d(256)

        self.conv8 = nn.Conv2d(256, 512, 3, 1, 1)
        self.bn8 = nn.BatchNorm2d(512)
        self.conv9 = nn.Conv2d(512, 512, 3, 1, 1)
        self.bn9 = nn.BatchNorm2d(512)
        self.conv10 = nn.Conv2d(512, 512, 3, 1, 1)
        self.bn10 = nn.BatchNorm2d(512)

        self.conv11 = nn.Conv2d(512, 512, 3, 1, 1)
        self.bn11 = nn.BatchNorm2d(512)
        self.conv12 = nn.Conv2d(512, 512, 3, 1, 1)
        self.bn12 = nn.BatchNorm2d(512)
        self.conv13 = nn.Conv2d(512, 512, 3, 1, 1)
        self.bn13 = nn.BatchNorm2d(512)

        self.conv14 = nn.Conv2d(512, 128, 3, 1, 1)
        self.bn14 = nn.BatchNorm2d(128)
        self.conv15 = nn.Conv2d(128, 128, 3, 1, 1)
        self.bn15 = nn.BatchNorm2d(128)

        self.conv16 = nn.Conv2d(128, 64, 3, 1, 1)
        self.bn16 = nn.BatchNorm2d(64)
        self.conv17 = nn.Conv2d(64, 64, 3, 1, 1)
        self.bn17 = nn.BatchNorm2d(64)
        self.conv18 = nn.Conv2d(64, 256, 3, 1, 1)
        self.bn18 = nn.BatchNorm2d(256)


    def forward(self, x):
        # thing about BN(Relu) vs Relu(BN)
        x = self.bn1(F.relu(self.conv1(x)))
        x = self.bn2(F.relu(self.conv2(x)))
        x = F.max_pool2d(x, 2, 2)
        x = self.bn3(F.relu(self.conv3(x)))
        x = self.bn4(F.relu(self.conv4(x)))
        x = F.max_pool2d(x, 2, 2)
        x = self.bn5(F.relu(self.conv5(x)))
        x = self.bn6(F.relu(self.conv6(x)))
        x = self.bn7(F.relu(self.conv7(x)))
        x = F.max_pool2d(x, 2, 2)
        x = self.bn8(F.relu(self.conv8(x)))
        x = self.bn9(F.relu(self.conv9(x)))
        x = self.bn10(F.relu(self.conv10(x)))
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.bn11(F.relu(self.conv11(x)))
        x = self.bn12(F.relu(self.conv12(x)))
        x = self.bn13(F.relu(self.conv13(x)))
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.bn14(F.relu(self.conv14(x)))
        x = self.bn15(F.relu(self.conv15(x)))
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.bn16(F.relu(self.conv16(x)))
        x = self.bn17(F.relu(self.conv17(x)))
        x = self.bn18(F.relu(self.conv18(x)))
        if not self.training:
            x = F.softmax(x, dim=1) #softmax along channels (256 chanels)

        return x