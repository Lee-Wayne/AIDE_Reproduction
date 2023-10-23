import torch
import torch.nn as nn


def UNet_up_conv_bn_relu(input_channel, output_channel, learned_bilinear=False):
    if learned_bilinear:
        return nn.Sequential(nn.ConvTranspose2d(input_channel, output_channel, kernel_size=2, stride=2),
                             nn.BatchNorm2d(output_channel),
                             nn.ReLU())
    else:
        return nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                             nn.Conv2d(input_channel, output_channel, kernel_size=3, padding=1),
                             nn.BatchNorm2d(output_channel),
                             nn.ReLU())


class Block(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(input_channel, output_channel, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(output_channel)
        self.conv2 = nn.Conv2d(output_channel, output_channel, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(output_channel)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x


class Down_Block(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(Down_Block, self).__init__()
        self.block = Block(input_channel, output_channel)

    def forward(self, x):
        x = self.block(x)
        return x


class Up_Block(nn.Module):
    def __init__(self, input_channel, output_channel, prev_channel, learned_bilinear=False):
        super(Up_Block, self).__init__()
        self.bilinear_up = UNet_up_conv_bn_relu(input_channel, output_channel, learned_bilinear)
        self.block = Block(prev_channel * 2, output_channel)

    def forward(self, pre_feature_map, x):
        x = self.bilinear_up(x)
        x = torch.cat((x, pre_feature_map), dim=1)
        x = self.block(x)
        return x
