import torch
import torch.nn as nn
from .netblock import Up_Block, Down_Block


class FuseUNet(nn.Module):
    def __init__(self, num_classes=2, learned_bilinear=False):
        super(FuseUNet, self).__init__()

        """ encode model1 """

        self.model1_down_block1 = Down_Block(3, 32)
        self.model1_maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.model1_down_block2 = Down_Block(64, 64)
        self.model1_maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.model1_down_block3 = Down_Block(128, 128)
        self.model1_maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.model1_down_block4 = Down_Block(256, 256)
        self.model1_maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.model1_down_block5 = Down_Block(512, 512)

        """ encode model2 """

        self.model2_down_block1 = Down_Block(3, 32)
        self.model2_maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.model2_down_block2 = Down_Block(32, 64)
        self.model2_maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.model2_down_block3 = Down_Block(64, 128)
        self.model2_maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.model2_down_block4 = Down_Block(128, 256)
        self.model2_maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.model2_down_block5 = Down_Block(256, 512)

        """ decode """

        self.up_block1 = Up_Block(1024, 512, 512, learned_bilinear)
        self.up_block2 = Up_Block(512, 256, 256, learned_bilinear)
        self.up_block3 = Up_Block(256, 128, 128, learned_bilinear)
        self.up_block4 = Up_Block(128, 64, 64, learned_bilinear)

        self.last_conv1 = nn.Conv2d(64, num_classes, 1, padding=0)

    def forward(self, model1_inputs, model2_inputs):
        """ encoder """
        y = self.model1_down_block1(model1_inputs)
        x = self.model2_down_block1(model2_inputs)

        y1 = torch.cat((y, x), dim=1)

        y = self.model1_maxpool1(y1)
        y = self.model1_down_block2(y)

        x = self.model2_maxpool1(x)
        x = self.model2_down_block2(x)

        y2 = torch.cat((y, x), dim=1)

        y = self.model1_maxpool2(y2)
        y = self.model1_down_block3(y)

        x = self.model2_maxpool2(x)
        x = self.model2_down_block3(x)

        y3 = torch.cat((y, x), dim=1)

        y = self.model1_maxpool3(y3)
        y = self.model1_down_block4(y)

        x = self.model2_maxpool3(x)
        x = self.model2_down_block4(x)

        y4 = torch.cat((y, x), dim=1)

        y = self.model1_maxpool4(y4)
        y = self.model1_down_block5(y)

        x = self.model2_maxpool4(x)
        x = self.model2_down_block5(x)

        y5 = torch.cat((y, x), dim=1)

        """ decoder """

        y = self.up_block1(y4, y5)
        y = self.up_block2(y3, y)
        y = self.up_block3(y2, y)
        y = self.up_block4(y1, y)
        y = self.last_conv1(y)

        return y
