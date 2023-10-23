from torch import nn
import torch.nn.functional as F
import torch


class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, reduction='mean', ignore_index=255):
        super(CrossEntropyLoss2d, self).__init__()
        self.crossentropy_loss = nn.CrossEntropyLoss(weight, reduction=reduction, ignore_index=ignore_index)

    def forward(self, inputs, targets):
        if len(targets.shape) > 3:
            targets = torch.argmax(targets.float(), dim=1)
        return self.crossentropy_loss(inputs, targets)


class DiceLoss(nn.Module):
    def __init__(self, epsilon=1.0, reduction='mean'):
        super(DiceLoss, self).__init__()
        self.epsilon = epsilon
        self.reduction = reduction

    def forward(self, inputs, targets):
        N = targets.size(0)
        if len(inputs.shape) > 3:
            inputs = F.softmax(inputs, dim=1)
            y = inputs[:, 1, :, :].view(N, -1).float()
        else:
            y = inputs.view(N, -1).float()
        t = targets.view(N, -1).float()
        I = torch.sum(y * t, dim=1)
        U = torch.sum(t, dim=1) + torch.sum(y, dim=1)
        loss = 1.0 - (2 * I + self.epsilon) / (U + self.epsilon)
        if self.reduction == 'mean':
            dice_loss = loss.sum() / inputs.shape[0]
        elif self.reduction == 'sum':
            dice_loss = loss.sum()
        elif self.reduction == 'none':
            dice_loss = loss
        else:
            print("ERROR")
        return dice_loss


class MultiClassDiceLoss(nn.Module):
    def __init__(self, weight=None, epsilon=1.0, reduction='mean'):
        super(MultiClassDiceLoss, self).__init__()
        self.weight = weight
        self.epsilon = epsilon
        self.reduction = reduction
        self.dice = DiceLoss(epsilon=self.epsilon, reduction=self.reduction)

    def forward(self, inputs, targets):
        inputs = F.softmax(inputs, dim=1)
        tot_loss = 0
        if len(targets.shape) > 3:
            C = targets.shape[1]
            for i in range(C):
                dice_loss = self.dice(inputs[:, i], targets[:, i])
                if self.weight is not None:
                    dice_loss *= self.weight[i]
                tot_loss += dice_loss
        else:
            tot_loss = self.dice(inputs[:, 1], targets)
        return tot_loss


class CEMDiceLoss(nn.Module):
    def __init__(self, cediceweight=None, ceclassweight=None, diceclassweight=None, reduction='mean'):
        super(CEMDiceLoss, self).__init__()
        self.cediceweight = cediceweight
        self.ceclassweight = ceclassweight
        self.diceclassweight = diceclassweight
        self.ce = CrossEntropyLoss2d(weight=ceclassweight, reduction=reduction)
        self.multidice = MultiClassDiceLoss(weight=diceclassweight, reduction=reduction)

    def forward(self, inputs, target):
        ce_loss = self.ce(inputs, target)
        multidice_loss = self.multidice(inputs, target)
        if self.cediceweight is not None:
            loss = ce_loss * self.cediceweight[0] + multidice_loss * self.cediceweight[1]
        else:
            loss = ce_loss + multidice_loss
        return loss
