import torch.nn.functional as F
import torch
import numpy as np
np.seterr(divide='ignore', invalid='ignore')

def Dice_fn(inputs, targets, threshold=0.5):
    inputs = F.softmax(inputs, dim=1)
    inputs = inputs[:, 1, :, :]
    inputs[inputs >= threshold] = 1
    inputs[inputs < threshold] = 0
    dice = torch.tensor(0.0, device="cuda" if torch.cuda.is_available() else "cpu")
    img_count = 0
    for input_, target_ in zip(inputs, targets):
        iflat = input_.view(-1).float()
        tflat = target_.view(-1).float()
        intersection = (iflat * tflat).sum()
        if tflat.sum() == 0:
            if iflat.sum() == 0:
                dice_single = torch.tensor(1.0)
            else:
                dice_single = torch.tensor(0.0)
                img_count += 1
        else:
            dice_single = ((2. * intersection) / (iflat.sum() + tflat.sum()))
            img_count += 1
        dice += dice_single
    return dice