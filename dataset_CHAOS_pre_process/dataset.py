import os
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import pandas as pd
import pydicom
import torch

palette = [[0], [63], [126], [189], [252]]


class CHAOS_seg(Dataset):
    def __init__(self, root, csv_file, transform=None):
        self.root = root
        img_mask = pd.read_csv(csv_file)
        self.inphase = img_mask['Inphase'].values.tolist()
        self.outphase = img_mask['Outphase'].values.tolist()
        self.masks = img_mask['Mask'].values.tolist()
        self.transform = transform

    def __getitem__(self, idx):
        inphase_img = pydicom.read_file(os.path.join(self.root, self.inphase[idx])).pixel_array
        inphase_img = Image.fromarray(inphase_img)

        if inphase_img.mode != 'RGB':
            inphase_img = inphase_img.convert('RGB')

        outphase_img = pydicom.read_file(os.path.join(self.root, self.outphase[idx])).pixel_array
        outphase_img = Image.fromarray(outphase_img)

        if outphase_img.mode != 'RGB':
            outphase_img = outphase_img.convert('RGB')

        mask = Image.open(os.path.join(self.root, self.masks[idx]))
        if mask.mode != 'L':
            mask = mask.convert('L')

        augset = {
            'augno': 4,
            'imgmodal11': inphase_img.copy(),
            'imgmodal21': outphase_img.copy(),
            'mask1': mask.copy(),
            'degree1': 0.0,
            'hflip1': 0,
            'imgmodal12': inphase_img.copy(),
            'imgmodal22': outphase_img.copy(),
            'mask2': mask.copy(),
            'degree2': 0.0,
            'hflip2': 0,
            'imgmodal13': inphase_img.copy(),
            'imgmodal23': outphase_img.copy(),
            'mask3': mask.copy(),
            'degree3': 0.0,
            'hflip3': 0,
            'imgmodal14': inphase_img.copy(),
            'imgmodal24': outphase_img.copy(),
            'mask4': mask.copy(),
            'degree4': 0.0,
            'hflip4': 0,
        }

        if self.transform:
            inphase_img, outphase_img, augset, mask = self.transform(inphase_img, outphase_img, augset, mask)

        mask_arr = np.array(mask)
        mask_arr = np.expand_dims(mask_arr, axis=2)
        mask_arr = one_hot_mask(mask_arr, palette)
        mask = mask_arr.transpose([2, 0, 1])
        mask = torch.from_numpy(mask).long()

        for i in range(augset['augno']):
            augmask_arr = np.array(augset[f'mask{i + 1}'])
            augmask_arr = np.expand_dims(augmask_arr, axis=2)
            augmask_arr = one_hot_mask(augmask_arr, palette)
            augmask = augmask_arr.transpose([2, 0, 1])
            augset[f'mask{i + 1}'] = torch.from_numpy(augmask).long()
        return inphase_img, outphase_img, augset, mask

    def __len__(self):
        return len(self.masks)


def one_hot_mask(label, label_values):
    semantic_map = []
    for color in label_values:
        equality = np.equal(label, color)
        class_map = np.all(equality, axis=-1)
        semantic_map.append(class_map.astype(np.uint8))
    semantic_map = np.stack(semantic_map, axis=-1)
    return semantic_map
