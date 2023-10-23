from PIL import Image
import numpy as np
import torch


class Compose():
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, img1, img2, augset, mask):
        for t in self.transform:
            img1, img2, augset, mask = t(img1, img2, augset, mask)
        return img1, img2, augset, mask


class Resize():
    def __init__(self, size):
        self.size = size

    def __call__(self, img1, img2, augset, mask):
        assert len(self.size) == 2
        ow, oh = self.size
        augset['imgmodal11'] = augset['imgmodal11'].resize((ow, oh), Image.BILINEAR)
        augset['imgmodal21'] = augset['imgmodal21'].resize((ow, oh), Image.BILINEAR)
        augset['mask1'] = augset['mask1'].resize((ow, oh), Image.NEAREST)

        augset['imgmodal12'] = augset['imgmodal12'].resize((ow, oh), Image.BILINEAR)
        augset['imgmodal22'] = augset['imgmodal22'].resize((ow, oh), Image.BILINEAR)
        augset['mask2'] = augset['mask2'].resize((ow, oh), Image.NEAREST)

        augset['imgmodal13'] = augset['imgmodal13'].resize((ow, oh), Image.BILINEAR)
        augset['imgmodal23'] = augset['imgmodal23'].resize((ow, oh), Image.BILINEAR)
        augset['mask3'] = augset['mask3'].resize((ow, oh), Image.NEAREST)

        augset['imgmodal14'] = augset['imgmodal14'].resize((ow, oh), Image.BILINEAR)
        augset['imgmodal24'] = augset['imgmodal24'].resize((ow, oh), Image.BILINEAR)
        augset['mask4'] = augset['mask4'].resize((ow, oh), Image.NEAREST)

        return img1.resize((ow, oh), Image.BILINEAR), img2.resize((ow, oh), Image.BILINEAR), \
            augset, mask.resize((ow, oh), Image.BILINEAR)


class ToTensor():
    def __call__(self, img1, img2, augset, mask):
        img1 = torch.from_numpy(np.array(img1).transpose(2, 0, 1)).float() / 255.0
        img2 = torch.from_numpy(np.array(img2).transpose(2, 0, 1)).float() / 255.0
        mask = np.array(mask)
        mask = torch.from_numpy(mask)

        augset['imgmodal11'] = torch.from_numpy(np.array(augset['imgmodal11']).transpose(2, 0, 1)).float() / 255.0
        augset['imgmodal21'] = torch.from_numpy(np.array(augset['imgmodal21']).transpose(2, 0, 1)).float() / 255.0
        augset['mask1'] = torch.from_numpy(np.array(augset['mask1']))

        augset['imgmodal12'] = torch.from_numpy(np.array(augset['imgmodal12']).transpose(2, 0, 1)).float() / 255.0
        augset['imgmodal22'] = torch.from_numpy(np.array(augset['imgmodal22']).transpose(2, 0, 1)).float() / 255.0
        augset['mask2'] = torch.from_numpy(np.array(augset['mask2']))

        augset['imgmodal13'] = torch.from_numpy(np.array(augset['imgmodal13']).transpose(2, 0, 1)).float() / 255.0
        augset['imgmodal23'] = torch.from_numpy(np.array(augset['imgmodal23']).transpose(2, 0, 1)).float() / 255.0
        augset['mask3'] = torch.from_numpy(np.array(augset['mask3']))

        augset['imgmodal14'] = torch.from_numpy(np.array(augset['imgmodal14']).transpose(2, 0, 1)).float() / 255.0
        augset['imgmodal24'] = torch.from_numpy(np.array(augset['imgmodal24']).transpose(2, 0, 1)).float() / 255.0
        augset['mask4'] = torch.from_numpy(np.array(augset['mask4']))

        return img1, img2, augset, mask


class Normalize():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, img1, img2, augset, mask):
        if self.mean is None:
            img1mean = img1.mean(dim=(1, 2)).unsqueeze(1).unsqueeze(2)
            img1std = img1.std(dim=(1, 2)).unsqueeze(1).unsqueeze(2)
            img2mean = img2.mean(dim=(1, 2)).unsqueeze(1).unsqueeze(2)
            img2std = img2.std(dim=(1, 2)).unsqueeze(1).unsqueeze(2)
        else:
            img1mean = torch.FloatTensor(self.mean).unsqueeze(1).unsqueeze(2)
            img2mean = img1mean
            img1std = torch.FloatTensor(self.std).unsqueeze(1).unsqueeze(2)
            img2std = img1std

        img1 = img1.sub(img1mean).div(img1std)
        img2 = img2.sub(img2mean).div(img2std)

        augset['imgmodal11'] = augset['imgmodal11'].sub(img1mean).div(img1std)
        augset['imgmodal21'] = augset['imgmodal21'].sub(img2mean).div(img2std)

        augset['imgmodal12'] = augset['imgmodal12'].sub(img1mean).div(img1std)
        augset['imgmodal22'] = augset['imgmodal22'].sub(img2mean).div(img2std)

        augset['imgmodal13'] = augset['imgmodal13'].sub(img1mean).div(img1std)
        augset['imgmodal23'] = augset['imgmodal23'].sub(img2mean).div(img2std)

        augset['imgmodal14'] = augset['imgmodal14'].sub(img1mean).div(img1std)
        augset['imgmodal24'] = augset['imgmodal24'].sub(img2mean).div(img2std)

        return img1, img2, augset, mask