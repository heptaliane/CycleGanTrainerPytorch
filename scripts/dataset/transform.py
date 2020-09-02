# -*- coding: utf-8 -*-
from torchvision import transforms


def setup_image_transform(img_size=None, flip=None, scale=None, jitter=None):
    trans = list()

    if flip is not None:
        if flip['horizontal']:
            trans.append(transforms.RandomHorizontalFlip())
        if flip['vertical']:
            trans.append(transforms.RandomVerticalFlip())

    
    if img_size is not None:
        if scale is not None:
            trans.append(transforms.RandomResizedCrop(img_size,
                                                      scale=scale,
                                                      ratio=(1.0, 1.0)))
        else:
            trans.append(transforms.Resize(img_size))

    if jitter is not None:
        trans.append(transforms.ColorJitter(**jitter))

    trans.append(transforms.ToTensor())

    return trans
