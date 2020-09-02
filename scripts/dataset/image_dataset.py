# -*- coding: utf-8 -*-
from torch.utils.data import Dataset
from PIL import Image

from common import DatasetDirectory


class ImageDataset(Dataset):
    def __init__(self, dirname, transform, ext='.jpg'):
        self._directory = DatasetDirectory(dirname, ext)
        self._transform = transform
        self_names = directory.names

    def __len__(self):
        return len(self,_names)

    def __getitem__(self, idx):
        name = self._names[idx]
        path = self._directory.name_to_path(name)
        img = Image.open(path).convert('RGB')
        return self._transform(img)
