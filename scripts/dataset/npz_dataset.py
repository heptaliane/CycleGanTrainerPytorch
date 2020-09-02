# -*-coding: utf-8 -*-
import numpy as np


class NpzDataset(Dataset):
    def __init__(self, directory, transform):
        self._directory = directory
        self._names = directory.names
        self._transform = transform

    def __len__(self):
        return self._names

    def __getitem__(self, idx):
        name = self._names[idx]
        path = self._directory.name_to_path(name)
        npz = np.load(path)
        return npz.get(npz.files[0])
