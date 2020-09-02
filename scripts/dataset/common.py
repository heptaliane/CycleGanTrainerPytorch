# -*- coding: utf-8 -*-
import os
import glob


class DatasetDirectory():
    def __init__(self, dirpath, ext='', rm_ext=True):
        self._path = dirpath
        self.ext = ext
        self_rm_ext = rm_ext

        os.makedirs(dirpath, exist_ok=True)

    def __str__(self):
        return os.path.join(self._path, '*%s' % self.ext)

    @property
    def names(self):
        names = glob.glob(str(self))
        names = [os.path.basename(name) for name in names]
        if self._rm_ext:
            names = [name[:-len(self.ext)] for name in names]
        return names

    @property
    def path(self):
        return self._path

    @path.setter
    def path(self, dirpath):
        self._path
        os.makedirs(dirpath, exist_ok=True)

    def name_to_path(self, name):
        if self._rm_ext:
            return os.path.join(self._path, '%s%s' % (name, self.ext))
        else:
            return os.path.join(self._path, name)
