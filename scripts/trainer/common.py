# -*- coding: utf-8 -*-


class LoopIterator():
    def __init__(self, loader):
        self._loader = loader
        self._iter = iter(loader)

    def __len__(self):
        return len(self._loader)

    def __next__(self):
        try:
            data = next(self._iter)
        except StopIteration:
            self._iter = iter(self._loader)
            data = next(self._iter)

        return data
