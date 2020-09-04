# -*- coding: utf-8 -*-
import os
import math
from collections import deque

import torch

# Logging
from logging import getLogger, NullHandler
logger = getLogger(__name__)
logger.addHandler(NullHandler())


class BestModelWriter():
    def __init__(self, save_dir, label='model'):
        self._loss = float('inf')
        self._dst_path = os.path.join(save_dir, 'best_%s.pth' % label)
        os.makedirs(save_dir, exist_ok=True)

    def load_state_dict(self, state):
        self._loss = state['loss']

    def state_dict(self):
        return dict(loss=self._loss)

    def __call__(self, model, loss):
        if loss < self._loss:
            self._loss = loss
            torch.save(model.state_dict(), self._dst_path)
            logger.info('Save best model (%s).', self._dst_path)


class LocalBestModelWriter():
    def __init__(self, save_dir, label='model', epochs=10):
        self._dst_path = os.path.join(save_dir, 'local_best_%s.pth' % label)
        self._loss = deque([float('inf')], epochs)
        os.makedirs(save_dir, exist_ok=True)

    def load_state_dict(self, state):
        self._loss = deque(state['loss'], state['epochs'])

    def state_dict(self):
        return {
            'loss': list(self._loss),
            'epochs': len(self._loss),
        }

    def __call__(self, model, loss):
        if loss < min(self._loss):
            torch.save(model.state_dict(), self._dst_path)
            logger.info('Save local best model (%s).', self._dst_path)
        self._loss.append(loss)


class RegularModelWriter():
    def __init__(self, save_dir, label='model', interval=50):
        self._name_fmt = '%s_epoch_%s.pth' % (label, '%05d')
        self._save_dir = save_dir
        self.interval = interval
        os.makedirs(save_dir, exist_ok=True)

    def __call__(self, model, epoch):
        if epoch % self.interval == 0:
            return

        x = epoch // self.interval
        log_interval = round(10 ** (round(math.log10(x) * 10) * 0.1))
        if log_interval == epoch:
            dst_path = os.path.join(self._save_dir, self._name_fmt % epoch)
            torch.save(model.state_dict(), dst_path)
            logger.info('Save model (%s).', dst_path)
