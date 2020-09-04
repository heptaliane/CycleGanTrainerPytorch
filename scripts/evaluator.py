# -*- coding: utf-8 -*-
import os

import torch
from torchvision.transforms import ToPILImage


class CycleGanImageEvaluator():
    def __init__(self, save_dir, interval=0):
        self.save_dir = os.path.join(save_dir, 'test_result')
        self.interval = interval
        self.offset = 0
        self.epoch = 0
        self.tensor_to_image = ToPILImage()
        os.makedirs(self.save_dir, exist_ok=True)

    def _normalize_input(self, tensor):
        tmin, tmax = float(tensor.min()), float(tensor.max())
        tensor.clamp_(min=tmin, max=tmax)
        return tensor.add_(-tmin).div(tmax - tmin + 1e-5)

    def __call__(self, a1, b1, a2, b2, epoch):
        if self.interval <= 0:
            dst_dir = os.path.join(self.save_dir, 'latest')
        elif epoch % self.interval == 0:
            dst_dir = os.path.join(self.save_dir, 'epoch_%04d' % epoch)
        else:
            return
        if self.epoch != epoch:
            os.makedirs(dst_dir, exist_ok=True)
            self.offset = 0
            self.epoch = epoch

        n, _, h, w = a1.shape
        thumb = torch.zeros((n, 3, h * 2, w * 2), dtype=torch.float32)
        thumb[:, :, 0:h, 0:w] = a1
        thumb[:, :, h:h * 2, 0:w] = b1
        thumb[:, :, 0:h, w:w * 2] = a2
        thumb[:, :, h:h * 2, w:w * 2] = b2

        dst_path = os.path.join(dst_dir, 'test_result_idx_%03d.jpg')
        for i in range(n):
            img = self.tensor_to_image(thumb[i])
            img.save(dst_path % (self.offset * n + i))
        self.offset += n
