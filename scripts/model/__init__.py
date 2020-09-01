# -*- coding: utf-8 -*-
import os

import torch

from .unet import UNet
from .resnet import ResNet
from .patch_discriminator import PatchDiscriminator

# Logging
from logging import getLogger, NullHandler
logger = getLogger(__name__)
logger.addHandler(NullHandler())


def _reshape_state_dict(src, target):
    assert src.dim() == target.dim()
    for d in range(src.dim()):
        chunk = list(torch.chunk(src, src.shape[d], dim=d))
        while len(chunk) < target.shape[d]:
            chunk.extend(chunk)
        src = torch.cat(chunk[:target.shape[d]], dim=d)
    return src


def load_pretrained_model(model, pretrained_path):
    if not os.path.exists(pretrained_path):
        return

    logger.info('Load pretrained model (%s)', pretrained_path)
    src = torch.load(pretrained_path)
    dst = model.state_dict()

    state = dict()
    for k in dst.keys():
        if k not in src:
            state[k] = dst[k]
        elif src[k].shape == dst[k].shape:
            state[k] = src[k]
        else:
            state[k] = _reshape_state_dict(src[k], dst[k])

    model.load_state_dict(state)
