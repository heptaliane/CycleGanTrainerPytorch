#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import argparse

from torch.utils.data import DataLoader
from torch.optim import Adam

from dataset import ImageDataset, setup_image_transform
from model import create_generator, create_discriminator


def setup_dataset(config, label_a, label_b):
    train_dir_a = os.path.join(config['dataset']['train']['dirname'], label_a)
    train_dir_b = os.path.join(config['dataset']['train']['dirname'], label_b)
    test_dir_a = os.path.join(config['dataset']['test']['dirname'], label_a)
    test_dir_b = os.path.join(config['dataset']['test']['dirname'], label_b)
    assert os.path.exists(train_dir_a) and os.path.exists(train_dir_b)
    assert os.path.exists(test_dir_a) and os.path.exists(test_dir_b)

    img_size = config['img_size']
    train_transform = setup_image_transform(img_size=img_size,
                                            **config['trainsform'])
    test_transform = setup_image_transform(img_size=img_size)

    train_dataset_a = ImageDataset(train_dir_a, train_transform,
                                   config['dataset']['train']['ext'])
    train_dataset_b = ImageDataset(train_dir_b, train_transform,
                                   config['dataset']['train']['ext'])
    test_dataset_a = ImageDataset(test_dir_a, test_transform,
                                  config['dataset']['test']['ext'])
    test_dataset_b = ImageDataset(test_dir_b, test_transform,
                                  config['dataset']['test']['ext'])

    train_loader_a = DataLoader(train_dataset_a, **config['loader'],
                                shuffle=True)
    train_loader_b = DataLoader(train_dataset_b, **config['loader'],
                                shuffle=True)
    test_loader_a = DataLoader(test_dataset_a, **config['loader'])
    test_loader_b = DataLoader(test_dataset_b, **config['loader'])

    return {
        'train_loader_a': train_loader_a,
        'train_loader_b': train_loader_b,
        'test_loader_a': test_loader_a,
        'test_loader_b': test_loader_b,
    }


def setup_model(config):
    gen_conf = config['model']['generator']
    dis_conf = config['model']['discriminator']

    # Create generator
    arch = gen_conf.get('arch')
    in_ch = gen_conf.get('in_ch')
    out_ch = gen_conf.get('out_ch')

    gen_a2b = create_model(arch, in_ch, out_ch,
                           gen_conf['pretrained']['a2b'],
                           **gen_conf['kwargs'])
    gen_b2a = create_model(arch, out_ch, in_ch,
                           gen_conf['pretrained']['b2a'],
                           **gen_conf['kwargs'])

    # Create discriminator
    arch = dis_conf.get('arch')
    dis_a2b = PatchDiscriminator(arch, in_ch, dis_conf['pretrained']['a2b'],
                                 **dis_conf['kwargs'])
    dis_b2a = PatchDiscriminator(arch, out_ch, dis_conf['pretrained']['b2a'],
                                 **dis_conf['kwargs'])

    if config['optimizer']['generator'] is not None:
        gen_optimizer = Adam(gen_a2b.parameters(), gen_b2a.parameters(),
                             **config['optimizer']['generator'])
    else:
        gen_optimizer = None

    if config['optimizer']['discriminator'] is not None:
        dis_optimizer = Adam(dis_a2b.parameters(), dis_b2a.parameters(),
                             **config['optimizer']['discriminator'])
    else:
        dis_optimizer = None

    return {
        'gen_a2b': gen_a2b,
        'gen_b2a': gen_b2a,
        'dis_a': dis_b2a,
        'dis_b': dis_a2b,
        'gen_optimizer': gen_optimizer,
        'dis_optimizer': dis_optimizer,
    }
