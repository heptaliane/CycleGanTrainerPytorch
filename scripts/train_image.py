#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import time
import argparse

from torch.utils.data import DataLoader
from torch.optim import Adam

from config import load_config
from common import write_json
from dataset import ImageDataset, setup_image_transform
from model import create_generator, create_discriminator
from trainer import CycleGanTrainer
from evaluator import CycleGanImageEvaluator

# Logging
from logging import getLogger, StreamHandler, INFO
logger = getLogger()
logger.setLevel(INFO)
logger.addHandler(StreamHandler())


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', default='config/image_default.json',
                        help='Path to configuration file')
    parser.add_argument('--output_dir', '--output', '--out', '-o',
                        default='result',
                        help='Path to output directory')
    parser.add_argument('--gpu', '-g', type=int, default=None,
                        help='GPU id (default is cpu)')
    parser.add_argument('--labels', '-l', required=True, nargs=2,
                        help='Training dataset label')
    parser.add_argument('--max_epoch', '-m', type=int, default=-1,
                        help='When the epoch reach this value, stop training,')
    args = parser.parse_args()
    return args


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

    gen_a2b = create_generator(arch, in_ch, out_ch,
                               gen_conf['pretrained']['a2b'],
                               **gen_conf['kwargs'])
    gen_b2a = create_generator(arch, out_ch, in_ch,
                               gen_conf['pretrained']['b2a'],
                               **gen_conf['kwargs'])

    # Create discriminator
    arch = dis_conf.get('arch')
    dis_a2b = create_discriminator(arch, in_ch, dis_conf['pretrained']['a2b'],
                                   **dis_conf['kwargs'])
    dis_b2a = create_discriminator(arch, out_ch, dis_conf['pretrained']['b2a'],
                                   **dis_conf['kwargs'])

    if config['optimizer']['generator'] is not None:
        gen_optimizer = Adam([*gen_a2b.parameters(), *gen_b2a.parameters()],
                             **config['optimizer']['generator'])
    else:
        gen_optimizer = None

    if config['optimizer']['discriminator'] is not None:
        dis_optimizer = Adam([*dis_a2b.parameters(), *dis_b2a.parameters()],
                             **config['optimizer']['discriminator'])
    else:
        dis_optimizer = None

    return {
        'gen_a2b': gen_a2b,
        'gen_b2a': gen_b2a,
        'dis_a2b': dis_a2b,
        'dis_b2a': dis_b2a,
        'gen_optimizer': gen_optimizer,
        'dis_optimizer': dis_optimizer,
    }


def setup_trainer(config, save_dir, device, datasets, models, labels):
    evaluator = CycleGanImageEvaluator(save_dir,
                                       config['save_interval']['evaluate'])
    interval = config['save_interval']['model']
    trainer = CycleGanTrainer(save_dir, **datasets, **models,
                              label_a=labels[0], label_b=labels[1],
                              device=device, evaluator=evaluator,
                              interval=interval)

    return trainer


def main(argv):
    args = parse_arguments(argv)
    config = load_config(args.config)

    ctime = time.strftime('%y%m%d_%H%M')
    dst_dir = os.path.join(args.output_dir, '%s_%s2%s' % (ctime, *args.labels))
    os.makedirs(dst_dir, exist_ok=True)
    write_json(os.path.join(dst_dir, 'config.json'), config)

    datasets = setup_dataset(config, args.labels[0], args.labels[1])
    models = setup_model(config)
    trainer = setup_trainer(config, dst_dir, args.gpu,
                            datasets, models, args.labels)
    trainer.run(1000, args.max_epoch)


if __name__ == '__main__':
    main(sys.argv[1:])
