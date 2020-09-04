# -*- coding: utf-8 -*-
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .common import LoopIterator
from .model_writer import BestModelWriter, LocalBestModelWriter, \
                          RegularModelWriter

# Logging
from logging import getLogger, NullHandler
logger = getLogger(__name__)
logger.addHandler(NullHandler())


class CycleGanTrainer():
    def __init__(self, save_dir, train_loader_a, train_loader_b,
                 test_loader_a, test_loader_b,
                 gen_a2b, gen_b2a,
                 dis_a2b, dis_b2a, gen_optimizer=None, dis_optimizer=None,
                 label_a=None, label_b=None, device=None, interval=50,
                 evaluator=None):
        self.device = torch.device('cpu') if device is None else device
        self.train_iter_a = LoopIterator(train_loader_a)
        self.train_iter_b = LoopIterator(train_loader_b)
        self.test_iter_a = LoopIterator(test_loader_a)
        self.test_iter_b = LoopIterator(test_loader_b)
        self.gen_a2b = gen_a2b.to(self.device)
        self.gen_b2a = gen_b2a.to(self.device)
        self.dis_a2b = dis_a2b.to(self.device)
        self.dis_b2a = dis_b2a.to(self.device)
        self.gen_optimizer = gen_optimizer
        self.dis_optimizer = dis_optimizer
        self.dis_loss = torch.nn.BCELoss()
        self.gen_loss = torch.nn.L1Loss()
        self.evaluator = evaluator
        self.batch_size = train_loader_a.batch_size
        self.epoch = 0

        # Setup writer
        lbl_a = 'A' if label_a is None else label_a
        lbl_b = 'B' if label_b is None else label_b
        if dis_optimizer is None:
            self.gen_a_writer = BestModelWriter(save_dir, 'G_%s' % lbl_a)
            self.gen_b_writer = BestModelWriter(save_dir, 'G_%s' % lbl_b)
        else:
            self.gen_a_writer = LocalBestModelWriter(save_dir, 'G_%s' % lbl_a)
            self.gen_b_writer = LocalBestModelWriter(save_dir, 'G_%s' % lbl_b)
            self.reg_dis_a_writer = RegularModelWriter(
                save_dir, 'D_%s' % lbl_a, interval)
            self.reg_dis_b_writer = RegularModelWriter(
                save_dir, 'D_%s' % lbl_b, interval)
        if gen_optimizer is None:
            self.dis_a_writer = BestModelWriter(save_dir, 'D_%s' % lbl_a)
            self.dis_b_writer = BestModelWriter(save_dir, 'D_%s' % lbl_b)
        else:
            self.dis_a_writer = LocalBestModelWriter(save_dir, 'D_%s' % lbl_a)
            self.dis_b_writer = LocalBestModelWriter(save_dir, 'D_%s' % lbl_b)
            self.reg_gen_a_writer = RegularModelWriter(
                save_dir, 'G_%s' % lbl_a, interval)
            self.reg_gen_b_writer = RegularModelWriter(
                save_dir, 'G_%s' % lbl_b, interval)
        self.logger = SummaryWriter(save_dir)

        # Setup loss keys
        self._loss_keys = list()
        if gen_optimizer is not None:
            self._loss_keys.extend(['gen_loss', 'gen_gan_loss',
                                    'gen_cycle_loss', 'gen_identity_loss',
                                    'gen_a2b_loss', 'gen_b2a_loss',
                                    'gen_a2a_loss', 'gen_b2b_loss',
                                    'gen_a2b2a_loss', 'gen_b2a2b_loss'])
        if dis_optimizer is not None:
            self._loss_keys.extend(['dis_a_loss', 'dis_b_loss',
                                    'judge_a_real', 'judge_a_fake',
                                    'judge_b_real', 'judge_b_fake'])

    def _train(self):
        if self.dis_optimizer is not None:
            self.dis_a2b.train()
            self.dis_b2a.train()
        if self.gen_optimizer is not None:
            self.gen_a2b.train()
            self.gen_b2a.train()

    def _eval(self):
        self.dis_a2b.eval()
        self.dis_b2a.eval()
        self.gen_a2b.eval()
        self.gen_b2a.eval()

    def _save_models(self, loss):
        if 'gen_loss' in loss:
            self.gen_a_writer(self.gen_a2b, loss['gen_loss'])
            self.gen_b_writer(self.gen_b2a, loss['gen_loss'])
            self.reg_gen_a_writer(self.gen_a2b, self.epoch)
            self.reg_gen_b_writer(self.gen_b2a, self.epoch)
        if 'dis_a_loss' in loss:
            self.dis_a_writer(self.dis_b2a, loss['dis_a_loss'])
            self.reg_dis_a_writer(self.dis_b2a, self.epoch)
        if 'dis_b_loss' in loss:
            self.dis_b_writer(self.dis_a2b, loss['dis_b_loss'])
            self.reg_dis_b_writer(self.dis_a2b, self.epoch)

    def _forward_discriminator(self, gen_inp, dis_inp, gen_model, dis_model,
                               backward):
        self.dis_optimizer.zero_grad()

        # Forward real
        judge_real = dis_model.forward(dis_inp)
        real_label = torch.full_like(judge_real, 1.0,
                                     dtype=torch.float32,
                                     device=self.device)
        dis_real_loss = self.dis_loss(judge_real, real_label)

        if backward:
            dis_real_loss.backward()

        # Generate fake
        fake = gen_model.forward(gen_inp)

        # Forward fake
        judge_fake = dis_model.forward(fake.detach())
        fake_label = torch.full_like(judge_fake, 0.0,
                                     dtype=torch.float32,
                                     device=self.device)
        dis_fake_loss = self.dis_loss(judge_fake, fake_label)

        if backward:
            dis_fake_loss.backward()
            self.dis_optimizer.step()

        return {
            'loss': dis_fake_loss.item() + dis_real_loss.item(),
            'judge_real': judge_real.mean().item(),
            'judge_fake': judge_fake.mean().item(),
        }

    def _forward_generator(self, inp_a, inp_b, backward):
        self.gen_optimizer.zero_grad()

        # Forward A to B
        a2b = self.gen_a2b.forward(inp_a)
        judge_a2b = self.dis_a2b.forward(a2b)
        real_label = torch.full_like(judge_a2b, 1.0,
                                     dtype=torch.float32,
                                     device=self.device)
        gen_a2b_loss = self.dis_loss(judge_a2b, real_label)

        # Forward B to A
        b2a = self.gen_b2a.forward(inp_b)
        judge_b2a = self.dis_b2a.forward(b2a)
        real_label = torch.full_like(judge_b2a, 1.0,
                                     dtype=torch.float32,
                                     device=self.device)
        gen_b2a_loss = self.dis_loss(judge_b2a, real_label)

        gen_gan_loss = gen_a2b_loss + gen_b2a_loss

        # Cycle loss
        a2b2a = self.gen_b2a.forward(a2b)
        b2a2b = self.gen_a2b.forward(b2a)
        gen_a2b2a_loss = self.gen_loss(a2b2a, inp_a)
        gen_b2a2b_loss = self.gen_loss(b2a2b, inp_b)

        gen_cycle_loss = gen_a2b2a_loss + gen_b2a2b_loss

        # Identity loss
        a = self.gen_b2a.forward(inp_a)
        b = self.gen_a2b.forward(inp_b)
        gen_a2a_loss = self.gen_loss(a, inp_a)
        gen_b2b_loss = self.gen_loss(b, inp_b)

        gen_identity_loss = gen_a2a_loss + gen_b2b_loss

        gen_loss = gen_gan_loss + gen_cycle_loss + gen_identity_loss

        if backward:
            gen_loss.backward()
            self.gen_optimizer.step()

        return {
            'gen_loss': gen_loss.item(),
            'gen_gan_loss': gen_gan_loss.item(),
            'gen_cycle_loss': gen_cycle_loss.item(),
            'gen_identity_loss': gen_identity_loss.item(),
            'gen_a2b_loss': gen_a2b_loss.item(),
            'gen_b2a_loss': gen_b2a_loss.item(),
            'gen_a2b2a_loss': gen_a2b2a_loss.item(),
            'gen_b2a2b_loss': gen_b2a2b_loss.item(),
            'gen_a2a_loss': gen_a2a_loss.item(),
            'gen_b2b_loss': gen_b2b_loss.item(),
        }, (a2b.detach().cpu(), b2a.detach().cpu())

    def _forward(self, inp_a, inp_b, backward):
        a = inp_a.to(device=self.device)
        b = inp_b.to(device=self.device)

        if self.dis_optimizer is not None:
            dis_a_loss = self._forward_discriminator(b, a, self.gen_b2a,
                                                     self.dis_b2a, backward)
            dis_b_loss = self._forward_discriminator(a, b, self.gen_a2b,
                                                     self.dis_a2b, backward)

        if self.gen_optimizer is not None:
            gen_loss, preds = self._forward_generator(a, b, backward)

        return {
            **gen_loss,
            'dis_a_loss': dis_a_loss['loss'],
            'dis_b_loss': dis_b_loss['loss'],
            'judge_a_real': dis_a_loss['judge_real'],
            'judge_a_fake': dis_a_loss['judge_fake'],
            'judge_b_real': dis_b_loss['judge_real'],
            'judge_b_fake': dis_b_loss['judge_fake'],
        }, preds

    def _train_step(self, n_train):
        self._train()

        avg_loss = {k: 0.0 for k in self._loss_keys}
        for _ in tqdm(range(n_train // self.batch_size)):
            inp_a = next(self.train_iter_a)
            inp_b = next(self.train_iter_b)
            loss, _ = self._forward(inp_a, inp_b, True)
            for k, v in loss.items():
                avg_loss[k] += v

        for k in avg_loss.keys():
            avg_loss[k] = avg_loss[k] / n_train * self.batch_size
            self.logger.add_scalar('train_%s' % k, avg_loss[k], self.epoch)
            logger.info('train_%s: %f', k, avg_loss[k])

    def _test_step(self):
        self._eval()

        preds = list()
        avg_loss = {k: 0.0 for k in self._loss_keys}
        n_test = min(len(self.test_iter_a), len(self.test_iter_b))
        for _ in tqdm(range(n_test)):
            inp_a = next(self.test_iter_a)
            inp_b = next(self.test_iter_b)
            loss, pred = self._forward(inp_a, inp_b, False)
            for k, v in loss.items():
                avg_loss[k] += v

        for k in avg_loss.keys():
            avg_loss[k] = avg_loss[k] / n_test
            self.logger.add_scalar('test_%s' % k, avg_loss[k], self.epoch)
            logger.info('test_%s: %f', k, avg_loss[k])

        self.save_models(avg_loss)
        if self.evaluator is not None:
            self.evaluator(preds, self.epoch)

    def run(self, n_train, max_epoch=-1):
        while True:
            self.epoch += 1
            logger.info('Epoch: %d', self.epoch)
            self._train_step(n_train)
            self._test_step()

            if 0 < max_epoch < self.epoch:
                logger.info('Reached max epoch')
                break
