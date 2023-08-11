# -*- coding:utf-8 -*-
# Created Time: Thu 05 Jul 2018 10:00:41 PM CST
# Author: Taihong Xiao <xiaotaihong@126.com>
from config import cfg

import torch
import torchvision
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, models, datasets
import numpy as np
import os
import argparse
from tqdm import tqdm


class Program(nn.Module):
    def __init__(self, cfg, gpu):
        super(Program, self).__init__()
        self.cfg = cfg
        self.attack_dims = (1, 28, 28)
        self.victim_dims = (3, 224, 224)
        self.gpu = gpu
        self.init_net()
        self.init_mask()
        self.M: torch.Tensor
        self.W = Parameter(torch.randn(self.M.shape), requires_grad=True)

    def init_net(self):
        if self.cfg.net == 'resnet50':
            self.net = models.resnet50(weights=models.resnet.ResNet50_Weights.DEFAULT)
            # mean and std for input
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(-1, 1, 1)
            std = np.array([0.229, 0.224, 0.225],dtype=np.float32).reshape(-1, 1, 1)
            self.mean = Parameter(torch.from_numpy(mean), requires_grad=False)
            self.std = Parameter(torch.from_numpy(std), requires_grad=False)

        elif self.cfg.net == 'vgg16':
            self.net = torchvision.models.vgg16(pretrained=False)
            self.net.load_state_dict(torch.load(os.path.join(self.cfg.models_dir, 'vgg16-397923af.pth')))

            # mean and std for input
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(-1, 1, 1)
            std = np.array([0.229, 0.224, 0.225],dtype=np.float32).reshape(-1, 1, 1)
            self.mean = Parameter(torch.from_numpy(mean), requires_grad=False)
            self.std = Parameter(torch.from_numpy(std), requires_grad=False)

        else:
            raise NotImplementedError()

        self.net.eval()
        for param in self.net.parameters():
            param.requires_grad = False

    def init_mask(self):
        victim_channels, victim_height, victim_width = self.victim_dims
        attack_channels, attack_height, attack_width = self.attack_dims
        height_offset = round((victim_height - attack_height) / 2)
        width_offset = round((victim_width - attack_height) / 2)

        M = torch.ones(victim_channels, victim_height, victim_width)
        M[:, height_offset:height_offset+attack_height, width_offset:width_offset+attack_width] = 0
        self.register_buffer('M', M)

    def output_mapper(self, output: torch.Tensor) -> torch.Tensor:
        return output[:,:10]

    def forward(self, image):
        victim_channels, victim_height, victim_width = self.victim_dims
        attack_channels, attack_height, attack_width = self.attack_dims
        height_offset = round((victim_height - attack_height) / 2)
        width_offset = round((victim_width - attack_height) / 2)

        image = image.repeat(1,3,1,1)
        X = torch.zeros(image.shape[0], victim_channels, victim_height, victim_width, device=image.device)
        X[:, :, height_offset:height_offset+attack_height, width_offset:width_offset+attack_width] = image.detach().clone()
        X.requires_grad_()

        P = torch.tanh(self.W * self.M)
        # P = torch.sigmoid(self.W * self.M)
        X_adv = X + P
        # X_adv = (X_adv - self.mean) / self.std
        Y_adv = self.net(X_adv)
        Y_adv = F.softmax(Y_adv, 1)
        return self.output_mapper(Y_adv)


class Adversarial_Reprogramming(object):
    def __init__(self, args, cfg=cfg):
        self.mode = args.mode
        self.gpu = args.gpu
        self.restore = args.restore
        self.cfg = cfg
        self.init_dataset()
        self.Program = Program(self.cfg, self.gpu)
        self.restore_from_file()
        self.set_mode_and_gpu()

    def init_dataset(self):
        if self.cfg.dataset == 'mnist':
            train_set = datasets.MNIST(os.path.join(self.cfg.data_dir, 'mnist'), train=True, transform=transforms.ToTensor(), download=True)
            test_set = datasets.MNIST(os.path.join(self.cfg.data_dir, 'mnist'), train=False, transform=transforms.ToTensor(), download=True)
            kwargs = {'num_workers': 1, 'pin_memory': True, 'drop_last': True}
            self.train_loader = DataLoader(train_set, batch_size=self.cfg.batch_size, shuffle=True, **kwargs)
            self.test_loader = DataLoader(test_set, batch_size=self.cfg.batch_size, shuffle=False, **kwargs)
        else:
            raise NotImplementedError()

    def restore_from_file(self):
        if self.restore is not None:
            ckpt = os.path.join(self.cfg.train_dir, 'W_%03d.pt' % self.restore)
            assert os.path.exists(ckpt)
            if self.gpu:
                self.Program.load_state_dict(torch.load(ckpt), strict=False)
            else:
                self.Program.load_state_dict(torch.load(ckpt, map_location='cpu'), strict=False)
            self.start_epoch = self.restore + 1
        else:
            self.start_epoch = 1

    def set_mode_and_gpu(self):
        if self.mode == 'train':
            # optimizer
            # self.BCE = torch.nn.BCELoss()
            self.BCE = torch.nn.CrossEntropyLoss()
            self.optimizer = torch.optim.Adam((self.Program.get_parameter('W'),), lr=self.cfg.lr, betas=(0.5, 0.999))
            self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=2, gamma=self.cfg.decay)
            if self.restore:
                self.lr_scheduler.step()
            if self.gpu:
                with torch.cuda.device(0):
                    self.BCE.cuda()
                    self.Program.cuda()

            if len(self.gpu) > 1:
                self.Program = torch.nn.DataParallel(self.Program, device_ids=list(range(len(self.gpu))))

        elif self.mode == 'validate' or self.mode == 'test':
            if self.gpu:
                with torch.cuda.device(0):
                    self.Program.cuda()

            if len(self.gpu) > 1:
                self.Program = torch.nn.DataParallel(self.Program, device_ids=list(range(len(self.gpu))))

        else:
            raise NotImplementedError()

    @property
    def get_W(self):
        for p in self.Program.parameters():
            if p.requires_grad:
                return p

    def tensor2var(self, tensor, requires_grad=False, volatile=False):
        if self.gpu:
            with torch.cuda.device(0):
                tensor = tensor.cuda()
        return Variable(tensor, requires_grad=requires_grad, volatile=volatile)

    def compute_loss(self, out, label):
        # label = torch.zeros(self.cfg.batch_size, 10).scatter_(1, label.view(-1,1), 1).to(device=label.device)
        return self.BCE(out, label.to(out.device)) + self.cfg.lmd * torch.norm(self.Program.get_parameter('W')) ** 2

    def validate(self):
        acc = 0.0
        for k, (image, label) in enumerate(self.test_loader):
            image = self.tensor2var(image)
            out = self.Program(image)
            pred = out.data.cpu().numpy().argmax(1)
            acc += sum(label.numpy() == pred) / float(len(label) * len(self.test_loader))
        print('test accuracy: %.6f' % acc)

    def train(self):
        for self.epoch in range(self.start_epoch, self.cfg.max_epoch + 1):
            for image, label in tqdm(self.train_loader):
                image = self.tensor2var(image)
                self.out = self.Program(image)
                self.loss = self.compute_loss(self.out, label)
                self.optimizer.zero_grad()
                self.loss.backward()
                self.optimizer.step()
            self.lr_scheduler.step()
            print('epoch: %03d/%03d, loss: %.6f' % (self.epoch, self.cfg.max_epoch, self.loss.data.cpu().numpy()))
            torch.save({'W': self.get_W}, os.path.join(self.cfg.train_dir, 'W_%03d.pt' % self.epoch))
            self.validate()

    def test(self):
        pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', default='train', type=str, choices=['train', 'validate', 'test'])
    parser.add_argument('-r', '--restore', default=None, action='store', type=int, help='Specify checkpoint id to restore.')
    parser.add_argument('-g', '--gpu', default=[], nargs='+', type=str, help='Specify GPU ids.')
    # test params
    print(cfg)

    args = parser.parse_args()
    # print(args)
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(args.gpu)
    AR = Adversarial_Reprogramming(args)
    if args.mode == 'train':
        AR.train()
    elif args.mode == 'validate':
        AR.validate()
    elif args.mode == 'test':
        AR.test()
    else:
        raise NotImplementedError()

if __name__ == "__main__":
    main()
