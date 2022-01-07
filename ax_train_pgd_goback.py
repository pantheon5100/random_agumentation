import argparse
import copy
import logging
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from apex import amp
import random
import contextlib

from networks.preact_resnet import PreActResNet18
from networks.resnet import *
from networks.wideresnet import WideResNet
from utils_zk import (upper_limit, lower_limit, clamp, get_loaders, attack_pgd, evaluate_pgd, evaluate_standard, std)
from train_utils import *

import setGPU

from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--dataset', default='cifar10', choices=['cifar10', 'svhn', 'cifar100'])
    parser.add_argument('--cifar10-data-dir', default='../../cifar-data', type=str)
    parser.add_argument('--svhn-data-dir', default='../../svhn-data', type=str)

    parser.add_argument('--epochs', default=30, type=int)
    parser.add_argument('--lr-schedule', default='cyclic', choices=['cyclic', 'piecewise'])
    parser.add_argument('--lr-min', default=0., type=float)
    # parser.add_argument('--lr-max', default=0.2, type=float)
    parser.add_argument('--lr-max', default=0.3, type=float)

    parser.add_argument('--weight-decay', default=5e-4, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    # parser.add_argument('--epsilon', default=8, type=int)
    parser.add_argument('--epsilon', default=16, type=int)

    # parser.add_argument('--alpha', default=4, type=float)
    parser.add_argument('--alpha', default=10, type=float)

    parser.add_argument('--attack-iters', default=2, type=int, help='Attack steps')

    parser.add_argument('--net-name', default='preact', choices=['preact', 'res18', 'wrn34'], type=str)
    parser.add_argument('--num-classes', default=10, type=int)

    parser.add_argument('--delta-init', default='random', type=str, choices=['zero', 'random', 'gauss'],
        help='Perturbation initialization method')

    parser.add_argument('--out-dir', default='train_pgd', type=str, help='Output directory')
    parser.add_argument('--seed', default=0, type=int, help='Random seed')

    parser.add_argument('--opt-level', default='O1', type=str, choices=['O0', 'O1'],
        help='O0 is FP32 training, O1 is Mixed Precision')
    parser.add_argument('--loss-scale', default='dynamic', type=str, choices=['1.0', 'dynamic'],
        help='If loss_scale is "dynamic", adaptively adjust the loss scale over time')
    parser.add_argument('--master-weights', action='store_true',
        help='Maintain FP32 master weights to accompany any FP16 model weights, not applicable for O1 opt level')
    parser.add_argument('--eval', action='store_true', help='evaluation against pgd every epoch')
    return parser.parse_args()

def compute_pgd_delta(model, X, y, delta, epsilon, alpha, opt, attack_iters):
    for iter_number in range(attack_iters):
        output = model(X + delta[:X.size(0)])
        loss = F.cross_entropy(output, y)
        with amp.scale_loss(loss, opt) as scaled_loss:
            grad = torch.autograd.grad(scaled_loss, delta)[0]
        grad = grad.detach()

        # Step size for different iteration
        # NoiseAugmentation_16-step2-StepSize_12_4
        # if iter_number == 0:
        #     alpha = 12
        # else:
        #     alpha = 4

        # NoiseAugmentation_16-step2-StepSize_8_8
        if iter_number == 0:
            alpha = (8/255)/0.25
        else:
            alpha = (8/255)/0.25

        delta.data = clamp(delta + alpha * torch.sign(grad), -epsilon, epsilon)
        delta.data = clamp(delta, lower_limit - X, upper_limit - X)

        # delta.data = torch.clamp(delta + alpha * torch.sign(grad), -epsilon, epsilon)
        # bug test
        # delta.data[:X.size(0)] = clamp(delta[:X.size(0)], 0.0 - X, 1.0 - X)
    delta = delta.detach()
    return delta, grad

def main(args, model_name):

    # create tensorboard summary writer
    writer = SummaryWriter(comment=args.experiment_name)

    if args.dataset == 'cifar10':
        train_loader, test_loader, val_loader = get_loaders(args.cifar10_data_dir, args.batch_size, dataset_name=args.dataset)
    elif args.dataset == 'svhn':
        train_loader, test_loader = get_loaders(args.svhn_data_dir, args.batch_size, dataset_name=args.dataset)
        n_alpha_warmup_epochs = 5
        n_iterations_max_alpha = n_alpha_warmup_epochs*len(train_loader)
    else:
        raise NameError('This program does not support this dataset')

    ## hyperparameters
    epsilon = (args.epsilon / 255.) / std
    alpha = (args.alpha / 255.) / std
    max_alpha = alpha
    pgd_alpha = epsilon/4.0 # for evaluation

    if args.net_name == 'preact':
        model = PreActResNet18().cuda()
    elif args.net_name == 'res18':
        model = ResNet18().cuda()
    elif args.net_name == 'wrn34':
        model = WideResNet().cuda()

    ## on SVHN, GradAlign will converge to a majority classifier with default initialization for eps=12 (our method won't)
    # if args.dataset == 'svhn': model.apply(normal_weight_init)
    model.apply(normal_weight_init)
    model.train()

    opt = torch.optim.SGD(model.parameters(), lr=args.lr_max, momentum=args.momentum, weight_decay=args.weight_decay)
    amp_args = dict(opt_level=args.opt_level)
    if args.opt_level == 'O2':
        amp_args['master_weights'] = args.master_weights
    if not args.opt_level == 'O0': amp.register_float_function(torch, 'batch_norm')
    model, opt = amp.initialize(model, opt, **amp_args)

    criterion = nn.CrossEntropyLoss()

    lr_steps = args.epochs * len(train_loader)
    if args.lr_schedule == 'cyclic':
        scheduler = torch.optim.lr_scheduler.CyclicLR(opt, base_lr=args.lr_min, max_lr=args.lr_max,
            step_size_up=lr_steps / 2, step_size_down=lr_steps / 2)
    elif args.lr_schedule == 'piecewise':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[lr_steps / 2, lr_steps * 3 / 4], gamma=0.1)

    # Training
    prev_robust_acc = 0.
    start_train_time = time.time()

    if args.eval:
        logger.info('Epoch \t Seconds \t LR \t \t Train Loss \t Train Acc \t Test Acc \t 20 step PGD test Acc')
    else:
        logger.info('Epoch \t Seconds \t LR \t \t Train Loss \t Train Acc')
    for epoch in range(args.epochs):
        start_epoch_time = time.time()
        train_loss = 0
        train_acc = 0
        train_n = 0
        for i, (X, y) in enumerate(train_loader):
            X, y = X.cuda(), y.cuda()
            if epoch==0 and i == 0: first_batch = (X, y)


            # baseline 
            # baseline_16-step_2-StepSize_10
            # noise augmentation
            # 2 times eps uniform noise
            # NoiseAugmentation_16-step2-StepSize_10_10
            noise = torch.zeros_like(X).cuda()
            # noise.uniform_(-2*(args.epsilon / 255.) / std, 2*(args.epsilon / 255.) / std)
            for i in range(len(epsilon)):
                noise[:, i, :, :].uniform_(-epsilon[i][0][0].item(), epsilon[i][0][0].item())

            ###########################################################
            # gaussian noise
            # noise = torch.randn(X.shape).cuda()
            # noise = (3*args.epsilon*noise)/255

            # with clamp
            # noise = torch.randn(X.shape).cuda()
            # noise = (3*args.epsilon*noise)/255
            # noise.data = clamp(noise, 0.0 - X, 1.0 - X)

            # failed
            # noise = 0.002 * torch.randn(X.shape).cuda()
            ###########################################################


            # 1 times eps uniform noise
            # noise.uniform_(-1*epsilon, 1*epsilon)
            # noise.data = clamp(noise, 0.0 - X, 1.0 - X)
            X = X + noise

            if args.dataset == 'svhn':
                ## svhn needs warm up
                iteration = epoch*len(train_loader)+i+1
                alpha = min(iteration / n_iterations_max_alpha * max_alpha, max_alpha)

            if args.delta_init == 'zero':
                delta = torch.zeros_like(X).cuda()
            if args.delta_init == 'random':
                delta = torch.zeros_like(X).cuda()
                # delta = torch.zeros_like(X).cuda().uniform_(-epsilon, epsilon)
                # delta.data = clamp(delta, 0.0 - X, 1.0 - X)
            if args.delta_init == 'gauss':
                delta = 0.001 * torch.randn(X.shape).cuda()
                delta.data = clamp(delta, 0.0 - X, 1.0 - X)

            delta.requires_grad = True

            ### train
            delta, grad = compute_pgd_delta(model, X, y, delta, epsilon,
                                            alpha, opt, args.attack_iters)

            output = model(X + delta[:X.size(0)])
            loss = criterion(output, y)

            opt.zero_grad()
            with amp.scale_loss(loss, opt) as scaled_loss:
                scaled_loss.backward()
            opt.step()
            train_loss += loss.item() * y.size(0)
            train_acc += (output.max(1)[1] == y).sum().item()
            train_n += y.size(0)
            scheduler.step()

        epoch_time = time.time()
        lr = scheduler.get_lr()[0]

        model.eval()
        pgd_loss, pgd_acc = evaluate_pgd(val_loader, model, 50, 1, epsilon, pgd_alpha, surrogate='ce', opt=opt)
        test_loss, test_acc = evaluate_standard(val_loader, model)
        model.train()

        writer.add_scalar("pgd_loss", pgd_loss, epoch)
        writer.add_scalar("pgd_acc", pgd_acc, epoch)
        writer.add_scalar("test_loss", test_loss, epoch)
        writer.add_scalar("test_acc", test_acc, epoch)
        # if args.eval:
        #     pgd_loss, pgd_acc = evaluate_pgd(test_loader, model, 20, 1, epsilon, pgd_alpha, surrogate='ce', opt=opt)
        #     test_loss, test_acc = evaluate_standard(test_loader, model)
        #     logger.info('%d \t %.1f \t \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f',
        #             epoch, epoch_time - start_epoch_time, lr, train_loss/train_n, train_acc/train_n, test_acc, pgd_acc)
        # else:
        #     logger.info('%d \t %.1f \t \t %.4f \t %.4f \t %.4f',
        #             epoch, epoch_time - start_epoch_time, lr, train_loss/train_n, train_acc/train_n)
        logger.info('%d \t %.1f \t \t %.4f \t %.4f \t %.4f',
                epoch, epoch_time - start_epoch_time, lr, train_loss/train_n, train_acc/train_n)



    train_time = time.time()
    torch.save(model.state_dict(), os.path.join(args.out_dir, '{}.pth'.format(model_name)))
    logger.info('Total train time: %.4f minutes', (train_time - start_train_time)/60)

    # Evaluation
    ### For reliable evaluate, we use single precision
    ### It is important to # disable global conversion to fp16 from amp.initialize() (https://github.com/NVIDIA/apex/issues/567)
    context_manager = amp.disable_casts() if not args.opt_level == 'O0' else contextlib.nullcontext()
    with context_manager:
        if args.net_name == 'preact':
            model_test = PreActResNet18().cuda()
        elif args.net_name == 'res18':
            model_test = ResNet18().cuda()
        elif args.net_name == 'wrn34':
            model_test = WideResNet().cuda()

        model_test.load_state_dict(torch.load(os.path.join(args.out_dir, '{}.pth'.format(model_name))))
        model_test.float()
        model_test.eval()

        test_loss, test_acc = evaluate_standard(test_loader, model_test)

        logger.info('Test Loss \t Test Acc \t PGD Loss \t PGD Acc')

        pgd_loss, pgd_acc = evaluate_pgd(test_loader, model_test, 50, 10, epsilon, pgd_alpha, surrogate='ce')

        logger.info('%.4f \t \t %.4f \t %.4f \t %.4f', test_loss, test_acc, pgd_loss, pgd_acc)


import shutil
if __name__ == "__main__":
    args = get_args()
    ### special name for storing the models
    model_name = 'pgd-{}_net_{}_schedule_{}_seed_{}_delta_init_{}_eps_{}_alpha_{}_lrmax_{}_wd_{}'.format(
        args.attack_iters, args.net_name, args.lr_schedule, args.seed, args.delta_init,
        args.epsilon, args.alpha, args.lr_max, args.weight_decay)

    args.experiment_name = args.out_dir
    args.out_dir = os.path.join("exp0105", args.out_dir)


    ### log information
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    logfile = os.path.join(args.out_dir, '{}_output.log'.format(model_name))
    if os.path.exists(logfile):
        os.remove(logfile)

    logging.basicConfig(format='[%(asctime)s] - %(message)s',
                        datefmt='%Y/%m/%d %H:%M:%S',
                        level=logging.INFO,
                        filename=logfile)
    logger.info(args)

    ### random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)

    shutil.copyfile("k_train_pgd.py", f"{args.out_dir}/k_train_pgd.py")

    main(args, model_name)
