import argparse
import logging
import os
from torchattacks.attacks.autoattack import AutoAttack
import setGPU
import apex.amp as amp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

from preact_resnet import PreActResNet18
from utils import (clamp, get_loaders, evaluate_standard, evaluate_pgd)

import shutil
import glob

from types import int, str, float


# from alive_progress import alive_it


logger = logging.getLogger(__name__)

from torch.utils.tensorboard import SummaryWriter
import shutil
from datetime import datetime

def get_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument('experiment', type=str)
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--data-dir', default='/dev/shm', type=str)
    parser.add_argument('--epochs', default=15, type=int)
    parser.add_argument('--lr-schedule', default='cyclic', type=str, choices=['cyclic', 'multistep'])
    parser.add_argument('--lr-min', default=0., type=float)
    parser.add_argument('--lr-max', default=0.2, type=float)
    parser.add_argument('--weight-decay', default=5e-4, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--epsilon', default=8, type=int)
    parser.add_argument('--attack-iters', default=7, type=int, help='Attack iterations')
    parser.add_argument('--restarts', default=1, type=int)
    parser.add_argument('--alpha', default=2, type=float, help='Step size')
    parser.add_argument('--delta-init', default='random', choices=['zero', 'random'],
        help='Perturbation initialization method')
    parser.add_argument('--out-dir', default='train_pgd_output', type=str, help='Output directory')
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    parser.add_argument('--opt-level', default='O2', type=str, choices=['O0', 'O1', 'O2'],
        help='O0 is FP32 training, O1 is Mixed Precision, and O2 is "Almost FP16" Mixed Precision')
    parser.add_argument('--loss-scale', default='1.0', type=str, choices=['1.0', 'dynamic'],
        help='If loss_scale is "dynamic", adaptively adjust the loss scale over time')
    parser.add_argument('--master-weights', action='store_true',
        help='Maintain FP32 master weights to accompany any FP16 model weights, not applicable for O1 opt level')

    # when we use noise agumentation, removing delta initialization yields better performance
    parser.add_argument('--noise_aug', action='store_true')
    parser.add_argument('--noise_aug_type', default='normal', type=str, help='Change noise augmentation type.')
    parser.add_argument('--noise_aug_size', default=1, type=float, help='Change noise augmenttion  size.')

    parser.add_argument('--early-stop', action='store_true', help='Early stop if overfitting occurs')

    # whether normalize image
    parser.add_argument('--image_normalize', action='store_true')
    parser.add_argument('--zero_one_clamp', default=1, type=int)
    
    return parser.parse_args()


def main():
    args = get_args()

    args.image_normalize = False

    # if args.image_normalize:
    if True:

        args.out_dir = args.out_dir + f"-image_normalize-"
        cifar10_mean = (0.4914, 0.4822, 0.4465)
        cifar10_std = (0.2471, 0.2435, 0.2616)
    else:
        args.out_dir = args.out_dir + f"-remove_image_normalize-"
        cifar10_mean = (0., 0., 0.)
        cifar10_std = (1., 1., 1.)

    mu = torch.tensor(cifar10_mean).view(3,1,1).cuda()
    std = torch.tensor(cifar10_std).view(3,1,1).cuda()

    upper_limit = (1 - mu)
    lower_limit = (0 - mu)

    #testing_file = "log_files/NoiseAug/PGD_baseline-image_normalize--NoiseAug-_type_uniform-noise_aug_size_2.0--epochs_40-lr_schedule_cyclic-lr_max_0.2-epsilon_16-attack_steps_2-alpha_8.0-delta_init_zero-zero_one_clamp_0-seed_1/20220108145735/model.pth"
    #testing_file = "log_files/NoiseAug/PGD_baseline-image_normalize--NoiseAug-_type_uniform-noise_aug_size_2.0--epochs_30-lr_schedule_cyclic-lr_max_0.3-epsilon_16-attack_steps_2-alpha_8.0-delta_init_zero-zero_one_clamp_0-seed_1/20220108145735/model.pth"
    testing_file ="log_files/NoiseAug/PGD_baseline-image_normalize--epochs_30-lr_schedule_cyclic-lr_max_0.3-epsilon_16-attack_steps_3-alpha_6.6666-delta_init_zero-zero_one_clamp_1-seed_1/20220108142635/model.pth"
    # TE
    #testing_file = "log_files/NoiseAug/PGD_baseline-image_normalize--NoiseAug-_type_uniform-noise_aug_size_2.0--epochs_40-lr_schedule_cyclic-lr_max_0.2-epsilon_16-attack_steps_2-alpha_8.0-delta_init_zero-zero_one_clamp_0-seed_2/20220108173245/model.pth"
    state_dict_loaded = torch.load(testing_file)

    logfile = "/".join(testing_file.split("/")[:-2]) + "/AutoAttack_testing.log"
    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.INFO,
        filename=logfile)
    logger.info(args)

    # eps 16
    epsilon = (16 / 255.)
    test_alpha = (4 / 255.)
    # eps 8
    # epsilon = (8 / 255.) / std
    # test_alpha = (2 / 255.) / std

    # Final Evaluation (record with tensorboard)
    model_test = PreActResNet18().cuda() ### to make sure that the robustness evaluation is done on single precision instead of half-precision
    model_test.load_state_dict(state_dict_loaded)
    model_test.float()
    model_test.eval()
    _, test_loader, _ = get_loaders(args.data_dir, args.batch_size, args.image_normalize, cifar10_mean, cifar10_std)


    pgd_loss, pgd_acc = evaluate_pgd(test_loader, model_test, 50, 10, epsilon, test_alpha, lower_limit, upper_limit, opt=None, logger=logger)
    #pgd_loss, pgd_acc = evaluate_pgd(test_loader, model_test, 50, 1, epsilon, test_alpha, lower_limit, upper_limit, opt=None, logger=logger)
    test_loss, test_acc = evaluate_standard(test_loader, model_test)


    logger.info('Test Loss \t Test Acc \t PGD Loss \t PGD Acc')
    logger.info('%.4f \t \t %.4f \t %.4f \t %.4f', test_loss, test_acc, pgd_loss, pgd_acc)


def evaluate_autoattack(test_loader, model):

    aa_acctacker = AutoAttack(model, norm='Linf', eps=, version='standard', n_classes=10, seed=None, verbose=True)



if __name__ == "__main__":
    main()
