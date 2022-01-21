import argparse
from cProfile import label
import logging
import os
import time
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

    args.image_normalize = True

    if args.image_normalize:
        args.out_dir = args.out_dir + f"-image_normalize-"
        cifar10_mean = (0.4914, 0.4822, 0.4465)
        cifar10_std = (0.2471, 0.2435, 0.2616)
    else:
        args.out_dir = args.out_dir + f"-remove_image_normalize-"
        cifar10_mean = (0., 0., 0.)
        cifar10_std = (1., 1., 1.)

    mu = torch.tensor(cifar10_mean).view(3,1,1).cuda()
    std = torch.tensor(cifar10_std).view(3,1,1).cuda()

    upper_limit = ((1 - mu)/ std)
    lower_limit = ((0 - mu)/ std)

    #testing_file = "log_files/NoiseAug/PGD_baseline-image_normalize--NoiseAug-_type_uniform-noise_aug_size_2.0--epochs_40-lr_schedule_cyclic-lr_max_0.2-epsilon_16-attack_steps_2-alpha_8.0-delta_init_zero-zero_one_clamp_0-seed_1/20220108145735/model.pth"
    #testing_file = "log_files/NoiseAug/PGD_baseline-image_normalize--NoiseAug-_type_uniform-noise_aug_size_2.0--epochs_30-lr_schedule_cyclic-lr_max_0.3-epsilon_16-attack_steps_2-alpha_8.0-delta_init_zero-zero_one_clamp_0-seed_1/20220108145735/model.pth"
    # testing_file ="log_files/NoiseAug/PGD_baseline-image_normalize--epochs_30-lr_schedule_cyclic-lr_max_0.3-epsilon_16-attack_steps_3-alpha_6.6666-delta_init_zero-zero_one_clamp_1-seed_1/20220108142635/model.pth"

    # PGD2 baseline eps 8
    # testing_file = "log_files/pgd2-baseline-eps8/pgd2-baseline-eps8-image_normalize--epochs_30-lr_schedule_cyclic-lr_max_0.3-epsilon_8-attack_steps_2-alpha_4.0-delta_init_zero-zero_one_clamp_1-seed_0/20220114084834/model.pth"
    # FGSM NoiseAug eps 8 step size 10
    # testing_file = "log_files/FGSM_NoiseAug_type_size-eps8-step_size10/FGSM_NoiseAug_type_size-eps8-step_size10-image_normalize--NoiseAug-_type_normal-noise_aug_size_1.0--epochs_30-lr_schedule_cyclic-lr_max_0.3-epsilon_8-attack_steps_7-alpha_10.0-delta_init_zero-zero_one_clamp_1-seed_0/20220112043450/model.pth"

    # FGSM NoiseAug eps 16 step size 20
    testing_file = "log_files/FGSM_NoiseAug_type_size-eps16-step_size20/FGSM_NoiseAug_type_size-eps16-step_size20-image_normalize--NoiseAug-_type_normal-noise_aug_size_1.0--epochs_30-lr_schedule_cyclic-lr_max_0.3-epsilon_16-attack_steps_7-alpha_20.0-delta_init_zero-zero_one_clamp_1-seed_0/20220112052332/model.pth"

    # TE
    #testing_file = "log_files/NoiseAug/PGD_baseline-image_normalize--NoiseAug-_type_uniform-noise_aug_size_2.0--epochs_40-lr_schedule_cyclic-lr_max_0.2-epsilon_16-attack_steps_2-alpha_8.0-delta_init_zero-zero_one_clamp_0-seed_2/20220108173245/model.pth"
    state_dict_loaded = torch.load(testing_file)

    # logfile = "/".join(testing_file.split("/")[:-2]) + "/testing.log"
    # logging.basicConfig(
    #     format='[%(asctime)s] - %(message)s',
    #     datefmt='%Y/%m/%d %H:%M:%S',
    #     level=logging.INFO,
    #     filename=logfile)
    # logger.info(args)

    # eps 16
    epsilon = (16 / 255.) / std
    test_alpha = (4 / 255.) / std
    # eps 8
    # epsilon = (8 / 255.) / std
    # test_alpha = (2 / 255.) / std

    # Final Evaluation (record with tensorboard)
    model_test = PreActResNet18().cuda() ### to make sure that the robustness evaluation is done on single precision instead of half-precision
    model_test.load_state_dict(state_dict_loaded)
    model_test.float()
    model_test.eval()
    _, test_loader, _ = get_loaders(args.data_dir, 512, args.image_normalize, cifar10_mean, cifar10_std)



    criterion = nn.CrossEntropyLoss(size_average=False)
    model_test.eval()


    max_alpha = 200
    alpha_step_size = 2

    avg_confidence = np.zeros((3, int(max_alpha/alpha_step_size)))
    for i, (X, y) in enumerate(test_loader):
        X, y = X.cuda(), y.cuda()
        # pgd_delta = attack_pgd(model_test, X, y, epsilon, test_alpha, attack_iters, restarts, lower_limit, upper_limit, opt=opt)

        for iter_number, alpha in enumerate(range(2, max_alpha, alpha_step_size)):
            print(f"iter number: {iter_number}")
            avg_confidence[2][iter_number] = alpha

            alpha = (alpha / 255.) / std

            delta = torch.zeros_like(X).cuda()
            # if args.delta_init == 'random':
            #     for i in range(len(epsilon)):
            #         delta[:, i, :, :].uniform_(-epsilon[i][0][0].item(), epsilon[i][0][0].item())
            #     delta.data = clamp(delta, lower_limit - X, upper_limit - X)
            delta.requires_grad = True

            # for att_iter_num in range(args.attack_iters):
            # Only used for FGSM Training
            output = model_test(X + delta[:X.size(0)])
            loss = criterion(output, y)
            loss.backward()
            # with amp.scale_loss(loss, opt) as scaled_loss:
            #     scaled_loss.backward()
            grad = delta.grad.detach()

            # remain clamp Baseline
            # delta.data = clamp(delta + alpha * torch.sign(grad), -epsilon, epsilon)
            # if args.zero_one_clamp:
                # import pdb; pdb.set_trace()
                # delta.data = clamp(delta, lower_limit - X, upper_limit - X)

            # remove clamp
            delta.data = delta + alpha * torch.sign(grad)
            delta.grad.zero_()
            delta = delta.detach()

            # output_clean = model_test(X)
            # prob_clean = F.softmax(output_clean, dim=-1)
            # correct_prob_clean = (F.one_hot(y) * prob_clean).sum(-1).mean()
            # avg_confidence[0][iter_number] = correct_prob_clean

            output_adv = model_test(X + delta)
            prob_adv = F.softmax(output_adv, dim=-1)
            correct_prob_adv = (F.one_hot(y) * prob_adv).sum(-1).mean()
            avg_confidence[1][iter_number] = correct_prob_adv



        break

    np.save("confidence.npy", avg_confidence)

    import matplotlib.pyplot as plt

    # plt.plot(avg_confidence[0, :], label="clean confidence")
    plt.plot(avg_confidence[1, :], label="adv confidence")

    plt.legend()

    plt.savefig("confidence.png")







if __name__ == "__main__":
    main()
