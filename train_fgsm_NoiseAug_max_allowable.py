import argparse
import logging
import os
import time
import setGPU
import apex.amp as amp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from preact_resnet import PreActResNet18
from utils import (clamp, get_loaders, evaluate_standard, evaluate_pgd)

import shutil
import glob

from sys import exit


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

    # paper method grad align
    parser.add_argument('--grad-align-cos-lambda', default=0.0, type=float)

    # output align method
    OUTPUT_ALIGN = ["none", "CS_AE_NAE", "KL_AE_NAE", "NAE2GT"]
    parser.add_argument('--out_align_method', default='none', type=str, choices=OUTPUT_ALIGN, help='Change random agumentation type.')
    parser.add_argument('--out_align_noise', default=2., type=float)

    # whether normalize image
    parser.add_argument('--image_normalize', action='store_true')
    parser.add_argument('--zero_one_clamp', default=1, type=int)
    

    return parser.parse_args()


def main():
    args = get_args()

    saving_prefix = args.out_dir

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

    # args.experiment = args.out_dir
    if args.noise_aug:
        assert (args.out_align_method == "none") and (args.grad_align_cos_lambda == 0.0)
        args.out_dir = args.out_dir + f"-NoiseAug-_type_{args.noise_aug_type}-noise_aug_size_{args.noise_aug_size}-"
    if args.grad_align_cos_lambda != 0.0:
        assert (args.out_align_method == "none") and (args.noise_aug == False)
        args.out_dir = args.out_dir + f"-grad_align_cos_lambda_{args.grad_align_cos_lambda}-"
    if args.out_align_method != "none":
        assert (args.grad_align_cos_lambda == 0.0) and (args.noise_aug == False)
        args.out_dir = args.out_dir + f"-out_align_method_{args.out_align_method}"

    args.out_dir = args.out_dir + f"-epochs_{args.epochs}-lr_schedule_{args.lr_schedule}-lr_max_{args.lr_max}-epsilon_{args.epsilon}-attack_steps_{args.attack_iters}-alpha_{args.alpha}-delta_init_{args.delta_init}-zero_one_clamp_{args.zero_one_clamp}-seed_{args.seed}"

    args.experiment_name = args.out_dir

    time_stamp = datetime.now().strftime("%Y%m%d%H%M%S")
    args.out_dir = os.path.join(saving_prefix, args.out_dir, time_stamp)

    # create tensorboard summary wirter
    writer = SummaryWriter(comment=args.experiment_name)


    ################################################################
    # training code saving
    ################################################################

    pathname = "./*.py"
    files = glob.glob(pathname, recursive=True)

    for file in files:
        dest_fpath = os.path.join("./log_files", args.out_dir, "code", file.split("/")[-1])
        try:
            shutil.copy(file, dest_fpath)
        except IOError as io_err:
            os.makedirs(os.path.dirname(dest_fpath))
            shutil.copy(file, dest_fpath)

    ################################################################
    # training code saving
    ################################################################

    args.out_dir = os.path.join("./log_files", args.out_dir) 
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    
    logfile = os.path.join(args.out_dir, 'output.log')
    if os.path.exists(logfile):
        os.remove(logfile)

    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.INFO,
        filename=logfile)
    logger.info(args)

    # random seed setting
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    train_loader, test_loader, val_loader = get_loaders(args.data_dir, args.batch_size, args.image_normalize, cifar10_mean, cifar10_std)

    epsilon = (args.epsilon / 255.) / std
    alpha = (args.alpha / 255.) / std

    if args.epsilon == 8:
        test_alpha = (2 / 255.) / std
    else:
        test_alpha = (4 / 255.) / std


    model = PreActResNet18().cuda()
    model.train()

    opt = torch.optim.SGD(model.parameters(), lr=args.lr_max, momentum=args.momentum, weight_decay=args.weight_decay)
    amp_args = dict(opt_level=args.opt_level, loss_scale=args.loss_scale, verbosity=False)
    if args.opt_level == 'O2':
        amp_args['master_weights'] = args.master_weights
    model, opt = amp.initialize(model, opt, **amp_args)
    criterion = nn.CrossEntropyLoss()

    lr_steps = args.epochs * len(train_loader)
    if args.lr_schedule == 'cyclic':
        scheduler = torch.optim.lr_scheduler.CyclicLR(opt, base_lr=args.lr_min, max_lr=args.lr_max,
            step_size_up=lr_steps / 2, step_size_down=lr_steps / 2)
    elif args.lr_schedule == 'multistep':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[lr_steps / 2, lr_steps * 3 / 4], gamma=0.1)

    # Training
    start_train_time = time.time()
    logger.info('Epoch \t Seconds \t LR \t \t Train Loss \t Train Acc')
    training_time_accumelater = 0
    for epoch in range(args.epochs):
        print(f"Epoch: {epoch}")
        start_epoch_time = time.time()
        train_loss = 0
        train_acc = 0
        train_n = 0
        model.train()
        for i, (X, y) in enumerate(train_loader):
            X, y = X.cuda(), y.cuda()

            if args.noise_aug:
                noise = torch.zeros_like(X).cuda()
                if args.noise_aug_type == "uniform":
                    noise.uniform_(-1, 1) 
                elif args.noise_aug_type == "normal":
                    noise = torch.randn(X.shape).cuda()
                for j in range(len(epsilon)):
                    noise[:, j, :, :] = args.noise_aug_size*epsilon[j][0][0].item() * noise[:, j, :, :]
                if args.zero_one_clamp:
                    noise.data = clamp(noise, lower_limit - X, upper_limit - X)
                X = X + noise


            delta = torch.zeros_like(X).cuda()
            if args.delta_init == 'random':
                for i in range(len(epsilon)):
                    delta[:, i, :, :].uniform_(-epsilon[i][0][0].item(), epsilon[i][0][0].item())
                delta.data = clamp(delta, lower_limit - X, upper_limit - X)
            delta.requires_grad = True

            # for att_iter_num in range(args.attack_iters):
            # Only used for FGSM Training
            output = model(X + delta[:X.size(0)])
            loss = criterion(output, y)
            with amp.scale_loss(loss, opt) as scaled_loss:
                scaled_loss.backward()
            grad = delta.grad.detach()

            # remain clamp Baseline
            # delta.data = clamp(delta + alpha * torch.sign(grad), -epsilon, epsilon)
            delta.data = delta + alpha * torch.sign(grad)
            if args.zero_one_clamp:
                # import pdb; pdb.set_trace()
                delta.data = clamp(delta, lower_limit - X, upper_limit - X)

            # remove clamp
            # delta.data = delta + alpha * torch.sign(grad)
            delta.grad.zero_()

            delta = delta.detach()
            output = model(X + delta)
            loss = criterion(output, y)

            # # add grad align method
            reg = torch.zeros(1).cuda()[0]  # for .item() to run correctly
            if args.grad_align_cos_lambda != 0.0:
                grad2 = get_input_grad(model, X, y, opt, epsilon, True, delta_init='random_uniform', backprop=True)
                grads_nnz_idx = ((grad**2).sum([1, 2, 3])**0.5 != 0) * ((grad2**2).sum([1, 2, 3])**0.5 != 0)
                grad1, grad2 = grad[grads_nnz_idx], grad2[grads_nnz_idx]
                grad1_norms, grad2_norms = l2_norm_batch(grad1), l2_norm_batch(grad2)
                grad1_normalized = grad1 / grad1_norms[:, None, None, None]
                grad2_normalized = grad2 / grad2_norms[:, None, None, None]
                cos = torch.sum(grad1_normalized * grad2_normalized, (1, 2, 3))
                reg += args.grad_align_cos_lambda * (1.0 - cos.mean())

            loss += reg

            # add output align method
            if args.out_align_method != "none":
                loss_out_align = output_align(X, y, output, epsilon, alpha, model, lower_limit, upper_limit, align_method=args.out_align_method, opt=opt, out_align_noise=args.out_align_noise)
                loss += loss_out_align

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
        logger.info('%d \t %.1f \t \t %.4f \t %.4f \t %.4f',
            epoch, epoch_time - start_epoch_time, lr, train_loss/train_n, train_acc/train_n)
        training_time_accumelater += epoch_time - start_epoch_time

        model.eval()
        pgd_loss, pgd_acc = evaluate_pgd(val_loader, model, 50, 1, epsilon, test_alpha, lower_limit, upper_limit, opt=opt, logger=logger)
        test_loss, test_acc = evaluate_standard(val_loader, model)
        model.train()

        writer.add_scalar("pgd_loss", pgd_loss, epoch)
        writer.add_scalar("pgd_acc", pgd_acc, epoch)
        writer.add_scalar("test_loss", test_loss, epoch)
        writer.add_scalar("test_acc", test_acc, epoch)

        if pgd_acc < 0.01:
            logger.info(f"Catastrophic happens at epoch {epoch}. Stop Training.")
            exit(0)

    train_time = time.time()
    torch.save(model.state_dict(), os.path.join(args.out_dir, 'model.pth'))
    logger.info('Total train time: %.4f minutes', (train_time - start_train_time)/60)

    # Final Evaluation (record with tensorboard)
    model_test = PreActResNet18().cuda() ### to make sure that the robustness evaluation is done on single precision instead of half-precision
    model_test.load_state_dict(model.state_dict())
    model_test.float()
    model_test.eval()

    pgd_loss, pgd_acc = evaluate_pgd(test_loader, model_test, 50, 10, epsilon, test_alpha, lower_limit, upper_limit, opt=None, logger=logger)
    test_loss, test_acc = evaluate_standard(test_loader, model_test)


    logger.info('Test Loss \t Test Acc \t PGD Loss \t PGD Acc')
    logger.info('%.4f \t \t %.4f \t %.4f \t %.4f', test_loss, test_acc, pgd_loss, pgd_acc)

    logger.info(f'Training Time Consuming:{training_time_accumelater}')


# function for grad align
def get_input_grad(model, X, y, opt, eps, half_prec, delta_init='none', backprop=False):
    if delta_init == 'none':
        delta = torch.zeros_like(X, requires_grad=True)
    elif delta_init == 'random_uniform':
        delta = get_uniform_delta(X.shape, eps, requires_grad=True)
    elif delta_init == 'random_corner':
        delta = get_uniform_delta(X.shape, eps, requires_grad=True)
        delta = eps * torch.sign(delta)
    else:
        raise ValueError('wrong delta init')

    output = model(X + delta)
    loss = F.cross_entropy(output, y)
    if half_prec:
        with amp.scale_loss(loss, opt) as scaled_loss:
            grad = torch.autograd.grad(scaled_loss, delta, create_graph=True if backprop else False)[0]
            grad /= scaled_loss / loss
    else:
        grad = torch.autograd.grad(loss, delta, create_graph=True if backprop else False)[0]
    if not backprop:
        grad, delta = grad.detach(), delta.detach()
    return grad
def get_uniform_delta(shape, eps, requires_grad=True):
    delta = torch.zeros(shape).cuda()
    delta.uniform_(-eps, eps)
    delta.requires_grad = requires_grad
    return delta
def l2_norm_batch(v):
    norms = (v ** 2).sum([1, 2, 3]) ** 0.5
    return norms


# function for output align experiment
def output_align(X, y, adv_logit, epsilon, alpha, model, lower_limit, upper_limit, align_method, opt, out_align_noise):
    # Generate X+noise adversarial example
    noise = torch.zeros_like(X).cuda()
    for j in range(len(epsilon)):
        noise[:, j, :, :].uniform_(-out_align_noise*epsilon[j][0][0].item(), out_align_noise*epsilon[j][0][0].item())

    delta_noise = torch.zeros_like(X).cuda()
    delta_noise.requires_grad = True

    output_noise = model(X + noise + delta_noise[:X.size(0)])
    loss_noise = F.cross_entropy(output_noise, y)
    with amp.scale_loss(loss_noise, opt) as scaled_loss:
        grad = torch.autograd.grad(scaled_loss, delta_noise)[0]
    grad = grad.detach()
    # delta_noise.data = torch.clamp(delta_noise + alpha * torch.sign(grad), -epsilon, epsilon)
    # delta_noise.data[:X.size(0)] = clamp(delta_noise[:X.size(0)], 0.0 - X, 1.0 - X)
    # delta_noise = delta_noise.detach()
    delta_noise.data = clamp(delta_noise + alpha * torch.sign(grad), -epsilon, epsilon)
    delta_noise.data[:X.size(0)] = clamp(delta_noise[:X.size(0)], lower_limit - X, upper_limit - X)
    delta_noise = delta_noise.detach()
    # CS
    # fgsm-CS_AE_NAE
    #import ipdb; ipdb.set_trace()
    if align_method == "CS_AE_NAE":
        output_noise = model(X + noise + delta_noise[:X.size(0)])
        loss = F.cosine_similarity(adv_logit, output_noise, dim=-1).mean()
    
    # KL_AE_NAE
    elif align_method == "KL_AE_NAE":
        output_noise = model(X + noise + delta_noise[:X.size(0)])
        criterion_kl = nn.KLDivLoss(size_average=False)
        loss_kl = criterion_kl(F.log_softmax(adv_logit, dim=1), F.softmax(output_noise, dim=1))
        loss = loss_kl

    # ce add NAE2GT
    elif align_method == "NAE2GT":

        output_noise = model(X + noise + delta_noise[:X.size(0)])
        loss = F.cross_entropy(output_noise, y)

    return loss

if __name__ == "__main__":
    main()
