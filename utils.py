import apex.amp as amp
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
from torchattacks.attacks import autoattack 

# cifar10_mean = (0.4914, 0.4822, 0.4465)
# cifar10_std = (0.2471, 0.2435, 0.2616)

# mu = torch.tensor(cifar10_mean).view(3,1,1).cuda()
# std = torch.tensor(cifar10_std).view(3,1,1).cuda()

# upper_limit = ((1 - mu)/ std)
# lower_limit = ((0 - mu)/ std)


def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)


import torch
import numpy as np


class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img

def get_loaders(dir_, batch_size, image_normalize, cifar10_mean, cifar10_std, cutout=False, n_holes=1, length=14):
    if image_normalize:
        train_transform = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(cifar10_mean, cifar10_std),
        ]
        if cutout:
            train_transform.append(Cutout(n_holes=n_holes, length=length))
            
        
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(cifar10_mean, cifar10_std),
        ])
    else:
        train_transform = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]

        if cutout:
            train_transform.append(Cutout(n_holes=n_holes, length=length))
            
        test_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
    
    
    train_transform = transforms.Compose(train_transform)

    num_workers = 2
    train_dataset = datasets.CIFAR10(
        dir_, train=True, transform=train_transform, download=True)
    test_dataset = datasets.CIFAR10(
        dir_, train=False, transform=test_transform, download=True)

    val_dataset = datasets.CIFAR10(
        dir_, train=False, transform=test_transform, download=True)
    val_dataset.data = val_dataset.data[::10]
    val_dataset.targets = val_dataset.targets[::10]
    # val_dataset.data = val_dataset.data[:1000]
    # val_dataset.targets = val_dataset.targets[:1000]

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=2,
    )

    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=2,
    )

    return train_loader, test_loader, val_loader


def attack_pgd(model, X, y, epsilon, alpha, attack_iters, restarts, lower_limit, upper_limit, opt=None):
    max_loss = torch.zeros(y.shape[0]).cuda()
    max_delta = torch.zeros_like(X).cuda()
    for zz in range(restarts):
        delta = torch.zeros_like(X).cuda()
        for i in range(len(epsilon)):
            delta[:, i, :, :].uniform_(-epsilon[i][0][0].item(), epsilon[i][0][0].item())
        delta.data = clamp(delta, lower_limit - X, upper_limit - X)
        delta.requires_grad = True
        for _ in range(attack_iters):
            output = model(X + delta)
            index = torch.where(output.max(1)[1] == y)
            if len(index[0]) == 0:
                break
            loss = F.cross_entropy(output, y)
            if opt is not None:
                with amp.scale_loss(loss, opt) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            grad = delta.grad.detach()
            d = delta[index[0], :, :, :]
            g = grad[index[0], :, :, :]
            d = clamp(d + alpha * torch.sign(g), -epsilon, epsilon)
            d = clamp(d, lower_limit - X[index[0], :, :, :], upper_limit - X[index[0], :, :, :])
            delta.data[index[0], :, :, :] = d
            delta.grad.zero_()
        all_loss = F.cross_entropy(model(X+delta), y, reduction='none').detach()
        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)
    return max_delta


def evaluate_pgd(test_loader, model, attack_iters, restarts, epsilon, alpha, lower_limit, upper_limit, opt=None, logger=None):
    # epsilon = (8 / 255.) / std
    # alpha = (2 / 255.) / std
    pgd_loss = 0
    pgd_acc = 0
    n = 0
    model.eval()
    for i, (X, y) in enumerate(test_loader):
        X, y = X.cuda(), y.cuda()
        pgd_delta = attack_pgd(model, X, y, epsilon, alpha, attack_iters, restarts, lower_limit, upper_limit, opt=opt)
        with torch.no_grad():
            output = model(X + pgd_delta)
            loss = F.cross_entropy(output, y)
            pgd_loss += loss.item() * y.size(0)
            pgd_acc += (output.max(1)[1] == y).sum().item()
            n += y.size(0)
            print('batch id: %d, pgd accuracy: %f'%(i, pgd_acc/n), flush=True)
            if logger is not None:
                logger.info('batch id: %d, pgd accuracy: %f', i, pgd_acc/n)

    return pgd_loss/n, pgd_acc/n


def evaluate_standard(test_loader, model):
    test_loss = 0
    test_acc = 0
    n = 0
    model.eval()
    with torch.no_grad():
        for i, (X, y) in enumerate(test_loader):
            X, y = X.cuda(), y.cuda()
            output = model(X)
            loss = F.cross_entropy(output, y)
            test_loss += loss.item() * y.size(0)
            test_acc += (output.max(1)[1] == y).sum().item()
            n += y.size(0)
    return test_loss/n, test_acc/n


def evaluate_auto(test_loader, model, epsilon, logger=None):
    # epsilon = (8 / 255.) / std
    # alpha = (2 / 255.) / std
    auto_loss = 0
    auto_acc = 0
    n = 0
    model.eval()
    attack = autoattack(model, norm='Linf', eps=epsilon, version='standard', n_classes=10, seed=None, verbose=False)
    for i, (X, y) in enumerate(test_loader):
        X, y = X.cuda(), y.cuda()
        #pgd_delta = attack_autoattack(model, X, y, epsilon, alpha, attack_iters, restarts, lower_limit, upper_limit, opt=opt)
        adv_images = attack(X, y)
        with torch.no_grad():
            # output = model(X + pgd_delta)
            output = model(adv_images)
            loss = F.cross_entropy(output, y)
            auto_loss += loss.item() * y.size(0)
            auto_acc += (output.max(1)[1] == y).sum().item()
            n += y.size(0)
            print('batch id: %d, autoattack accuracy: %f'%(i, auto_acc/n), flush=True)
            if logger is not None:
                logger.info('batch id: %d, autoattack accuracy: %f', i, auto_acc/n)

    return auto_loss/n, auto_acc/n



