import apex.amp as amp
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
from torchattacks.attacks import autoattack 
from collections import namedtuple

# cifar10_mean = (0.4914, 0.4822, 0.4465)
# cifar10_std = (0.2471, 0.2435, 0.2616)

# mu = torch.tensor(cifar10_mean).view(3,1,1).cuda()
# std = torch.tensor(cifar10_std).view(3,1,1).cuda()

# upper_limit = ((1 - mu)/ std)
# lower_limit = ((0 - mu)/ std)

class Transform():
    def __init__(self, dataset, transforms):
        self.dataset, self.transforms = dataset, transforms
        self.choices = None
        
    def __len__(self):
        return len(self.dataset)
           
    def __getitem__(self, index):
        data, labels = self.dataset[index]
        for choices, f in zip(self.choices, self.transforms):
            args = {k: v[index] for (k,v) in choices.items()}
            data = f(data, **args)
        return data, labels
    
    def set_random_choices(self):
        self.choices = []
        x_shape = self.dataset[0][0].shape
        N = len(self)
        for t in self.transforms:
            options = t.options(x_shape)
            x_shape = t.output_shape(x_shape) if hasattr(t, 'output_shape') else x_shape
            self.choices.append({k:np.random.choice(v, size=N) for (k,v) in options.items()})



def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)

class Cutout(namedtuple('Cutout', ('h', 'w'))):
    def __call__(self, x, x0, y0):
        x = x.copy()
        x[:,y0:y0+self.h,x0:x0+self.w].fill(0.0)
        return x

    def options(self, x_shape):
        C, H, W = x_shape
        return {'x0': range(W+1-self.w), 'y0': range(H+1-self.h)} 

def get_loaders(dir_, batch_size, image_normalize, cifar10_mean, cifar10_std, cutout, cutout_len):
    if image_normalize:
        # train_transform = transforms.Compose([
        #     transforms.RandomCrop(32, padding=4),
        #     transforms.RandomHorizontalFlip(),
        #     transforms.ToTensor(),
        #     transforms.Normalize(cifar10_mean, cifar10_std),
        # ])
        train_transform = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(cifar10_mean, cifar10_std),
        ]
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(cifar10_mean, cifar10_std),
        ])
    else:
        # train_transform = transforms.Compose([
        #     transforms.RandomCrop(32, padding=4),
        #     transforms.RandomHorizontalFlip(),
        #     transforms.ToTensor(),
        # ])
        train_transform = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]

        test_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
    if cutout:
        train_transform.append(Cutout(cutout_len, cutout_len))
    train_transform = transforms.Compose(train_transform)

    num_workers = 2
    train_dataset = datasets.CIFAR10(
        dir_, train=True, transform=train_transform, download=True)
    test_dataset = datasets.CIFAR10(
        dir_, train=False, transform=test_transform, download=True)

    val_dataset = datasets.CIFAR10(
        dir_, train=False, transform=test_transform, download=True)
    val_dataset.data = val_dataset.data[:1000]
    val_dataset.targets = val_dataset.targets[:1000]

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



    transforms_ct = [Crop(32, 32), FlipLR()]
    if cutout:
        transforms_ct.append(Cutout(cutout_len, cutout_len))

    dataset = cifar10(dir_)
    train_set = list(zip(transpose(pad(dataset['train']['data'], 4)/255.),
        dataset['train']['labels']))
    train_set_x = Transform(train_set, transforms_ct)
    train_batches = Batches(train_set_x, batch_size, shuffle=True, set_random_choices=True, num_workers=2)


    # return train_loader, test_loader, val_loader
    return train_batches, test_loader, val_loader

def pad(x, border=4):
    return np.pad(x, [(0, 0), (border, border), (border, border), (0, 0)], mode='reflect')

def transpose(x, source='NHWC', target='NCHW'):
    return x.transpose([source.index(d) for d in target]) 

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def attack_pgd(model, X, y, epsilon, alpha, attack_iters, restarts, lower_limit, upper_limit, opt=None, mixup=False, y_a=None, y_b=None, lam=None):
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
            if mixup:
                criterion = F.CrossEntropyLoss()
                loss = mixup_criterion(criterion, model(X+delta), y_a, y_b, lam)
            else:
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
        if mixup:
            criterion = F.CrossEntropyLoss(reduction='none')
            all_loss = mixup_criterion(criterion, model(X+delta), y_a, y_b, lam)
        else:
            all_loss = F.cross_entropy(model(X+delta), y, reduction='none')
        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)

        # all_loss = F.cross_entropy(model(X+delta), y, reduction='none').detach()
        # max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        # max_loss = torch.max(max_loss, all_loss)
    return max_delta

# def attack_pgd(model, X, y, epsilon, alpha, attack_iters, restarts, lower_limit, upper_limit, opt=None):
#     max_loss = torch.zeros(y.shape[0]).cuda()
#     max_delta = torch.zeros_like(X).cuda()
#     for zz in range(restarts):
#         delta = torch.zeros_like(X).cuda()
#         for i in range(len(epsilon)):
#             delta[:, i, :, :].uniform_(-epsilon[i][0][0].item(), epsilon[i][0][0].item())
#         delta.data = clamp(delta, lower_limit - X, upper_limit - X)
#         delta.requires_grad = True
#         for _ in range(attack_iters):
#             output = model(X + delta)
#             index = torch.where(output.max(1)[1] == y)
#             if len(index[0]) == 0:
#                 break
#             loss = F.cross_entropy(output, y)
#             if opt is not None:
#                 with amp.scale_loss(loss, opt) as scaled_loss:
#                     scaled_loss.backward()
#             else:
#                 loss.backward()
#             grad = delta.grad.detach()
#             d = delta[index[0], :, :, :]
#             g = grad[index[0], :, :, :]
#             d = clamp(d + alpha * torch.sign(g), -epsilon, epsilon)
#             d = clamp(d, lower_limit - X[index[0], :, :, :], upper_limit - X[index[0], :, :, :])
#             delta.data[index[0], :, :, :] = d
#             delta.grad.zero_()
#         all_loss = F.cross_entropy(model(X+delta), y, reduction='none').detach()
#         max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
#         max_loss = torch.max(max_loss, all_loss)
#     return max_delta


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




#####################
## data augmentation
#####################

class Crop(namedtuple('Crop', ('h', 'w'))):
    def __call__(self, x, x0, y0):
        return x[:,y0:y0+self.h,x0:x0+self.w]

    def options(self, x_shape):
        C, H, W = x_shape
        return {'x0': range(W+1-self.w), 'y0': range(H+1-self.h)}
    
    def output_shape(self, x_shape):
        C, H, W = x_shape
        return (C, self.h, self.w)
    
class FlipLR(namedtuple('FlipLR', ())):
    def __call__(self, x, choice):
        return x[:, :, ::-1].copy() if choice else x 
        
    def options(self, x_shape):
        return {'choice': [True, False]}

class Cutout(namedtuple('Cutout', ('h', 'w'))):
    def __call__(self, x, x0, y0):
        x = x.copy()
        x[:,y0:y0+self.h,x0:x0+self.w].fill(0.0)
        return x

    def options(self, x_shape):
        C, H, W = x_shape
        return {'x0': range(W+1-self.w), 'y0': range(H+1-self.h)} 
    
    
class Transform():
    def __init__(self, dataset, transforms):
        self.dataset, self.transforms = dataset, transforms
        self.choices = None
        
    def __len__(self):
        return len(self.dataset)
           
    def __getitem__(self, index):
        data, labels = self.dataset[index]
        for choices, f in zip(self.choices, self.transforms):
            args = {k: v[index] for (k,v) in choices.items()}
            data = f(data, **args)
        return data, labels
    
    def set_random_choices(self):
        self.choices = []
        x_shape = self.dataset[0][0].shape
        N = len(self)
        for t in self.transforms:
            options = t.options(x_shape)
            x_shape = t.output_shape(x_shape) if hasattr(t, 'output_shape') else x_shape
            self.choices.append({k:np.random.choice(v, size=N) for (k,v) in options.items()})

#####################
## dataset
#####################
import torchvision
def cifar10(root):
    train_set = torchvision.datasets.CIFAR10(root=root, train=True, download=True)
    test_set = torchvision.datasets.CIFAR10(root=root, train=False, download=True)
    return {
        'train': {'data': train_set.data, 'labels': train_set.targets},
        'test': {'data': test_set.data, 'labels': test_set.targets}
    }

#####################
## data loading
#####################

class Batches():
    def __init__(self, dataset, batch_size, shuffle, set_random_choices=False, num_workers=0, drop_last=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.set_random_choices = set_random_choices
        self.dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True, shuffle=shuffle, drop_last=drop_last
        )
    
    def __iter__(self):
        if self.set_random_choices:
            self.dataset.set_random_choices() 
        return ({'input': x.to("cuda").half(), 'target': y.to("cuda").long()} for (x,y) in self.dataloader)
    
    def __len__(self): 
        return len(self.dataloader)


