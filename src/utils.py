import os
import random
import types
import yaml
from functools import wraps
import time

import matplotlib.pyplot as plt
from munch import DefaultMunch
import numpy as np
import torch
from torch import optim
from torch.optim import lr_scheduler
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair, _quadruple


def get_device():
    """Get device."""
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    if torch.cuda.is_available():
        print(f'GPU : {torch.cuda.get_device_name(device=device)}')
    return device

def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        return_values = func(*args, **kwargs)
        total_s = time.time() - start
        total_m = total_s/60
        if total_s < 3:
            print('Execution of {} took {:.2f} ms.'.format(func.__name__, total_s*1000))
        elif total_m > 60:
            total_h = int(total_m//60)
            print('Execution of {} took {}h {}min.'.format(func.__name__, total_h, int(round(total_m%60))))
        else:
            print('Execution of {} took {:.2f} minutes.'.format(func.__name__, total_m))
        return return_values
    return wrapper

def build_config(config_file):
    with open(config_file, 'r') as yaml_file:
        data = yaml.load(yaml_file, Loader=yaml.Loader)
    return DefaultMunch.fromDict(data, None)
    

def set_seeds(seed=42):
    torch.manual_seed(seed + 1)
    torch.cuda.manual_seed(seed + 2)
    torch.cuda.manual_seed_all(seed + 3)
    np.random.seed(seed + 4)
    torch.cuda.manual_seed_all(seed + 5)
    random.seed(seed + 6)
    print(f'Seed was set to {seed}')

def get_dataset_mean_dev_tensors(dataset):
    constants = build_config(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'constants.yaml'))    
    mean = eval(f'constants.{dataset}.mean')
    std = eval(f'constants.{dataset}.std')
    if mean is None or std is None:
        raise ValueError(f'Do not have the constants stored for the {dataset} dataset. Try one of {list(constants.keys())} (See constants.yaml).')
    dm = torch.as_tensor(mean)[:, None, None]
    ds = torch.as_tensor(std)[:, None, None]
    return dm, ds

def set_optimizer(model, optimizer_config):
    def set_lr(optim, lr):
        for g in optim.param_groups:
            g['lr'] = lr
    parameters = model if type(model) == list else model.parameters()
    if optimizer_config.name == 'SGD':
        momentum = optimizer_config.momentum if optimizer_config is None else 0
        weight_decay = optimizer_config.weight_decay if optimizer_config.weight_decay is None else 0
        optimizer = optim.SGD(parameters, lr=optimizer_config.lr, momentum=momentum, weight_decay = weight_decay)
    elif optimizer_config.name == 'AdamW':
        optimizer = optim.AdamW(parameters, lr=optimizer_config.lr, betas=(optimizer_config.beta1, optimizer_config.beta2), weight_decay = optimizer_config.weight_decay)
    elif optimizer_config.name == 'Adam':
        optimizer = optim.Adam(parameters, lr=optimizer_config.lr, betas=(optimizer_config.beta1, optimizer_config.beta2), weight_decay = optimizer_config.weight_decay)
    elif optimizer_config.name == 'LBFGS':
        optimizer = optim.LBFGS(parameters, optimizer_config.lr)
    else:
        raise ValueError(f'The optimizer {optimizer_config.name} is not implemented yet..')

    print(f'Using {optimizer_config.name} optimizer with learning rate {optimizer_config.lr}')
    optimizer.lr = lambda: optimizer.param_groups[0]['lr']
    optimizer.set_lr = lambda lr: set_lr(optimizer, lr)

    return optimizer

def set_scheduler(optimizer, validation_frequency, scheduler_config=None, use_val=False):
    def last_lr(self):
        return self.state_dict()['_last_lr'][0]
    if scheduler_config is None:
        return None
    else:
        name = scheduler_config.name

    if name == 'StepLR':
        scheduler = lr_scheduler.StepLR(optimizer, scheduler_config.step_size)
        scheduler.after_val = False
    elif name == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode = scheduler_config.mode, factor = scheduler_config.factor, patience=scheduler_config.patience)
        scheduler.after_val = True
    elif name == 'MultiStepLR':
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones = scheduler_config.milestones, gamma = scheduler_config.gamma)
        scheduler.after_val = False
    else:
        raise ValueError(f'The scheduler {name} is not implemented yet..')

    print(f'Using {name} as a learning rate scheduler.')
    if scheduler.after_val and validation_frequency != 1:
        print(f'You use a learning rate scheduler that only steps after each evaluation but you only evaluate after every {validation_frequency} rounds!')
    m = scheduler_config.metric
    if scheduler_config.use_loss: m = m + '_L'
    m = 'VAL_' + m if use_val else 'TST_' + m
    scheduler.metric = m
    scheduler.lr = types.MethodType(last_lr, scheduler)
    return scheduler

class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    Code inspired by https://github.com/Bjarten/early-stopping-pytorch 26.03.2021
    """
    def __init__(self, patience=7, delta=0, metric='CrossEntropy', use_loss=True, subject_to='min', use_val=True, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            metric (str): string of the metric we are looking at in our history
            subject_to (str): Defines whether the metric is subject to minimazation or maximization; 'min' or 'max' (defaut or when misspelled: 'min')
            verbose (bool): If True, logs a message for each validation loss improvement.
                            Default: False
        """
        self.patience = patience
        self.use_loss = use_loss
        self.use_val = use_val        
        if use_loss: metric = metric + '_L'
        if use_val is not None:
            metric = 'VAL_' + metric if use_val else 'TST_' + metric
        self.metric = metric
        self.subject_to = subject_to
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.stop = False
        self.improved = False
        self.delta = delta

    def get_state(self):
        return self.__dict__

    def set_state(self, state_dict):
        self.__dict__ = state_dict

    def __call__(self, metric):
        if self.subject_to == 'max':
            score = -metric
        else:
            score = metric

        if self.best_score is None:
            self.improved = True
            self.best_score = score
        elif score >= self.best_score + self.delta:
            self.improved = False
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.stop = True
        else:
            self.improved = True
            self.best_score = score
            self.counter = 0

    def reset(self):
        self.counter = 0
        self.best_score = None
        self.stop = False
        self.improved = False
                

def show(img, scale=None, ret_ax=False, savefig=None):
    img = image_preprint(img, scale)
    if len(img.shape) == 2:
        ax = plt.imshow(img, cmap='gray');
    else:
        ax = plt.imshow(img);
    plt.axis('off')
    if ret_ax:
        return ax
    if savefig:
        savefig = savefig if savefig.endswith('.pdf') else savefig+'.pdf'
        plt.savefig(savefig, bbox_inches='tight', transparent=True)
    plt.show()
    plt.close()

def show_multiple(images, columns, rows, img_scale=None, titles=[], savefig=None, figsize=(10,5), dpi=150):
    """Creates a matrix of images

    Args:
        images (list): List of images
        columns (int): number of columns
        rows ([int): number of rows
        img_scale (str, optional): preprint string. Defaults to None.
        titles (list, optional): list of strings that is used as column titles. Defaults to [].
        savefig (str, optional): path to save img to. Defaults to None.
        figsize (tuple, optional): plt.figsize. Defaults to (10,5).
        dpi (int, optional): plt.dpi. Defaults to 150.
    """
    assert len(images) == columns*rows, f'The number of images does not match the number of columns and rows! {len(images)} != {columns*rows}'
    fig=plt.figure(figsize=figsize, dpi=dpi)
    for i in range(1, columns*rows +1):
        img = image_preprint(images[i-1], scale = img_scale)
        ax = fig.add_subplot(rows, columns, i)
        plt.axis('off')
        plt.imshow(img)

    axes = fig.axes
    for ax, title in zip(axes, titles):
        ax.set_title(title)

    plt.subplots_adjust(wspace=0.1, hspace=0.05)
    if savefig:
        savefig = savefig if savefig.endswith('.pdf') else savefig+'.pdf'
        plt.savefig(savefig, bbox_inches='tight')
    plt.show()
    plt.close()

def image_preprint(img, scale=None):
    img = img.clone().detach().cpu()
    if scale == None:
        pass
    elif scale == '01':
        n = Normalizer(mode='single')
        img = n(img)
    elif scale == 'clip':
        img = img.clamp(0,1)
    else:
        dm, ds = get_dataset_mean_dev_tensors(scale)
        dn = DeNormalizer(dm.cpu(), ds.cpu())
        img = dn(img)
        img = img.clamp(0,1)
    img = img.squeeze()
    img = img.permute(1, 2, 0).cpu() if len(img.shape) == 3 else img.cpu()
    return img

class Normalizer(object):
    def __init__(self, mean=None, std=None, mode='batch'):
        self.mean = mean
        self.std = std
        self.mode = mode
        if mode == 'stat':
            assert mean is not None and std is not None, 'If data is to be normalized based on specific statistics, you need to provide mean and std.mean'
            self.norm_fn = self.stat_norm
        elif mode == 'batch':
            self.norm_fn = self.scale_01
        elif mode == 'single':
            self.norm_fn = self.scale_single_01
        else: 
            raise ValueError(f'The mode {mode} is not implemented yet.')
    
    def __call__(self, tensor):
        return self.norm_fn(tensor)

    def stat_norm(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.div_(s).sub_(m)
        return tensor
    
    def scale_01(self, tensor):
        return (tensor.max()-tensor) / ((tensor.max() - tensor.min())+1e-5)
   
    def scale_single_01(self, tensor):
        t = []
        for x in tensor:
            t.append(self.scale_01(x))
        return torch.stack(t)
 
class DeNormalizer(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor