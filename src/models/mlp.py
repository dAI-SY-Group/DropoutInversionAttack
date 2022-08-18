from typing import Iterable

import numpy as np
from torch import nn

from src.models.modules import MaskDropout

def build_MLP(config):
    assert config is not None
    widths = [config.hidden_size for _ in range(config.depth)] if config.depth is not None and config.hidden_size is not None else config.widths
    biases = [config.bias for _ in range(config.depth+1)] if config.depth is not None and config.bias is not None else config.biases
    assert isinstance(widths, Iterable), f'widths {(widths)} must be an Iterable of integers!'
    assert isinstance(biases, Iterable), f'biases {(biases)} must be an Iterable of bools!'
    model = MLP(config.data_shape, config.num_classes, widths, biases, config.p)
    return model

class MLP(nn.Module):
    def __init__(self, data_shape=(3,32,32), num_classes=10, widths=[1024,1024], biases=[True, True, True], dropout_rate=0., *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert len(widths)+1 == len(biases), f'You must provide one more bias than widths (because of the input layer)! Got {len(widths)} widths and {len(biases)} biases instead!'
        self.data_shape = data_shape
        self.num_classes = num_classes
        self.flatten = nn.Flatten()
        self.layers = nn.ModuleList()
        for i in range(len(widths)):
            in_channels = np.prod(data_shape) if i == 0 else widths[i-1]
            out_channels = widths[i]
            self.layers.append(MLPBlock(in_channels, out_channels, nn.GELU(), MaskDropout(dropout_rate), biases[i]))
        self.layers.append(MLPBlock(widths[i], num_classes, None, None, biases[i+1])) #last layer

    def forward(self, x):
        x = self.flatten(x)
        for layer in self.layers:
            x = layer(x)
        return x

class MLPBlock(nn.Module):
    def __init__(self, in_channels, out_channels, regularization_layer=None, activation_function=None, use_bias=True, dense=nn.Linear):
        super().__init__()
        self.linear = dense(in_channels, out_channels, bias=use_bias)
        self.regularization_layer = regularization_layer
        self.activation_function = activation_function

    def forward(self, x):
        x = self.linear(x)
        if self.regularization_layer is not None:
            x = self.regularization_layer(x)
        if self.activation_function is not None:
            x = self.activation_function(x)
        return x