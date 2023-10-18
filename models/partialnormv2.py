import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.parameter import Parameter

__all__ = ['PartialNorm2D']


class _PartialNorm(nn.Module):

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True, var_thresh=0):
        super(_PartialNorm, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        self.var_thresh = var_thresh
        if self.affine:
            self.weight = Parameter(torch.Tensor(num_features))
            self.bias = Parameter(torch.Tensor(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.ones(num_features))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)
            self.register_buffer('num_batches_tracked', None)

        self.pretrain_mean = None
        self.pretrain_var = None

        self.reset_parameters()

    def reset_parameters(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)
        if self.affine:
            self.weight.data.uniform_()
            self.bias.data.zero_()
        self.pretrain_mean = None
        self.pretrain_var = None

    def _check_input_dim(self, input):
        return NotImplemented

    def forward(self, input):
        self._check_input_dim(input)
        B, C = input.shape[0], input.shape[1]
        reduce_dim = list(range(input.dim()))
        reduce_dim.pop(1)

        if self.pretrain_mean is None:
            self.pretrain_mean = self.running_mean.clone()
            self.pretrain_var = self.running_var.clone()
            self.min_var, _ = torch.stack(
                [self.pretrain_var, torch.ones_like(self.pretrain_var) * self.var_thresh]).min(dim=0)
        if self.training:
            batch_mean = torch.zeros_like(self.pretrain_mean)
            batch_var = torch.zeros_like(self.pretrain_var)

            output_batch = F.batch_norm(input, batch_mean, batch_var, self.weight, self.bias,
                                        True, 1, self.eps)

            batch_mean_grad = input.mean(dim=reduce_dim, keepdims=True)

            output_pretrain = F.batch_norm(input - batch_mean_grad, torch.zeros_like(batch_mean), self.min_var,
                                           self.weight, self.bias,
                                           False, self.momentum, self.eps)

            use_batch_channel = (batch_var > self.min_var).float()

            output = output_batch * use_batch_channel.view(1, -1, 1, 1) + \
                     output_pretrain * (1 - use_batch_channel.view(1, -1, 1, 1))

            # mean = batch_mean * self.use_batch_channel + self.pretrain_mean * (1 - use_batch_channel)
            var = batch_var * use_batch_channel + self.min_var * (1 - use_batch_channel)

            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var

        else:
            output = F.batch_norm(input, self.running_mean, self.running_var, self.weight, self.bias,
                                  False, self.momentum, self.eps)

        return output


class PartialNorm2D(_PartialNorm):
    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))
