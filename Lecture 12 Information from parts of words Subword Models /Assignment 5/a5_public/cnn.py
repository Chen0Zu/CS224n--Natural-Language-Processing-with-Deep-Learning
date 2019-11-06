#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

### YOUR CODE HERE for part 1i
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CNN(nn.Module):
    def __init__(self, char_embed_size, num_filters, max_word_length, kernel_size = 5):
        super(CNN, self).__init__()
        self.char_embed_size = char_embed_size
        self.num_filters = num_filters
        self.max_word_length = max_word_length
        self.kernel_size = kernel_size

        self.conv1d = nn.Conv1d(in_channels = self.char_embed_size,
                                out_channels = self.num_filters,
                                kernel_size = self.kernel_size,
                                bias = True)
        self.max_pool_1d = nn.MaxPool1d(max_word_length - kernel_size + 1)
    def forward(self, x_reshaped):
        x_conv = self.conv1d(x_reshaped)
        x_conv_out = self.max_pool_1d(F.relu_(x_conv)).squeeze()
        return x_conv_out
### END YOUR CODE

