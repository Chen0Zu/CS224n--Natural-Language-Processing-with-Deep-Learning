#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

### YOUR CODE HERE for part 1h
import torch
import torch.nn as nn
import torch.functional as F

class Highway(nn.Module):
    def __init__(self, word_embed_size):
        super(Highway, self).__init__()
        self.word_embed_size = word_embed_size
        self.proj = nn.Linear(in_features = self.word_embed_size, out_features = self.word_embed_size)
        self.gate = nn.Linear(in_features = self.word_embed_size, out_features = self.word_embed_size)

    def forward(self, x_conv:torch.Tensor) -> torch.Tensor:
        x_proj = torch.relu_(self.proj(x_conv))
        x_gate = torch.sigmoid(self.gate(x_conv))
        x_highway = x_gate * x_proj + (1-x_gate) * x_conv
        return x_highway
### END YOUR CODE 

