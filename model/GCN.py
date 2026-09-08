#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function

import torch.nn as nn
import torch
from torch.nn.parameter import Parameter
import math
from model.Transformer import Transformer
from model.Scale import *

class GCN(nn.Module):
    def __init__(self, input_feature, hidden_feature, p_dropout, num_stage=1, node_n=48):
        """
        :param input_feature: num of input feature
        :param hidden_feature: num of hidden feature
        :param p_dropout: drop out prob.
        :param num_stage: number of Transformer blocks
        :param node_n: number of nodes in graph
        """
        super(GCN, self).__init__()
        self.num_stage = num_stage

        self.linear_in = nn.Linear(input_feature, hidden_feature)
        self.bn_in = nn.BatchNorm1d(node_n * hidden_feature)

        self.spatial_pos_embed = nn.Parameter(torch.randn(1, node_n, hidden_feature) * 0.02)

        self.trans_blocks = nn.ModuleList([
            Transformer(
                d_input=hidden_feature,
                d_model=hidden_feature,
                d_output=hidden_feature,
                n_layers=1,
                n_heads=8,
                d_k=hidden_feature // 8,
                d_v=hidden_feature // 8,
                d_ff=hidden_feature * 2
            ) for _ in range(num_stage)
        ])

        self.linear_out = nn.Linear(hidden_feature, input_feature)

        self.do = nn.Dropout(p_dropout)
        self.act_f = nn.Tanh()

    def forward(self, x):
        res = x
        y = self.linear_in(x)
        b, n, f = y.shape
        y = self.bn_in(y.view(b, -1)).view(b, n, f)
        y = self.act_f(y)
        y = self.do(y)

        y = y + self.spatial_pos_embed

        for i in range(self.num_stage):
            y_in = y
            y = self.trans_blocks[i](y)
            y = y + y_in

        y = self.linear_out(y)
        y = y + res
        return y