#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function

import torch.nn as nn
import torch
from torch.nn.parameter import Parameter
import math
from model.Transformer import Transformer
# 即使不使用，也保留原始 import 以免触发全局路径错误
from model.Scale import *

alpha = 0.2
lamda = 1.5


# =====================================================================
# 纯净版消融架构：Pure Spatial Transformer (伪装为 GCN 供外部调用)
# 核心消融点：
# 1. 移除所有 GraphConvolution (放弃固定的图邻接矩阵特征聚合)
# 2. 移除所有多尺度池化与人工融合 (放弃 0.3 的硬编码躯干/肢体规则)
# 3. 将骨架的 node_n 个关节视为 Sequence，完全依赖自注意力机制挖掘空间拓扑
# =====================================================================
class GCN(nn.Module):
    def __init__(self, input_feature, hidden_feature, p_dropout, num_stage=1, node_n=48):
        """
        :param input_feature: num of input feature
        :param hidden_feature: num of hidden feature
        :param p_dropout: drop out prob.
        :param num_stage: number of Transformer blocks (对齐原 num_stage)
        :param node_n: number of nodes in graph (视为 Sequence Length)
        """
        super(GCN, self).__init__()
        self.num_stage = num_stage

        # 1. 初始特征映射 (替代原第一层 GraphConvolution)
        self.linear_in = nn.Linear(input_feature, hidden_feature)
        self.bn_in = nn.BatchNorm1d(node_n * hidden_feature)

        # 2. 堆叠纯 Transformer 编码器块
        # 这里复用了你项目中原有的 Transformer 模块
        # d_k 和 d_v 根据 hidden_feature 动态自适应，避免原版硬编码 256 的隐患
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

        # 3. 输出特征映射 (替代原最后一层 gc7)
        self.linear_out = nn.Linear(hidden_feature, input_feature)

        self.do = nn.Dropout(p_dropout)
        self.act_f = nn.Tanh()

    def forward(self, x):
        """
        输入 x 形状: [batch_size, node_n, input_feature]
        """
        # 保存全局残差
        res = x

        # -------------------------------------------------------------
        # 步骤 1：升维到隐层空间 (Embedding)
        # -------------------------------------------------------------
        y = self.linear_in(x)
        b, n, f = y.shape
        y = self.bn_in(y.view(b, -1)).view(b, n, f)
        y = self.act_f(y)
        y = self.do(y)

        # -------------------------------------------------------------
        # 步骤 2：全空间自注意力交互 (Spatial Self-Attention)
        # 彻底抛弃了图结构的邻接矩阵和多尺度池化
        # -------------------------------------------------------------
        for i in range(self.num_stage):
            y_in = y
            # Transformer 将 n 个关节视为 sequence 计算注意力得分
            y = self.trans_blocks[i](y)
            # 添加 Block 级残差连接 (对齐原 GC_Block 的残差逻辑)
            y = y + y_in

        # -------------------------------------------------------------
        # 步骤 3：降维并输出
        # -------------------------------------------------------------
        y = self.linear_out(y)

        # 加上全局输入残差
        y = y + res

        return y