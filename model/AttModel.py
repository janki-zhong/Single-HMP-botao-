from torch.nn import Module
from torch import nn
import torch
# import model.transformer_base
import math
from model import GCN
import utils.util as util
import numpy as np
from model.Transformer import Transformer
import torch.nn.functional as F

from model import GCN1


class AttModel(Module):

    def __init__(self, in_features=48, kernel_size=5, d_model=512, num_stage=2, dct_n=10):
        super(AttModel, self).__init__()

        self.kernel_size = kernel_size
        self.d_model = d_model
        # self.seq_in = seq_in
        self.dct_n = dct_n
        # ks = int((kernel_size + 1) / 2)
        assert kernel_size == 10

        self.convQ = nn.Sequential(nn.Conv1d(in_channels=in_features, out_channels=d_model, kernel_size=6,
                                             bias=False),
                                   nn.ReLU(),
                                   nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=5,
                                             bias=False),
                                   nn.ReLU())

        self.convK = nn.Sequential(nn.Conv1d(in_channels=in_features, out_channels=d_model, kernel_size=6,
                                             bias=False),
                                   nn.ReLU(),
                                   nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=5,
                                             bias=False),
                                   nn.ReLU())

        self.gcn = GCN.GCN(input_feature=(dct_n) * 2, hidden_feature=d_model, p_dropout=0.3,
                           num_stage=num_stage,
                           node_n=in_features)
        self.gcn1 = GCN1.GCN1(input_feature=(dct_n) * 2, hidden_feature=d_model, p_dropout=0.3,
                           num_stage=num_stage,
                           node_n=in_features, joint=False)

        # MLP for fusion
        # self.mlp = MLP(input_dim=198, hidden_dim=d_model, output_dim=66)

    def forward(self, src, output_n=25, input_n=50, itera=1, dev='cuda:0', is_train=0, bs_i=0):
        """

        :param src: [batch_size,seq_len,feat_dim]
        :param output_n:
        :param input_n:
        :param frame_n:
        :param dct_n:
        :param itera:
        :return:
        """
        dct_n = self.dct_n
        # dim = src.shape[2]
        # logits = src[:, -20:-10]
        logits = src[:, -output_n:]  # [bs,51:60,dim]
        logits_cat = torch.stack((logits, logits, logits), dim=-1)
        # logits_test_cat = torch.stack((logits_test, logits_test, logits_test), dim=-1)
        src = src[:, :input_n]  # [bs,0:50,dim]
        src_tmp = src.clone()
        bs = src.shape[0]
        src_key_tmp = src_tmp.transpose(1, 2)[:, :, :(input_n - output_n)].clone()  # [bs,66,0:40]
        src_query_tmp = src_tmp.transpose(1, 2)[:, :, -self.kernel_size:].clone()  # [bs,66,41:50]

        dct_m, idct_m = util.get_dct_matrix(self.kernel_size + output_n)
        dct_m = torch.from_numpy(dct_m).float().to(dev)
        idct_m = torch.from_numpy(idct_m).float().to(dev)

        vn = input_n - self.kernel_size - output_n + 1  # 滑动窗口数量
        vl = self.kernel_size + output_n  # 每个窗口的长度
        idx = np.expand_dims(np.arange(vl), axis=0) + \
              np.expand_dims(np.arange(vn), axis=1)
        src_value_tmp = src_tmp[:, idx].clone().reshape(
            [bs * vn, vl, -1])  # [bs*vn, vl, 66]
        src_value_tmp = torch.matmul(dct_m[:dct_n].unsqueeze(dim=0), src_value_tmp).reshape(
            [bs, vn, dct_n, -1]).transpose(2, 3).reshape(
            [bs, vn, -1])  # [bs, vn, 66*vl]
        # 对每个窗口做DCT降维，得到V
        idx = list(range(-self.kernel_size, 0, 1)) + [-1] * output_n  # 预测窗口的索引：前kernel_size个是最近历史，后output_n个是待预测位置
        outputs = []
        # 对Key输入做卷积，提取特征（除以1000是归一化，避免数值过大）
        key_tmp = self.convK(src_key_tmp / 1000.0)
        for i in range(itera):
            query_tmp = self.convQ(src_query_tmp / 1000.0)
            score_tmp = torch.matmul(query_tmp.transpose(1, 2), key_tmp) + 1e-15
            att_tmp = score_tmp / (torch.sum(score_tmp, dim=2, keepdim=True))
            dct_att_tmp = torch.matmul(att_tmp, src_value_tmp)[:, 0].reshape(
                [bs, -1, dct_n])
            input_gcn = src_tmp[:, idx]
            dct_in_tmp = torch.matmul(dct_m[:dct_n].unsqueeze(dim=0), input_gcn).transpose(1, 2)

        # 差分
            # att差分
            dct_att_tmp_diff = torch.diff(dct_att_tmp)   # 生成差分特征（速度维度）
            # dct_att_tmp_diff = torch.cat([dct_att_tmp[:, :, 0].unsqueeze(-1), dct_att_tmp_diff], dim=-1)      # 拼第一帧
            dct_att_tmp_diff = torch.cat([dct_att_tmp_diff, dct_att_tmp[:, :, -1].unsqueeze(-1)], dim=-1)  # 拼最后一帧
            # in差分
            # input_gcn_diff = torch.diff(input_gcn, dim=1)
            # input_gcn_diff = torch.cat([input_gcn[:, 0, :].unsqueeze(1), input_gcn_diff], dim=1)
            # dct_in_tmp_diff = torch.matmul(dct_m[:dct_n].unsqueeze(dim=0), input_gcn_diff).transpose(1, 2)
        # 二差分
            # att差分
            # dct_att_tmp_diff2 = torch.diff(dct_att_tmp_diff)
            # dct_att_tmp_diff = torch.cat([dct_att_tmp[:, :, 0].unsqueeze(-1), dct_att_tmp_diff], dim=-1)      # 拼第一帧
            # dct_att_tmp_diff2 = torch.cat([dct_att_tmp_diff2, dct_att_tmp_diff[:, :, -1].unsqueeze(-1)],dim=-1)  # 拼最后一帧

            # 拼接
            dct_in_t = torch.cat([dct_in_tmp, dct_att_tmp], dim=-1)  # 原始+注意力
            dct_in_t_diff = torch.cat([dct_in_tmp, dct_att_tmp_diff], dim=-1)  # 原始+差分注意力
            # dct_in_tmp_diff_a = torch.cat([dct_in_tmp, dct_att_tmp_diff2], dim=-1)
            # dct_in_tmp_diff2 = torch.cat([dct_att_tmp_diff2, input_gcn_diff2], dim=-1)
            # 加噪
            mean = 0
            std_dev = 1
            gaussian_noise = torch.normal(mean, std_dev, dct_in_t.shape).to(dev)

            # 将高斯噪声加入到原始数据中
            dct_in_t_noise = dct_in_t + gaussian_noise

            # 进入GCN
            dct_out_t = self.gcn(dct_in_t)   # 空间维度
            dct_out_v = self.gcn1(dct_in_t_diff)   # 速度维度
            # dct_out_a = self.gcn1(dct_in_tmp_diff_a)   # 加速度维度
            # dct_out_j = self.gcn1(dct_in_j, joint=True)  # 时间维度
            dct_out_t_noise = self.gcn1(dct_in_t_noise)   # 加噪维度
            # dct_out_tmp_diff2 = self.gcn(dct_in_tmp_diff2)

            # 相加
            # dct_out_fused = 0.9 * dct_out_t + 0.1 * dct_out_tv
            # 融合三个输出

            out_gcn_t = torch.matmul(idct_m[:, :dct_n].unsqueeze(dim=0), dct_out_t[:, :, :dct_n].transpose(1, 2))[:, 10:, :]  # (bs,10,66)
            out_gcn_v = torch.matmul(idct_m[:, :dct_n].unsqueeze(dim=0), dct_out_v[:, :, :dct_n].transpose(1, 2))[:, 10:, :]
            # out_gcn_a = torch.matmul(idct_m[:, :dct_n].unsqueeze(dim=0), dct_out_a[:, :, :dct_n].transpose(1, 2))[:, 10:, :]
            out_gcn_t_noise = torch.matmul(idct_m[:, :dct_n].unsqueeze(dim=0), dct_out_t_noise[:, :, :dct_n].transpose(1, 2))[:, 10:, :]
            # out_gcn_j = torch.matmul(idct_m[:, :dct_n].unsqueeze(dim=0), dct_out_j[:, :, :dct_n].transpose(1, 2))[:, 10:, :]
            out_gcn_cat = torch.stack((out_gcn_t, out_gcn_v, out_gcn_t_noise), dim=-1)
            # out_gcn_cat = torch.stack((out_gcn_t, out_gcn_v), dim=-1)
            out_gcn_gum, gum_index = gumbel_softmax(out_gcn_cat, logits_cat, tau=1, hard=True)
            out_gcn = out_gcn_gum

            outputs.append(out_gcn.unsqueeze(2))
        outputs = torch.cat(outputs, dim=2)
        return outputs

def gumbel_softmax(logits, sele_in, tau=1, hard=False, eps=1e-10, dim=-1):
    gumbels = (-torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log())  # ~Gumbel(0,1)
    gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
    y_soft = F.softmax(gumbels, dim)

    if hard:
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        ret = logits

    sele_out = ret * logits
    result = sele_out.sum(dim=-1)
    return result, index