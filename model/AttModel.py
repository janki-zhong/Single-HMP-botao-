from torch.nn import Module
from torch import nn
import torch
import math
from model import GCN
import utils.util as util
import numpy as np
import torch.nn.functional as F


# =====================================================================
# 模块 1：物理运动学模糊先验 (Physical Kinematic Fuzzy Prior)
# =====================================================================
class FuzzyKinematicPrior(nn.Module):
    def __init__(self, dct_n=10):
        super(FuzzyKinematicPrior, self).__init__()
        self.dct_n = dct_n
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.beta = nn.Parameter(torch.tensor(1.0))
        self.sigma = nn.Parameter(torch.tensor(0.5))

        self.freq_penalty = nn.Parameter(torch.linspace(0.0, 1.0, dct_n).view(1, 1, dct_n))
        self.noise_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, src_tmp, dct_in_t):
        bs, seq_len, dim = src_tmp.shape
        reshaped_src = src_tmp.view(bs, seq_len, -1, 3)

        vel = torch.diff(reshaped_src, dim=1)
        acc = torch.diff(vel, dim=1)

        vel = torch.cat([vel[:, 0:1], vel], dim=1)
        acc = torch.cat([acc[:, 0:1], acc[:, 0:1], acc], dim=1)

        v_norm = torch.sqrt(torch.sum(vel ** 2, dim=-1) + 1e-8).mean(dim=1)
        a_norm = torch.sqrt(torch.sum(acc ** 2, dim=-1) + 1e-8).mean(dim=1)

        E = self.alpha * v_norm + self.beta * a_norm

        mu = torch.exp(-(E ** 2) / (2 * self.sigma ** 2 + 1e-6))
        mu_spatial = mu.repeat_interleave(3, dim=1)

        mu_t = mu.unsqueeze(1).repeat(1, 1, 3).view(bs, 1, -1).transpose(1, 2)
        freq_p = torch.sigmoid(self.freq_penalty)
        fuzzy_mask = 1.0 - (1.0 - mu_t) * freq_p

        dct_part = dct_in_t[:, :, :self.dct_n]
        att_part = dct_in_t[:, :, self.dct_n:]

        dct_part_filtered = dct_part * fuzzy_mask
        dct_in_t_filtered = torch.cat([dct_part_filtered, att_part], dim=-1)

        std_dev = torch.abs(self.noise_scale) * torch.clamp(1.0 - mu_t, min=1e-8)
        std_dev = std_dev.expand_as(dct_in_t_filtered)

        dynamic_gaussian = torch.normal(mean=0.0, std=std_dev).to(src_tmp.device)
        dct_in_t_dynamic_noise = dct_in_t_filtered + dynamic_gaussian

        return dct_in_t_filtered, dct_in_t_dynamic_noise, mu_spatial


# =====================================================================
# 模块 2：特征分布模糊注意力 (Feature Entropy Fuzzy Attention)
# =====================================================================
class FuzzyFeatureAttention(nn.Module):
    def __init__(self, c=20, r=4):
        super(FuzzyFeatureAttention, self).__init__()
        self.mlp_dim = max(int(c / r), 8)
        self.ch = c
        self.c_param = nn.Parameter(torch.rand([1, 1, 1]), requires_grad=True)
        self.u = nn.Parameter(torch.rand([1, 3, 1, 1]), requires_grad=True)
        self.sig = nn.Parameter(torch.rand([1, 3, 1, 1]), requires_grad=True)
        self.fc1 = nn.Linear(c, self.mlp_dim)
        self.fc2 = nn.Linear(self.mlp_dim, c)

    def maxs_for(self, u_fuzzy, xf):
        s = torch.max(2 * torch.abs(u_fuzzy - 0.5), dim=1)[0]
        out = xf + s - xf * s
        return out

    def fuzzify(self, x):
        x_exp = x.unsqueeze(dim=1)
        u = torch.div(1, 1 + torch.exp(self.u * (x_exp + self.sig)))
        return u

    def forward(self, x):
        x1 = F.relu6(x)
        u = self.fuzzify(x1)
        mu = torch.mean(x1, dim=1, keepdim=True)
        xf = torch.div(1, 1 + torch.exp(self.c_param * (mu - x1)))
        p = self.maxs_for(u, xf)
        x1_defuzzy = torch.div((p * x1).sum(dim=1, keepdim=True), p.sum(dim=1, keepdim=True) + 1e-6)
        x2 = self.fc2(F.relu(self.fc1(x1_defuzzy)))
        x2 = x2.sigmoid()
        out = x * x2
        return out


# =====================================================================
# 模块 3：全局协同混合器
# =====================================================================
class SynergyMixer(nn.Module):
    def __init__(self, node_n, in_features, d_model):
        super(SynergyMixer, self).__init__()
        self.feat_mixer = nn.Sequential(nn.Linear(in_features, d_model), nn.GELU(), nn.Linear(d_model, in_features))
        self.node_mixer = nn.Sequential(nn.Linear(node_n, node_n), nn.GELU(), nn.Linear(node_n, node_n))

    def forward(self, x):
        h = self.feat_mixer(x)
        h_spatial = self.node_mixer(h.transpose(1, 2)).transpose(1, 2)
        return h + h_spatial


# =====================================================================
# 主模型 AttModel (消融：移除基于物理先验的动态门控，回退至静态权重)
# =====================================================================
class AttModel(Module):

    def __init__(self, in_features=48, kernel_size=10, d_model=512, num_stage=2, dct_n=10):
        super(AttModel, self).__init__()

        self.kernel_size = kernel_size
        self.d_model = d_model
        self.dct_n = dct_n
        assert kernel_size == 10

        self.convQ = nn.Sequential(nn.Conv1d(in_channels=in_features, out_channels=d_model, kernel_size=6, bias=False),
                                   nn.ReLU(),
                                   nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=5, bias=False),
                                   nn.ReLU())
        self.convK = nn.Sequential(nn.Conv1d(in_channels=in_features, out_channels=d_model, kernel_size=6, bias=False),
                                   nn.ReLU(),
                                   nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=5, bias=False),
                                   nn.ReLU())

        self.fuzzy_prior = FuzzyKinematicPrior(dct_n=self.dct_n)
        self.fuzzy_feature_attention = FuzzyFeatureAttention(c=(dct_n) * 2, r=4)

        self.gcn = GCN.GCN(input_feature=(dct_n) * 2, hidden_feature=d_model, p_dropout=0.3,
                           num_stage=num_stage, node_n=in_features)

        self.synergy_mixer = SynergyMixer(node_n=in_features, in_features=(dct_n) * 2, d_model=d_model)

        # -------------------------------------------------------------
        # [消融修改]：移除动态 dynamic_gate 模块，使用静态可学习参数作为基线融合机制
        # -------------------------------------------------------------
        self.fusion_weight = nn.Parameter(torch.tensor(0.5))

    def forward(self, src, output_n=10, input_n=50, itera=1, dev='cuda:0', is_train=0, bs_i=0):
        dct_n = self.dct_n

        src = src[:, :input_n]
        src_tmp = src.clone()
        bs = src.shape[0]

        src_key_tmp = src_tmp.transpose(1, 2)[:, :, :(input_n - output_n)].contiguous()
        src_query_tmp = src_tmp.transpose(1, 2)[:, :, -self.kernel_size:].contiguous()

        dct_m, idct_m = util.get_dct_matrix(self.kernel_size + output_n)
        dct_m = torch.from_numpy(dct_m).float().to(dev)
        idct_m = torch.from_numpy(idct_m).float().to(dev)

        vn = input_n - self.kernel_size - output_n + 1
        vl = self.kernel_size + output_n
        idx = np.expand_dims(np.arange(vl), axis=0) + np.expand_dims(np.arange(vn), axis=1)
        src_value_tmp = src_tmp[:, idx].clone().reshape([bs * vn, vl, -1])
        src_value_tmp = torch.matmul(dct_m[:dct_n].unsqueeze(dim=0), src_value_tmp).reshape(
            [bs, vn, dct_n, -1]).transpose(2, 3).reshape([bs, vn, -1])

        idx_pred = list(range(-self.kernel_size, 0, 1)) + [-1] * output_n
        outputs = []

        key_tmp = self.convK(src_key_tmp / 1000.0)

        for i in range(itera):
            query_tmp = self.convQ(src_query_tmp / 1000.0)

            score_tmp = torch.matmul(query_tmp.transpose(1, 2), key_tmp)
            score_tmp = score_tmp + 1e-15

            att_tmp = score_tmp / (torch.sum(score_tmp, dim=2, keepdim=True))
            dct_att_tmp = torch.matmul(att_tmp, src_value_tmp)[:, 0].reshape([bs, -1, dct_n])

            input_gcn = src_tmp[:, idx_pred]
            dct_in_tmp = torch.matmul(dct_m[:dct_n].unsqueeze(dim=0), input_gcn).transpose(1, 2)

            dct_in_t = torch.cat([dct_in_tmp, dct_att_tmp], dim=-1)

            # --------------------------------------------------
            # 【双重模糊串联特征流】
            # --------------------------------------------------
            dct_in_t_filtered, _, mu_spatial = self.fuzzy_prior(input_gcn, dct_in_t)
            dct_in_t_refined = self.fuzzy_feature_attention(dct_in_t_filtered)
            dct_out_gcn = self.gcn(dct_in_t_refined)
            dct_out_mixer = self.synergy_mixer(dct_in_t)

            fused_dct =dct_out_gcn + dct_out_mixer
            # --------------------------------------------------

            out_fused_t = torch.matmul(idct_m[:, :dct_n].unsqueeze(dim=0),
                                       fused_dct[:, :, :dct_n].transpose(1, 2))[:, 10:, :]

            outputs.append(out_fused_t.unsqueeze(2))

        outputs = torch.cat(outputs, dim=2)
        return outputs