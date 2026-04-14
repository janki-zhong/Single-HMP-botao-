import torch
import torch.nn as nn
import torch.nn.functional as F


class FuzzyAttention(nn.Module):
    def __init__(self, c, r=16):  # c输入通道数 r通道压缩比例，默认为16
        super(FuzzyAttention, self).__init__()
        self.mlp_dim = max(int(c / r), 8)
        self.ch = c
        self.c = nn.Parameter(torch.rand([1, 1, 1, 1, 1]), requires_grad=True)  # 控制 sigmoid 的陡峭度
        self.u = nn.Parameter(torch.rand([1, 3, 1, 1, 1, 1]), requires_grad=True)
        self.sig = nn.Parameter(torch.rand([1, 3, 1, 1, 1, 1]), requires_grad=True)  # 三个通道的模糊隶属度函数参数（目标/模糊/背景）
        self.cv1 = nn.Conv3d(c, self.mlp_dim, 1)
        self.cv2 = nn.Conv3d(self.mlp_dim, c, 1)    # 1x1x1 卷积，用于通道压缩与恢复

    # 计算基于模糊熵思想的不确定性加权
    def maxs_for(self, x, xf):  # x 是模糊隶属度值（越靠近0.5越模糊），xf 是空间加权；
        s = torch.max(2 * torch.abs(x - 0.5), dim=1)[0]
        out = xf + s - xf * s  # 使用 fuzzy algebraic sum：out = xf ⊕ s
        return out

    # 该函数将输入 x 转化为三个 fuzzy 隶属度图（通道维度为3）
    def fuzzify(self, x):
        u = torch.div(1, 1 + torch.exp(self.u * (x.unsqueeze(dim=1) + self.sig)))
        # 模拟“目标中心”、“模糊边缘”、“背景”三种语义区域；使用可学习sigmoid函数分别映射每个点的隶属度值，范围在[0,1]
        return u

    def forward(self, x):
        x1 = F.relu6(x)  # 将输入特征限制在 [0,6]，以稳定训练
        u = self.fuzzify(x1)  # 模糊化：生成3个通道的 fuzzy 隶属度
        mu = torch.mean(x1, axis=(-1, -2), keepdim=True)    # 计算输入特征图 x1 在空间维度（即高度 H 和宽度 W）上的 均值 mu
        xf = torch.div(1, 1 + torch.exp(self.c * (mu - x1)))    # 通过 mu 和 x1 的差值，以及一个可学习的参数 self.c，对每个体素进行 sigmoid 模糊化
        # 计算每通道均值 mu，以此为中心，再次通过 sigmoid 模糊化，得到 xf：每个体素与均值的相似程度。
        # xf = torch.clamp(self.a * x + self.b, min=0, max=1)
        p = self.maxs_for(u, xf)  # 计算加权图 p，强调低不确定区域
        x1_defuzzy = torch.div((p * x1).sum(axis=(2, 3, 4), keepdim=True), p.sum(axis=(2, 3, 4), keepdim=True) + 1e-6)
        # 去模糊化过程：依据不确定性权重 p 重新加权特征，获得特征中心信息
        x2 = self.cv2(F.relu(self.cv1(x1_defuzzy)))
        x2 = x2.sigmoid()
        # MLP + sigmoid 输出 attention mask，范围在(0,1)
        out = x * x2  # 对输入特征进行通道重加权融合，强化关键区域
        return out


if __name__ == '__main__':
    net = FuzzyAttention(c=8)
    net = net.cuda()
    data = torch.randn((1, 8, 16, 160, 192)).cuda()  # 随机生成一个形状为 [1, 8, 16, 160, 192] 的3D张量
    with torch.no_grad():
        res = net(data)

    print(res.size())