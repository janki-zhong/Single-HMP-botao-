from torch.utils.data import Dataset
import numpy as np
import torch
from utils import data_utils

class Datasets(Dataset):

    def __init__(self, opt, split, actions='all'):
        """
        融合 H36M 内存管理机制和 GPU 加速的 CMU 数据集类
        """
        self.path_to_data = "./datasets/cmu_mocap/"
        self.in_n = opt.input_n
        self.out_n = opt.output_n
        self.split = split

        # 仿照 H36M 引入设备参数，用于提前将数据放到 GPU 上
        self.dev = opt.dev

        # 仿照 H36M 初始化字典和索引列表
        self.p3d = {}
        self.data_idx = []
        seq_len = self.in_n + self.out_n

        is_all = actions
        actions = data_utils.define_actions_cmu(actions)
        # actions = ['walking']

        # 区分训练集与测试集路径
        if split == 0:
            path_to_data = self.path_to_data + '/train/'
            is_test = False
        else:
            path_to_data = self.path_to_data + '/test/'
            is_test = True

        print(f">>> Loading CMU data from {path_to_data}")

        if not is_test:
            all_seqs, dim_ignore, dim_use = data_utils.load_data_cmu_3d_all(opt, path_to_data, actions,
                                                                            self.in_n, self.out_n,
                                                                            is_test=is_test)
        else:
            # 测试集使用 _n 或 _all 取决于你的需求，这里保留你的原有逻辑
            all_seqs, dim_ignore, dim_use = data_utils.load_data_cmu_3d_n(opt, path_to_data, actions,
                                                                          self.in_n, self.out_n,
                                                                          is_test=is_test)

        self.dim_used = dim_use

        # ==========================================================
        # 核心改造区：融合 H3.6M 的 p3d 字典与 data_idx 索引机制
        # ==========================================================
        # CMU 的 all_seqs 通常是切分好的 numpy 数组: [样本数量, seq_len, 特征维度]
        # 我们将其转为 FloatTensor，并直接放到 opt.dev (如 'cuda:0') 中加速

        all_seqs_tensor = torch.from_numpy(all_seqs).float().to(self.dev)
        num_samples = all_seqs_tensor.shape[0]

        for key in range(num_samples):
            # 1. 将独立的序列样本存入字典
            self.p3d[key] = all_seqs_tensor[key]

            # 2. 构建滑动窗口索引。由于 CMU 工具函数内部已经完成了滑动窗口切片，
            # 这里的起始帧 start_frame 恒定为 0。
            # 这保持了与 H36M 代码完全一致的元组结构 (key, start_frame)
            self.data_idx.append((key, 0))

    def __len__(self):
        # 仿照 H36M，返回 data_idx 的长度
        return len(self.data_idx)

    def __getitem__(self, item):
        # 仿照 H36M 的读取逻辑：通过 item 获取 key 和起始帧
        key, start_frame = self.data_idx[item]

        # 构造需要的帧索引序列 (CMU 切片完毕，通常这里 fs 就是 0 到 seq_len-1)
        fs = np.arange(start_frame, start_frame + self.in_n + self.out_n)

        # 返回字典中对应的张量片段
        return self.p3d[key][fs]