from torch.utils.data import Dataset
import numpy as np
import torch
from utils import data_utils

class Datasets(Dataset):

    def __init__(self, opt, split, actions='all'):

        self.path_to_data = "./datasets/cmu_mocap/"
        self.in_n = opt.input_n
        self.out_n = opt.output_n
        self.split = split
        self.dev = opt.dev

        self.p3d = {}
        self.data_idx = []
        seq_len = self.in_n + self.out_n

        is_all = actions
        actions = data_utils.define_actions_cmu(actions)

        if split == 0:
            path_to_data = self.path_to_data + '/train/'
            is_test = False
        else:
            path_to_data = self.path_to_data + '/test/'
            is_test = True

        print(f">>> Loading CMU data from {path_to_data}")

        if not is_test:
            all_seqs, dim_ignore, dim_use = data_utils.load_data_cmu_3d_all(opt, path_to_data, actions, self.in_n, self.out_n, is_test=is_test)
            # all_seqs, dim_ignore, dim_use = data_utils.load_data_cmu_3d_reordered(opt, path_to_data, actions, self.in_n, self.out_n, is_test=is_test)
        else:
            all_seqs, dim_ignore, dim_use = data_utils.load_data_cmu_3d_n(opt, path_to_data, actions, self.in_n, self.out_n, is_test=is_test)
            # all_seqs, dim_ignore, dim_use = data_utils.load_data_cmu_3d_n_1(opt, path_to_data, actions, self.in_n, self.out_n, is_test=is_test)

        self.dim_used = dim_use
        all_seqs_tensor = torch.from_numpy(all_seqs).float().to(self.dev)
        num_samples = all_seqs_tensor.shape[0]

        for key in range(num_samples):
            self.p3d[key] = all_seqs_tensor[key]

            self.data_idx.append((key, 0))

    def __len__(self):
        return len(self.data_idx)

    def __getitem__(self, item):
        key, start_frame = self.data_idx[item]
        fs = np.arange(start_frame, start_frame + self.in_n + self.out_n)
        return self.p3d[key][fs]