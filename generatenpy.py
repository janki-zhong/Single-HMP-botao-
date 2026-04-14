import os
import torch
import numpy as np
from utils.opt import Options
from utils import h36motion3d as datasets
from torch.utils.data import DataLoader


def save_shared_data():
    opt = Options().parse()
    # 强制进入评估模式
    opt.is_eval = True

    acts = ["walking", "eating", "smoking", "discussion", "directions",
            "greeting", "phoning", "posing", "purchases", "sitting",
            "sittingdown", "takingphoto", "waiting", "walkingdog",
            "walkingtogether"]

    save_dir = './shared_test_data'
    os.makedirs(save_dir, exist_ok=True)

    for act in acts:
        # 使用你 Ours 的数据集类读取数据
        test_dataset = datasets.Datasets(opt, split=2, actions=[act])
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

        for p3d_h36 in test_loader:
            # p3d_h36 shape 应该是 [1, seq_len, 96]
            save_path = os.path.join(save_dir, f'{act}.npy')
            np.save(save_path, p3d_h36.cpu().numpy())
            print(f">>> Saved common test data for: {act} -> {save_path}")
            break  # 每个动作只取严格的第一个 batch (第一个样本)


if __name__ == '__main__':
    save_shared_data()