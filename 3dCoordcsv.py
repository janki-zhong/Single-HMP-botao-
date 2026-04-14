from utils import h36motion3d as datasets
from model import AttModel
from utils.opt import Options
from utils import util
from utils import log

from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import numpy as np
import time
import h5py
import csv
import os


def main(opt):
    print('>>> create models')
    # 依然是 66，因为网络结构的输入/输出尺寸没变
    in_features = 66
    d_model = opt.d_model
    kernel_size = opt.kernel_size
    dev = opt.dev
    net_pred = AttModel.AttModel(in_features=in_features, kernel_size=kernel_size, d_model=d_model,
                                 num_stage=opt.num_stage, dct_n=opt.dct_n)
    net_pred.to(dev)

    model_path_len = './checkpoint/2026-03-20_08-41-37_main_h36m_3d_in50_out25_ks10_dctn20_14Llongtime/ckpt_best.pth.tar'
    print(">>> loading ckpt len from '{}'".format(model_path_len))

    if torch.cuda.is_available():
        ckpt = torch.load(model_path_len, map_location=torch.device('cuda:0'))
    else:
        ckpt = torch.load(model_path_len, map_location=torch.device('cpu'))

    dirty_state_dict = ckpt['state_dict']
    clean_state_dict = {
        k: v for k, v in dirty_state_dict.items()
        if not (k.endswith('total_ops') or k.endswith('total_params'))
    }
    net_pred.load_state_dict(clean_state_dict, strict=True)
    print(">>> ckpt len loaded (epoch: {} | err: {})".format(ckpt['epoch'], ckpt['err']))

    acts = ["walking", "eating", "smoking", "discussion", "directions",
            "greeting", "phoning", "posing", "purchases", "sitting",
            "sittingdown", "takingphoto", "waiting", "walkingdog",
            "walkingtogether"]

    # 保存完整的 96 维（32 个关节点 × 3）数据
    out_dim = 96

    gt_csv_path = 'single_sample_gt_96dim.csv'
    pred_csv_path = 'single_sample_pred_96dim.csv'

    # 修改表头，生成 96 个维度的列名
    csv_head = ["Action", "Sample_ID", "Frame"] + [f"Dim_{i}" for i in range(out_dim)]

    for path in [gt_csv_path, pred_csv_path]:
        with open(path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(csv_head)

    global_sample_id = 0

    for i, act in enumerate(acts):
        gt_data_96, pred_data_96 = run_model(net_pred, act=act, opt=opt)

        if gt_data_96 is not None and pred_data_96 is not None:
            # 分别获取 GT 和 Pred 的帧数 (GT: 28, Pred: 25)
            _, gt_frames, _ = gt_data_96.shape
            _, pred_frames, _ = pred_data_96.shape

            # 构建 GT 时间戳: -400ms, -200ms, 0ms, 40ms, 80ms ... 1000ms
            gt_time_steps = [-400, -200, 0] + [(j + 1) * 40 for j in range(gt_frames - 3)]

            # 构建 Pred 时间戳: 40ms, 80ms ... 1000ms
            pred_time_steps = [(j + 1) * 40 for j in range(pred_frames)]

            with open(gt_csv_path, mode='a', newline='') as f:
                writer = csv.writer(f)
                for t_idx in range(gt_frames):
                    row = [act, global_sample_id, gt_time_steps[t_idx]] + gt_data_96[0, t_idx].tolist()
                    writer.writerow(row)

            with open(pred_csv_path, mode='a', newline='') as f:
                writer = csv.writer(f)
                for t_idx in range(pred_frames):
                    row = [act, global_sample_id, pred_time_steps[t_idx]] + pred_data_96[0, t_idx].tolist()
                    writer.writerow(row)

            global_sample_id += 1
            print(f'>>> Extracted sequence for: {act} (GT: {gt_frames} frames, Pred: {pred_frames} frames)')

    print(">>> Finished! Generated 96-dim CSV files with decoupled timestamps.")


def run_model(net_pred, act, opt=None):
    net_pred.eval()
    in_n = opt.input_n
    out_n = opt.output_n

    dim_used = np.array([6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 21, 22, 23, 24, 25,
                         26, 27, 28, 29, 30, 31, 32, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45,
                         46, 47, 51, 52, 53, 54, 55, 56, 57, 58, 59, 63, 64, 65, 66, 67, 68,
                         75, 76, 77, 78, 79, 80, 81, 82, 83, 87, 88, 89, 90, 91, 92])
    dev = opt.dev

    joint_to_ignore = np.array([16, 20, 23, 24, 28, 31])
    index_to_ignore = np.concatenate((joint_to_ignore * 3, joint_to_ignore * 3 + 1, joint_to_ignore * 3 + 2))
    joint_equal = np.array([13, 19, 22, 13, 27, 30])
    index_to_equal = np.concatenate((joint_equal * 3, joint_equal * 3 + 1, joint_equal * 3 + 2))

    data_path = f'./shared_test_data/{act}.npy'
    if not os.path.exists(data_path):
        print(f"Warning: File {data_path} not found. Skipping action {act}.")
        return None, None

    p3d_h36 = torch.from_numpy(np.load(data_path)).float().to(dev)

    p3d_src = p3d_h36.clone()[:, :, dim_used]

    with torch.no_grad():
        p3d_out_all = net_pred(p3d_src, input_n=in_n, output_n=out_n, itera=1, dev=dev, is_train=3, bs_i=0)

    # 提取历史 3 帧的 GT (-400ms, -200ms, 0ms)
    past_indices = [in_n - 11, in_n - 6, in_n - 1]
    gt_past = p3d_h36.clone()[0:1, past_indices].cpu().detach().numpy()

    # 提取未来的 25 帧 GT
    gt_future = p3d_h36.clone()[0:1, in_n:in_n + out_n].cpu().detach().numpy()

    # 提取未来的 25 帧 Pred 并在显存中重构为 96 维
    p3d_out = p3d_h36.clone()[:, in_n:in_n + out_n]
    p3d_out[:, :, dim_used] = p3d_out_all[:, :, 0]
    p3d_out[:, :, index_to_ignore] = p3d_out[:, :, index_to_equal]

    # 核心修改：Pred 只保留未来的 25 帧预测，不再拼接历史输入
    pred_future = p3d_out[0:1, :, :].cpu().detach().numpy()

    # GT 拼接完整序列 (3 + 25 = 28帧)
    gt_96 = np.concatenate([gt_past, gt_future], axis=1)

    # Pred 直接返回纯未来预测序列 (25帧)
    pred_96 = pred_future

    return gt_96, pred_96


if __name__ == '__main__':
    option = Options().parse()
    main(option)