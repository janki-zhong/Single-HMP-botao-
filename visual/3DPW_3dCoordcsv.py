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
    # 3DPW 通常采用 24个关节点 × 3 = 72维
    in_features = opt.in_features
    d_model = opt.d_model
    kernel_size = opt.kernel_size
    dev = opt.dev
    net_pred = AttModel.AttModel(in_features=in_features, kernel_size=kernel_size, d_model=d_model,
                                 num_stage=opt.num_stage, dct_n=opt.dct_n)
    net_pred.to(dev)

    # 替换为你实际的 3DPW 模型权重路径
    model_path_len = './checkpoint/2026-05-21_06-37-51_main_3DPW_in50_out30_ks10_dctn20/ckpt_best.pth.tar'
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
    print(">>> ckpt len loaded (epoch: {} | err: {})".format(ckpt.get('epoch', 'N/A'), ckpt.get('err', 'N/A')))

    # 3DPW 没有特定动作类别，通常按测试序列名(Sequence)或批次ID提取
    seqs = ["downtown_walk_00", "courtyard_basketball_00", "courtyard_arguing_00"]

    # 3DPW 输出保留 72 维 (24个关节点 × 3)
    out_dim = 54

    gt_csv_path = '3dpw_sample_gt_72dim.csv'
    pred_csv_path = '3dpw_sample_pred_72dim.csv'

    # 修改表头，生成 72 个维度的列名
    csv_head = ["Sequence", "Sample_ID", "Frame"] + [f"Dim_{i}" for i in range(out_dim)]

    for path in [gt_csv_path, pred_csv_path]:
        with open(path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(csv_head)

    global_sample_id = 0
    # 3DPW 一般为 30 FPS，因此每帧间隔约 33.3ms，这里取整数 33
    FPS_MS = 33

    for i, seq in enumerate(seqs):
        gt_data_72, pred_data_72 = run_model(net_pred, seq=seq, opt=opt)

        if gt_data_72 is not None and pred_data_72 is not None:
            _, gt_frames, _ = gt_data_72.shape
            _, pred_frames, _ = pred_data_72.shape

            # 构建 GT 时间戳: 历史帧对应大概 -400ms(12帧), -200ms(6帧), 0ms
            gt_time_steps = [-396, -198, 0] + [(j + 1) * FPS_MS for j in range(gt_frames - 3)]

            # 构建 Pred 时间戳: 33ms, 66ms ...
            pred_time_steps = [(j + 1) * FPS_MS for j in range(pred_frames)]

            with open(gt_csv_path, mode='a', newline='') as f:
                writer = csv.writer(f)
                for t_idx in range(gt_frames):
                    row = [seq, global_sample_id, gt_time_steps[t_idx]] + gt_data_72[0, t_idx].tolist()
                    writer.writerow(row)

            with open(pred_csv_path, mode='a', newline='') as f:
                writer = csv.writer(f)
                for t_idx in range(pred_frames):
                    row = [seq, global_sample_id, pred_time_steps[t_idx]] + pred_data_72[0, t_idx].tolist()
                    writer.writerow(row)

            global_sample_id += 1
            print(f'>>> Extracted sequence for: {seq} (GT: {gt_frames} frames, Pred: {pred_frames} frames)')

    print(">>> Finished! Generated 72-dim CSV files for 3DPW.")


def run_model(net_pred, seq, opt=None):
    net_pred.eval()
    in_n = opt.input_n
    out_n = opt.output_n

    # 3DPW 通常不屏蔽关节点，全量输入/输出
    dim_used = np.arange(72)
    dev = opt.dev

    # 测试数据路径，如果你的3dpw文件名为其他格式请直接修改
    data_path = f'./shared_test_data/3dpw_{seq}.npy'
    if not os.path.exists(data_path):
        print(f"Warning: File {data_path} not found. Skipping sequence {seq}.")
        return None, None

    p3d_3dpw = torch.from_numpy(np.load(data_path)).float().to(dev)

    # 提取网络输入
    p3d_src = p3d_3dpw.clone()[:, :, dim_used]

    with torch.no_grad():
        p3d_out_all = net_pred(p3d_src, input_n=in_n, output_n=out_n, itera=1, dev=dev, is_train=3, bs_i=0)

    # 提取历史 3 帧的 GT (基于30帧/秒，分别取大概 -400ms, -200ms, 0ms)
    idx_1 = max(0, in_n - 13)
    idx_2 = max(0, in_n - 7)
    idx_3 = in_n - 1
    past_indices = [idx_1, idx_2, idx_3]
    gt_past = p3d_3dpw.clone()[0:1, past_indices].cpu().detach().numpy()

    # 提取未来的 out_n 帧 GT
    gt_future = p3d_3dpw.clone()[0:1, in_n:in_n + out_n].cpu().detach().numpy()

    # 提取未来的 out_n 帧 Pred 并在显存中重构
    p3d_out = p3d_3dpw.clone()[:, in_n:in_n + out_n]
    p3d_out[:, :, dim_used] = p3d_out_all[:, :, 0]

    # Pred 只保留未来的预测帧，无需还原被丢弃的关节点(因为3DPW未丢弃)
    pred_future = p3d_out[0:1, :, :].cpu().detach().numpy()

    # GT 拼接完整序列 (3帧历史 + 预测总帧数)
    gt_72 = np.concatenate([gt_past, gt_future], axis=1)

    # Pred 直接返回纯未来预测序列
    pred_72 = pred_future

    return gt_72, pred_72


if __name__ == '__main__':
    option = Options().parse()
    main(option)