from utils import cmumotion3d as datasets
from model import AttModel
from utils.opt import Options
from utils import util
from utils import log

from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import numpy as np
import time
import torch.optim as optim


# 运行示例:
# python main_cmu_3d_eval.py --is_eval --kernel_size 10 --dct_n 20 --input_n 50 --output_n 25 --ckpt ./checkpoint/pretrained/cmu_model_path/

def main(opt):
    print('>>> create models')

    # CMU 专属维度
    in_features = opt.in_features  # 应该是 75
    d_model = opt.d_model
    kernel_size = opt.kernel_size
    dev = opt.dev
    net_pred = AttModel.AttModel(in_features=in_features, kernel_size=kernel_size, d_model=d_model,
                                 num_stage=opt.num_stage, dct_n=opt.dct_n)
    net_pred.to(dev)

    # 默认加载 best 权重
    model_path_len = '{}/ckpt_best.pth.tar'.format(opt.ckpt)
    print(">>> loading ckpt len from '{}'".format(model_path_len))

    if torch.cuda.is_available():
        ckpt = torch.load(model_path_len, map_location=torch.device(dev))
    else:
        ckpt = torch.load(model_path_len, map_location=torch.device('cpu'))

    # 清理可能存在的额外统计参数
    dirty_state_dict = ckpt['state_dict']
    clean_state_dict = {
        k: v for k, v in dirty_state_dict.items()
        if not (k.endswith('total_ops') or k.endswith('total_params'))
    }
    net_pred.load_state_dict(clean_state_dict, strict=True)
    print(">>> ckpt len loaded (epoch: {} | err: {})".format(ckpt['epoch'], ckpt['err']))

    print('>>> loading datasets')

    # 构建 CSV 表头
    head = np.array(['act'])
    for k in range(1, opt.output_n + 1):
        head = np.append(head, [f'#{k}'])

    # CMU 专属的 8 个评测动作
    acts = ["basketball", "basketball_signal", "directing_traffic", "jumping",
            "running", "soccer", "walking", "washwindow"]

    errs = np.zeros([len(acts) + 1, opt.output_n])

    for i, act in enumerate(acts):
        # CMU 测试集对应 split=1
        test_dataset = datasets.Datasets(opt, split=1, actions=act)
        print('>>> Testing dataset length for {}: {:d}'.format(act, test_dataset.__len__()))

        # 注意：CMU 数据集已在 init 时推入 GPU，必须使用 num_workers=0 和 pin_memory=False
        test_loader = DataLoader(test_dataset, batch_size=opt.test_batch_size, shuffle=False, num_workers=0,
                                 pin_memory=False)

        ret_test = run_model(net_pred, is_train=3, data_loader=test_loader, opt=opt)

        # 打印当前动作最后一帧的误差 (可根据需要修改为打印平均误差)
        print('testing error ({}): {:.3f}'.format(act, ret_test[f'#{opt.output_n}']))

        ret_log = np.array([])
        for k in ret_test.keys():
            ret_log = np.append(ret_log, [ret_test[k]])
        errs[i] = ret_log

    # 计算所有动作的平均值
    errs[-1] = np.mean(errs[:-1], axis=0)

    acts_column = np.expand_dims(np.array(acts + ["average"]), axis=1)
    value = np.concatenate([acts_column, errs.astype(np.str_)], axis=1)

    print("\n--- Final Results ---")
    print(value)

    # 保存结果到 CSV
    log.save_csv_log(opt, head, value, is_create=True, file_name='test_cmu_action_eval')


def run_model(net_pred, optimizer=None, is_train=0, data_loader=None, epo=1, opt=None):
    net_pred.eval()
    titles = np.array(range(opt.output_n)) + 1
    m_p3d_cmu = np.zeros([opt.output_n])
    n = 0
    in_n = opt.input_n
    out_n = opt.output_n
    dev = opt.dev

    # 动态获取有效的特征维度 (通常为 75)
    dim_used = data_loader.dataset.dim_used
    joint_num = len(dim_used) // 3

    itera = 1

    for i, p3d_cmu in enumerate(data_loader):
        batch_size, seq_n, _ = p3d_cmu.shape
        n += batch_size

        # 确保数据格式并转移到设备
        p3d_cmu = p3d_cmu.float().to(dev)

        # 切片提取所需维度 [batch, seq_len, 75]
        p3d_src = p3d_cmu[:, :, dim_used]

        # 构造 Ground Truth，形状 [batch, out_n, 25, 3]
        p3d_sup = p3d_src[:, -out_n:].reshape([-1, out_n, joint_num, 3])

        with torch.no_grad():
            p3d_out_all = net_pred(p3d_src, input_n=in_n, output_n=out_n, itera=itera, dev=dev, is_train=is_train,
                                   bs_i=i)

        # 整理输出形状 [batch, out_n, itera, 25, 3]
        p3d_out_all = p3d_out_all.reshape([batch_size, out_n, itera, joint_num, 3])

        # 仅取最后一次迭代的结果 [batch, out_n, 25, 3]
        p3d_out = p3d_out_all[:, :, 0]

        # 计算 MPJPE 误差
        mpjpe_p3d = torch.sum(torch.mean(torch.norm(p3d_sup - p3d_out, dim=3), dim=2), dim=0)
        m_p3d_cmu += mpjpe_p3d.cpu().data.numpy()

    ret = {}
    m_p3d_cmu = m_p3d_cmu / n
    for j in range(out_n):
        ret["#{:d}".format(titles[j])] = m_p3d_cmu[j]

    return ret


if __name__ == '__main__':
    option = Options().parse()

    # 强制覆盖参数以适配 CMU
    option.cuda_idx = option.dev
    option.in_features = 75
    option.is_eval = True

    main(option)