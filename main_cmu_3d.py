from utils import cmumotion3d as datasets
from model import AttModel
from utils.opt import Options
from utils import util
from utils import log
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
import time
import torch.optim as optim
from thop import profile, clever_format

# 【修改 1】导入你自己编写的 CMU 数据集类 (请确保文件路径正确)
# 假设你将刚刚写的类保存在了 utils/cmu_motion_3d.py 中
from utils.cmumotion3d import Datasets


def main(opt):
    lr_now = opt.lr_now
    print('>>> create models')

    # 【注意】运行 CMU 数据集时，请确保启动参数 --in_features 是 75 (25个关节 * 3)
    in_features = opt.in_features
    d_model = opt.d_model
    kernel_size = opt.kernel_size
    dev = opt.dev
    net_pred = AttModel.AttModel(in_features=in_features, kernel_size=kernel_size, d_model=d_model,
                                 num_stage=opt.num_stage, dct_n=opt.dct_n)
    net_pred.to(dev)

    optimizer = optim.Adam(filter(lambda x: x.requires_grad, net_pred.parameters()), lr=opt.lr_now)
    print(">>> total params: {:.2f}M".format(sum(p.numel() for p in net_pred.parameters()) / 1000000.0))

    torch.cuda.synchronize()
    input_data = torch.randn(1, in_features, in_features).to(dev)
    flops, params = profile(net_pred, inputs=(input_data,))
    flops, params = clever_format([flops, params], "%.3f")

    print(f"FLOPs: {flops}")
    print(f"Params: {params}")

    times = 0
    for i in range(30):
        torch.cuda.synchronize()
        net_pred.eval()
        start_time = time.time()
        torch.cuda.synchronize()
        with torch.no_grad():
            output = net_pred(input_data)
        end_time = time.time()
        time1 = end_time - start_time
        if i > 9:
            times += time1
        i += 1
    average_times = times / 20
    print(f'average_cost={average_times}')

    start_epoch = 1

    if opt.is_load or opt.is_eval:
        model_path_len = './{}/ckpt_best.pth.tar'.format(opt.ckpt)
        print(">>> loading ckpt len from '{}'".format(model_path_len))
        ckpt = torch.load(model_path_len)
        start_epoch = ckpt['epoch'] + 1
        err_best = ckpt['err']
        lr_now = ckpt['lr']
        net_pred.load_state_dict(ckpt['state_dict'])
        print(">>> ckpt len loaded (epoch: {} | err: {})".format(ckpt['epoch'], ckpt['err']))

    print('>>> loading datasets')

    # 【修改 2】CMU 专属动作列表
    cmu_actions = ["basketball", "basketball_signal", "directing_traffic", "jumping",
                   "running", "soccer", "walking", "washwindow"]

    if not opt.is_eval:
        # 【修改 3】因为数据已经在 Dataset 的 __init__ 中送入 GPU，
        # 必须设置 num_workers=0 和 pin_memory=False，否则 PyTorch 多进程会崩溃！
        train_dataset = datasets.Datasets(opt, split=0, actions='all')
        print('>>> Training dataset length: {:d}'.format(train_dataset.__len__()))
        data_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True,
                                 num_workers=0, pin_memory=False)

        valid_dataset = datasets.Datasets(opt, split=1, actions='all')
        print('>>> Validation dataset length: {:d}'.format(valid_dataset.__len__()))
        valid_loader = DataLoader(valid_dataset, batch_size=opt.test_batch_size, shuffle=True,
                                  num_workers=0, pin_memory=False)

    # 构建针对每个 CMU 动作的测试字典，这符合论文标准测评方法
    test_loaders = {}
    for act in cmu_actions:
        test_dataset = datasets.Datasets(opt, split=1, actions=act)
        test_loaders[act] = DataLoader(test_dataset, batch_size=opt.test_batch_size, shuffle=False,
                                       num_workers=0, pin_memory=False)

    # evaluation
    if opt.is_eval:
        ret_log = np.array([])
        head = np.array([])
        for act in cmu_actions:
            ret_test = run_model(net_pred, is_train=3, data_loader=test_loaders[act], opt=opt)
            for k in ret_test.keys():
                ret_log = np.append(ret_log, [ret_test[k]])
                head = np.append(head, [act + '_' + k])
        log.save_csv_log(opt, head, ret_log, is_create=True, file_name='test_cmu')

    # training
    if not opt.is_eval:
        err_best = 1000
        for epo in range(start_epoch, opt.epoch + 1):
            is_best = False
            lr_now = util.lr_decay_mine(optimizer, lr_now, 0.1 ** (1 / opt.epoch))
            print('>>> training epoch: {:d}'.format(epo))

            ret_train = run_model(net_pred, optimizer, is_train=0, data_loader=data_loader, epo=epo, opt=opt)
            print('train error: {:.3f}'.format(ret_train['m_p3d']))

            ret_valid = run_model(net_pred, is_train=1, data_loader=valid_loader, opt=opt, epo=epo)
            print('validation error: {:.3f}'.format(ret_valid['m_p3d']))

            # 分别测试 CMU 的各个动作并拼接到 log 中
            ret_log = np.array([epo, lr_now])
            head = np.array(['epoch', 'lr'])

            for k in ret_train.keys():
                ret_log = np.append(ret_log, [ret_train[k]])
                head = np.append(head, ['train_' + k])
            for k in ret_valid.keys():
                ret_log = np.append(ret_log, [ret_valid[k]])
                head = np.append(head, ['valid_' + k])

            avg_test_err = 0
            for act in cmu_actions:
                ret_test = run_model(net_pred, is_train=3, data_loader=test_loaders[act], opt=opt, epo=epo)
                avg_test_err += ret_test[f'#{opt.output_n}']
                for k in ret_test.keys():
                    ret_log = np.append(ret_log, [ret_test[k]])
                    head = np.append(head, [act + '_test_' + k])

            avg_test_err /= len(cmu_actions)
            print(f'testing error (Avg over all actions): {avg_test_err:.3f}')

            log.save_csv_log(opt, head, ret_log, is_create=(epo == 1))
            if ret_valid['m_p3d'] < err_best:
                err_best = ret_valid['m_p3d']
                is_best = True
            log.save_ckpt({'epoch': epo,
                           'lr': lr_now,
                           'err': avg_test_err,
                           'state_dict': net_pred.state_dict(),
                           'optimizer': optimizer.state_dict()},
                          is_best=is_best, opt=opt)


def run_model(net_pred, optimizer=None, is_train=0, data_loader=None, epo=1, opt=None):
    dev = opt.dev

    if is_train == 0:
        net_pred.train()
    else:
        net_pred.eval()

    l_p3d = 0
    if is_train <= 1:
        m_p3d = 0
    else:
        titles = np.array(range(opt.output_n)) + 1
        m_p3d = np.zeros([opt.output_n])

    n = 0
    in_n = opt.input_n
    out_n = opt.output_n

    # 【修改 4】动态获取 CMU 数据集的特征维度，删除所有写死的数字
    dim_used = data_loader.dataset.dim_used
    joint_num = len(dim_used) // 3  # 通常 CMU 是 75 / 3 = 25

    itera = 1
    st = time.time()

    for i, p3d_cmu in enumerate(data_loader):
        batch_size, seq_n, _ = p3d_cmu.shape
        if batch_size == 1 and is_train == 0:
            continue
        n += batch_size
        bt = time.time()

        # 数据已在 GPU，只需要确保类型
        p3d_cmu = p3d_cmu.float()

        # 提取有效维度 [batch, seq_len, 75]
        p3d_src = p3d_cmu[:, :, dim_used]

        # 【修改 5】Ground Truth 切片与重组，形状：[batch, out_n, 25, 3]
        p3d_sup = p3d_src[:, -out_n:].reshape([-1, out_n, joint_num, 3])

        # 模型前向传播，输出所有的序列信息
        p3d_out_all = net_pred(p3d_src, input_n=in_n, output_n=out_n, itera=itera, dev=dev, is_train=is_train, bs_i=i)

        # 整理预测输出，形状：[batch, out_n, itera, 25, 3]
        p3d_out_all = p3d_out_all.reshape([batch_size, out_n, itera, joint_num, 3])

        # 仅取最后一次迭代的结果
        p3d_out = p3d_out_all[:, :, 0]  # [batch, out_n, 25, 3]

        # 【修改 6】完全移除了 H36M 的 joint_to_ignore 覆盖逻辑，直接计算 Loss！

        grad_norm = 0
        if is_train == 0:
            # 训练 Loss
            loss_p3d = torch.mean(torch.norm(p3d_out - p3d_sup, dim=3))
            loss_all = loss_p3d
            optimizer.zero_grad()
            loss_all.backward()
            nn.utils.clip_grad_norm_(list(net_pred.parameters()), max_norm=opt.max_norm)
            optimizer.step()
            l_p3d += loss_p3d.item() * batch_size

        # 测试与验证 MPJPE
        if is_train <= 1:
            mpjpe_p3d = torch.mean(torch.norm(p3d_sup - p3d_out, dim=3))
            m_p3d += mpjpe_p3d.item() * batch_size
        else:
            mpjpe_p3d_test = torch.sum(torch.mean(torch.norm(p3d_sup - p3d_out, dim=3), dim=2), dim=0)
            m_p3d += mpjpe_p3d_test.cpu().data.numpy()

        if i % 1000 == 0:
            print('{}/{}|bt {:.3f}s|tt{:.0f}s|gn{}'.format(i + 1, len(data_loader), time.time() - bt,
                                                           time.time() - st, grad_norm))
    ret = {}
    if is_train == 0:
        ret["l_p3d"] = l_p3d / n
    if is_train <= 1:
        ret["m_p3d"] = m_p3d / n
    else:
        m_p3d = m_p3d / n
        for j in range(out_n):
            ret["#{:d}".format(titles[j])] = m_p3d[j]

    return ret


if __name__ == '__main__':
    option = Options().parse()
    main(option)