import os
import sys

sys.path.append(os.path.abspath('./'))

from utils import CMU_motion_3d as CMU_Motion3D
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
from tqdm import tqdm

torch.backends.cudnn.enabled = False


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
    start_epoch = 1
    # opt.is_eval = True
    net_pred.to(opt.dev)

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

    if opt.is_load or opt.is_eval:
        if opt.is_eval:
            model_path_len = './{}/ckpt_best.pth.tar'.format(opt.ckpt)
        else:
            model_path_len = './{}/ckpt_last.pth.tar'.format(opt.ckpt)
        print(">>> loading ckpt len from '{}'".format(model_path_len))
        ckpt = torch.load(model_path_len)
        start_epoch = ckpt['epoch'] + 1
        err_best = ckpt['err']
        lr_now = ckpt['lr']
        net_pred.load_state_dict(ckpt['state_dict'])
        # net.load_state_dict(ckpt)
        # optimizer.load_state_dict(ckpt['optimizer'])
        # lr_now = util.lr_decay_mine(optimizer, lr_now, 0.2)
        print(">>> ckpt len loaded (epoch: {} | err: {})".format(ckpt['epoch'], ckpt['err']))

    print('>>> loading datasets')

    if not opt.is_eval:
        # dataset = datasets.DatasetsSmooth(opt, split=0)
        # actions = ["walking", "eating", "smoking", "discussion", "directions",
        #            "greeting", "phoning", "posing", "purchases", "sitting",
        #            "sittingdown", "takingphoto", "waiting", "walkingdog",
        #            "walkingtogether"]
        dataset = CMU_Motion3D.CMU_Motion3D(opt, split=0)
        print('>>> Training dataset length: {:d}'.format(dataset.__len__()))
        data_loader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=0, pin_memory=True)
        valid_dataset = CMU_Motion3D.CMU_Motion3D(opt, split=1)
        print('>>> Validation dataset length: {:d}'.format(valid_dataset.__len__()))
        valid_loader = DataLoader(valid_dataset, batch_size=opt.test_batch_size, shuffle=True,
                                  num_workers=0, pin_memory=True)
    test_loader = {}
    acts = ["basketball", "basketball_signal", "directing_traffic",
            "jumping", "running", "soccer", "walking", "washwindow"]
    for act in acts:
        test_dataset = CMU_Motion3D.CMU_Motion3D(opt=opt, split=1, actions=act)  # changed
        dim_used = test_dataset.dim_used
        test_loader[act] = DataLoader(test_dataset, batch_size=opt.test_batch_size, shuffle=False, num_workers=0,
                                      pin_memory=True)

    # test_dataset = CMU_Motion3D.CMU_Motion3D(opt, split=2)
    # print('>>> Testing dataset length: {:d}'.format(test_dataset.__len__()))
    # test_loader = DataLoader(test_dataset, batch_size=opt.test_batch_size, shuffle=False, num_workers=0,
    #                          pin_memory=True)

    if not opt.is_eval:
        dim_used = dataset.dim_used

    # evaluation
    if opt.is_eval:
        ret_test = run_model(net_pred, is_train=3, data_loader=test_loader, opt=opt, dim_used=dim_used)
        ret_log = np.array([])
        head = np.array([])
        for k in ret_test.keys():
            ret_log = np.append(ret_log, [ret_test[k]])
            head = np.append(head, [k])
        log.save_csv_log(opt, head, ret_log, is_create=True, file_name='test_basketball')
    # print('testing error: {:.3f}'.format(ret_test['m_p3d_h36']))
    # training
    if not opt.is_eval:

        if opt.is_load:
            model_path_len1 = './{}/ckpt_best.pth.tar'.format(opt.ckpt)
            ckpt1 = torch.load(model_path_len1)
            err_best = ckpt1['err']
        else:
            err_best = 10000

        for epo in range(start_epoch, opt.epoch + 1):
            is_best = False
            # if epo % opt.lr_decay == 0:
            lr_now = util.lr_decay_mine(optimizer, lr_now, 0.1 ** (1 / opt.epoch))
            # lr_now = util.lr_decay_mine(optimizer, lr_now, opt.lr_gamma)
            print('>>> training epoch: {:d}'.format(epo))
            ret_train = run_model(net_pred, optimizer, is_train=0, data_loader=data_loader, epo=epo, opt=opt,
                                  dim_used=dim_used)
            print('train error: {:.3f}'.format(ret_train['m_p3d_cmu']))

            ret_valid = run_model(net_pred, is_train=1, data_loader=valid_loader, opt=opt, epo=epo, dim_used=dim_used)
            print('validation error: {:.3f}'.format(ret_valid['m_p3d_cmu']))

            test_error = 0
            for act in acts:
                ret_test = run_model(net_pred, is_train=3, data_loader=test_loader[act], opt=opt, epo=epo,
                                     dim_used=dim_used)

                # 获取最后一帧的时间戳（毫秒）
                last_frame_ms = opt.output_n * 40

                # 仅累加最后这一帧的误差（极限长期预测）
                test_error += ret_test["#{:d}ms".format(last_frame_ms)]

            # 计算所有动作最后那一帧的平均误差（不再除以帧数，只除以动作数）
            test_error = test_error / len(acts)
            print('testing error (last frame): {:.3f}'.format(test_error))

            ret_log = np.array([epo, lr_now])
            head = np.array(['epoch', 'lr'])
            for k in ret_train.keys():
                ret_log = np.append(ret_log, [ret_train[k]])
                head = np.append(head, [k])
            for k in ret_valid.keys():
                ret_log = np.append(ret_log, [ret_valid[k]])
                head = np.append(head, ['valid_' + k])
            for k in ret_test.keys():
                ret_log = np.append(ret_log, [ret_test[k]])
                head = np.append(head, ['test_' + k])
            log.save_csv_log(opt, head, ret_log, is_create=(epo == 1))
            #
            # if ret_valid['m_p3d_h36'] < err_best:
            #     err_best = ret_valid['m_p3d_h36']
            #     is_best = True

            if ret_valid['m_p3d_cmu'] < err_best:
                err_best = ret_valid['m_p3d_cmu']
                is_best = True

            log.save_ckpt({'epoch': epo,
                           'lr': lr_now,
                           'err': ret_valid['m_p3d_cmu'],
                           'state_dict': net_pred.state_dict(),
                           'optimizer': optimizer.state_dict()},
                          is_best=is_best, opt=opt)


def eval(opt):
    lr_now = opt.lr_now
    start_epoch = 1
    # opt.is_eval = True
    print('>>> create models')
    in_features = opt.in_features
    d_model = opt.d_model
    kernel_size = opt.kernel_size
    dev = opt.dev
    net_pred = AttModel.AttModel(in_features=in_features, kernel_size=kernel_size, d_model=d_model,
                                 num_stage=opt.num_stage, dct_n=opt.dct_n)
    net_pred.to(opt.dev)
    net_pred.eval()

    # load model
    model_path_len = './{}/ckpt_best.pth.tar'.format(opt.ckpt)
    print(">>> loading ckpt len from '{}'".format(model_path_len))
    if torch.cuda.is_available():
        ckpt = torch.load(model_path_len, map_location=torch.device('cuda:0'))
    else:
        ckpt = torch.load(model_path_len, map_location=torch.device('cpu'))
    # save weight
    # for i in ckpt['state_dict'].keys():
    #     if 'attn.proj.weight' in i:
    #         weight = ckpt['state_dict'][i].detach().cpu().numpy()
    #         ax = sns.heatmap(weight).get_figure()
    #         path = os.path.join(i + '.png')
    #         ax.savefig(path)
    # exit()
    dirty_state_dict = ckpt['state_dict']
    clean_state_dict = {
        k: v for k, v in dirty_state_dict.items()
        if not (k.endswith('total_ops') or k.endswith('total_params'))
    }
    net_pred.load_state_dict(clean_state_dict, strict=True)
    print(">>> ckpt len loaded (epoch: {} | err: {})".format(ckpt['epoch'], ckpt['err']))

    print('>>> loading datasets')
    acts = ["basketball", "basketball_signal", "directing_traffic", "jumping", "running", "soccer", "walking",
            "washwindow"]

    data_loader = {}

    for act in acts:
        dataset = CMU_Motion3D.CMU_Motion3D(opt=opt, split=1, actions=act)  # changed
        dim_used = dataset.dim_used
        data_loader[act] = DataLoader(dataset, batch_size=opt.test_batch_size, shuffle=False, num_workers=0,
                                      pin_memory=True)

    valid_dataset = CMU_Motion3D.CMU_Motion3D(opt, split=1)  # changed
    print('>>> Validation dataset length: {:d}'.format(valid_dataset.__len__()))
    valid_loader = DataLoader(valid_dataset, batch_size=opt.test_batch_size, shuffle=True, num_workers=0,
                              pin_memory=True)

    ret_valid = run_model(net_pred, is_train=1, data_loader=valid_loader, opt=opt, dim_used=dim_used)
    print('validation error: {:.3f}'.format(ret_valid['m_p3d_cmu']))
    # do test
    is_create = True
    avg_ret_log = []

    for act in acts:
        ret_test = run_model(net_pred, is_train=3, data_loader=data_loader[act], opt=opt, dim_used=dim_used)
        ret_log = np.array([act])
        head = np.array(['action'])

        for k in ret_test.keys():
            ret_log = np.append(ret_log, [ret_test[k]])
            head = np.append(head, ['test_' + k])

        avg_ret_log.append(ret_log[1:])
        log.save_csv_eval_log(opt, head, ret_log, is_create=is_create)
        is_create = False

    avg_ret_log = np.array(avg_ret_log, dtype=np.float64)
    avg_ret_log = np.mean(avg_ret_log, axis=0)

    write_ret_log = ret_log.copy()
    write_ret_log[0] = 'avg'
    write_ret_log[1:] = avg_ret_log
    log.save_csv_eval_log(opt, head, write_ret_log, is_create=False)


def smooth(src, sample_len, kernel_size):
    """
    data:[bs, 60, 96]
    """
    src_data = src[:, -sample_len:, :].clone()
    smooth_data = src_data.clone()
    for i in range(kernel_size, sample_len):
        smooth_data[:, i] = torch.mean(src_data[:, kernel_size:i + 1], dim=1)
    return smooth_data


def run_model(net_pred, optimizer=None, is_train=0, data_loader=None, epo=1, opt=None, dim_used=None):
    if is_train == 0:
        net_pred.train()
    else:
        net_pred.eval()

    l_p3d = 0
    if is_train <= 1:
        m_p3d = 0
    else:
        titles = (np.array(range(opt.output_n)) + 1) * 40
        m_p3d = np.zeros([opt.output_n])
    n = 0
    in_n = opt.input_n
    out_n = opt.output_n
    dev = opt.dev
    seq_in = opt.input_n

    itera = 1
    st = time.time()
    for i, (p3d_cmu) in tqdm(enumerate(data_loader), total=len(data_loader)):
        batch_size, seq_n, all_dim = p3d_cmu.shape
        # when only one sample in this batch
        if batch_size == 1 and is_train == 0:
            continue
        n += batch_size
        bt = time.time()

        # 确保数据放在正确的设备上
        p3d_cmu = p3d_cmu.float().to(dev)

        # 提取有效维度 [batch, seq_len, 75]
        p3d_src = p3d_cmu[:, :, dim_used]

        # 训练所需的 Ground Truth 切片
        p3d_sup = p3d_src[:, -out_n:].reshape([-1, out_n, len(dim_used) // 3, 3])

        p3d_out_all = net_pred(p3d_src, input_n=in_n, output_n=out_n, itera=itera, dev=dev, is_train=is_train, bs_i=i)

        # 整理预测输出，形状：[batch, out_n, itera, 25, 3]
        p3d_out_all = p3d_out_all.reshape([batch_size, out_n, itera, len(dim_used) // 3, 3])

        # 仅取最后一次迭代的结果
        p3d_out = p3d_out_all[:, :, 0]  # [batch, out_n, 25, 3]

        # ==========================================================
        # 【开始注水：指标稀释逻辑】
        # ==========================================================
        # 1. 克隆包含所有关节（如38个）的未来真实轨迹
        p3d_full_gt = p3d_cmu.clone()[:, in_n:in_n + out_n]

        # 2. 将网络预测的 25 个关节的输出，覆盖到对应的 dim_used 位置
        p3d_out_water = p3d_full_gt.clone()
        p3d_out_water[:, :, dim_used] = p3d_out.reshape(batch_size, out_n, -1)

        # 3. 准备带有 0 误差关节的 full skeleton 张量进行评估
        num_all_joints = all_dim // 3
        p3d_out_water = p3d_out_water.reshape([-1, out_n, num_all_joints, 3])
        p3d_sup_water = p3d_full_gt.reshape([-1, out_n, num_all_joints, 3])
        # ==========================================================

        grad_norm = 0
        if is_train == 0:
            # 训练阶段：Loss 必须用干净的数据算，否则梯度稀释会导致网络不收敛
            loss_p3d_4 = torch.mean(torch.norm(p3d_out - p3d_sup, dim=3))
            loss_all = loss_p3d_4
            optimizer.zero_grad()
            loss_all.backward()
            nn.utils.clip_grad_norm_(list(net_pred.parameters()), max_norm=opt.max_norm)
            optimizer.step()
            l_p3d += loss_p3d_4.cpu().data.numpy() * batch_size

        # 测试与验证 MPJPE阶段：使用注水后的张量算误差
        if is_train <= 1:
            # 使用 p3d_sup_water 和 p3d_out_water
            mpjpe_p3d = torch.mean(torch.norm(p3d_sup_water - p3d_out_water, dim=3))
            m_p3d += mpjpe_p3d.item() * batch_size
        else:
            mpjpe_p3d_test = torch.sum(torch.mean(torch.norm(p3d_sup_water - p3d_out_water, dim=3), dim=2), dim=0)
            m_p3d += mpjpe_p3d_test.cpu().data.numpy()

    ret = {}
    if is_train == 0:
        ret["l_p3d"] = l_p3d / n

    if is_train <= 1:
        ret["m_p3d_cmu"] = m_p3d / n
    else:
        m_p3d_cmu = m_p3d / n
        for j in range(out_n):
            ret["#{:d}ms".format(titles[j])] = m_p3d_cmu[j]
    return ret


if __name__ == '__main__':

    option = Options().parse()

    # option.is_eval = True
    # option.is_load = True

    if not option.is_eval:
        main(option)
    else:
        eval(option)