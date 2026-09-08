from utils import dpw3d as datasets
from model import AttModel
from utils.opt import Options
from utils import util
from utils import log
import os
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import numpy as np
import time
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_motion_trend(pred, gt, seq_id, save_dir='./eval_vis'):
    os.makedirs(save_dir, exist_ok=True)
    fig = plt.figure(figsize=(14, 6))
    ax_pred = fig.add_subplot(121, projection='3d')
    ax_gt   = fig.add_subplot(122, projection='3d')
    for j in range(pred.shape[1]):
        ax_pred.plot(pred[:, j, 0], pred[:, j, 1], pred[:, j, 2],
                     color='red', alpha=0.6, linewidth=0.8)
        ax_gt.plot(gt[:, j, 0], gt[:, j, 1], gt[:, j, 2],
                   color='green', alpha=0.6, linewidth=0.8)
    ax_pred.set_title('Predicted Motion Trajectories')
    ax_gt.set_title('Ground Truth Motion Trajectories')
    plt.suptitle(f'Sequence {seq_id}   ({pred.shape[0]} frames)', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'seq_{seq_id:04d}.png'), dpi=150)
    plt.close()


def compute_similarity_transform_batch(S1, S2):
    mu1 = S1.mean(dim=1, keepdim=True)
    mu2 = S2.mean(dim=1, keepdim=True)
    X1 = S1 - mu1
    X2 = S2 - mu2
    var1 = torch.sum(X1 ** 2, dim=(1, 2))
    K = torch.bmm(X1.transpose(1, 2), X2)
    U, S, Vh = torch.linalg.svd(K)
    V = Vh.transpose(1, 2)
    R = torch.bmm(V, U.transpose(1, 2))
    det = torch.linalg.det(R)
    correction = torch.eye(3, device=S1.device).unsqueeze(0).repeat(S1.shape[0], 1, 1)
    correction[:, 2, 2] = torch.sign(det)
    R = torch.bmm(torch.bmm(V, correction), U.transpose(1, 2))
    S_corrected = S.clone()
    S_corrected[:, 2] *= torch.sign(det)
    trace = torch.sum(S_corrected, dim=1)
    c = trace / torch.clamp(var1, min=1e-8)
    mu1_rot = torch.bmm(mu1, R.transpose(1, 2))
    t = mu2 - c.view(-1, 1, 1) * mu1_rot
    return c.view(-1, 1, 1) * torch.bmm(S1, R.transpose(1, 2)) + t


def main(opt):
    lr_now = opt.lr_now
    print('>>> create models')
    in_features = opt.in_features
    d_model = opt.d_model
    kernel_size = opt.kernel_size
    dev = opt.dev
    net_pred = AttModel.AttModel(in_features=in_features, kernel_size=kernel_size, d_model=d_model,
                                 num_stage=opt.num_stage, dct_n=opt.dct_n, output_n=opt.output_n)
    net_pred.to(dev)

    optimizer = optim.Adam(filter(lambda x: x.requires_grad, net_pred.parameters()),
                           lr=opt.lr_now, weight_decay=1e-4)
    print(">>> total params: {:.2f}M".format(sum(p.numel() for p in net_pred.parameters()) / 1000000.0))

    start_epoch = 1
    if opt.is_load or opt.is_eval:
        ckpt_path = './{}/ckpt_best.pth.tar'.format(opt.ckpt)
        print(">>> loading ckpt from '{}'".format(ckpt_path))
        ckpt = torch.load(ckpt_path, map_location=dev)
        start_epoch = ckpt['epoch'] + 1
        err_best = ckpt['err']
        lr_now = ckpt['lr']
        clean_dict = {k: v for k, v in ckpt['state_dict'].items()
                      if not (k.endswith('total_ops') or k.endswith('total_params'))}
        net_pred.load_state_dict(clean_dict, strict=True)
        print(">>> loaded ckpt (epoch: {} | err: {})".format(ckpt['epoch'], ckpt['err']))
        if opt.is_load and not opt.is_eval and 'optimizer' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer'])
            print(">>> optimizer state loaded.")

    print('>>> loading datasets')
    if not opt.is_eval:
        train_dataset = datasets.Datasets(opt, split=0)
        train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True,
                                  num_workers=1, pin_memory=True)
        valid_dataset = datasets.Datasets(opt, split=1)
        valid_loader = DataLoader(valid_dataset, batch_size=opt.test_batch_size, shuffle=False,
                                  num_workers=1, pin_memory=True)
    test_dataset = datasets.Datasets(opt, split=2)
    test_loader = DataLoader(test_dataset, batch_size=opt.test_batch_size, shuffle=False,
                             num_workers=1, pin_memory=True)

    if opt.is_eval:
        ret_test = run_model(net_pred, is_train=3, data_loader=test_loader, opt=opt)
        print('>>> Testing Evaluation Results <<<')
        print('Overall MPJPE: {:.3f} | PA-MPJPE: {:.3f}'.format(
            ret_test.get("Overall_MPJPE", 0), ret_test.get("Overall_PA-MPJPE", 0)))
        last_key = '#{:d}'.format(opt.output_n)
        if last_key in ret_test:
            print('last frame {}: MPJPE {:.3f} | PA-MPJPE {:.3f}'.format(
                opt.output_n, ret_test[last_key], ret_test['PA-'+last_key]))
        mpjpe_keys = [k for k in ret_test if 'ms' in k and 'PA-' not in k]
        for k in mpjpe_keys:
            print(f'{k}: {ret_test[k]:.1f}')
        # 保存日志
        log.save_csv_log(opt, np.array(list(ret_test.keys())), np.array(list(ret_test.values())),
                         is_create=True, file_name='test_3dpw_eval')
        return

    err_best = 1000
    for epo in range(start_epoch, opt.epoch + 1):
        is_best = False
        lr_now = util.lr_decay_mine(optimizer, lr_now, 0.1 ** (1 / opt.epoch))
        print('>>> training epoch: {:d}'.format(epo))

        ret_train = run_model(net_pred, optimizer, is_train=0, data_loader=train_loader, epo=epo, opt=opt)
        ret_valid = run_model(net_pred, is_train=1, data_loader=valid_loader, opt=opt, epo=epo)
        ret_test  = run_model(net_pred, is_train=3, data_loader=test_loader, opt=opt, epo=epo)

        print('train: {:.3f} | valid: {:.3f} | test: {:.3f}'.format(
            ret_train['Overall_MPJPE'], ret_valid['Overall_MPJPE'], ret_test['Overall_MPJPE']))

        if ret_valid['Overall_MPJPE'] < err_best:
            err_best = ret_valid['Overall_MPJPE']
            is_best = True

        log.save_ckpt({'epoch': epo, 'lr': lr_now, 'err': ret_valid['Overall_MPJPE'],
                       'state_dict': net_pred.state_dict(), 'optimizer': optimizer.state_dict()},
                      is_best=is_best, opt=opt)


def run_model(net_pred, optimizer=None, is_train=0, data_loader=None, epo=1, opt=None):
    dev = opt.dev
    if is_train == 0:
        net_pred.train()
    else:
        net_pred.eval()

    out_n = opt.output_n
    in_n = opt.input_n
    joint_used = np.arange(4, 22)
    num_joints = len(joint_used)
    dim_used = num_joints * 3

    m_p3d = np.zeros(out_n)
    pa_m_p3d = np.zeros(out_n)
    n = 0
    l_p3d = 0
    itera = 1
    st = time.time()
    grad_norm = 0
    visualized = set()

    with torch.set_grad_enabled(is_train == 0):
        for i, batch_data in enumerate(data_loader):
            if isinstance(batch_data, (list, tuple)):
                p3d_seq, seq_ids = batch_data
            else:
                p3d_seq = batch_data
                seq_ids = None

            batch_size, seq_n, _, _ = p3d_seq.shape
            if seq_ids is None:
                seq_ids = torch.zeros(batch_size, dtype=torch.long)

            if batch_size == 1 and is_train == 0:
                continue
            n += batch_size
            bt = time.time()

            p3d_seq = p3d_seq.float().to(dev)
            p3d_used = p3d_seq[:, :, joint_used, :]
            p3d_src = p3d_used.reshape(batch_size, seq_n, dim_used)
            p3d_sup = p3d_used[:, -out_n:, :, :]

            p3d_out = net_pred(p3d_src, input_n=in_n, output_n=out_n, itera=itera,
                               dev=dev, is_train=is_train, bs_i=i)
            p3d_out = p3d_out[:, :, 0].reshape(batch_size, out_n, num_joints, 3)

            if is_train == 0:
                diff = p3d_out - p3d_sup
                norm_diff = torch.norm(diff, dim=3)
                max_gamma, warmup = 1.5, 15
                gamma = 0.0 if epo <= warmup else max_gamma * (epo - warmup) / (opt.epoch - warmup + 1e-6)
                time_w = torch.exp(torch.linspace(0, gamma, out_n, device=dev)).view(1, out_n, 1)
                loss_pos = torch.mean(norm_diff * time_w)

                last_in = p3d_src[:, -1:, :].reshape(batch_size, 1, num_joints, 3)
                full_out = torch.cat([last_in, p3d_out], dim=1)
                full_sup = torch.cat([last_in, p3d_sup], dim=1)
                v_out = full_out[:, 1:] - full_out[:, :-1]
                v_sup = full_sup[:, 1:] - full_sup[:, :-1]
                loss_vel = torch.mean(torch.sqrt(torch.sum((v_out - v_sup)**2, dim=3) + 1e-8))
                a_out = v_out[:, 1:] - v_out[:, :-1]
                a_sup = v_sup[:, 1:] - v_sup[:, :-1]
                loss_acc = torch.mean(torch.sqrt(torch.sum((a_out - a_sup)**2, dim=3) + 1e-8))
                loss = loss_pos + 0.5 * loss_vel + 0.1 * loss_acc

                optimizer.zero_grad()
                loss.backward()
                total_norm = nn.utils.clip_grad_norm_(net_pred.parameters(), max_norm=opt.max_norm)
                grad_norm = total_norm.item() if isinstance(total_norm, torch.Tensor) else float(total_norm)
                optimizer.step()
                l_p3d += torch.mean(norm_diff).item() * batch_size

            with torch.no_grad():
                mpjpe = torch.norm(p3d_sup - p3d_out, dim=3)

                if is_train <= 1:
                    m_p3d += mpjpe.mean(dim=(0, 2)).cpu().numpy() * batch_size
                else:
                    m_p3d += mpjpe.mean(dim=(0, 2)).detach().cpu().numpy() * batch_size

                    pred_flat = p3d_out.reshape(-1, num_joints, 3)
                    sup_flat = p3d_sup.reshape(-1, num_joints, 3)
                    pred_aligned = compute_similarity_transform_batch(pred_flat, sup_flat)
                    pred_aligned = pred_aligned.reshape(batch_size, out_n, num_joints, 3)
                    pampjpe = torch.norm(p3d_sup - pred_aligned, dim=3)

                    pa_m_p3d += pampjpe.mean(dim=(0, 2)).detach().cpu().numpy() * batch_size

                    if opt.is_eval:
                        for b in range(batch_size):
                            sid = seq_ids[b].item()
                            if sid not in visualized:
                                visualized.add(sid)
                                pred_np = p3d_out[b].detach().cpu().numpy()
                                sup_np  = p3d_sup[b].detach().cpu().numpy()
                                plot_motion_trend(pred_np, sup_np, sid)

            if i % 1000 == 0:
                print('{}/{}|bt {:.3f}s|tt{:.0f}s|gn{:.2f}'.format(
                    i+1, len(data_loader), time.time()-bt, time.time()-st, grad_norm))

    ret = {}
    if is_train == 0:
        ret["l_p3d"] = l_p3d / n
    m_p3d /= n
    pa_m_p3d /= n
    ret["Overall_MPJPE"] = np.mean(m_p3d)

    if is_train > 1:
        ret["Overall_PA-MPJPE"] = np.mean(pa_m_p3d)
        titles = np.arange(1, out_n+1)
        for j, t in enumerate(titles):
            ret[f"#{t}"] = m_p3d[j]
            ret[f"PA-#{t}"] = pa_m_p3d[j]
        eval_frames = [2,5,8,11,14,17,20,23,26,29]
        for idx_k, f_idx in enumerate(eval_frames):
            if f_idx < out_n:
                time_key = f"{(idx_k+1)*100}ms"
                ret[time_key] = m_p3d[f_idx]
                ret[f"PA-{time_key}"] = pa_m_p3d[f_idx]
    return ret


if __name__ == '__main__':
    option = Options().parse()
    main(option)