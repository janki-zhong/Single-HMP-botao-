from utils import dpw3d as datasets
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


def compute_similarity_transform_batch(S1, S2):

    mu1 = S1.mean(dim=1, keepdim=True)
    mu2 = S2.mean(dim=1, keepdim=True)
    X1 = S1 - mu1
    X2 = S2 - mu2

    var1 = torch.sum(X1 ** 2, dim=(1, 2))

    K = torch.bmm(X1.transpose(1, 2), X2)

    U, S, Vh = torch.linalg.svd(K)
    V = Vh.transpose(1, 2)

    det = torch.linalg.det(R := torch.bmm(V, U.transpose(1, 2)))
    correction = torch.eye(3, device=S1.device, dtype=S1.dtype).unsqueeze(0).repeat(S1.shape[0], 1, 1)

    signs = torch.sign(det)
    signs[signs == 0] = 1.0
    correction[:, 2, 2] = signs

    R = torch.bmm(torch.bmm(V, correction), U.transpose(1, 2))

    S_corrected = S.clone()
    S_corrected[:, 2] *= signs
    trace = torch.sum(S_corrected, dim=1)
    c = trace / torch.clamp(var1, min=1e-8)
    mu1_rot = torch.bmm(mu1, R.transpose(1, 2))
    t = mu2 - c.view(-1, 1, 1) * mu1_rot
    S1_aligned = c.view(-1, 1, 1) * torch.bmm(S1, R.transpose(1, 2)) + t
    return S1_aligned


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

    optimizer = optim.Adam(
        filter(lambda x: x.requires_grad, net_pred.parameters()),
        lr=opt.lr_now,
        weight_decay=1e-4
    )
    print(">>> total params: {:.2f}M".format(sum(p.numel() for p in net_pred.parameters()) / 1000000.0))

    start_epoch = 1

    if opt.is_load or opt.is_eval:
        model_path_len = './{}/ckpt_best.pth.tar'.format(opt.ckpt)
        print(">>> loading ckpt len from '{}'".format(model_path_len))
        ckpt = torch.load(model_path_len, map_location=dev)
        start_epoch = ckpt['epoch'] + 1
        err_best = ckpt['err']
        lr_now = ckpt['lr']

        dirty_state_dict = ckpt['state_dict']
        clean_state_dict = {
            k: v for k, v in dirty_state_dict.items()
            if not (k.endswith('total_ops') or k.endswith('total_params'))
        }
        net_pred.load_state_dict(clean_state_dict, strict=True)
        print(">>> ckpt len loaded (epoch: {} | err: {})".format(ckpt['epoch'], ckpt['err']))

        if opt.is_load and not opt.is_eval and 'optimizer' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer'])
            print(">>> optimizer state loaded.")

    print('>>> loading datasets')

    if not opt.is_eval:
        dataset = datasets.Datasets(opt, split=0)
        data_loader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=1, pin_memory=True)
        valid_dataset = datasets.Datasets(opt, split=1)
        valid_loader = DataLoader(valid_dataset, batch_size=opt.test_batch_size, shuffle=False, num_workers=1,
                                  pin_memory=True)

    test_dataset = datasets.Datasets(opt, split=2)
    test_loader = DataLoader(test_dataset, batch_size=opt.test_batch_size, shuffle=False, num_workers=1,
                             pin_memory=True)

    if opt.is_eval:
        ret_test = run_model(net_pred, is_train=3, data_loader=test_loader, opt=opt)

        overall_mpjpe = ret_test.get("Overall_MPJPE", 0)
        overall_pampjpe = ret_test.get("Overall_PA-MPJPE", 0)

        print('>>> Testing Evaluation Results <<<')
        print('Overall Average (across all {} frames): MPJPE {:.3f} | PA-MPJPE {:.3f}'.format(
            opt.output_n, overall_mpjpe, overall_pampjpe))

        last_frame_key = '#{:d}'.format(opt.output_n)
        pa_last_frame_key = 'PA-#{:d}'.format(opt.output_n)
        if last_frame_key in ret_test:
            print('testing error (last frame {}): MPJPE {:.3f} | PA-MPJPE {:.3f}'.format(
                opt.output_n, ret_test[last_frame_key], ret_test[pa_last_frame_key]))

        mpjpe_keys = [k for k in ret_test.keys() if 'ms' in k and 'PA-' not in k]
        pa_keys = ['PA-' + k for k in mpjpe_keys]

        if mpjpe_keys:
            mpjpe_str = ' '.join([f'{k}:{ret_test[k]:.1f}' for k in mpjpe_keys])
            pa_str = ' '.join([f'{k}:{ret_test[pa_k]:.1f}' for k, pa_k in zip(mpjpe_keys, pa_keys)])
            print('testing temporal MPJPE:    ' + mpjpe_str)
            print('testing temporal PA-MPJPE: ' + pa_str)

        ret_log = np.array([])
        head = np.array([])
        for k in ret_test.keys():
            ret_log = np.append(ret_log, [ret_test[k]])
            head = np.append(head, [k])
        log.save_csv_log(opt, head, ret_log, is_create=True, file_name='test_3dpw_eval')
        return

    err_best = 1000
    last_frame_key = '#{:d}'.format(opt.output_n)

    for epo in range(start_epoch, opt.epoch + 1):
        is_best = False
        lr_now = util.lr_decay_mine(optimizer, lr_now, 0.1 ** (1 / opt.epoch))
        print('>>> training epoch: {:d}'.format(epo))

        ret_train = run_model(net_pred, optimizer, is_train=0, data_loader=data_loader, epo=epo, opt=opt)
        print('train error: {:.3f}'.format(ret_train['Overall_MPJPE']))

        ret_valid = run_model(net_pred, is_train=1, data_loader=valid_loader, opt=opt, epo=epo)
        print('validation error: {:.3f}'.format(ret_valid['Overall_MPJPE']))

        ret_test = run_model(net_pred, is_train=3, data_loader=test_loader, opt=opt, epo=epo)
        print('testing error (overall): {:.3f}'.format(ret_test['Overall_MPJPE']))

        ms_keys = [k for k in ret_test.keys() if 'ms' in k and 'PA-' not in k]
        ms_vals = [ret_test[k] for k in ms_keys]
        if ms_keys:
            print('testing temporal MPJPE: ' + ' '.join([f'{k}:{v:.1f}' for k, v in zip(ms_keys, ms_vals)]))

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

        if ret_valid['Overall_MPJPE'] < err_best:
            err_best = ret_valid['Overall_MPJPE']
            is_best = True

        log.save_ckpt({'epoch': epo,
                       'lr': lr_now,
                       'err': ret_valid['Overall_MPJPE'],
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
    out_n = opt.output_n
    in_n = opt.input_n

    titles = np.array(range(out_n)) + 1
    m_p3d_3dpw = np.zeros([out_n])
    pa_m_p3d_3dpw = np.zeros([out_n])
    n = 0

    joint_used = np.arange(4, 22)
    num_joints = len(joint_used)
    dim_used = num_joints * 3

    itera = 1
    st = time.time()
    grad_norm = 0.0

    with torch.set_grad_enabled(is_train == 0):
        for i, batch_data in enumerate(data_loader):
            if isinstance(batch_data, (list, tuple)):
                p3d_seq, seq_ids = batch_data
            else:
                p3d_seq = batch_data
                seq_ids = None

            batch_size, seq_n, _, _ = p3d_seq.shape
            if batch_size == 1 and is_train == 0:
                continue
            n += batch_size
            bt = time.time()

            p3d_seq = p3d_seq.float().to(dev)
            p3d_used = p3d_seq[:, :, joint_used, :]
            p3d_src = p3d_used.reshape([batch_size, seq_n, dim_used])
            p3d_sup = p3d_used[:, -out_n:, :, :]

            p3d_out_all = net_pred(p3d_src, input_n=in_n, output_n=out_n, itera=itera, dev=dev, is_train=is_train,
                                   bs_i=i)
            p3d_out_all = p3d_out_all[:, :, 0].reshape([batch_size, out_n, num_joints, 3])

            if is_train == 0:
                diff = p3d_out_all - p3d_sup
                norm_diff = torch.sqrt(torch.sum(diff ** 2, dim=3) + 1e-8)

                max_gamma = 1.5
                warmup_epochs = 15
                if epo <= warmup_epochs:
                    current_gamma = 0.0
                else:
                    progress = (epo - warmup_epochs) / (opt.epoch - warmup_epochs + 1e-6)
                    current_gamma = max_gamma * progress
                time_weights = torch.exp(torch.linspace(0, current_gamma, out_n)).to(dev)
                time_weights = time_weights.view(1, out_n, 1)

                loss_pos = torch.mean(norm_diff * time_weights)

                last_in = p3d_src[:, -1, :].reshape(batch_size, 1, num_joints, 3)

                full_out = torch.cat([last_in, p3d_out_all], dim=1)
                full_sup = torch.cat([last_in, p3d_sup], dim=1)

                v_out = full_out[:, 1:, :, :] - full_out[:, :-1, :, :]
                v_sup = full_sup[:, 1:, :, :] - full_sup[:, :-1, :, :]
                loss_vel = torch.mean(torch.sqrt(torch.sum((v_out - v_sup) ** 2, dim=3) + 1e-8))

                a_out = v_out[:, 1:, :, :] - v_out[:, :-1, :, :]
                a_sup = v_sup[:, 1:, :, :] - v_sup[:, :-1, :, :]
                loss_acc = torch.mean(torch.sqrt(torch.sum((a_out - a_sup) ** 2, dim=3) + 1e-8))
                loss_all = loss_pos + 0.5 * loss_vel + 0.1 * loss_acc

                optimizer.zero_grad()
                loss_all.backward()

                total_norm = nn.utils.clip_grad_norm_(net_pred.parameters(), max_norm=opt.max_norm)
                if isinstance(total_norm, torch.Tensor):
                    grad_norm = total_norm.item()
                else:
                    grad_norm = float(total_norm)

                optimizer.step()
                l_p3d += torch.mean(norm_diff).cpu().data.numpy() * batch_size

            if is_train <= 1:
                mpjpe_p3d = torch.mean(torch.norm(p3d_sup - p3d_out_all, dim=3), dim=(0, 2))
                m_p3d_3dpw += mpjpe_p3d.cpu().data.numpy() * batch_size
            else:
                mpjpe_p3d_test = torch.sum(torch.mean(torch.norm(p3d_sup - p3d_out_all, dim=3), dim=2), dim=0)
                m_p3d_3dpw += mpjpe_p3d_test.cpu().data.numpy()

                pred_flat = p3d_out_all.reshape(-1, num_joints, 3)
                sup_flat = p3d_sup.reshape(-1, num_joints, 3)
                pred_aligned_flat = compute_similarity_transform_batch(pred_flat, sup_flat)
                pred_aligned = pred_aligned_flat.reshape(batch_size, out_n, num_joints, 3)

                pampjpe_p3d_test = torch.sum(torch.mean(torch.norm(p3d_sup - pred_aligned, dim=3), dim=2), dim=0)
                pa_m_p3d_3dpw += pampjpe_p3d_test.cpu().data.numpy()

            if i % 1000 == 0:
                print('{}/{}|bt {:.3f}s|tt{:.0f}s|gn{:.2f}'.format(i + 1, len(data_loader), time.time() - bt,
                                                                   time.time() - st, grad_norm))

    ret = {}
    if is_train == 0:
        ret["l_p3d"] = l_p3d / n

    m_p3d_3dpw = m_p3d_3dpw / n
    pa_m_p3d_3dpw = pa_m_p3d_3dpw / n

    ret["Overall_MPJPE"] = np.mean(m_p3d_3dpw)

    if is_train > 1:
        ret["Overall_PA-MPJPE"] = np.mean(pa_m_p3d_3dpw)
        for j in range(out_n):
            ret["#{:d}".format(titles[j])] = m_p3d_3dpw[j]
            ret["PA-#{:d}".format(titles[j])] = pa_m_p3d_3dpw[j]

        eval_frames = [2, 5, 8, 11, 14, 17, 20, 23, 26, 29]
        for idx_k, f_idx in enumerate(eval_frames):
            if f_idx < out_n:
                time_key = "{}ms".format((idx_k + 1) * 100)
                ret[time_key] = m_p3d_3dpw[f_idx]
                ret["PA-" + time_key] = pa_m_p3d_3dpw[f_idx]

    return ret


if __name__ == '__main__':
    option = Options().parse()
    main(option)