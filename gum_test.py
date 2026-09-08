import torch

file_path = 'gum_ret.pth'
metadata = {
    'num_zeros': 3,
    # 'entropy_t_noise': entropy_t_noise,
    'num_ones': 2,
    'num_twos': 1
}
torch.save(metadata, file_path)


path_checkpoint = "gum_ret.pth"  # 断点权重文件的所在路径
checkpoint = torch.load(path_checkpoint)
a = 1