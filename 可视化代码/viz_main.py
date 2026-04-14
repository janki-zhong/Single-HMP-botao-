import numpy as np
from progress.bar import Bar
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # 使用Agg后端而不是Tkinter
# matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt

import utils.data_utils as data_utils

import viz

actions = ["walking", "eating", "smoking", "discussion", "directions",
            "greeting", "phoning", "posing", "purchases", "sitting",
            "sittingdown", "takingphoto", "waiting", "walkingdog",
            "walkingtogether"]


import pandas as pd
def main():
    fig = plt.figure(figsize=(150,150))
    #ax = plt.gca(projection='3d')
    ax = fig.add_subplot(projection ='3d')

    # for k in range(8):
    #     plt.cla()
    #     figure_title = "action:{}, seq:{},".format(actions, (k + 1))
    #     viz.plot_predictions(targ_expmap[k, :, :], pred_expmap[k, :, :], fig, ax, figure_title)
    #     plt.pause(1)
    #
    #  # .npz 处理
    # # gt_pose = np.load('./data/vis_data/sitting_gt.npz')['arr_0']
    # # pred_pose = np.load('./data/vis_data/sitting_pred.npz')['arr_0']
    # gt_pose = np.load('E:/PDANet/body_models/smpl_skeleton.npz')['arr_0']
    # pred_pose = np.load('E:/PDANet/body_models/smpl_skeleton.npz')['arr_0']
    # target_xyz = gt_pose[26,:,:]    # 10,96
    # pred_xyz = pred_pose[26,:,:]    # 10,96
    #  .npz 处理

    #  .csv 处理
    # target_xyz = pd.read_csv('/home/user0/Public/yyx/python_workspace/HisRepItself-master/vis/data/ground_truth/waiting_gt.csv').to_numpy()[:,1:]  # 25,96
    # pred_xyz = pd.read_csv('/home/user0/Public/yyx/python_workspace/HisRepItself-master/vis/data/pre_pose/waiting_pred.csv').to_numpy()[:,1:]  # 25,96
    target_xyz = pd.read_csv('E:/微信下载/WeChat Files/wxid_3661xmuclne022/FileStorage/File/2024-09/walkingtogether_gt.csv').to_numpy()[:,1:]  # 25,96
    pred_xyz = pd.read_csv('E:/Desktop/ckpt/h3.6m/创新点/深层次/644/test_pre_action.csv').to_numpy()[:,1:]  # 25,96

    #  .csv 处理


    plt.cla()
    figure_title = "action:{},".format(actions[0])

    viz.plot_xyz_predictions(target_xyz, pred_xyz, fig, ax, figure_title)  # T,96
    plt.pause(0.1)





if __name__ == "__main__":
    main()