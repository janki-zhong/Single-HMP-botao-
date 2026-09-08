import pandas as pd
import numpy as np
import matplotlib
import math

matplotlib.use('Agg')

# 强制输出 TrueType 字体，确保导出 PDF/EPS 时的矢量文字兼容性
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

# ----------------- 配置区 -----------------
FILE_BASELINE = 'baseline_pred_96dim.csv'
FILE_STSGCN = 'stsgcn_single_sample_pred_96dim.csv'
FILE_OURS = 'single_sample_pred_96dim.csv'
FILE_GT = 'single_sample_gt_96dim.csv'

OUTPUT_DIR = './visualizations_pdf'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 目标时间戳：3个历史时刻 (-400, -200, 0) + 25个未来时刻 (40, 80 ... 1000)
TARGET_TIMESTAMPS = [-400, -200, 0] + [(i + 1) * 40 for i in range(25)]
NUM_TIMES = len(TARGET_TIMESTAMPS)  # 总计 28 帧

# 仅保留指定的 4 个动作
TARGET_ACTIONS = [
    "eating", "discussion", "waiting", "directions"
]

TOTAL_COLS = NUM_TIMES  # 动态列数：28

# 17 个关键节点映射及拓扑关系
JOINT_MAPPING = [0, 1, 2, 3, 6, 7, 8, 12, 13, 14, 15, 17, 18, 19, 25, 26, 27]
NEW_PARENTS = [-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15]
NEW_LEFT_JOINTS = [4, 5, 6, 11, 12, 13]
NEW_RIGHT_JOINTS = [1, 2, 3, 14, 15, 16]


def draw_skeleton(ax, pose_3d, is_gt=False):
    for i, parent in enumerate(NEW_PARENTS):
        if parent == -1:
            continue

        x = [pose_3d[parent, 0], pose_3d[i, 0]]
        y = [pose_3d[parent, 2], pose_3d[i, 2]]
        z = [pose_3d[parent, 1], pose_3d[i, 1]]

        if is_gt:
            ax.plot(x, y, z, c='#808080', linewidth=4, alpha=0.5, zorder=1)
        else:
            if i in NEW_RIGHT_JOINTS:
                c = 'blue'
            elif i in NEW_LEFT_JOINTS:
                c = 'red'
            else:
                c = 'black'
            ax.plot(x, y, z, c=c, linewidth=2, zorder=2)


def main():
    print(">>> 正在加载 CSV 数据...")
    df_base = pd.read_csv(FILE_BASELINE)
    df_stsgcn = pd.read_csv(FILE_STSGCN)
    df_ours = pd.read_csv(FILE_OURS)
    df_gt = pd.read_csv(FILE_GT)

    all_acts = df_gt['Action'].unique()
    matched_acts = []
    for ta in TARGET_ACTIONS:
        for act in all_acts:
            if ta.lower() in act.lower() and act not in matched_acts:
                matched_acts.append(act)
                break

    if not matched_acts:
        print("未找到指定动作，请检查 CSV 数据。")
        return

    num_matched = len(matched_acts)
    TOTAL_ROWS = num_matched * 4

    # 【修改 1】：适度减小整体画布宽度和高度，因为我们要用负间距把它们“挤”在一起
    fig_width = NUM_TIMES * 1.6  # 原为 3.0，缩小宽度
    fig_height = num_matched * 8.0  # 原为 8.0，缩小高度
    fig = plt.figure(figsize=(fig_width, fig_height))

    for act_idx, act in enumerate(matched_acts):
        df_gt_act = df_gt[df_gt['Action'] == act]
        df_base_act = df_base[df_base['Action'] == act]
        df_stsgcn_act = df_stsgcn[df_stsgcn['Action'] == act]
        df_ours_act = df_ours[df_ours['Action'] == act]

        gt_plot = np.zeros((NUM_TIMES, 17, 3))
        base_plot = np.zeros((NUM_TIMES, 17, 3))
        stsgcn_plot = np.zeros((NUM_TIMES, 17, 3))
        ours_plot = np.zeros((NUM_TIMES, 17, 3))

        base_exists = [False] * NUM_TIMES
        stsgcn_exists = [False] * NUM_TIMES
        ours_exists = [False] * NUM_TIMES
        valid_points = []

        for c, t in enumerate(TARGET_TIMESTAMPS):
            # GT
            gt_row = df_gt_act[df_gt_act['Frame'] == t].iloc[:, 3:].values
            if len(gt_row) > 0:
                gt_plot[c] = gt_row[0].reshape(32, 3)[JOINT_MAPPING]
                valid_points.append(gt_plot[c])

            # PGBIG
            base_row = df_base_act[df_base_act['Frame'] == t].iloc[:, 3:].values
            if len(base_row) > 0:
                base_plot[c] = base_row[0].reshape(32, 3)[JOINT_MAPPING]
                base_exists[c] = True
                valid_points.append(base_plot[c])

            # STSGCN
            stsgcn_row = df_stsgcn_act[df_stsgcn_act['Frame'] == t].iloc[:, 3:].values
            if len(stsgcn_row) > 0:
                stsgcn_plot[c] = stsgcn_row[0].reshape(32, 3)[JOINT_MAPPING]
                stsgcn_exists[c] = True
                valid_points.append(stsgcn_plot[c])

            # Ours
            ours_row = df_ours_act[df_ours_act['Frame'] == t].iloc[:, 3:].values
            if len(ours_row) > 0:
                ours_plot[c] = ours_row[0].reshape(32, 3)[JOINT_MAPPING]
                ours_exists[c] = True
                valid_points.append(ours_plot[c])

        if len(valid_points) == 0:
            continue

        all_points = np.stack(valid_points, axis=0)

        # 【修改 2】：除数从 2.8 改为 3.5。这会缩小坐标轴的包围盒大小，相当于在 3D 空间内把镜头“拉近”，让骨架显得更大。
        max_range = (all_points.max(axis=(0, 1)) - all_points.min(axis=(0, 1))).max() / 3.5

        mid_x = (all_points[:, 0].max() + all_points[:, 0].min()) * 0.5
        mid_y = (all_points[:, 1].max() + all_points[:, 1].min()) * 0.5
        mid_z = (all_points[:, 2].max() + all_points[:, 2].min()) * 0.5

        row_offset = act_idx * 4

        for c in range(NUM_TIMES):
            # ================= 1. 绘制 PGBIG (第 1 行) =================
            ax_p_idx = (row_offset + 0) * TOTAL_COLS + c + 1
            ax_p = fig.add_subplot(TOTAL_ROWS, TOTAL_COLS, ax_p_idx, projection='3d')

            draw_skeleton(ax_p, gt_plot[c], is_gt=True)
            if base_exists[c]:
                draw_skeleton(ax_p, base_plot[c], is_gt=False)

            ax_p.set_xlim(mid_x - max_range, mid_x + max_range)
            ax_p.set_ylim(mid_z - max_range, mid_z + max_range)
            ax_p.set_zlim(mid_y - max_range, mid_y + max_range)
            try:
                ax_p.set_box_aspect([1, 1, 1])
            except AttributeError:
                pass
            ax_p.view_init(elev=15, azim=70)
            ax_p.set_axis_off()

            if c == 0:
                ax_p.text2D(-0.25, 0.5, f"{act.capitalize()}\n\nPGBIG", transform=ax_p.transAxes,
                            fontsize=18, fontweight='bold', va='center', ha='right')

            # ================= 2. 绘制 STSGCN (第 2 行) =================
            ax_s_idx = (row_offset + 1) * TOTAL_COLS + c + 1
            ax_s = fig.add_subplot(TOTAL_ROWS, TOTAL_COLS, ax_s_idx, projection='3d')

            draw_skeleton(ax_s, gt_plot[c], is_gt=True)
            if stsgcn_exists[c]:
                draw_skeleton(ax_s, stsgcn_plot[c], is_gt=False)

            ax_s.set_xlim(mid_x - max_range, mid_x + max_range)
            ax_s.set_ylim(mid_z - max_range, mid_z + max_range)
            ax_s.set_zlim(mid_y - max_range, mid_y + max_range)
            try:
                ax_s.set_box_aspect([1, 1, 1])
            except AttributeError:
                pass
            ax_s.view_init(elev=15, azim=70)
            ax_s.set_axis_off()

            if c == 0:
                ax_s.text2D(-0.25, 0.5, "STSGCN", transform=ax_s.transAxes,
                            fontsize=18, fontweight='bold', va='center', ha='right')

            # ================= 3. 绘制 Ours (第 3 行) =================
            ax_o_idx = (row_offset + 2) * TOTAL_COLS + c + 1
            ax_o = fig.add_subplot(TOTAL_ROWS, TOTAL_COLS, ax_o_idx, projection='3d')

            draw_skeleton(ax_o, gt_plot[c], is_gt=True)
            if ours_exists[c]:
                draw_skeleton(ax_o, ours_plot[c], is_gt=False)

            ax_o.set_xlim(mid_x - max_range, mid_x + max_range)
            ax_o.set_ylim(mid_z - max_range, mid_z + max_range)
            ax_o.set_zlim(mid_y - max_range, mid_y + max_range)
            try:
                ax_o.set_box_aspect([1, 1, 1])
            except AttributeError:
                pass
            ax_o.view_init(elev=15, azim=70)
            ax_o.set_axis_off()

            if c == 0:
                ax_o.text2D(-0.25, 0.5, "Ours", transform=ax_o.transAxes,
                            fontsize=18, fontweight='bold', va='center', ha='right')

            # ================= 4. 绘制 Ground Truth (第 4 行) =================
            ax_gt_idx = (row_offset + 3) * TOTAL_COLS + c + 1
            ax_gt = fig.add_subplot(TOTAL_ROWS, TOTAL_COLS, ax_gt_idx, projection='3d')

            draw_skeleton(ax_gt, gt_plot[c], is_gt=True)

            ax_gt.set_xlim(mid_x - max_range, mid_x + max_range)
            ax_gt.set_ylim(mid_z - max_range, mid_z + max_range)
            ax_gt.set_zlim(mid_y - max_range, mid_y + max_range)
            try:
                ax_gt.set_box_aspect([1, 1, 1])
            except AttributeError:
                pass
            ax_gt.view_init(elev=15, azim=70)
            ax_gt.set_axis_off()

            if c == 0:
                ax_gt.text2D(-0.25, 0.5, "GT", transform=ax_gt.transAxes,
                             fontsize=18, fontweight='bold', va='center', ha='right')

    # 【修改 3】：利用负数的 wspace 和 hspace 来强行吞并 3D 子图边缘的透明区域，间距收缩的同时骨架尺寸不变。
    plt.subplots_adjust(wspace=-0.40, hspace=0.05, left=0.08, right=0.98, bottom=0.02, top=0.98)

    pdf_filename = os.path.join(OUTPUT_DIR, f'tight_spacing_4_actions_4rows_{NUM_TIMES}frames.pdf')
    fig.savefig(pdf_filename, format='pdf', bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    print(f">>> 指定的 {num_matched} 个动作渲染完毕，保存至: {pdf_filename}")


if __name__ == '__main__':
    main()