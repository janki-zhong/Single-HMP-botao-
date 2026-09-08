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
FILE_OURS = 'single_sample_pred_96dim.csv'
FILE_GT = 'single_sample_gt_96dim.csv'

OUTPUT_DIR = './visualizations_pdf'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 目标时间戳：包含 3 个历史时刻和 5 个未来时刻
TARGET_TIMESTAMPS = [-400, -200, 0, 200, 400, 600, 800, 1000]
TIME_LABELS = [f"{t}ms" for t in TARGET_TIMESTAMPS]
NUM_TIMES = len(TARGET_TIMESTAMPS)

TARGET_ACTIONS = [
    "walking", "eating", "smoking", "discussion", "directions",
    "greeting", "phoning", "posing", "purchases", "sitting",
    "sittingdown", "takingphoto", "waiting", "walkingdog", "walkingtogether"
]

TOTAL_COLS = 18
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
            ax.plot(x, y, z, c='#808080', linewidth=4, alpha=0.6, zorder=1)
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
    df_ours = pd.read_csv(FILE_OURS)
    df_gt = pd.read_csv(FILE_GT)
    print("CSV中实际包含的动作:", df_gt['Action'].unique())

    all_acts = df_gt['Action'].unique()

    matched_acts = []
    for ta in TARGET_ACTIONS:
        for act in all_acts:
            # 兼容包含关系并去重
            if ta.lower() in act.lower() and act not in matched_acts:
                matched_acts.append(act)
                break

    if not matched_acts:
        print("未找到指定动作，请检查 CSV 数据。")
        return

    num_matched = len(matched_acts)
    # 计算需要的逻辑行数（每行放左、右2个动作）
    action_rows = math.ceil(num_matched / 2.0)
    # 实际的绘图行数：每个动作占2排（PGBIG 和 Ours）
    TOTAL_ROWS = action_rows * 2

    # 动态调整画布高度：每行动作大概分配 7 的高度
    fig = plt.figure(figsize=(48, 7 * action_rows))

    for act_idx, act in enumerate(matched_acts):
        df_gt_act = df_gt[df_gt['Action'] == act]
        df_base_act = df_base[df_base['Action'] == act]
        df_ours_act = df_ours[df_ours['Action'] == act]

        gt_plot = np.zeros((NUM_TIMES, 17, 3))
        base_plot = np.zeros((NUM_TIMES, 17, 3))
        ours_plot = np.zeros((NUM_TIMES, 17, 3))

        base_exists = [False] * NUM_TIMES
        ours_exists = [False] * NUM_TIMES
        valid_points = []

        for c, t in enumerate(TARGET_TIMESTAMPS):
            gt_row = df_gt_act[df_gt_act['Frame'] == t].iloc[:, 3:].values
            if len(gt_row) > 0:
                gt_plot[c] = gt_row[0].reshape(32, 3)[JOINT_MAPPING]
                valid_points.append(gt_plot[c])

            base_row = df_base_act[df_base_act['Frame'] == t].iloc[:, 3:].values
            if len(base_row) > 0:
                base_plot[c] = base_row[0].reshape(32, 3)[JOINT_MAPPING]
                base_exists[c] = True
                valid_points.append(base_plot[c])

            ours_row = df_ours_act[df_ours_act['Frame'] == t].iloc[:, 3:].values
            if len(ours_row) > 0:
                ours_plot[c] = ours_row[0].reshape(32, 3)[JOINT_MAPPING]
                ours_exists[c] = True
                valid_points.append(ours_plot[c])

        if len(valid_points) == 0:
            continue

        all_points = np.stack(valid_points, axis=0)
        max_range = (all_points.max(axis=(0, 1)) - all_points.min(axis=(0, 1))).max() / 2.8
        mid_x = (all_points[:, 0].max() + all_points[:, 0].min()) * 0.5
        mid_y = (all_points[:, 1].max() + all_points[:, 1].min()) * 0.5
        mid_z = (all_points[:, 2].max() + all_points[:, 2].min()) * 0.5

        col_offset = 0 if (act_idx % 2) == 0 else 10
        row_offset = (act_idx // 2) * 2

        for c in range(NUM_TIMES):
            # ================= 绘制 PGBIG (上排) =================
            ax_p_idx = (row_offset + 0) * TOTAL_COLS + (col_offset + c) + 1
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
                ax_p.text2D(-0.20, 0.5, f"{act.capitalize()}\n\nPGBIG", transform=ax_p.transAxes,
                            fontsize=20, fontweight='bold', va='center', ha='right')

            # ================= 绘制 Ours (下排) =================
            ax_o_idx = (row_offset + 1) * TOTAL_COLS + (col_offset + c) + 1
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
                ax_o.text2D(-0.20, 0.5, "Ours", transform=ax_o.transAxes,
                            fontsize=20, fontweight='bold', va='center', ha='right')

            ax_o.text2D(0.5, -0.25, TIME_LABELS[c], transform=ax_o.transAxes,
                        fontsize=15, fontweight='bold', ha='center', va='top')

    plt.subplots_adjust(wspace=-0.25, hspace=-0.05, left=0.08, right=0.98, bottom=0.05, top=0.95)

    # 动态生成时间轴箭头
    # 计算当前图表的垂直有效绘制区域，避免将坐标写死导致排版错乱
    for row_idx in range(action_rows):
        # 推导基于整个画布的高度分数，从上向下分布
        # bottom 为 0，top 为 1。由于顶部有 margin，实际可用区间为 [0.05, 0.95]
        usable_height = 0.95 - 0.05
        step_height = usable_height / action_rows

        # y 坐标处于每个动作块的下边缘附近 (稍微往上偏移一点对齐文本)
        y_pos = 0.95 - (row_idx + 1) * step_height + (step_height * 0.15)

        # 左列箭头
        plt.annotate('', xy=(0.45, y_pos), xytext=(0.08, y_pos), xycoords='figure fraction',
                     arrowprops=dict(arrowstyle="->", color='black', lw=2))
        # 判断如果最后一行只有左边一个动作，就不画右边的箭头
        if (row_idx * 2 + 1) < num_matched:
            # 右列箭头
            plt.annotate('', xy=(0.96, y_pos), xytext=(0.58, y_pos), xycoords='figure fraction',
                         arrowprops=dict(arrowstyle="->", color='black', lw=2))

    pdf_filename = os.path.join(OUTPUT_DIR, f'all_{num_matched}_actions_perfect.pdf')
    fig.savefig(pdf_filename, format='pdf', bbox_inches='tight')
    plt.close(fig)
    print(f">>> {num_matched} 个动作渲染完毕，保存至: {pdf_filename}")

if __name__ == '__main__':
    main()