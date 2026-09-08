import os
import pandas as pd
import numpy as np
import matplotlib

# 明确声明使用无界面后端并导出高质量矢量字体
matplotlib.use('Agg')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt

# ================= 配置区域 =================
# 指向你刚刚生成的双文件 CSV
FILE_GT = '3DPW_single_sample_gt_54dim.csv'
FILE_OURS = '3DPW_single_sample_pred_54dim.csv'

OUTPUT_DIR = './visualizations_pdf'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 目标时间戳 (ms) - 我们希望在图表上展示的关键帧
TARGET_TIMESTAMPS = [-200, -100, 0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
TIME_LABELS = [f"{t}ms" for t in TARGET_TIMESTAMPS]
NUM_TIMES = len(TARGET_TIMESTAMPS)

# 布局参数: 左右共 16 帧，中间隔开 2 列缓冲区，总计 18 列，4 行
TOTAL_COLS = NUM_TIMES * 2 + 2
TOTAL_ROWS = 4
TARGET_ACTIONS = ['courtyard_seq', 'downtown_seq', 'outdoors_seq', 'office_seq']

# === 3DPW 18关节 骨骼拓扑树 ===
BONES = [
    (0, 3), (3, 6),  # 左腿
    (1, 4), (4, 7),  # 右腿
    (2, 5), (5, 8), (8, 11),  # 躯干与头部
    (5, 9), (9, 12), (12, 14), (14, 16),  # 左臂
    (5, 10), (10, 13), (13, 15), (15, 17),  # 右臂
    (0, 2), (1, 2)  # 视觉补偿连线 (骨盆连接)
]

LEFT_JOINTS = {0, 3, 6, 9, 12, 14, 16}
RIGHT_JOINTS = {1, 4, 7, 10, 13, 15, 17}


# ================= 核心功能函数 =================
def get_bone_color(p1, p2):
    """判断骨骼连线颜色：左红、右蓝、中躯干黑"""
    if p1 in LEFT_JOINTS and p2 in LEFT_JOINTS:
        return 'red'
    elif p1 in RIGHT_JOINTS and p2 in RIGHT_JOINTS:
        return 'blue'
    elif (p1 == 0 and p2 == 2) or (p1 == 2 and p2 == 0):
        return 'red'
    elif (p1 == 1 and p2 == 2) or (p1 == 2 and p2 == 1):
        return 'blue'
    return 'black'


def draw_skeleton(ax, pose_3d, is_gt=False):
    """在指定的 3D 坐标轴上绘制人体骨架"""
    if pose_3d is None or len(pose_3d) == 0: return

    for bone in BONES:
        p1, p2 = bone
        # 提取坐标：交换 Y 和 Z 以符合 matplotlib 的 3D 视角习惯
        x = [pose_3d[p1, 0], pose_3d[p2, 0]]
        y = [pose_3d[p1, 2], pose_3d[p2, 2]]
        z = [pose_3d[p1, 1], pose_3d[p2, 1]]

        if is_gt:
            ax.plot(x, y, z, c='#808080', linewidth=4, alpha=0.5, zorder=1)
        else:
            ax.plot(x, y, z, c=get_bone_color(p1, p2), linewidth=2, zorder=2)


def find_closest_frame(df, target_t, threshold=20):
    """基于数值寻找最接近目标时间戳的数据行 (解决 33.33ms 无法精确匹配 100ms 的问题)"""
    if df.empty: return np.array([])
    times = df['Frame'].values
    idx = np.argmin(np.abs(times - target_t))

    if np.abs(times[idx] - target_t) <= threshold:
        # Dim_0 开始于第 4 列 (索引 3)
        return df.iloc[idx, 3:].values.astype(float)
    return np.array([])


def main():
    print(f">>> 正在加载并校验双文件 CSV 数据...")
    if not os.path.exists(FILE_GT) or not os.path.exists(FILE_OURS):
        print(f"❌ 错误: 找不到 {FILE_GT} 或 {FILE_OURS}！")
        return

    df_gt = pd.read_csv(FILE_GT)
    df_ours = pd.read_csv(FILE_OURS)

    # 匹配动作序列
    available_acts = df_gt['Action'].unique()
    matched_acts = [act for act in TARGET_ACTIONS if act in available_acts][:4]

    if not matched_acts:
        print("❌ 错误: 未在 CSV 中匹配到指定的动作序列。")
        return

    print(f">>> 成功匹配动作: {matched_acts}")
    fig = plt.figure(figsize=(48, 14))

    for act_idx, act in enumerate(matched_acts):
        df_gt_act = df_gt[df_gt['Action'] == act]
        df_ours_act = df_ours[df_ours['Action'] == act]

        gt_plot = np.zeros((NUM_TIMES, 18, 3))
        ours_plot = np.zeros((NUM_TIMES, 18, 3))

        gt_exists = [False] * NUM_TIMES
        ours_exists = [False] * NUM_TIMES
        valid_points = []

        for c, t in enumerate(TARGET_TIMESTAMPS):
            # 提取 GT
            gt_row = find_closest_frame(df_gt_act, t)
            if len(gt_row) > 0:
                gt_plot[c] = gt_row.reshape(18, 3)
                gt_exists[c] = True
                valid_points.append(gt_plot[c])

            # 提取 Ours (Pred)
            ours_row = find_closest_frame(df_ours_act, t)
            if len(ours_row) > 0:
                ours_plot[c] = ours_row.reshape(18, 3)
                ours_exists[c] = True
                valid_points.append(ours_plot[c])

        if len(valid_points) == 0: continue

        # 计算全局包围盒，确保同动作视口比例一致
        all_points = np.stack(valid_points, axis=0)
        max_range = (all_points.max(axis=(0, 1)) - all_points.min(axis=(0, 1))).max() / 2.5
        mid_x = (all_points[:, 0].max() + all_points[:, 0].min()) * 0.5
        mid_y = (all_points[:, 1].max() + all_points[:, 1].min()) * 0.5
        mid_z = (all_points[:, 2].max() + all_points[:, 2].min()) * 0.5

        col_offset = 0 if (act_idx % 2) == 0 else 10
        row_offset = (act_idx // 2) * 2

        for c in range(NUM_TIMES):
            # 绘制当前帧的 3D 子图
            ax_o_idx = (row_offset + 1) * TOTAL_COLS + (col_offset + c) + 1
            ax_o = fig.add_subplot(TOTAL_ROWS, TOTAL_COLS, ax_o_idx, projection='3d')

            # 绘制灰色背景真值
            if gt_exists[c]:
                draw_skeleton(ax_o, gt_plot[c], is_gt=True)
            # 绘制彩色预测骨架 (t<=0时通常Pred为空，只显示GT)
            if ours_exists[c]:
                draw_skeleton(ax_o, ours_plot[c], is_gt=False)

            ax_o.set_xlim(mid_x - max_range, mid_x + max_range)
            ax_o.set_ylim(mid_z - max_range, mid_z + max_range)
            ax_o.set_zlim(mid_y - max_range, mid_y + max_range)
            try:
                ax_o.set_box_aspect([1, 1, 1])
            except:
                pass

            ax_o.view_init(elev=15, azim=70)
            ax_o.set_axis_off()

            if c == 0:
                act_display = act.split('_')[0].capitalize()
                ax_o.text2D(-0.20, 0.5, f"{act_display}\n\nOurs", transform=ax_o.transAxes,
                            fontsize=20, fontweight='bold', va='center', ha='right')

            ax_o.text2D(0.5, -0.25, TIME_LABELS[c], transform=ax_o.transAxes,
                        fontsize=15, fontweight='bold', ha='center', va='top')

    # 渲染排版与装饰箭头
    plt.subplots_adjust(wspace=-0.25, hspace=-0.05, left=0.08, right=0.98, bottom=0.10, top=0.95)
    arrow_props = dict(arrowstyle="->", color='black', lw=2)
    plt.annotate('', xy=(0.45, 0.50), xytext=(0.08, 0.50), xycoords='figure fraction', arrowprops=arrow_props)
    plt.annotate('', xy=(0.96, 0.50), xytext=(0.58, 0.50), xycoords='figure fraction', arrowprops=arrow_props)
    plt.annotate('', xy=(0.45, 0.06), xytext=(0.08, 0.06), xycoords='figure fraction', arrowprops=arrow_props)
    plt.annotate('', xy=(0.96, 0.06), xytext=(0.58, 0.06), xycoords='figure fraction', arrowprops=arrow_props)

    pdf_filename = os.path.join(OUTPUT_DIR, '3DPW_actions_DualFile_Visualized.pdf')
    fig.savefig(pdf_filename, format='pdf', bbox_inches='tight')
    plt.close(fig)
    print(f">>> 🎉 3DPW 可视化图表生成完毕！保存至: {pdf_filename}")


if __name__ == '__main__':
    main()