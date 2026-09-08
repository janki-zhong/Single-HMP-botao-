import pandas as pd
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_single_type_skeletons(csv_file, output_path):
    print(f">>> Loading data from {csv_file}...")
    df = pd.read_csv(csv_file)

    actions = df['Action'].unique()

    # 【需求 1】指定的时间帧: 40, 280, 520, 760, 1000 ms
    # 对应在 CSV 中的 Frame 索引 (1=40ms, 7=280ms, 13=520ms, 19=760ms, 25=1000ms)
    target_frames = [1, 7, 13, 19, 25]
    frames = [f for f in target_frames if f in df['Frame'].unique()]

    nrows = len(actions)
    ncols = len(frames)

    print(f">>> Found {nrows} actions. Plotting {ncols} frames per action...")

    # 【需求 2】调整画布比例，使得骨架间隔更近但不紧挨着
    # 单个子图宽高从 2.2 压到了 2.0，拉近水平距离
    fig = plt.figure(figsize=(ncols * 2.0, nrows * 2.2))

    # 17 节点连接树
    parents = [-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15]

    # H3.6M 标准 17 关节对应的 32 关节原始索引
    indices_17 = [0, 1, 2, 3, 6, 7, 8, 12, 13, 14, 15, 17, 18, 19, 25, 26, 27]

    # 【需求 3】指定要画红色的原 32 节点索引
    # 它们在 indices_17 (即画图用的 0-16 索引) 中对应的相对位置：
    # 0,1,2,3(右腿) -> 0,1,2,3
    # 12,13,14,15(躯干/头) -> 7,8,9,10
    # 25,26,27(右臂) -> 14,15,16
    red_idx_17 = {0, 1, 2, 3, 7, 8, 9, 10, 14, 15, 16}

    for row_idx, action in enumerate(actions):
        act_data = df[df['Action'] == action].sort_values('Frame')

        for col_idx, frame in enumerate(frames):
            ax_idx = row_idx * ncols + col_idx + 1
            ax = fig.add_subplot(nrows, ncols, ax_idx, projection='3d')

            # 提取 96 维坐标并重塑为 (32, 3)
            coords_32 = act_data[act_data['Frame'] == frame].iloc[0, 3:].values.astype(float).reshape(32, 3)

            # 交换 Y 和 Z 轴
            coords_32 = coords_32[:, [0, 2, 1]]
            # 如果画出来依然是倒立的，请取消下面这行的注释：
            # coords_32[:, 2] = -coords_32[:, 2]

            # 提取 17 节点
            coords = coords_32[indices_17]

            for i, p in enumerate(parents):
                if p != -1:
                    # 判断当前节点是否属于红色指定组
                    if i in red_idx_17:
                        bone_color = '#E74C3C'  # 红色
                    else:
                        bone_color = '#3498DB'  # 蓝色

                    # 绘制连线 (因为不再是重叠图，线宽调粗为 3.0，透明度调为不透明 1.0)
                    ax.plot([coords[i, 0], coords[p, 0]],
                            [coords[i, 1], coords[p, 1]],
                            [coords[i, 2], coords[p, 2]],
                            color=bone_color, linewidth=3.0, alpha=1.0)

            root = coords[0]
            # 控制骨架显示大小
            RADIUS = 500
            ax.set_xlim3d([-RADIUS + root[0], RADIUS + root[0]])
            ax.set_ylim3d([-RADIUS + root[1], RADIUS + root[1]])
            ax.set_zlim3d([-RADIUS + root[2], RADIUS + root[2]])

            ax.view_init(elev=15, azim=70)
            ax.set_axis_off()

            # 只在第一行标明毫秒数标题
            if row_idx == 0:
                ax.set_title(f"{frame * 40} ms", fontsize=14, fontweight='bold', pad=0)

            # 只在最左侧标明动作名称
            if col_idx == 0:
                ax.text2D(-0.15, 0.5, action, transform=ax.transAxes,
                          fontsize=14, fontweight='bold', rotation=90, va='center')

    # 【间隔控制核心】：wspace=-0.1 让子图拉近，但不会粘连
    plt.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0.05, wspace=-0.1, hspace=0.1)

    print(f">>> Saving vector image to {output_path}...")
    plt.savefig(output_path, format='pdf', bbox_inches='tight', facecolor='white')
    plt.close()


if __name__ == "__main__":
    gt_file = 'single_sample_gt_96dim.csv'
    pred_file = 'single_sample_pred_96dim.csv'

    # 分别调用函数生成两个独立的 PDF 文件
    plot_single_type_skeletons(gt_file, 'skeleton_GT_separate.pdf')
    plot_single_type_skeletons(pred_file, 'skeleton_Pred_separate.pdf')
    print(">>> All Done! generated 'skeleton_GT_separate.pdf' and 'skeleton_Pred_separate.pdf'")