import pandas as pd
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_all_skeletons(gt_csv, pred_csv, output_path='skeleton_comparison_contrast.pdf'):
    print(f">>> Loading data from {gt_csv} and {pred_csv}...")
    df_gt = pd.read_csv(gt_csv)
    df_pred = pd.read_csv(pred_csv)

    actions = df_gt['Action'].unique()
    frames = sorted(df_gt['Frame'].unique())

    nrows = len(actions)
    ncols = len(frames)

    print(f">>> Found {nrows} actions and {ncols} frames.")
    print(f">>> Generating {nrows}x{ncols} image grid. This will take a moment...")

    # 保留大图、窄间隔的画布系数
    fig = plt.figure(figsize=(ncols * 2.2, nrows * 2.5))

    parents = [-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15]
    indices_17 = [0, 1, 2, 3, 6, 7, 8, 12, 13, 14, 15, 17, 18, 19, 25, 26, 27]

    for row_idx, action in enumerate(actions):
        gt_act = df_gt[df_gt['Action'] == action].sort_values('Frame')
        pred_act = df_pred[df_pred['Action'] == action].sort_values('Frame')

        for col_idx, frame in enumerate(frames):
            ax_idx = row_idx * ncols + col_idx + 1
            ax = fig.add_subplot(nrows, ncols, ax_idx, projection='3d')

            gt_coords_32 = gt_act[gt_act['Frame'] == frame].iloc[0, 3:].values.astype(float).reshape(32, 3)
            pred_coords_32 = pred_act[pred_act['Frame'] == frame].iloc[0, 3:].values.astype(float).reshape(32, 3)

            # 交换 Y 和 Z 轴，让人物“站”起来
            gt_coords_32 = gt_coords_32[:, [0, 2, 1]]
            pred_coords_32 = pred_coords_32[:, [0, 2, 1]]

            gt_coords = gt_coords_32[indices_17]
            pred_coords = pred_coords_32[indices_17]

            for i, p in enumerate(parents):
                if p != -1:
                    # 【核心修改点】：GT 颜色改为红色的对比色（青色 `#18B3C3`）
                    # 依然保持粗线宽和半透明，作为 Pred 的对比底色
                    ax.plot([gt_coords[i, 0], gt_coords[p, 0]],
                            [gt_coords[i, 1], gt_coords[p, 1]],
                            [gt_coords[i, 2], gt_coords[p, 2]],
                            color='#18B3C3', linewidth=4.5, alpha=0.5, label='GT' if i == 1 else "")

                    # Pred 保持红色
                    ax.plot([pred_coords[i, 0], pred_coords[p, 0]],
                            [pred_coords[i, 1], pred_coords[p, 1]],
                            [pred_coords[i, 2], pred_coords[p, 2]],
                            color='#E74C3C', linewidth=2.0, alpha=1.0, label='Pred' if i == 1 else "")

            # 保留放大镜头的 RADIUS 设置
            root_gt = gt_coords[0]
            RADIUS = 550
            ax.set_xlim3d([-RADIUS + root_gt[0], RADIUS + root_gt[0]])
            ax.set_ylim3d([-RADIUS + root_gt[1], RADIUS + root_gt[1]])
            ax.set_zlim3d([-RADIUS + root_gt[2], RADIUS + root_gt[2]])

            ax.view_init(elev=15, azim=70)
            ax.set_axis_off()

            # 调整标题和标签边距
            if row_idx == 0:
                ax.set_title(f"{frame * 40} ms", fontsize=14, fontweight='bold', pad=0)

            if col_idx == 0:
                ax.text2D(-0.15, 0.5, action, transform=ax.transAxes,
                          fontsize=16, fontweight='bold', rotation=90, va='center')

            if row_idx == 0 and col_idx == 0:
                # 更新图例位置，避免遮挡
                ax.legend(loc='upper left', frameon=False, fontsize=12)

    # 保留窄间隔设置
    plt.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0.05, wspace=-0.2, hspace=0.1)

    print(f">>> Saving vector image with contrast colors to {output_path}...")
    plt.savefig(output_path, format='pdf', bbox_inches='tight', facecolor='white')
    plt.close()
    print(">>> Done!")


if __name__ == "__main__":
    # 使用你之前生成的 96 维 CSV 文件
    gt_file = 'single_sample_gt_96dim.csv'
    pred_file = 'single_sample_pred_96dim.csv'

    plot_all_skeletons(gt_file, pred_file)