import numpy as np
import matplotlib.pyplot as plt

# 1. 定义自变量 E (范围从 -3 到 3) 和 参数 sigma
E = np.linspace(-3, 3, 1000)
sigma = 1.0  # 你可以根据需要修改 sigma 的值

# 2. 核心公式
epsilon = 1e-6
mu = np.exp(-(E**2) / (2 * sigma**2 + epsilon))

# 3. 学术风图表配置
fig, ax = plt.subplots(figsize=(6, 3))

# 绘制曲线，使用顶会常用的莫兰迪蓝 (#4285F4)
ax.plot(E, mu, color='#4285F4', linewidth=2.5)

# 隐藏上边框和右边框 (学术极简风)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(1.2)
ax.spines['bottom'].set_linewidth(1.2)

# ================= 修改区域 =================
# 隐藏 X 轴和 Y 轴的坐标数字
ax.set_xticklabels([])
ax.set_yticklabels([])

# 隐藏坐标轴上的凸起刻度小短线 (使其变成纯粹的两条相交线)
ax.tick_params(axis='both', which='both', length=0)
# ============================================

# 添加网格线辅助视觉对齐 (即使去掉了坐标数字，底层的网格线依然会按默认刻度渲染)
ax.grid(True, linestyle='--', alpha=0.5)

# 4. 导出为矢量图
plt.tight_layout()
plt.savefig('gaussian_mu.svg', format='svg', transparent=True)
plt.show()