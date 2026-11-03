import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import japanize_matplotlib

fig = plt.figure(figsize=(15, 6))

# ----------------- 1. 問題設定の幾何・境界条件の模式図 (3D円盤) -----------------
ax1 = fig.add_subplot(1, 2, 1, projection='3d')

a = 1.0  # 半径
h = 0.1  # 厚さ

# 円盤の上面と下面のメッシュ
theta = np.linspace(0, 2 * np.pi, 60)
r = np.linspace(0, a, 20)
Theta, R = np.meshgrid(theta, r)
X = R * np.cos(Theta)
Y = R * np.sin(Theta)

# 上面 (z = h/2) と 下面 (z = -h/2)
Z_top = np.full_like(X, h / 2.0)
Z_bot = np.full_like(X, -h / 2.0)

# 上面・下面のプロット
ax1.plot_surface(X, Y, Z_top, color='coral', alpha=0.6, shade=True)
ax1.plot_surface(X, Y, Z_bot, color='coral', alpha=0.4, shade=True)

# 側面（円柱側面）
theta_cyl = np.linspace(0, 2 * np.pi, 60)
z_cyl = np.linspace(-h / 2.0, h / 2.0, 10)
Theta_c, Z_c = np.meshgrid(theta_cyl, z_cyl)
X_c = a * np.cos(Theta_c)
Y_c = a * np.sin(Theta_c)
ax1.plot_surface(X_c, Y_c, Z_c, color='cyan', alpha=0.7, shade=True)

# 外周ライン（境界条件強調）
ax1.plot(a * np.cos(theta), a * np.sin(theta), h / 2.0, 'b-', linewidth=2.5, label='外周境界: T = 0 ℃')
ax1.plot(a * np.cos(theta), a * np.sin(theta), -h / 2.0, 'b-', linewidth=2.5)

# 半径 a の矢印とラベル
ax1.plot([0, a], [0, 0], [h / 2.0, h / 2.0], 'k--', linewidth=1.5)
ax1.text(a / 2, 0.05, h / 2.0 + 0.02, '半径 a = 1.0 m', color='black', fontsize=11, fontweight='bold')

# 厚さ h の矢印とラベル
ax1.plot([a * 1.08, a * 1.08], [0, 0], [-h / 2.0, h / 2.0], 'k-', linewidth=1.5)
ax1.plot([a * 1.05, a * 1.11], [0, 0], [h / 2.0, h / 2.0], 'k-', linewidth=1.0)
ax1.plot([a * 1.05, a * 1.11], [0, 0], [-h / 2.0, -h / 2.0], 'k-', linewidth=1.0)
ax1.text(a * 1.12, 0, 0, '厚さ h = 0.1 m', color='black', fontsize=11, fontweight='bold')

# 領域内初期温度の注記
ax1.text(0, 0, h / 2.0 + 0.04, '初期温度: T(t=0) = 500 ℃\n(全域均一)', color='darkred', 
         fontsize=11, ha='center', va='bottom', bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

# 外周境界条件の吹き出し注記
ax1.text(0, -a * 1.25, 0, '外周境界条件 (t > 0):\nT = 0 ℃ (急冷)', color='blue',
         fontsize=11, ha='center', bbox=dict(boxstyle='round,pad=0.3', facecolor='aliceblue', edgecolor='blue'))

ax1.set_xlim([-1.3, 1.3])
ax1.set_ylim([-1.3, 1.3])
ax1.set_zlim([-0.3, 0.3])
ax1.set_box_aspect((1, 1, 0.4))
ax1.view_init(elev=28, azim=-55)
ax1.set_title('(a) 問題設定の幾何形状と境界・初期条件', fontsize=13, pad=10)
ax1.set_xlabel('x [m]')
ax1.set_ylabel('y [m]')
ax1.set_zlabel('z [m]')

# ----------------- 2. FEM解析モデルとメッシュ分割図 (2D平面) -----------------
ax2 = fig.add_subplot(1, 2, 2)

# メッシュ生成
from scipy.spatial import Delaunay
nodes = [[0.0, 0.0]]
boundary_indices = []
nr = 15  # 視認性向上のため程よい粗さで表示
for i in range(1, nr + 1):
    r_val = a * (i / nr)
    ntheta = max(6, int(6 * i))
    thetas = np.linspace(0, 2 * np.pi, ntheta, endpoint=False)
    start_idx = len(nodes)
    for th in thetas:
        nodes.append([r_val * np.cos(th), r_val * np.sin(th)])
    if i == nr:
        boundary_indices.extend(range(start_idx, len(nodes)))

nodes = np.array(nodes)
tri = Delaunay(nodes)
elements = tri.simplices
centroids = np.mean(nodes[elements], axis=1)
elements = elements[np.hypot(centroids[:, 0], centroids[:, 1]) < a]

# 三角形メッシュの描画
ax2.triplot(nodes[:, 0], nodes[:, 1], elements, color='#555555', linewidth=0.7, alpha=0.8, zorder=2)

# 節点の描画
interior_indices = np.setdiff1d(np.arange(len(nodes)), boundary_indices)
ax2.scatter(nodes[interior_indices, 0], nodes[interior_indices, 1], s=12, color='#e65100', 
            label='内部節点 (未知自由度 T)', zorder=3)
ax2.scatter(nodes[boundary_indices, 0], nodes[boundary_indices, 1], s=25, color='blue', 
            label='外周境界節点 (Dirichlet: T = 0 ℃)', zorder=4)

# 円の外枠
circle = plt.Circle((0, 0), a, color='blue', fill=False, linewidth=2.0, zorder=5)
ax2.add_patch(circle)

# 座標軸とアスペクト比
ax2.set_aspect('equal')
ax2.set_xlim([-1.25, 1.25])
ax2.set_ylim([-1.25, 1.25])
ax2.grid(True, linestyle=':', alpha=0.5)
ax2.set_title('(b) 2次元FEM解析モデル（離散化メッシュ）', fontsize=13)
ax2.set_xlabel('x [m]', fontsize=11)
ax2.set_ylabel('y [m]', fontsize=11)
ax2.legend(loc='upper right', fontsize=9, framealpha=0.9)

# 補足テキスト
ax2.text(0, -1.18, f'全節点数: {len(nodes)} / 要素数: {len(elements)} (一次三角形要素 CST)', 
         ha='center', fontsize=10, bbox=dict(boxstyle='square,pad=0.2', facecolor='white', edgecolor='gray'))

plt.tight_layout()
plt.savefig('c:/Users/sudak/src/fem_lesson/problem_schematic.png', dpi=200)
print("problem_schematic.png saved successfully.")

