"""
================================================================================
有限要素法ライブラリ (scikit-fem) を用いた円盤非定常熱伝導解析サンプル
================================================================================
【特徴】
  - メッシュ生成から弱形式の定義、行列アセンブリまでライブラリが自動化
  - 自作版（約180行）に比べ、わずか数十行で直感的に記述可能
  - 厳密解（ベッセル級数）との比較グラフを生成
================================================================================
"""

import numpy as np
import scipy.special as sp
from scipy.sparse.linalg import factorized
import matplotlib.pyplot as plt
import japanize_matplotlib

# scikit-fem の主要モジュールをインポート
from skfem import MeshTri, Basis, ElementTriP1, BilinearForm, asm
from skfem.helpers import dot, grad

# ==============================================================================
# 1. 物理パラメータの設定
# ==============================================================================
a = 1.0               # 半径 [m]
h = 0.1               # 厚さ [m]
lam = 48.0            # 熱伝導率 [W/(m·K)]
c = 480.0             # 比熱 [J/(kg·K)]
rho = 7.86e3          # 密度 [kg/m³]
T0 = 500.0            # 初期温度 [℃]
alpha = lam / (c * rho)

# ==============================================================================
# 2. scikit-fem によるメッシュ生成 (1行で円形メッシュを作成！)
# ==============================================================================
# init_circle(nrefs=4): 単位円の三角形メッシュを4回細分化 (要素数: 1024, 節点数: 545)
mesh = MeshTri.init_circle(nrefs=4)
# 1次三角形要素 (P1要素) 上の有限要素基底関数を定義
basis = Basis(mesh, ElementTriP1())
print(f"[scikit-fem] メッシュ生成完了: 節点数 = {mesh.nvertices}, 要素数 = {mesh.nelements}")

# ==============================================================================
# 3. 弱形式の定義 (数学の表記そのままデコレータで記述！)
# ==============================================================================
@BilinearForm
def stiffness(u, v, w):
    # 熱伝導項: ∇u · ∇v
    return dot(grad(u), grad(v))

@BilinearForm
def mass(u, v, w):
    # 熱容量項: u * v
    return u * v

# ==============================================================================
# 4. 全体行列の自動アセンブリ (asm 関数)
# ==============================================================================
# 積分計算や疎行列の構築はライブラリが内部で完全自動実行
K = asm(stiffness, basis) * (lam * h)
C = asm(mass, basis) * (rho * c * h)
print("[scikit-fem] 熱伝導行列 K および熱容量行列 C のアセンブリ完了")

# ==============================================================================
# 5. 境界条件と時間積分スキームの設定
# ==============================================================================
dt = 20.0           # 時間刻み [s]
t_max = 12000.0     # 解析終了時刻 [s]
theta = 0.5         # Crank-Nicolson法
n_steps = int(t_max / dt)

# 外周境界 (r ≒ 1.0) の自由度番号を取得
boundary_dofs = basis.get_dofs(lambda x: np.hypot(x[0], x[1]) >= 0.99)
interior_dofs = basis.complement_dofs(boundary_dofs)

# Crank-Nicolson法の時間進行行列: A * T^(n+1) = B * T^n
A = C / dt + theta * K
B = C / dt - (1.0 - theta) * K

# 内部自由度ブロックの事前LU分解 (高速化)
A_int = A[interior_dofs, :][:, interior_dofs]
solve_step = factorized(A_int.tocsc())

# 初期温度ベクトル (外周 0 ℃, 内部 500 ℃)
T = np.full(mesh.nvertices, T0)
T[boundary_dofs] = 0.0

# 中心点 (r=0) の節点番号を特定
r_all = np.hypot(mesh.p[0], mesh.p[1])
center_idx = np.argmin(r_all)

# ==============================================================================
# 6. 時間ステップループ
# ==============================================================================
times = [0.0]
center_T_fem = [T[center_idx]]
output_snapshots = [600.0, 1800.0, 3600.0, 7200.0, 10800.0]
snapshot_fem = {}

print(f"[scikit-fem] 時間発展計算開始 (全 {n_steps} ステップ)...")
for step in range(1, n_steps + 1):
    current_time = step * dt
    
    # 1. 右辺計算
    rhs = B @ T
    # 2. 内部節点の更新 (前進後退代入)
    T[interior_dofs] = solve_step(rhs[interior_dofs])
    # 3. 外周境界ディリクレ条件の再適用
    T[boundary_dofs] = 0.0
    
    times.append(current_time)
    center_T_fem.append(T[center_idx])
    
    for ts in output_snapshots:
        if np.isclose(current_time, ts, atol=dt * 0.1):
            snapshot_fem[ts] = T.copy()

print("[scikit-fem] 計算完了。厳密解との比較プロットを作成します。")

# ==============================================================================
# 7. 厳密解の計算と結果の可視化
# ==============================================================================
beta_roots = sp.jn_zeros(0, 100)
def exact_solution(r, t):
    r_arr, t_arr = np.atleast_1d(r), np.atleast_1d(t)
    res = np.zeros((len(t_arr), len(r_arr)), dtype=float)
    for beta in beta_roots:
        An = 2.0 / (beta * sp.jv(1, beta))
        pn = alpha * (beta / a)**2
        res += An * np.outer(np.exp(-pn * t_arr), sp.jv(0, beta * r_arr / a))
    res *= T0
    return res[0] if np.isscalar(t) else (res[:, 0] if np.isscalar(r) else res)

times = np.array(times)
center_T_exact = exact_solution(0.0, times)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# (1) 中心冷却曲線の比較
axes[0].plot(times / 60.0, center_T_exact, 'r-', linewidth=2.2, label='厳密解 (Bessel級数)')
axes[0].plot(times / 60.0, center_T_fem, 'b--', linewidth=1.6, label='scikit-fem (FEM)')
axes[0].set_xlabel('経過時間 [分]', fontsize=11)
axes[0].set_ylabel('中心温度 T(0, t) [℃]', fontsize=11)
axes[0].set_title('中心温度の冷却曲線比較 (scikit-fem vs 厳密解)', fontsize=12)
axes[0].grid(True, linestyle='--', alpha=0.6)
axes[0].legend(fontsize=10)

# (2) t = 30 分の2次元温度コンター図 (scikit-fem のメッシュ描画)
t_show = 1800.0
sol_show = snapshot_fem[t_show]
cnt = axes[1].tricontourf(mesh.p[0], mesh.p[1], mesh.t.T, sol_show, levels=25, cmap='plasma')
fig.colorbar(cnt, ax=axes[1], label='温度 [℃]')
axes[1].set_aspect('equal')
axes[1].set_title(f'scikit-fem 温度コンター図 (t = {int(t_show/60)} 分)', fontsize=12)
axes[1].set_xlabel('x [m]', fontsize=11)
axes[1].set_ylabel('y [m]', fontsize=11)

plt.tight_layout()
plt.savefig('fem_skfem_comparison.png', dpi=200)
print("fem_skfem_comparison.png を保存しました。")

