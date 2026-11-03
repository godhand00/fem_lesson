"""
================================================================================
円盤の非定常熱伝導解析ソルバー (2次元有限要素法 FEM & ベッセル級数厳密解)
================================================================================
【支配方程式】
    ρ * c * (∂T/∂t) = λ * ∇²T = λ * (∂²T/∂x² + ∂²T/∂y²)

【初期条件】
    t = 0 で 円盤全域 T(x, y, 0) = 500 ℃

【境界条件】
    t > 0 で 外周境界 (r = a) において T = 0 ℃ (ディリクレ境界条件・急冷)
    上下面 (厚さ h) は断熱 (∂T/∂z = 0)

【解法アルゴリズム】
    - 空間離散化: 1次三角形要素 (CST: Constant Strain Triangle)
    - 時間離散化: Crank-Nicolson法 (θ = 0.5, 無条件安定・時間2次精度)
    - 連立方程式解法: 疎行列LU分解の事前計算 (scipy.sparse.linalg.factorized)
================================================================================
"""

import numpy as np
import scipy.special as sp
import scipy.sparse as sp_sparse
import scipy.sparse.linalg as sla
import matplotlib.pyplot as plt
import japanize_matplotlib

# ==============================================================================
# 1. 物理パラメータおよび材料定数の設定
# ==============================================================================
a = 1.0               # 円盤の半径 [m]
h = 0.1               # 円盤の厚さ [m] (体積積分で面積分に掛ける厚みパラメータ)
lam = 48.0            # 熱伝導率 λ [W/(m·K)] (鉄の物性値)
c = 480.0             # 比熱 c [J/(kg·K)] (鉄の物性値)
rho = 7.86e3          # 密度 ρ [kg/m³] (鉄の物性値)
T0 = 500.0            # 初期均一温度 [℃]

# 熱拡散率（温度伝導率） α = λ / (ρ * c) [m²/s]
# 熱が物体内を拡散していく速さを表す指標 (鉄の場合 約 1.27e-5 m²/s)
alpha = lam / (c * rho)

# ==============================================================================
# 2. 厳密解（理論解）の計算モジュール
# ==============================================================================
# 0次の第1種ベッセル関数 J_0(x) の正の零点 β_n を最初の100項まで取得
# (J_0(β_n) = 0 となる根: β_1 ≈ 2.4048, β_2 ≈ 5.5201, ...)
beta_roots = sp.jn_zeros(0, 100)

def exact_solution(r, t):
    """
    ベッセル級数展開による円盤非定常熱伝導の厳密解を計算する関数
    
    数式:
        T(r, t) = T0 * Σ [ A_n * J_0(β_n * r / a) * exp(-p_n * t) ]
        A_n = 2 / (β_n * J_1(β_n))
        p_n = α * (β_n / a)²
        
    Parameters:
        r : float または 1D array_like, 半径位置 [m]
        t : float または 1D array_like, 経過時間 [s]
        
    Returns:
        T : 指定した (r, t) における理論温度 [℃]
    """
    r_arr = np.atleast_1d(r)
    t_arr = np.atleast_1d(t)
    
    # 初期時刻 t = 0 の特異処理 (外周以外は T0, 外周は 0)
    if len(t_arr) == 1 and t_arr[0] == 0.0:
        res = np.full_like(r_arr, T0, dtype=float)
        res[np.isclose(r_arr, a)] = 0.0
        return res[0] if np.isscalar(r) else res

    # 級数の総和をベクトル外積演算で高速に計算
    # T のサイズ: (時間数, 半径点数)
    T = np.zeros((len(t_arr), len(r_arr)), dtype=float)
    for beta in beta_roots:
        J1_beta = sp.jv(1, beta)        # 1次ベッセル関数 J_1(β_n)
        An = 2.0 / (beta * J1_beta)     # 展開係数 A_n
        pn = alpha * (beta / a)**2      # 時間減衰係数 p_n [1/s]
        
        J0_val = sp.jv(0, beta * r_arr / a)  # 空間モード J_0(β_n * r / a)
        exp_val = np.exp(-pn * t_arr)        # 時間減衰項 exp(-p_n * t)
        
        # 外積により全 (t, r) ペアに一度に足し合わせ
        T += An * np.outer(exp_val, J0_val)
        
    res = T0 * T
    
    # 入力形状に応じてスカラーまたは配列を返却
    if np.isscalar(t) and np.isscalar(r):
        return res[0, 0]
    elif np.isscalar(t):
        return res[0]
    elif np.isscalar(r):
        return res[:, 0]
    return res

# ==============================================================================
# 3. 有限要素メッシュの自動生成
# ==============================================================================
def generate_mesh(radius=1.0, nr=30):
    """
    円盤領域の三角形メッシュを生成する
    
    方針:
      - 半径方向に同心円を配置し、外側ほど円周上の分割数を増やして
        三角形のアスペクト比（歪み）を均等に保つ。
      - scipy.spatial.Delaunay によりドローネ三角分割を実施。
      - 最外周のノード番号を境界節点（Dirichlet境界）として記録。
      
    Parameters:
        radius : 円盤半径 [m]
        nr     : 半径方向の分割同心円数 (デフォルト: 30)
        
    Returns:
        nodes          : 節点座標配列 shape: (N_nodes, 2)
        elements       : 要素節点インデックス配列 shape: (N_elements, 3)
        boundary_nodes : 外周境界節点のインデックス配列
    """
    from scipy.spatial import Delaunay
    
    # 原点 (0, 0) を最初の節点 (index: 0) として登録
    nodes = [[0.0, 0.0]]
    boundary_nodes = []
    
    # 同心円状に節点を配置
    for i in range(1, nr + 1):
        r = radius * (i / nr)
        # 外周に向かうほど円周が長くなるため、節点数も半径に比例して増やす
        ntheta = max(6, int(6 * i))
        thetas = np.linspace(0, 2 * np.pi, ntheta, endpoint=False)
        start_idx = len(nodes)
        for th in thetas:
            nodes.append([r * np.cos(th), r * np.sin(th)])
        # 最外周 (i == nr) の節点番号を境界節点リストに保存
        if i == nr:
            boundary_nodes.extend(range(start_idx, len(nodes)))
            
    nodes = np.array(nodes)
    
    # ドローネ三角分割で三角形要素を自動接続
    tri = Delaunay(nodes)
    elements = tri.simplices
    
    # 三角形の重心が半径 radius 内にある要素のみを採用
    centroids = np.mean(nodes[elements], axis=1)
    inside = np.hypot(centroids[:, 0], centroids[:, 1]) < radius
    elements = elements[inside]
    
    return nodes, elements, np.array(boundary_nodes)

# メッシュ生成の実行
nodes, elements, boundary_nodes = generate_mesh(a, nr=30)
num_nodes = len(nodes)
num_elements = len(elements)
print(f"メッシュ生成完了: 全節点数 = {num_nodes}, 要素数 = {num_elements}, 外周境界節点数 = {len(boundary_nodes)}")

# ==============================================================================
# 4. 要素行列の算出と全体行列のアセンブリ
# ==============================================================================
# 全体系常微分方程式: C * (dT/dt) + K * T = 0
# 疎行列（CSR形式）を効率よく構築するため、COO形式のリスト (I, J, V) で集約
I = []      # 行インデックス
J = []      # 列インデックス
V_K = []    # 熱伝導行列（剛性行列）成分
V_C = []    # 熱容量行列（質量行列）成分

for elem in elements:
    pts = nodes[elem]       # 要素を構成する3頂点の座標 shape: (3, 2)
    x = pts[:, 0]
    y = pts[:, 1]
    
    # 三角形のヤコビアン行列式 det(J) と要素面積 Area
    # det(J) = (x2 - x1)*(y3 - y1) - (x3 - x1)*(y2 - y1)
    detJ = (x[1] - x[0]) * (y[2] - y[0]) - (x[2] - x[0]) * (y[1] - y[0])
    Area = 0.5 * np.abs(detJ)
    
    # 形状関数の勾配ベクトル [∂N_i/∂x, ∂N_i/∂y] の係数計算
    b = np.array([y[1] - y[2], y[2] - y[0], y[0] - y[1]])
    c_grad = np.array([x[2] - x[1], x[0] - x[2], x[1] - x[0]])
    
    # 頂点の回り順（時計回り／反時計回り）による符号補正
    if detJ < 0:
        b = -b
        c_grad = -c_grad
        
    # 形状関数の勾配マトリクス shape: (3, 2)
    grad_N = np.column_stack([b, c_grad]) / (2.0 * Area)
    
    # --------------------------------------------------------------------------
    # 要素熱伝導行列 K^e:
    #   K^e_ij = ∫_e λ * (∇N_i · ∇N_j) * h dA = λ * h * Area * (∇N_i · ∇N_j)
    # --------------------------------------------------------------------------
    Ke = Area * h * lam * (grad_N @ grad_N.T)
    
    # --------------------------------------------------------------------------
    # 要素熱容量行列 C^e:
    #   C^e_ij = ∫_e ρ * c * (N_i * N_j) * h dA
    #   一次三角形要素の面積積分公式より: (ρ * c * Area * h / 12) * [[2,1,1],[1,2,1],[1,1,2]]
    # --------------------------------------------------------------------------
    Ce = (rho * c * Area * h / 12.0) * np.array([
        [2.0, 1.0, 1.0],
        [1.0, 2.0, 1.0],
        [1.0, 1.0, 2.0]
    ])
    
    # 要素内の3×3成分を全体座標系 (I, J) に割り当て
    for i in range(3):
        for j in range(3):
            I.append(elem[i])
            J.append(elem[j])
            V_K.append(Ke[i, j])
            V_C.append(Ce[i, j])

# COO形式から高速演算用のCSR (Compressed Sparse Row) 形式へ変換
K = sp_sparse.coo_matrix((V_K, (I, J)), shape=(num_nodes, num_nodes)).tocsr()
C = sp_sparse.coo_matrix((V_C, (I, J)), shape=(num_nodes, num_nodes)).tocsr()
print("全体熱伝導行列 K および熱容量行列 C のアセンブリ完了")

# ==============================================================================
# 5. 時間積分スキーム (Crank-Nicolson法) と境界条件の適用
# ==============================================================================
theta = 0.5         # 0.5: Crank-Nicolson法 (2次精度・無条件安定), 1.0: 後退オイラー
dt = 20.0           # 時間刻み Δt [s]
t_max = 12000.0     # 解析終了時刻 [s] (約 3.3 時間)
n_steps = int(t_max / dt)

# 初期温度ベクトルの設定: 全ノード 500 ℃、外周境界のみ 0 ℃
T = np.full(num_nodes, T0)
T[boundary_nodes] = 0.0

# 節点を「内部未知節点 Ω_I」と「拘束外周節点 Γ_D」に分類
interior_nodes = np.setdiff1d(np.arange(num_nodes), boundary_nodes)

# 一般化台形則の方程式:
#   (C / Δt + θ * K) * T^(n+1) = (C / Δt - (1 - θ) * K) * T^n
M_lhs = (C / dt + theta * K).tocsr()
M_rhs = (C / dt - (1.0 - theta) * K).tocsr()

# 内部節点ブロック A_II = M_lhs[interior, interior] を抽出
A_int = M_lhs[interior_nodes, :][:, interior_nodes]

# 【高速化の要】
# A_int は時間ステップを通じて一定であるため、ループ前に一度だけLU分解を実施。
# これにより各ステップでは O(N³) の逆行列計算ではなく、O(N) の前進後退代入のみで解ける。
solve_int = sla.factorized(A_int.tocsc())

# ==============================================================================
# 6. 時間ステップループ (時間発展解析)
# ==============================================================================
# 温度履歴の保存用配列
times = [0.0]
center_T_fem = [T[0]]  # 節点 0 は原点 (0, 0) の中心温度

# スナップショットを保存する指定時刻 [s] (10分, 30分, 1時間, 2時間, 3時間)
output_snapshots = [600.0, 1800.0, 3600.0, 7200.0, 10800.0]
snapshot_fem = {}

print(f"時間積分開始: 全 {n_steps} ステップ (Δt = {dt} s, t_max = {t_max} s)...")

for step in range(1, n_steps + 1):
    current_time = step * dt
    
    # 1. 右辺ベクトルの更新: b = M_rhs * T^n
    rhs_vec = M_rhs.dot(T)
    
    # 2. 内部未知節点の解 T^(n+1) を前進後退代入で求解
    T[interior_nodes] = solve_int(rhs_vec[interior_nodes])
    
    # 3. 外周ディリクレ境界条件の再代入: T_boundary = 0 ℃
    T[boundary_nodes] = 0.0
    
    # 4. 温度履歴の記録
    times.append(current_time)
    center_T_fem.append(T[0])
    
    # 指定時刻に達したらスナップショットを保存
    for ts in output_snapshots:
        if np.isclose(current_time, ts, atol=dt * 0.1):
            snapshot_fem[ts] = T.copy()

print("時間発展解析完了。グラフ作成と厳密解比較を行います。")

# ==============================================================================
# 7. 厳密解との比較および可視化プロット
# ==============================================================================
times = np.array(times)
center_T_fem = np.array(center_T_fem)

# 中心位置 (r = 0) における厳密解の冷却曲線を一括計算
center_T_exact = exact_solution(0.0, times)

# 3枚並列のグラフウィンドウを作成
fig = plt.figure(figsize=(16, 5.2))

# ------------------------------------------------------------------------------
# グラフ 1: 円盤中心温度の時間変化 (冷却曲線)
# ------------------------------------------------------------------------------
ax1 = fig.add_subplot(1, 3, 1)
ax1.plot(times / 60.0, center_T_exact, 'r-', linewidth=2.2, label='厳密解 (Bessel級数)')
ax1.plot(times / 60.0, center_T_fem, 'b--', linewidth=1.6, label='FEM (Crank-Nicolson)')
ax1.set_xlabel('経過時間 [分]', fontsize=11)
ax1.set_ylabel('中心温度 T(0, t) [℃]', fontsize=11)
ax1.set_title('(a) 円盤中心の冷却曲線 (温度時間履歴)', fontsize=12)
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.legend(fontsize=10)

# ------------------------------------------------------------------------------
# グラフ 2: 各時刻における半径方向温度分布 (FEM点 vs 厳密解曲線)
# ------------------------------------------------------------------------------
ax2 = fig.add_subplot(1, 3, 2)
r_nodes = np.hypot(nodes[:, 0], nodes[:, 1])  # 各節点の半径座標 r = √(x² + y²)
r_fine = np.linspace(0, a, 150)               # 厳密解用の微細半径配列
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

for ts, color in zip(output_snapshots, colors):
    t_min = int(ts / 60.0)
    # 厳密解の理論曲線を実線で描画
    T_ex = exact_solution(r_fine, ts)
    ax2.plot(r_fine, T_ex, '-', color=color, linewidth=1.5, label=f'{t_min}分 (厳密解)')
    # 全節点のFEM計算値を散布点で重ね合わせ
    ax2.plot(r_nodes, snapshot_fem[ts], 'o', color=color, markersize=2.5, alpha=0.35)

ax2.set_xlabel('半径位置 r [m]', fontsize=11)
ax2.set_ylabel('温度 T [℃]', fontsize=11)
ax2.set_title('(b) 半径方向温度分布 (線:厳密解, 点:FEM節点値)', fontsize=12)
ax2.grid(True, linestyle='--', alpha=0.6)
ax2.legend(fontsize=8.5, loc='upper right')

# ------------------------------------------------------------------------------
# グラフ 3: 2次元温度コンター図 (t = 30分 = 1800s)
# ------------------------------------------------------------------------------
ax3 = fig.add_subplot(1, 3, 3)
t_show = 1800.0
sol_show = snapshot_fem[t_show]

# 三角形メッシュ上のコンター塗りつぶし
cnt = ax3.tricontourf(nodes[:, 0], nodes[:, 1], elements, sol_show, levels=25, cmap='inferno')
fig.colorbar(cnt, ax=ax3, label='温度 [℃]')
ax3.set_aspect('equal')
ax3.set_title(f'(c) 2次元温度コンター図 (t = {int(t_show/60)} 分)', fontsize=12)
ax3.set_xlabel('x [m]', fontsize=11)
ax3.set_ylabel('y [m]', fontsize=11)

plt.tight_layout()
plt.savefig('fem_heat_conduction_comparison.png', dpi=200)
print("fem_heat_conduction_comparison.png を正常に保存しました。")
