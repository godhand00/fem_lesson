import numpy as np
import scipy.special as sp
import scipy.sparse as sp_sparse
import scipy.sparse.linalg as sla
import matplotlib.pyplot as plt
import japanize_matplotlib

# 物理パラメータ
a = 1.0               # 半径 [m]
h = 0.1               # 厚さ [m]
lam = 48.0            # 熱伝導率 [W/(m K)]
c = 480.0             # 比熱 [J/(kg K)]
rho = 7.86e3          # 密度 [kg/m^3]
T0 = 500.0            # 初期温度 [℃]
alpha = lam / (c * rho) # 熱拡散率 [m^2/s]

beta_roots = sp.jn_zeros(0, 100)

def exact_solution(r, t):
    r = np.asarray(r)
    if np.isscalar(t) and t == 0.0:
        res = np.full_like(r, T0, dtype=float)
        res[np.isclose(r, a)] = 0.0
        return res
    
    # rが1次元またはスカラー、tが1次元またはスカラー両方に対応
    # t: (M,), r: (N,) -> (M, N) or (N,)
    r_arr = np.atleast_1d(r)
    t_arr = np.atleast_1d(t)
    
    # T shape: (len(t_arr), len(r_arr))
    T = np.zeros((len(t_arr), len(r_arr)), dtype=float)
    for beta in beta_roots:
        J1_beta = sp.jv(1, beta)
        An = 2.0 / (beta * J1_beta)
        pn = alpha * (beta / a)**2
        J0_val = sp.jv(0, beta * r_arr / a) # (len(r_arr),)
        exp_val = np.exp(-pn * t_arr)       # (len(t_arr),)
        T += An * np.outer(exp_val, J0_val)
    res = T0 * T
    if np.isscalar(t) and np.isscalar(r):
        return res[0, 0]
    elif np.isscalar(t):
        return res[0]
    elif np.isscalar(r):
        return res[:, 0]
    return res

def generate_mesh(radius=1.0, nr=30):
    from scipy.spatial import Delaunay
    nodes = [[0.0, 0.0]]
    boundary_nodes = []
    
    for i in range(1, nr + 1):
        r = radius * (i / nr)
        ntheta = max(6, int(6 * i))
        thetas = np.linspace(0, 2 * np.pi, ntheta, endpoint=False)
        start_idx = len(nodes)
        for th in thetas:
            nodes.append([r * np.cos(th), r * np.sin(th)])
        if i == nr:
            boundary_nodes.extend(range(start_idx, len(nodes)))
            
    nodes = np.array(nodes)
    tri = Delaunay(nodes)
    elements = tri.simplices
    
    centroids = np.mean(nodes[elements], axis=1)
    inside = np.hypot(centroids[:, 0], centroids[:, 1]) < radius
    elements = elements[inside]
    boundary_nodes = np.array(boundary_nodes)
    return nodes, elements, boundary_nodes

nodes, elements, boundary_nodes = generate_mesh(a, nr=30)
num_nodes = len(nodes)

# 行列組み立て
I, J, V_K, V_C = [], [], [], []
for elem in elements:
    pts = nodes[elem]
    x, y = pts[:, 0], pts[:, 1]
    detJ = (x[1] - x[0]) * (y[2] - y[0]) - (x[2] - x[0]) * (y[1] - y[0])
    Area = 0.5 * np.abs(detJ)
    b = np.array([y[1] - y[2], y[2] - y[0], y[0] - y[1]])
    c_grad = np.array([x[2] - x[1], x[0] - x[2], x[1] - x[0]])
    if detJ < 0:
        b, c_grad = -b, -c_grad
    grad_N = np.column_stack([b, c_grad]) / (2.0 * Area)
    Ke = Area * h * lam * (grad_N @ grad_N.T)
    Ce = (rho * c * Area * h / 12.0) * np.array([
        [2.0, 1.0, 1.0],
        [1.0, 2.0, 1.0],
        [1.0, 1.0, 2.0]
    ])
    for i in range(3):
        for j in range(3):
            I.append(elem[i])
            J.append(elem[j])
            V_K.append(Ke[i, j])
            V_C.append(Ce[i, j])

K = sp_sparse.coo_matrix((V_K, (I, J)), shape=(num_nodes, num_nodes)).tocsr()
C = sp_sparse.coo_matrix((V_C, (I, J)), shape=(num_nodes, num_nodes)).tocsr()

theta = 0.5
dt = 20.0
t_max = 12000.0
n_steps = int(t_max / dt)

T = np.full(num_nodes, T0)
T[boundary_nodes] = 0.0

interior_nodes = np.setdiff1d(np.arange(num_nodes), boundary_nodes)
M_lhs = (C / dt + theta * K).tocsr()
M_rhs = (C / dt - (1.0 - theta) * K).tocsr()
A_int = M_lhs[interior_nodes, :][:, interior_nodes]
solve_int = sla.factorized(A_int.tocsc())

# 時間履歴の記録
times = [0.0]
center_T_fem = [T[0]] # ノード0が中心 (0, 0)

output_snapshots = [600.0, 1800.0, 3600.0, 7200.0, 10800.0]
snapshot_fem = {}

for step in range(1, n_steps + 1):
    current_time = step * dt
    rhs_vec = M_rhs.dot(T)
    T[interior_nodes] = solve_int(rhs_vec[interior_nodes])
    T[boundary_nodes] = 0.0
    
    times.append(current_time)
    center_T_fem.append(T[0])
    
    for ts in output_snapshots:
        if np.isclose(current_time, ts, atol=dt*0.1):
            snapshot_fem[ts] = T.copy()

times = np.array(times)
center_T_fem = np.array(center_T_fem)
center_T_exact = exact_solution(0.0, times)

# グラフ作成 (3分割)
fig = plt.figure(figsize=(16, 5))

# 1. 中心温度の時間変化
ax1 = fig.add_subplot(1, 3, 1)
ax1.plot(times / 60, center_T_exact, 'r-', linewidth=2, label='厳密解 (Exact)')
ax1.plot(times / 60, center_T_fem, 'b--', linewidth=1.5, label='FEM (Crank-Nicolson)')
ax1.set_xlabel('時間 [分]', fontsize=11)
ax1.set_ylabel('中心温度 T(0, t) [℃]', fontsize=11)
ax1.set_title('円盤中心の冷却曲線 (温度の時間履歴)', fontsize=12)
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.legend(fontsize=10)

# 2. 半径方向温度プロファイルの比較
ax2 = fig.add_subplot(1, 3, 2)
r_nodes = np.hypot(nodes[:, 0], nodes[:, 1])
r_fine = np.linspace(0, a, 100)
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

for ts, color in zip(output_snapshots, colors):
    t_min = int(ts / 60)
    T_ex = exact_solution(r_fine, ts)
    ax2.plot(r_fine, T_ex, '-', color=color, label=f'{t_min}分 (厳密解)')
    # 半径方向のノード値をサンプリングしてプロット
    ax2.plot(r_nodes, snapshot_fem[ts], 'o', color=color, markersize=2.5, alpha=0.35)

ax2.set_xlabel('半径 r [m]', fontsize=11)
ax2.set_ylabel('温度 T [℃]', fontsize=11)
ax2.set_title('各時刻における半径方向温度分布 (線:厳密解, 点:FEM)', fontsize=12)
ax2.grid(True, linestyle='--', alpha=0.6)
ax2.legend(fontsize=8, loc='upper right')

# 3. 2Dコンター図 (t = 30分 = 1800s)
ax3 = fig.add_subplot(1, 3, 3)
t_show = 1800.0
sol_show = snapshot_fem[t_show]
cnt = ax3.tricontourf(nodes[:, 0], nodes[:, 1], elements, sol_show, levels=25, cmap='inferno')
fig.colorbar(cnt, ax=ax3, label='温度 [℃]')
ax3.set_aspect('equal')
ax3.set_title(f'FEM 温度コンター (t = {int(t_show/60)} 分)', fontsize=12)
ax3.set_xlabel('x [m]', fontsize=11)
ax3.set_ylabel('y [m]', fontsize=11)

plt.tight_layout()
plt.savefig('fem_heat_conduction_comparison.png', dpi=200)
print("fem_heat_conduction_comparison.png を保存しました。")
