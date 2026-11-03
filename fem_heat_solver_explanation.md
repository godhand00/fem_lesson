# fem_heat_solver.py 詳細コード解説書

本ドキュメントでは、有限要素法（FEM）を用いた円盤の2次元非定常熱伝導解析スクリプト [`fem_heat_solver.py`](fem_heat_solver.py) について、**数理モデル・有限要素理論とPythonコードの各行がどのように対応しているか**を体系的かつ詳細に解説します。

---

## 1. プログラムの全体構造

スクリプトは以下の7つの機能モジュールで構成されています。

```
┌────────────────────────────────────────────────────────┐
│ 1. 物理パラメータ設定 (物性値・形状寸法の定義)         │
├────────────────────────────────────────────────────────┤
│ 2. 厳密解モジュール (ベッセル関数の級数計算)           │
├────────────────────────────────────────────────────────┤
│ 3. メッシュ生成 (同心円配置 + ドローネ三角分割)        │
├────────────────────────────────────────────────────────┤
│ 4. 行列アセンブリ (要素行列 Kᵉ, Cᵉ の計算と疎行列構築)  │
├────────────────────────────────────────────────────────┤
│ 5. ソルバー準備 (Crank-Nicolson法・境界条件・事前LU分解)│
├────────────────────────────────────────────────────────┤
│ 6. 時間ステップループ (前進後退代入による解の更新)      │
├────────────────────────────────────────────────────────┤
│ 7. 結果の可視化 (厳密解との3分割比較プロット)          │
└────────────────────────────────────────────────────────┘
```

---

## 2. セクション別コード詳細解説

### 2.1 物理パラメータおよび材料定数の設定（Lines 27–42）

```python
a = 1.0               # 円盤の半径 [m]
h = 0.1               # 円盤の厚さ [m] (体積積分で面積分に掛ける厚みパラメータ)
lam = 48.0            # 熱伝導率 λ [W/(m·K)] (鉄の物性値)
c = 480.0             # 比熱 c [J/(kg·K)] (鉄の物性値)
rho = 7.86e3          # 密度 ρ [kg/m³] (鉄の物性値)
T0 = 500.0            # 初期均一温度 [℃]

# 熱拡散率（温度伝導率） α = λ / (ρ * c) [m²/s]
alpha = lam / (c * rho)
```

#### 【解説と数理背景】
- **熱拡散率 $\alpha$**:
  $$
  \alpha = \frac{\lambda}{\rho c} \approx 1.27226 \times 10^{-5}\ \text{m}^2/\text{s}
  $$
  物質がどれだけ早く熱を周囲へ拡散させるかを表す重要な指標です。
- **厚さ $h = 0.1\ \text{m}$**:
  本問題は2次元平面熱伝導問題として解きますが、有限要素法で実体積の熱容量（エネルギー蓄積量）を正しく評価するため、面積分に厚さ $h$ を乗じて体積積分に換算しています。

---

### 2.2 厳密解（理論解）の計算モジュール（Lines 44–98）

```python
# 0次の第1種ベッセル関数 J_0(x) の正の零点 β_n を最初の100項まで取得
beta_roots = sp.jn_zeros(0, 100)

def exact_solution(r, t):
    ...
    T = np.zeros((len(t_arr), len(r_arr)), dtype=float)
    for beta in beta_roots:
        J1_beta = sp.jv(1, beta)        # 1次ベッセル関数 J_1(β_n)
        An = 2.0 / (beta * J1_beta)     # 展開係数 A_n
        pn = alpha * (beta / a)**2      # 時間減衰係数 p_n [1/s]
        
        J0_val = sp.jv(0, beta * r_arr / a)  # 空間モード J_0(β_n * r / a)
        exp_val = np.exp(-pn * t_arr)        # 時間減衰項 exp(-p_n * t)
        
        # 外積により全 (t, r) ペアに一度に足し合わせ
        T += An * np.outer(exp_val, J0_val)
```

#### 【解説と数理背景】
- **数式との対応**:
  $$
  T(r, t) = T_0 \sum_{n=1}^{100} A_n J_0\left(\beta_n \frac{r}{a}\right) e^{-p_n t}, \quad A_n = \frac{2}{\beta_n J_1(\beta_n)}, \quad p_n = \alpha \frac{\beta_n^2}{a^2}
  $$
- `scipy.special.jn_zeros(0, 100)`:
  $J_0(\beta_n) = 0$ となる根（固有値）$\beta_1 \approx 2.4048, \beta_2 \approx 5.5201, \dots$ を一括取得します。100項まで足し合わせることで、初期時刻付近の急峻な温度変化も極めて正確に表現できます。
- `np.outer(exp_val, J0_val)`:
  時間ベクトル $\mathbf{e} = [e^{-p_n t_1}, \dots]^T$ と空間ベクトル $\mathbf{j} = [J_0(\beta_n r_1/a), \dots]$ の外積を取ることで、2重ループを使わずにすべての時間・位置の組み合わせを高速行列演算（ベクトル化）しています。

---

### 2.3 有限要素メッシュの自動生成（Lines 100–156）

```python
def generate_mesh(radius=1.0, nr=30):
    from scipy.spatial import Delaunay
    nodes = [[0.0, 0.0]]
    boundary_nodes = []
    
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
    tri = Delaunay(nodes)
    elements = tri.simplices
    ...
```

#### 【解説と幾何学的工夫】
1. **なぜ外側ほど円周分割数を増やすのか？（`ntheta = int(6 * i)`）**:
   円周の長さは $2\pi r$ なので、中心付近と外周で同じ分割数にすると、外側の要素が細長くなって潰れてしまいます（アスペクト比の悪化）。半径 $r$（層番号 $i$）に比例して節点数を増やすことで、**円盤全体でほぼ正三角形に近い良質な要素網目**が自動生成されます。
2. **ドローネ三角分割（`Delaunay`）**:
   配置した点群に対して、外接円の中に他の点を含まない「最も歪みが小さい三角形接続」を数学的に自動生成します。
3. **境界節点の自動記録**:
   最外周（$i = nr$）の節点番号を `boundary_nodes` に記録しておき、後のディリクレ境界条件適用（$T = 0^\circ\text{C}$）で使用します。

---

### 2.4 要素行列の算出と全体行列のアセンブリ（Lines 158–226）

```python
for elem in elements:
    pts = nodes[elem]       # 3頂点座標 shape: (3, 2)
    x = pts[:, 0]
    y = pts[:, 1]
    
    # ヤコビアン行列式 det(J) と要素面積 Area
    detJ = (x[1] - x[0]) * (y[2] - y[0]) - (x[2] - x[0]) * (y[1] - y[0])
    Area = 0.5 * np.abs(detJ)
    
    # 形状関数の勾配ベクトル
    b = np.array([y[1] - y[2], y[2] - y[0], y[0] - y[1]])
    c_grad = np.array([x[2] - x[1], x[0] - x[2], x[1] - x[0]])
    if detJ < 0:
        b, c_grad = -b, -c_grad
    grad_N = np.column_stack([b, c_grad]) / (2.0 * Area)
    
    # 要素熱伝導行列 (剛性行列)
    Ke = Area * h * lam * (grad_N @ grad_N.T)
    
    # 要素熱容量行列 (質量行列)
    Ce = (rho * c * Area * h / 12.0) * np.array([
        [2.0, 1.0, 1.0],
        [1.0, 2.0, 1.0],
        [1.0, 1.0, 2.0]
    ])
```

#### 【解説と数理背景】
一次三角形要素（CST要素）内の形状関数 $N_i(x, y)$ の勾配 $\nabla N_i$ は要素内で定数となります：
$$
\begin{bmatrix}
\frac{\partial N_1}{\partial x} \\
\frac{\partial N_2}{\partial x} \\
\frac{\partial N_3}{\partial x}
\end{bmatrix}
= \frac{1}{2 A_e}
\begin{bmatrix}
y_2 - y_3 \\
y_3 - y_1 \\
y_1 - y_2
\end{bmatrix}, \quad
\begin{bmatrix}
\frac{\partial N_1}{\partial y} \\
\frac{\partial N_2}{\partial y} \\
\frac{\partial N_3}{\partial y}
\end{bmatrix}
= \frac{1}{2 A_e}
\begin{bmatrix}
x_3 - x_2 \\
x_1 - x_3 \\
x_2 - x_1
\end{bmatrix}
$$

- **要素熱伝導行列 $K^e$（剛性行列）**:
  $$
  K^e_{ij} = \int_e \lambda (\nabla N_i \cdot \nabla N_j) h \, dA = \lambda h A_e (\nabla N_i \cdot \nabla N_j)
  $$
  Pythonでは行列積 `grad_N @ grad_N.T` で $3 \times 3$ 行列を一括計算しています。
- **要素熱容量行列 $C^e$（質量行列）**:
  面積座標の積分公式 $\int_e L_i L_j \, dA$ より、対角成分は $2$、非対角成分は $1$ となる厳密解が得られます：
  $$
  C^e = \frac{\rho c h A_e}{12}
  \begin{bmatrix}
  2 & 1 & 1 \\
  1 & 2 & 1 \\
  1 & 1 & 2
  \end{bmatrix}
  $$
- **COO形式（Coordinate format）**:
  全要素の $3 \times 3$ 成分の「行番号 `I`」「列番号 `J`」「値 `V`」をリストに追加し、最後に `sp_sparse.coo_matrix` で一気に組み立ててから高速な CSR 形式へ変換します。これにより、巨大なゼロ行列を確保することなく $O(N)$ のメモリで全体行列を構築できます。

---

### 2.5 時間積分スキームとディリクレ境界条件の処理（Lines 228–261）

```python
theta = 0.5         # Crank-Nicolson法 (時間2次精度・無条件安定)
dt = 20.0           # 時間刻み Δt [s]
t_max = 12000.0     # 解析終了時刻 [s]
n_steps = int(t_max / dt)

# 初期条件
T = np.full(num_nodes, T0)
T[boundary_nodes] = 0.0

interior_nodes = np.setdiff1d(np.arange(num_nodes), boundary_nodes)

# 一般化台形則: (C/Δt + θ*K) T^(n+1) = (C/Δt - (1-θ)*K) T^n
M_lhs = (C / dt + theta * K).tocsr()
M_rhs = (C / dt - (1.0 - theta) * K).tocsr()

# 内部節点ブロックのみ抽出
A_int = M_lhs[interior_nodes, :][:, interior_nodes]

# 【最重要の高速化】事前LU分解
solve_int = sla.factorized(A_int.tocsc())
```

#### 【解説とアルゴリズムの急所】
1. **Crank-Nicolson法（$\theta = 0.5$）**:
   前進オイラー法（$\theta=0$）のような時間刻み制限（クーラン条件 $\Delta t \le \Delta x^2 / 2\alpha$）がなく、大きな時間刻み（$\Delta t = 20\,\text{s}$）でも解が発散しません。
2. **境界条件による自由度縮約**:
   外周境界節点 $\Gamma_D$ では常に $T = 0^\circ\text{C}$ と分かっているため、未知数から除外します。解くべき方程式は内部節点集合 $\Omega_I$ のみに関する連立一次方程式：
   $$
   \mathbf{A}_{II} \mathbf{T}_I^{n+1} = \mathbf{b}_I
   $$
   に縮約されます。
3. **事前LU分解（`sla.factorized`）**:
   係数行列 $\mathbf{A}_{II} = \frac{\mathbf{C}_{II}}{\Delta t} + \theta \mathbf{K}_{II}$ は時間ステップを通じて**一切変化しません**。  
   もし毎ステップ連立方程式を愚直に解くと、計算量はステップ数 $\times O(N^3)$ となり莫大な時間がかかります。  
   ループの直前に1度だけLU分解（因数分解）をキャッシュしておくことで、各ステップの計算量を $O(N)$（前進後退代入のみ）に削減し、**全体の計算を数秒で瞬時に完了**させています。

---

### 2.6 時間ステップループ（Lines 263–298）

```python
for step in range(1, n_steps + 1):
    current_time = step * dt
    
    # 1. 右辺ベクトルの更新
    rhs_vec = M_rhs.dot(T)
    
    # 2. 内部未知節点の解 T^(n+1) を前進後退代入で求解
    T[interior_nodes] = solve_int(rhs_vec[interior_nodes])
    
    # 3. 外周ディリクレ境界条件の再代入
    T[boundary_nodes] = 0.0
    
    # 4. 温度履歴・スナップショットの記録
    times.append(current_time)
    center_T_fem.append(T[0])
    ...
```

#### 【ステップごとの実行内容】
1. **右辺ベクトルの評価**: 前ステップの温度分布 $\mathbf{T}^n$ に右辺作用素 $\mathbf{M}_{\text{rhs}}$ を掛けて、次ステップの励起項を求めます。
2. **求解**: キャッシュされた分解オブジェクト `solve_int` を呼び出し、内部節点の温度 $\mathbf{T}_I^{n+1}$ を更新します。
3. **境界値の固定**: 外周節点に $0^\circ\text{C}$ をセットします。
4. **中心温度の記録**: メッシュ生成時に原点 $(0, 0)$ をインデックス $0$ として登録してあるため、`T[0]` を参照するだけで中心温度を追跡できます。

---

### 2.7 結果の可視化とプロット作成（Lines 300–363）

```python
fig = plt.figure(figsize=(16, 5.2))
# (a) 中心冷却曲線の比較 (線グラフ)
ax1.plot(times / 60.0, center_T_exact, 'r-', label='厳密解 (Bessel級数)')
ax1.plot(times / 60.0, center_T_fem, 'b--', label='FEM (Crank-Nicolson)')

# (b) 半径方向プロファイルの比較 (散布図 vs 理論曲線)
r_nodes = np.hypot(nodes[:, 0], nodes[:, 1])
...

# (c) 2次元温度コンター図 (tricontourf)
cnt = ax3.tricontourf(nodes[:, 0], nodes[:, 1], elements, sol_show, levels=25, cmap='inferno')
```

#### 【可視化のポイント】
- **(a) 冷却曲線**:
  全2,791節点のうち、最も冷却が遅い中心点（$r=0$）の温度変化を分単位でプロット。理論解とFEMが完全に重なることを実証します。
- **(b) 半径方向プロファイル**:
  2次元平面上に散らばる全節点について、中心からの距離 $r = \sqrt{x^2 + y^2}$ を計算し、各時刻における温度分布をプロット。理論曲線の上に全点が乗っていることを確認します。
- **(c) 非構造格子コンター図 (`tricontourf`)**:
  正方形グリッド用ではなく、非構造三角形メッシュを直接可視化できる `tricontourf` を使用して、熱が外周から均等に抜けていく同心円状の温度分布を美しく描画します。

---

## 3. まとめ：プログラムの優れた実装設計

1. **追加CAD/メッシュツール不要の完結設計**:
   外部のメッシュジェネレータ（Gmshなど）を使わず、Python標準ライブラリ群（`scipy.spatial.Delaunay`）のみで歪みのない高品質メッシュを自前生成。
2. **大規模問題に耐えうる疎行列・ベクトル化**:
   `scipy.sparse` と `np.outer` を駆使し、メモリ消費を最小限に抑えつつ高次ベッセル級数も一瞬で計算。
3. **事前LU分解による圧倒的な高速化**:
   Crank-Nicolson法の方程式構造を活かし、時間不変な行列分解をループ外に追い出すことで、600ステップの時間発展計算をわずか数秒で完遂。

