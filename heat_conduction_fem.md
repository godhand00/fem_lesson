# 円盤の非定常熱伝導問題における有限要素法（FEM）解析と厳密解の比較

本資料では、初期高温状態にある鉄製円盤が外周から急冷される非定常熱伝導問題について、基礎方程式から弱形式の導出、有限要素法（FEM）による空間・時間離散化、Pythonスクリプトによる数値計算、およびベッセル関数を用いた厳密解との比較検証までを体系的にまとめます。

---

## 1. 問題設定

### 1.1 概要と模式図
半径 $a = 1.0\ \text{m}$、厚さ $h = 0.1\ \text{m}$ の鉄製円盤を対象とします。  
初期状態で全体が均一に $500^\circ\text{C}$ に熱せられており、時刻 $t > 0$ で外周面が瞬時に $0^\circ\text{C}$ に冷却・保持される急冷（クエンチ）過程を解析します。

![問題設定の幾何形状・境界条件およびFEMメッシュモデル](problem_schematic.png)

| パラメータ | 記号 | 設定値 | 単位 | 備考 |
| :--- | :---: | :---: | :---: | :--- |
| 円盤半径 | $a$ | $1.0$ | $\text{m}$ | 円盤の外径 |
| 円盤厚さ | $h$ | $0.1$ | $\text{m}$ | $z$ 方向寸法（一様厚さ） |
| 熱伝導率 | $\lambda$ | $48.0$ | $\text{W/(m}\cdot\text{K)}$ | 鉄の標準物性値 |
| 比熱 | $c$ | $480.0$ | $\text{J/(kg}\cdot\text{K)}$ | 鉄の標準物性値 |
| 密度 | $\rho$ | $7.86 \times 10^3$ | $\text{kg/m}^3$ | 鉄の標準物性値 |
| 初期温度 | $T_0$ | $500.0$ | $^\circ\text{C}$ | $t = 0$ における領域内均一温度 |
| 外周冷却温度 | $T_{\text{boundary}}$ | $0.0$ | $^\circ\text{C}$ | $t > 0$ における外周拘束温度 |

#### 熱拡散率（温度伝導率） α
熱の伝わりやすさを表す物性パラメータ：

$$
\alpha = \frac{\lambda}{\rho c} = \frac{48.0}{7.86 \times 10^3 \times 480} \approx 1.27226 \times 10^{-5}\ \text{m}^2/\text{s}
$$

---

### 1.2 支配方程式と初期・境界条件
厚み $h = 0.1\ \text{m}$ に対して上下面は断熱（または厚み方向の熱流がゼロで一様）とみなせるため、2次元平面（$xy$ 直交座標系）の非定常熱伝導問題として定式化できます。

- **支配方程式**:

$$
\rho c \frac{\partial T}{\partial t} = \lambda \nabla^2 T = \lambda \left( \frac{\partial^2 T}{\partial x^2} + \frac{\partial^2 T}{\partial y^2} \right)
$$

- **初期条件 ($t = 0$)**:

$$
T(x, y, 0) = T_0 = 500^\circ\text{C} \quad (\text{領域内全域})
$$

- **境界条件 ($t > 0$)**:

$$
T(x, y, t) = 0^\circ\text{C} \quad (\text{外周境界 } r = \sqrt{x^2+y^2} = a)
$$

$$
\frac{\partial T}{\partial z} = 0 \quad (\text{上下面断熱})
$$

---

## 2. 厳密解（理論解）

本問題は円周方向に対称（軸対称）であるため、極座標表示における拡散方程式：

$$
\frac{\partial T}{\partial t} = \alpha \left( \frac{\partial^2 T}{\partial r^2} + \frac{1}{r} \frac{\partial T}{\partial r} \right)
$$

を変数分離法で解くことにより、第1種ベッセル関数 $J_0(x), J_1(x)$ の級数解として厳密解が導出されます。

$$
T(r, t) = T_0 \sum_{n=1}^{\infty} A_n J_0\left(\beta_n \frac{r}{a}\right) e^{-p_n t}
$$

ここで：
- $\beta_n$ : 0次の第1種ベッセル関数 $J_0(x)$ の第 $n$ 番目の正の零点（$J_0(\beta_n) = 0$）
  - $\beta_1 \approx 2.4048,\ \beta_2 \approx 5.5201,\ \beta_3 \approx 8.6537, \dots$
- $J_1(x)$ : 1次の第1種ベッセル関数
- 級数展開係数 $A_n$:

$$
A_n = \frac{2}{\beta_n J_1(\beta_n)}
$$

- 時間減衰係数 $p_n$:

$$
p_n = \alpha \frac{\beta_n^2}{a^2} = \frac{\lambda}{\rho c} \frac{\beta_n^2}{a^2}
$$

> **時定数の目安**:  
> 最小の減衰率 $p_1 \approx 1.272 \times 10^{-5} \times (2.4048)^2 \approx 7.357 \times 10^{-5}\ \text{s}^{-1}$。  
> 基準時定数 $\tau_1 = 1 / p_1 \approx 13590\ \text{s} \approx 3.77\ \text{時間}$ であり、円盤の中心まで冷却が進むには数時間オーダーを要します。

---

## 3. 有限要素法（FEM）による定式化

解析全体のワークフローは以下の4つのフェーズ（理論モデルの導出 $\to$ FEM空間離散化 $\to$ 時間積分と高速数値解法 $\to$ 厳密解との比較検証）に分類されます。

![FEM解析の全体ワークフロー詳細図](fem_workflow_diagram.png)

### 3.1 弱形式（Weak Form）の導出
任意の仮想温度（重み関数） $w(x, y)$ を支配方程式の両辺に乗じ、円盤領域 $\Omega$ 全体で面積積分します：

$$
\int_{\Omega} \rho c \frac{\partial T}{\partial t} w \, dA = \int_{\Omega} \lambda (\nabla^2 T) w \, dA
$$

右辺にグリーンの定理（2次元部分積分）を適用すると：

$$
\int_{\Omega} \lambda (\nabla^2 T) w \, dA = - \int_{\Omega} \lambda \nabla T \cdot \nabla w \, dA + \oint_{\partial \Omega} \lambda \frac{\partial T}{\partial n} w \, ds
$$

ディリクレ境界条件が課されている外周境界 $\partial \Omega$ 上では、仮想温度の条件として $w|_{\partial \Omega} = 0$ を設定するため、境界積分項（熱流束項）は自然に消滅します。両辺に円盤の厚さ $h$ を乗じて整理すると、以下の弱形式が得られます：

$$
\int_{\Omega} \rho c \frac{\partial T}{\partial t} w \, h \, dA + \int_{\Omega} \lambda \nabla T \cdot \nabla w \, h \, dA = 0
$$

---

### 3.2 空間離散化（一次三角形要素 / CST）
円盤領域 $\Omega$ を $N_e$ 個の三角形要素に分割します（上図(b)参照）。  
要素 $e$ 内の温度場 $T^e(x, y, t)$ を3頂点の節点温度 $T_1, T_2, T_3$ と形状関数 $N_i(x, y)$ で線形補間します：

$$
T^e(x, y, t) = \sum_{i=1}^{3} N_i(x, y) T_i(t) = \mathbf{N} \mathbf{T}^e
$$

形状関数の勾配 $\nabla N_i = \left[ \frac{\partial N_i}{\partial x}, \frac{\partial N_i}{\partial y} \right]^T$ は要素内で定数となります：

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

ここで $A_e = \frac{1}{2} |(x_2 - x_1)(y_3 - y_1) - (x_3 - x_1)(y_2 - y_1)|$ は要素の面積です。

#### 要素熱伝導行列（剛性行列 Kᵉ）

$$
K^e_{ij} = \int_{e} \lambda (\nabla N_i \cdot \nabla N_j) \, h \, dA = \lambda h A_e (\nabla N_i \cdot \nabla N_j)
$$

#### 要素熱容量行列（質量行列 Cᵉ）
一次三角形要素の面積座標公式より厳密に積分できます：

$$
C^e_{ij} = \int_{e} \rho c N_i N_j \, h \, dA = \frac{\rho c h A_e}{12}
\begin{bmatrix}
2 & 1 & 1 \\
1 & 2 & 1 \\
1 & 1 & 2
\end{bmatrix}
$$

---

### 3.3 全体方程式系と時間積分スキーム
全要素の要素行列を全体節点番号に対応付けて足し合わせることで、以下の連立常微分方程式系が得られます：

$$
\mathbf{C} \frac{d\mathbf{T}}{dt} + \mathbf{K} \mathbf{T} = \mathbf{0}
$$

時間方向の離散化に一般化台形則（$\theta$ 法）を適用します：

$$
\frac{\mathbf{T}^{n+1} - \mathbf{T}^n}{\Delta t} = - \mathbf{C}^{-1} \mathbf{K} \left[ \theta \mathbf{T}^{n+1} + (1 - \theta) \mathbf{T}^n \right]
$$

これを整理すると、ステップ $n \to n+1$ を解く線形方程式系が得られます：

$$
\left( \frac{\mathbf{C}}{\Delta t} + \theta \mathbf{K} \right) \mathbf{T}^{n+1} = \left( \frac{\mathbf{C}}{\Delta t} - (1 - \theta) \mathbf{K} \right) \mathbf{T}^n
$$

- $\theta = 0.5$（**Crank-Nicolson法**）を採用：時間2次精度を有し、無条件安定な優れたスキームです。

---

### 3.4 ディリクレ境界条件の適用
節点を「内部自由度節点 $\Omega_I$」と「外周境界節点 $\Gamma_D$」に分割します。

境界節点では常に $T_j^{n+1} = 0$ であるため、内部節点のみを未知数とする縮約連立方程式：

$$
\mathbf{A}_{II} \mathbf{T}_I^{n+1} = \mathbf{b}_I
$$

を解きます。左辺行列 $\mathbf{A}_{II} = \left( \frac{\mathbf{C}}{\Delta t} + \theta \mathbf{K} \right)_{II}$ は時間ステップを通じて不変であるため、あらかじめLU分解（`scipy.sparse.linalg.factorized`）を行っておくことで、各時間ステップを高速な前進後退代入のみで解くことができます。

---

## 4. Python実装コードの構成

実装スクリプト：[`fem_heat_solver.py`](file:///c:/Users/sudak/src/fem_lesson/fem_heat_solver.py)

### 主なコードブロック
1. **物性値・計算条件の定義**:
   - 半径 $a=1.0\ \text{m}$, 厚さ $h=0.1\ \text{m}$, 物性値 $\lambda, c, \rho$
   - 時間刻み $\Delta t = 20.0\ \text{s}$, 解析終了時刻 $t_{\max} = 12000.0\ \text{s}$（約3.3時間）
2. **厳密解関数の実装**:
   - `scipy.special.jn_zeros(0, 100)` で零点を取得し、ベッセル級数解を高速ベクトル計算
3. **円盤メッシュ自動生成**:
   - 半径方向・円周方向に均等比率で節点を配置し、`scipy.spatial.Delaunay` で高品質な三角形要素を自動生成
4. **全体行列の組み立て**:
   - COO形式で要素行列を展開・格納し、CSR形式の疎行列として構築
5. **時間ステップループ**:
   - 事前LU分解を用いたCrank-Nicolson時間進行法で温度分布を更新
6. **結果のプロット・可視化**:
   - 中心温度冷却曲線、各時刻の半径方向分布、2次元コンター図を並列描画

---

## 5. 計算結果と厳密解の比較検証

スクリプトを実行して得られた解析結果プロットです：

![FEM解析結果とベッセル級数厳密解の比較](fem_heat_conduction_comparison.png)

### 考察と検証ポイント
1. **中心温度の冷却曲線（左図）**:
   - 解析初期の約20分間は外周の冷気が中心まで到達せず、中心温度は $500^\circ\text{C}$ を保持します。
   - その後、滑らかに指数関数的に低下し、3時間後には約 $100^\circ\text{C}$ まで冷却されます。
   - **FEM計算値（青破線）と厳密解（赤実線）は完全に一致**しており、高い時間精度が確認できます。
2. **半径方向温度分布（中央図）**:
   - 10分、30分、60分、120分、180分の各スナップショットにおいて、厳密解の理論曲線（実線）と全節点のFEM温度値（散布点）をプロットしています。
   - 円盤全体にわたって理論解のベッセル関数形状と極めて高精度に一致しています。
3. **2次元温度コンター図（右図）**:
   - $t = 30\ \text{分}$ における温度コンターを表示。直交座標系の三角形メッシュでありながら、歪みのない綺麗な同心円状の等温線が得られています。

---

## 6. スクリプトの実行方法

ワークスペース内のターミナルで以下を実行します：

```bash
# 問題設定模式図の生成
python generate_schematic.py

# 詳細ワークフロー図の生成
python generate_workflow_diagram.py

# FEM解析の実行・厳密解比較プロットの生成
python fem_heat_solver.py
```

実行後、同ディレクトリに `problem_schematic.png`、`fem_workflow_diagram.png`、および `fem_heat_conduction_comparison.png` が生成・更新されます。
