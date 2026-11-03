import matplotlib.pyplot as plt
import matplotlib.patches as patches
import japanize_matplotlib

# キャンバス高さを 17.5 に拡張して各カード内に十分なゆとりを確保
fig, ax = plt.subplots(figsize=(16, 17.5), dpi=200)
ax.set_xlim(0, 16)
ax.set_ylim(0, 17.5)
ax.axis('off')

# 全体タイトル
ax.text(8, 16.9, "円盤非定常熱伝導解析の全体ワークフロー（FEM定式化・数値計算・厳密解比較）", 
        ha='center', va='center', fontsize=16, fontweight='bold', color='#1a237e', zorder=10)

# 各フェーズの配色スタイル
colors = {
    'phase1': {'bg': '#e8eaf6', 'border': '#3f51b5', 'header': '#283593'},
    'phase2': {'bg': '#e0f2f1', 'border': '#009688', 'header': '#00695c'},
    'phase3': {'bg': '#fff3e0', 'border': '#ff9800', 'header': '#e65100'},
    'phase4': {'bg': '#f3e5f5', 'border': '#ab47bc', 'header': '#6a1b9a'}
}

def draw_phase_box(x, y, w, h, title, key):
    # フェーズ背景ボックス
    rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.12", 
                                  facecolor=colors[key]['bg'], edgecolor=colors[key]['border'], linewidth=2.2, zorder=1)
    ax.add_patch(rect)
    # フェーズヘッダー帯 (最上部に配置)
    header_rect = patches.FancyBboxPatch((x, y + h - 0.7), w, 0.7, boxstyle="round,pad=0.08",
                                         facecolor=colors[key]['header'], edgecolor=colors[key]['border'], linewidth=1.2, zorder=2)
    ax.add_patch(header_rect)
    # ヘッダータイトルテキスト
    ax.text(x + w/2, y + h - 0.35, title, ha='center', va='center', 
            fontsize=12.5, fontweight='bold', color='white', zorder=3)

def draw_step_card(x, y, w, h, title, details, eq=None):
    # カード背景パッチ
    rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1", 
                                  facecolor='white', edgecolor='#90a4ae', linewidth=1.2, zorder=4)
    ax.add_patch(rect)
    
    # カードタイトル (上端から 0.28 下げた位置)
    ax.text(x + 0.25, y + h - 0.28, title, ha='left', va='center', fontsize=10.5, fontweight='bold', color='#263238', zorder=5)
    
    if eq:
        # TeX数式ボックス (上端から 0.80 下げた位置を中心)
        ax.text(x + w/2, y + h - 0.80, eq, ha='center', va='center', fontsize=10.2, color='#b71c1c',
                bbox=dict(boxstyle='square,pad=0.2', facecolor='#fffde7', edgecolor='#ffd54f', linewidth=0.8), zorder=5)
        # 説明テキスト: 数式ボックスの下（上端から 1.25 下げた位置から top 揃えで開始）
        ax.text(x + 0.25, y + h - 1.25, details, ha='left', va='top', fontsize=8.6, color='#455a64', linespacing=1.38, zorder=5)
    else:
        # 数式なしカード: タイトルの下から top 揃えで開始
        ax.text(x + 0.25, y + h - 0.52, details, ha='left', va='top', fontsize=8.6, color='#455a64', linespacing=1.38, zorder=5)

def draw_arrow(x1, y1, x2, y2, label=None):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color='#37474f', lw=2.0, mutation_scale=15), zorder=6)
    if label:
        ax.text((x1 + x2)/2, (y1 + y2)/2 + 0.15, label, ha='center', va='bottom', fontsize=8.5, color='#37474f', zorder=7)

# ==================== 左列: Phase 1 & Phase 2 ====================

# ----------------- Phase 1: 数理モデルと弱形式導出 -----------------
# 枠: y=9.4 to 16.3 (h=6.9), ヘッダー: 15.6 to 16.3
draw_phase_box(0.6, 9.4, 7.0, 6.9, "Phase 1: 数理モデルと弱形式の導出 (理論)", "phase1")

# カード1: y=12.8, h=2.3 (上端 15.1, ヘッダー下端 15.6 と 0.5 余白, 下端マージン 0.4)
draw_step_card(0.9, 12.8, 6.4, 2.3, "1. 支配方程式と初期・境界条件", 
               r"・初期条件: $T(x, y, 0) = 500\ ^\circ\mathrm{C}$ (全域一様)" "\n"
               r"・外周境界 ($r=1.0\,\mathrm{m}$): $T = 0\ ^\circ\mathrm{C}$ (急冷保持)" "\n"
               r"・上下面 (厚さ $h=0.1\,\mathrm{m}$): 断熱条件 $\partial T / \partial z = 0$", 
               eq=r"$\rho c \frac{\partial T}{\partial t} = \lambda \left(\frac{\partial^2 T}{\partial x^2} + \frac{\partial^2 T}{\partial y^2}\right)$")

# カード2: y=9.9, h=2.4 (上端 12.3, 下端マージン 0.5)
draw_step_card(0.9, 9.9, 6.4, 2.4, "2. 重み付き残差法・グリーンの定理（弱形式化）", 
               r"・仮想温度（重み関数）$w(x, y)$ を乗じて全領域で面積積分" "\n"
               r"・2次元部分積分により外周の熱流束境界項が自然消滅 ($w=0$)" "\n"
               r"・方程式の要求滑らかさを2階微分から1階微分へ低減", 
               eq=r"$\int_{\Omega} \rho c \frac{\partial T}{\partial t} w \, dA + \int_{\Omega} \lambda \nabla T \cdot \nabla w \, dA = 0$")

draw_arrow(4.1, 12.8, 4.1, 12.3)

# ----------------- Phase 2: 空間離散化と要素行列 -----------------
# 枠: y=0.5 to 8.8 (h=8.3), ヘッダー: 8.1 to 8.8
draw_phase_box(0.6, 0.5, 7.0, 8.3, "Phase 2: 空間離散化と全体行列の構築 (FEM)", "phase2")

# カード3: y=6.2, h=1.5 (上端 7.7, ヘッダー下端 8.1 と 0.4 余白, 下端マージン 0.35)
draw_step_card(0.9, 6.2, 6.4, 1.5, "3. 円盤のメッシュ分割 (一次三角形要素 CST)", 
               r"・同心円状に節点を配置し、Delaunay三角分割を実施" "\n"
               r"・要素数 5,400 / 節点数 2,791（要素歪みのない良質な網目）" "\n"
               r"・要素内の温度を3節点の形状関数 $N_i(x,y)$ で線形補間")

# カード4: y=3.3, h=2.4 (上端 5.7, 下端マージン 0.5)
draw_step_card(0.9, 3.3, 6.4, 2.4, "4. 要素行列の解析的積分", 
               r"・要素熱伝導行列: $\mathbf{K}^e = \lambda h A_e (\nabla \mathbf{N} \nabla \mathbf{N}^T)$" "\n"
               r"・要素熱容量行列: $\mathbf{C}^e = \frac{\rho c h A_e}{12} \mathbf{M}_0$  ($\mathbf{M}_0$: 節点結合係数)" "\n"
               r"・厚さ $h=0.1\,\mathrm{m}$ と要素面積 $A_e$ を乗じて体積積分を完了",
               eq=r"$T^e(x, y, t) = \sum_{i=1}^3 N_i(x, y) T_i(t)$")

# カード5: y=0.8, h=2.0 (上端 2.8, 下端マージン 0.3)
draw_step_card(0.9, 0.8, 6.4, 2.0, "5. 全体行列のアセンブリ (疎行列 CSR)", 
               r"・全要素の局所行列を全体自由度へ足し合わせ" "\n"
               r"・疎行列（CSR形式）として $\mathbf{C}$ と $\mathbf{K}$ を高速構築", 
               eq=r"$\mathbf{C} \frac{d\mathbf{T}}{dt} + \mathbf{K} \mathbf{T} = \mathbf{0}$")

draw_arrow(4.1, 9.4, 4.1, 8.8, "弱形式への代入")
draw_arrow(4.1, 6.2, 4.1, 5.7)
draw_arrow(4.1, 3.3, 4.1, 2.8)


# ==================== 右列: Phase 3 & Phase 4 ====================

# ----------------- Phase 3: 時間離散化と解法 -----------------
# 枠: y=6.1 to 16.3 (h=10.2), ヘッダー: 15.6 to 16.3
draw_phase_box(8.4, 6.1, 7.0, 10.2, "Phase 3: 時間離散化と高速ソルバー (数値解法)", "phase3")

# カード6: y=12.8, h=2.3 (上端 15.1, ヘッダー下端 15.6 と 0.5 余白, 下端マージン 0.4)
draw_step_card(8.7, 12.8, 6.4, 2.3, "6. 時間積分スキーム (Crank-Nicolson法)", 
               r"・一般化台形則 ($\theta = 0.5$) を採用" "\n"
               r"・時間方向2次精度の無条件安定スキーム" "\n"
               r"・大きな時間刻み $\Delta t = 20\,\mathrm{s}$ でも発散せず高精度を維持",
               eq=r"$\left(\frac{\mathbf{C}}{\Delta t} + 0.5 \mathbf{K}\right) \mathbf{T}^{n+1} = \left(\frac{\mathbf{C}}{\Delta t} - 0.5 \mathbf{K}\right) \mathbf{T}^n$")

# カード7: y=10.0, h=2.3 (上端 12.3, 下端マージン 0.4)
draw_step_card(8.7, 10.0, 6.4, 2.3, "7. 外周ディリクレ境界条件の適用 (自由度縮約)", 
               r"・節点を「内部未知節点 $\Omega_I$」と「拘束外周節点 $\Gamma_D$」に分割" "\n"
               r"・外周温度 $T_D = 0\ ^\circ\mathrm{C}$ を固定拘束" "\n"
               r"・解くべき連立方程式を内部節点ブロックのみに縮約",
               eq=r"$\mathbf{A}_{II} \mathbf{T}_I^{n+1} = \mathbf{b}_I$")

# カード8: y=8.2, h=1.3 (上端 9.5, 下端マージン 0.35)
draw_step_card(8.7, 8.2, 6.4, 1.3, r"8. 左辺行列 $\mathbf{A}_{II}$ の事前LU分解 (factorized)", 
               r"・係数行列 $\mathbf{A}_{II}$ は時間不変なためループ前に1度だけLU分解" "\n"
               r"・各時間ステップでの逆行列再計算を無くし計算速度を飛躍的に向上")

# カード9: y=6.4, h=1.3 (上端 7.7, 下端マージン 0.35)
draw_step_card(8.7, 6.4, 6.4, 1.3, r"9. 時間ステップループ ($t$: 0 $\to$ 12,000 s)", 
               r"・右辺 $\mathbf{b}_I$ を更新 $\to$ 前進後退代入で解 $\mathbf{T}_I^{n+1}$ を高速求解" "\n"
               r"・各指定時刻（10分, 30分, 1時間, 2時間, 3時間）の解を保存")

draw_arrow(4.1, 0.8, 8.4, 13.95, "時間発展方程式へ")
draw_arrow(11.9, 12.8, 11.9, 12.3)
draw_arrow(11.9, 10.0, 11.9, 9.5)
draw_arrow(11.9, 8.2, 11.9, 7.7)

# ----------------- Phase 4: 理論解との比較・検証 -----------------
# 枠: y=0.5 to 5.5 (h=5.0), ヘッダー: 4.8 to 5.5
draw_phase_box(8.4, 0.5, 7.0, 5.0, "Phase 4: 厳密解との比較検証と可視化", "phase4")

# カード10: y=2.4, h=2.0 (上端 4.4, ヘッダー下端 4.8 と 0.4 余白, 下端マージン 0.3)
draw_step_card(8.7, 2.4, 6.4, 2.0, "10. ベッセル級数厳密解の算出", 
               r"・scipy.special.jn_zeros でベッセル零点 $\beta_n$ を算出 (100項)" "\n"
               r"・級数理論解をベクトル計算で高速評価",
               eq=r"$T(r, t) = T_0 \sum_{n=1}^\infty A_n J_0\left(\beta_n \frac{r}{a}\right) e^{-p_n t}$")

# カード11: y=0.8, h=1.2 (上端 2.0, 下端マージン 0.25)
draw_step_card(8.7, 0.8, 6.4, 1.2, "11. 精度検証・多角的プロット出力", 
               r"① 中心温度冷却曲線の比較 (FEM vs 厳密解)" "\n"
               r"② 半径方向プロファイルの重ね合わせ比較 / ③ 2次元等温線コンター")

draw_arrow(11.9, 6.1, 11.9, 5.5, "FEM計算結果")
draw_arrow(11.9, 2.4, 11.9, 2.0)

plt.savefig('c:/Users/sudak/src/fem_lesson/fem_workflow_diagram.png', dpi=200, bbox_inches='tight')
print("fem_workflow_diagram.png successfully regenerated without text overflow.")
