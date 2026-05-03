"""u16 LeakyRelu オーバーフロー解説用の図を生成"""
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from pathlib import Path

matplotlib.rcParams['font.family'] = ['Noto Sans CJK JP', 'IPAGothic', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
OUT = Path(__file__).parent

# ================================================================
# Fig 1: LeakyRelu transfer function (実数空間)
# ================================================================
fig, ax = plt.subplots(figsize=(7, 5))
x = np.linspace(-1.2, 1.2, 500)
y = np.where(x >= 0, x, 0.1 * x)
ax.plot(x, y, "C0-", linewidth=2.5, label="LeakyRelu(α=0.1)")
ax.plot(x, x, "k--", alpha=0.3, linewidth=1, label="y=x (参考)")
ax.axhline(0, color="gray", linewidth=0.5)
ax.axvline(0, color="gray", linewidth=0.5)
ax.fill_between(x[x >= 0], 0, y[x >= 0], alpha=0.15, color="C2", label="正の枝: 傾き 1")
ax.fill_between(x[x < 0], 0, y[x < 0], alpha=0.15, color="C3", label="負の枝: 傾き 0.1")
ax.set_xlabel("x (入力, 実数)")
ax.set_ylabel("y (出力, 実数)")
ax.set_title("LeakyRelu(α=0.1) の伝達関数\n2 つの枝で傾きが 10 倍違う")
ax.legend(loc="upper left")
ax.grid(alpha=0.3)
ax.set_xlim(-1.2, 1.2)
ax.set_ylim(-0.15, 1.2)
plt.tight_layout()
plt.savefig(OUT / "01_leakyrelu_transfer.png", dpi=110)
plt.close()

# ================================================================
# Fig 2: 量子化グリッド (u8 vs u16) — 同じ実数レンジを表現
# ================================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# 同じ実数レンジ [-1.155, 0.489] を u8 と u16 で量子化
# (実際の resblocks.6/LeakyRelu_1 入力に近い値)
x_min, x_max = -1.155, 0.489
data = np.random.RandomState(42).randn(2000) * 0.3 - 0.15
data = np.clip(data, x_min, x_max)

# u8: 256 levels
ax = axes[0]
u8_levels = np.linspace(x_min, x_max, 256)
ax.hist(data, bins=80, color="C0", alpha=0.6, density=True)
for lvl in u8_levels[::8]:  # 1/8 だけ描画 (見やすさのため)
    ax.axvline(lvl, color="gray", linewidth=0.3, alpha=0.4)
ax.axvline(0, color="red", linewidth=2, label="x=0 境界 (LeakyRelu の折れ点)")
# x_zp 位置を矢印で
x_zp_u8_real = 0  # x_zp は実数で 0 に対応
ax.annotate(f"x_zp ≈ 180\n(整数 [0,255] の中の位置)", xy=(0, 1.5), xytext=(0.15, 2.5),
            arrowprops=dict(arrowstyle="->", color="red"), fontsize=9, color="red")
ax.set_title("u8 量子化: 256 段階\n  グリッド間隔 (scale) ≈ 6.45e-3", fontsize=11)
ax.set_xlabel("実数値 x")
ax.set_ylabel("分布密度")
ax.set_xlim(x_min, x_max)
ax.legend(loc="upper left", fontsize=9)
ax.grid(alpha=0.3)

# u16: 65536 levels
ax = axes[1]
ax.hist(data, bins=80, color="C0", alpha=0.6, density=True)
# u16 の grid は密すぎるので、目盛りだけ示す
u16_levels = np.linspace(x_min, x_max, 65536)
for i in range(0, 65536, 2048):
    ax.axvline(u16_levels[i], color="gray", linewidth=0.3, alpha=0.3)
ax.axvline(0, color="red", linewidth=2, label="x=0 境界 (LeakyRelu の折れ点)")
ax.annotate(f"x_zp ≈ 46039\n(整数 [0,65535] の中の位置)", xy=(0, 1.5), xytext=(0.15, 2.5),
            arrowprops=dict(arrowstyle="->", color="red"), fontsize=9, color="red")
ax.set_title("u16 量子化: 65536 段階\nグリッド間隔 (scale) ≈ 2.51e-5", fontsize=11)
ax.set_xlabel("実数値 x")
ax.set_ylabel("分布密度")
ax.set_xlim(x_min, x_max)
ax.legend(loc="upper left", fontsize=9)
ax.grid(alpha=0.3)

plt.suptitle("同じ実数レンジ [−1.155, 0.489] を u8 / u16 で量子化\n(scale 比は両者とも約 2.72 で同じ — 比率は LeakyRelu が決める)",
             fontsize=11)
plt.tight_layout()
plt.savefig(OUT / "02_quantization_grid.png", dpi=110)
plt.close()

# ================================================================
# Fig 3: delta = (x_q − x_zp) の振れ幅比較
# ================================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# u8
ax = axes[0]
x_zp_u8 = 180
x_q_u8 = np.arange(0, 256)
delta_u8 = x_q_u8 - x_zp_u8
ax.plot(x_q_u8, delta_u8, "C0-", linewidth=2)
ax.axhline(0, color="gray", linewidth=0.5)
ax.axhline(255 - x_zp_u8, color="C2", linestyle="--", alpha=0.6, label=f"delta_max = {255-x_zp_u8}")
ax.axhline(-x_zp_u8, color="C3", linestyle="--", alpha=0.6, label=f"delta_min = {-x_zp_u8}")
ax.fill_between(x_q_u8, 0, delta_u8, alpha=0.2)
ax.set_xlabel("x_q (量子化整数)")
ax.set_ylabel("delta = x_q − x_zp")
ax.set_title(f"u8: x_q ∈ [0, 255], x_zp = {x_zp_u8}\n|delta|_max = {max(x_zp_u8, 255-x_zp_u8)} (≈ 180)", fontsize=11)
ax.legend()
ax.grid(alpha=0.3)
ax.set_ylim(-50000, 50000)

# u16
ax = axes[1]
x_zp_u16 = 46039
x_q_u16 = np.arange(0, 65536, 64)
delta_u16 = x_q_u16 - x_zp_u16
ax.plot(x_q_u16, delta_u16, "C0-", linewidth=2)
ax.axhline(0, color="gray", linewidth=0.5)
ax.axhline(65535 - x_zp_u16, color="C2", linestyle="--", alpha=0.6, label=f"delta_max = {65535-x_zp_u16}")
ax.axhline(-x_zp_u16, color="C3", linestyle="--", alpha=0.6, label=f"delta_min = {-x_zp_u16}")
ax.fill_between(x_q_u16, 0, delta_u16, alpha=0.2)
ax.set_xlabel("x_q (量子化整数)")
ax.set_ylabel("delta = x_q − x_zp")
ax.set_title(f"u16: x_q ∈ [0, 65535], x_zp = {x_zp_u16}\n|delta|_max = {max(x_zp_u16, 65535-x_zp_u16)} (≈ 46039)", fontsize=11)
ax.legend()
ax.grid(alpha=0.3)
ax.set_ylim(-50000, 50000)

plt.suptitle("delta の振れ幅: u8 と u16 で 256 倍違う", fontsize=12)
plt.tight_layout()
plt.savefig(OUT / "03_delta_range.png", dpi=110)
plt.close()

# ================================================================
# Fig 4: 中間計算値 delta × M_pos と 16 bit 上限
# ================================================================
fig, ax = plt.subplots(figsize=(11, 6))

# scale_ratio = x_scale / r_scale = 2.72
# M_pos = 2.72 (実数)。delta × M_pos が 16 bit (max 65535) に収まるか?
# 実機の固定小数点だと M_pos = round(2.72 * 2^S) を 16bit で持つので、shift 含めると上限は約 65535/(M_int / 2^S)
# 簡単化: |delta| × 2.72 が 65535 を超えるかで判定

deltas = np.arange(0, 50000, 100)
intermediate = deltas * 2.72

ax.plot(deltas, intermediate, "C0-", linewidth=2.5, label="|delta| × M_pos (中間計算値)")
ax.axhline(65535, color="red", linewidth=2, linestyle="--", label="16 bit レジスタ上限 (65535)")
ax.axhline(131071, color="darkred", linewidth=1, linestyle=":", label="17 bit 上限 (131071)")

# u8 の点
delta_u8 = 180
ax.scatter([delta_u8], [delta_u8 * 2.72], color="C2", s=200, zorder=5,
           label=f"u8: delta={delta_u8}, 中間値≈{int(delta_u8*2.72)} ✓ 余裕で OK", marker="o")

# u16 の点
delta_u16 = 46039
ax.scatter([delta_u16], [delta_u16 * 2.72], color="C3", s=200, zorder=5,
           label=f"u16: delta={delta_u16}, 中間値≈{int(delta_u16*2.72)} ✗ オーバーフロー", marker="X")

# 領域シェード
ax.axhspan(0, 65535, alpha=0.1, color="green")
ax.axhspan(65535, 200000, alpha=0.1, color="red")
ax.text(45000, 30000, "OK 領域", color="darkgreen", fontsize=11, fontweight="bold")
ax.text(45000, 100000, "オーバーフロー領域", color="darkred", fontsize=11, fontweight="bold")

ax.set_xlabel("|delta| = |x_q − x_zp|")
ax.set_ylabel("中間計算値 |delta| × M_pos")
ax.set_title("u16 LeakyRelu のオーバーフロー: 中間計算値が 16 bit を超える\n(M_pos = x_scale/r_scale ≈ 2.72 は両 bit 幅で同じ)",
             fontsize=12)
ax.legend(loc="upper left", fontsize=9)
ax.grid(alpha=0.3)
ax.set_xlim(0, 50000)
ax.set_ylim(0, 150000)

plt.tight_layout()
plt.savefig(OUT / "04_intermediate_overflow.png", dpi=110)
plt.close()

# ================================================================
# Fig 5: ハードウェア実装の違い (LUT vs ランタイム算術)
# ================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

# u8: LUT 実装
ax = axes[0]
ax.set_title("u8 LeakyRelu: 事前計算 LUT (256 エントリ)", fontsize=12, fontweight="bold")
ax.text(0.5, 0.95, "計算は事前 (オフライン)", ha="center", fontsize=10, transform=ax.transAxes,
        bbox=dict(boxstyle="round", facecolor="lightyellow"))

# 左側: 事前計算
ax.text(0.15, 0.78, "オフラインで全 256 通り計算:", fontsize=10, transform=ax.transAxes, fontweight="bold")
for i, x_q in enumerate([0, 1, 2, "...", 254, 255]):
    if x_q == "...":
        ax.text(0.15, 0.70 - i*0.05, "  ...", fontsize=9, transform=ax.transAxes, family="monospace")
        continue
    real = (x_q - 180) * 6.45e-3
    if real >= 0:
        out_real = real
    else:
        out_real = 0.1 * real
    out_q = max(0, min(255, int(round(out_real / 2.37e-3) + 80)))
    ax.text(0.15, 0.70 - i*0.05, f"  LUT[{x_q:>3}] = {out_q}", fontsize=9, transform=ax.transAxes, family="monospace")

# 右側: 実行時
ax.text(0.55, 0.78, "実行時:", fontsize=10, transform=ax.transAxes, fontweight="bold")
ax.text(0.55, 0.70, "y_q = LUT[x_q]", fontsize=11, transform=ax.transAxes, family="monospace",
        bbox=dict(boxstyle="round", facecolor="lightgreen"))
ax.text(0.55, 0.55, "メモリ参照のみ\n→ 算術なし\n→ オーバーフロー\n  発生しようがない",
        fontsize=10, transform=ax.transAxes, color="darkgreen")
ax.text(0.55, 0.30, "LUT サイズ:\n256 entry × 1 byte\n= 256 byte (誤差ゼロ)",
        fontsize=9, transform=ax.transAxes,
        bbox=dict(boxstyle="round", facecolor="lightyellow"))
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.axis("off")

# u16: ランタイム算術
ax = axes[1]
ax.set_title("u16 LeakyRelu: ランタイム算術 (LUT は非現実的)", fontsize=12, fontweight="bold")
ax.text(0.5, 0.95, "実行時に毎サンプル計算", ha="center", fontsize=10, transform=ax.transAxes,
        bbox=dict(boxstyle="round", facecolor="mistyrose"))

ax.text(0.05, 0.83, "実行時 (各サンプル毎):", fontsize=10, transform=ax.transAxes, fontweight="bold")
ax.text(0.05, 0.75, "1) delta = x_q − x_zp", fontsize=10, transform=ax.transAxes, family="monospace")
ax.text(0.05, 0.68, "2) tmp = delta × M_pos      ← ★★★ ここ", fontsize=10, transform=ax.transAxes,
        family="monospace", color="darkred", fontweight="bold")
ax.text(0.05, 0.61, "3) tmp = tmp >> S", fontsize=10, transform=ax.transAxes, family="monospace")
ax.text(0.05, 0.54, "4) y_q = tmp + r_zp", fontsize=10, transform=ax.transAxes, family="monospace")
ax.text(0.05, 0.47, "5) negative branch なら × 0.1", fontsize=10, transform=ax.transAxes, family="monospace")

ax.text(0.05, 0.32, "問題: ステップ 2 で\ndelta = 46039, M_pos = 2.72 (固定小数点)\n→ tmp ≈ 125,200 が 16bit 乗算器に入らない",
        fontsize=10, transform=ax.transAxes, color="darkred",
        bbox=dict(boxstyle="round", facecolor="lightcoral", alpha=0.4))

ax.text(0.05, 0.10, "もし LUT 化するなら:\n65536 entry × 2 byte = 128 KB\n(各 op 毎に 128 KB は非現実的)",
        fontsize=9, transform=ax.transAxes,
        bbox=dict(boxstyle="round", facecolor="lightyellow"))
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.axis("off")

plt.tight_layout()
plt.savefig(OUT / "05_hw_implementation.png", dpi=110)
plt.close()

print("生成完了:")
for p in sorted(OUT.glob("*.png")):
    print(f"  {p.name}: {p.stat().st_size//1024} KB")
