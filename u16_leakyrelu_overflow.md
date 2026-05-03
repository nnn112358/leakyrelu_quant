# u16 量子化 LeakyRelu のオーバーフロー解析

**問題**: piper-plus tsukuyomi-chan の HiFiGAN decoder を u16 量子化で AX650 NPU 用に pulsar2 でビルドすると、`compiler.check=2` 時に以下のエラーで失敗:

```
AxQuantizedLeakyRelu, integer 131071 does not fit 'uint16_t'
op: /dec/resblocks.6/LeakyRelu_1
x_scale=2.5076835e-05, x_zeropoint=46039
r_scale=9.221705e-06,  r_zeropoint=12520
negative_slope=0.10000000149011612
```

ビルドは check=0 (検証スキップ) で通せば成功するが、**実機 NPU で execute 時に `AXCL_ERR_ENGINE_EXECUTE_FAIL` で失敗**する。

このレポートでは「なぜ u16 は overflow して u8 は OK なのか」を 5 枚の図で順を追って説明する。

---

## TL;DR

| 観点 | u8 | u16 |
|---|---|---|
| 量子化レベル数 | 256 | 65,536 |
| `\|delta\| = \|x_q − x_zp\|` の最大値 | 約 180 | 約 46,039 |
| 中間計算値 `delta × M_pos` (M_pos ≈ 2.72) | 約 490 | **約 125,200** |
| 16 bit 乗算器 (max 65,535) に収まるか | ✅ 余裕 | ❌ オーバーフロー |
| ハードウェア実装 | LUT (256 byte の事前計算表) | ランタイム算術 (毎サンプル乗算) |

---

## 1. 出発点: LeakyRelu(α=0.1) の伝達関数

LeakyRelu は入力の符号で**傾きが変わる**非線形関数:

```
y = x         (x ≥ 0 のとき, 傾き 1)
y = 0.1 × x   (x < 0 のとき, 傾き 0.1)
```

![LeakyRelu の伝達関数](figs/01_leakyrelu_transfer.png)

正の枝と負の枝で**傾きが 10 倍違う**。後でこの「10 倍」が量子化計算の bit 幅に響いてくる。

---

## 2. 量子化グリッド: u8 と u16 で精度は違うが、scale 比は同じ

実数値 `x ∈ [−1.155, 0.489]` を量子化する:
- **u8** : 256 段階 → `x_scale` (1 段階あたりの実数幅) ≈ **6.45 × 10⁻³**
- **u16**: 65,536 段階 → `x_scale` ≈ **2.51 × 10⁻⁵** (256 倍細かい)

![量子化グリッド比較](figs/02_quantization_grid.png)

LeakyRelu の出力レンジは入力の **約 1/2.72** 倍 (LeakyRelu の正の枝を通る成分が支配的) になり、

```
x_scale / r_scale ≈ 2.72   ← この比は u8/u16 どちらでも同じ
```

この比 2.72 は **LeakyRelu の伝達特性そのものから決まる量** で、calibration 法 (MinMax / Percentile / MSE) を変えても変わらない (実際に 3 方式で検証して全部 ≈ 2.72 だった)。

---

## 3. ハードウェアで何を計算しているか

量子化 LeakyRelu の数式 (正の枝):

```
y_real = x_real
       = (x_q − x_zp) × x_scale          ← 実数復号
y_q    = round(y_real / r_scale) + r_zp   ← 再量子化
       = round((x_q − x_zp) × M_pos) + r_zp
ただし M_pos = x_scale / r_scale (≈ 2.72)
```

ハードウェアは固定小数点で `M_pos = round(2.72 × 2^S)` を整数化して 16 bit 乗算器に持つ。
よって**毎サンプル 1 回の整数乗算が走る**:

```
delta = x_q − x_zp           ← 引き算
tmp   = delta × M_pos         ★★★ ここで bit 幅が決まる
tmp   = tmp >> S             ← shift で実数 scale を復元
y_q   = tmp + r_zp           ← zero point 加算
(負の枝なら × 0.1 が追加で入る)
```

---

## 4. 核心: `delta` の振れ幅が u8 と u16 で 256 倍違う

```
delta = x_q − x_zp
```

このグラフが今回の主役:

![delta の振れ幅](figs/03_delta_range.png)

**左 (u8)**:
- `x_q ∈ [0, 255]`, `x_zp ≈ 180` → `delta ∈ [−180, +75]` → `|delta|_max = 180`

**右 (u16)**:
- `x_q ∈ [0, 65535]`, `x_zp ≈ 46039` → `delta ∈ [−46039, +19496]` → `|delta|_max = 46039`

縦軸スケールを揃えてあるので一目瞭然: **u8 の delta はほぼ 0 にしか見えない** ほど u16 と桁が違う。

これは当然で、`x_q` の取りうる整数値そのものが 256 倍違うため。`x_zp` も非対称量子化で偏った位置に来るので、`|delta|_max` は実質的に `x_q` レンジの大半を占める値になる。

---

## 5. 中間値 `delta × M_pos` が 16 bit に収まるか

ステップ ★★★ の中間値 `delta × M_pos` を u8/u16 それぞれでプロット:

![中間計算値とオーバーフロー](figs/04_intermediate_overflow.png)

- **u8 の点 (緑 ○)**: `delta = 180`, 中間値 ≈ **490** — 16 bit 上限 (65,535) のたった 0.7%
- **u16 の点 (赤 ✗)**: `delta = 46,039`, 中間値 ≈ **125,226** — 16 bit 上限を **約 1.9 倍超過** (これが pulsar2 の言う `131071 does not fit uint16_t` の出所、131071 = 2¹⁷−1)

**つまり u16 LeakyRelu のオーバーフローは:**
- 比 `M_pos ≈ 2.72` 自体は問題なし (16 bit 乗算器の固定小数点としては余裕)
- でも `delta` が**ほぼ uint16 全レンジ** (46,039) を取るので、それを `M_pos` 倍すると 17 bit 必要になる
- 16 bit レジスタには絶対に収まらない → 構造的に避けられない問題

---

## 6. なぜ u8 はそもそも問題にならないのか — ハードウェア実装の違い

ここが根本の理解:

![HW 実装の違い](figs/05_hw_implementation.png)

### u8 LeakyRelu — **LUT (Look-Up Table) 実装**

- 入力 `x_q` の取りうる値はたったの **256 通り**
- すべてオフラインで計算できる: `LUT[0]`, `LUT[1]`, ..., `LUT[255]`
- 各エントリは **「正しく u8 [0,255] にクランプされた出力」**を事前に格納
- 実行時は **メモリ参照だけ** (`y_q = LUT[x_q]`) — 算術演算は走らない
- LUT サイズ: 256 byte (op ごとに持っても全然問題ない)

→ **計算段階そのものが存在しないので、オーバーフローしようがない**

### u16 LeakyRelu — **ランタイム算術実装**

- 入力 `x_q` の取りうる値は **65,536 通り**
- LUT を作るなら 65,536 entry × 2 byte = **128 KB / op**
- HiFiGAN decoder には LeakyRelu が ~25 個ある → 全部 LUT 化したら 3 MB 以上、現実的ではない
- そのため **毎サンプル**で `delta × M_pos >> S + r_zp` を計算する
- → 16 bit 乗算器制約に引っかかり、本ケースのような scale/zp の組合わせで詰む

---

## 7. 結論と回避策

| 案 | 効果 | 採否 |
|---|---|---|
| calibration 法を変える (MinMax → Percentile/MSE) | `M_pos = x_scale/r_scale` の比は LeakyRelu の伝達特性で決まり不動 → 効かない | ✗ |
| `enable_onnxsim: false` | onnxsim は overflow に関与していない | ✗ |
| `transformer_opt_level: 1` | Transformer 系の最適化なので CNN/HiFiGAN には無関係 | ✗ |
| 単一 `LeakyRelu_1` のみ U8 降格 | 同じく overflow する別の LeakyRelu (`resblocks.7/LeakyRelu` 等) で詰む | ✗ |
| **全 LeakyRelu を U8 降格** (`op_type: LeakyRelu` → `data_type: U8`) | LUT 実装に切替わり overflow 消滅、品質劣化は decoder cos 0.995 で許容範囲 | ✅ **採用** |

最終 config (抜粋):

```json
"layer_configs": [
  { "start_tensor_names": ["DEFAULT"], "end_tensor_names": ["DEFAULT"], "data_type": "U16" },
  { "op_type": "LeakyRelu", "data_type": "U8" }
]
```

成果物:
- `axmodel/tsukuyomi-decoder-u16_ax650.axmodel` (3.27 MB, mixed precision: 全体 U16 + LeakyRelu のみ U8)
- 実機 1Core で 41 ms / E2E パイプライン 64 ms / RTF 0.019 で動作

---

## Appendix A: 検証した 7 通りのビルド結果

| # | 試行 | calibration | layer_configs | 結果 |
|---|---|---|---|---|
| 1 | 元 (check=0) | MinMax | 全 U16 | ビルド OK / 実機 execute fail |
| 2 | check=2 だけ追加 | MinMax | 全 U16 | overflow 検出 (resblocks.6/LRelu_1) |
| 3 | onnxsim 無効 | MinMax | 全 U16 | 同じ overflow (onnxsim 無関係) |
| 4 | Percentile | Percentile | 全 U16 | 同じ overflow (scale tight 化、悪化方向) |
| 5 | MSE | MSE | 全 U16 | 同じ overflow (delta 振幅増、悪化方向) |
| 6 | transformer_opt_level=1 | MinMax | 全 U16 | 同じ overflow (HiFiGAN に効果なし) |
| 7 | 単一 LRelu_1 のみ U8 | MinMax | resblocks.6/LRelu_1 → U8 | 別 LRelu (resblocks.7) で同じ overflow |
| **8** | **全 LeakyRelu を U8** | **MinMax** | **op_type=LeakyRelu → U8** | ✅ **ビルド成功 + 実機動作 (RTF 0.019)** |

## Appendix B: 3 つの calibration 法での scale 値

`/dec/resblocks.6/LeakyRelu_1` の量子化パラメータ:

| 方式 | x_scale | x_zp | r_scale | r_zp | x_scale/r_scale |
|---|---:|---:|---:|---:|---:|
| MinMax | 2.51 × 10⁻⁵ | 46,039 | 9.22 × 10⁻⁶ | 12,520 | 2.72 |
| Percentile | 1.87 × 10⁻⁵ | 45,919 | 6.91 × 10⁻⁶ | 12,431 | 2.71 |
| MSE | 8.02 × 10⁻⁴ | 1,439 | 2.95 × 10⁻⁴ | 391 | 2.72 |

→ scale 値そのものは calibration 法で大きく変わるが、**比 (M_pos) は LeakyRelu の伝達関数で決まる定数**で動かない。
