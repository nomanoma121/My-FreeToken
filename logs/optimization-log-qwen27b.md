# Qwen27B (llama.cpp) 最適化ログ

Qwen3.8-27B-UD-Q4_K_M を 2x RTX3060 (llama.cpp, tensor split, MTP) で動かし、
decode 70 tok/s を狙う作業記録。3.8 (FreeToken) 側は `logs/optimization-log.md`。
実測値のみ記載、捏造厳禁。

## ハード・環境 (gpu-node-02)

- GPU: RTX 3060 12GB x2 (NVLinkなし、PHB接続、P2P DMAは有効)
- エンジン: ~/llama.cpp (build/bin/llama-server)
- モデル: unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M (17GB, HF cache)
- ベースライン構成: `-ngl all -c 131072 -np 1 --split-mode tensor -fit off --cache-type-k q8_0 --cache-type-v q8_0 --flash-attn on --no-mmproj --spec-type draft-mtp --spec-draft-n-max 2` → prose 49 / code 55 tok/s

## 作業記録

#### 2. 27B (llama.cpp, Qwen3.8-27B-UD-Q4_K_M) 速度スイープ全記録

サーバー: `./build/bin/llama-server -hf unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M -ngl all`
ベンチ: prose/code各384token (`/tmp/bench_llama.py`, server predicted_tps)。

| 構成 | prose | code | 備考 |
|---|---|---|---|
| tensor n-max 2, ctx131k (採用) | 48.9 | 54.5 | accept 57-69%, mean len 2.14-2.38 |
| tensor n-max 1 | 45.0 | 46.8 |  |
| tensor n-max 3 | 46.1 | 51.9 | accept 47-57%に低下、mean len 2.39-2.70 |
| layer split, ctx32k | 31.7 | 33.8 | 大幅悪化 |
| tensor, ctx32k | 49.2 | 54.8 | ctxサイズは無影響 |
| row split | 起動不可 | — | `device CUDA0 does not support split buffers` |

n-max 2が最適。layer split 131kはKV確保でOOM (`cudaMalloc failed`, device1で512MB確保失敗)。
ついでに `backend offload failed ... using CPU sampler` の警告あり (サンプラーがCPU実行)。

#### 3. nsysによる27B tensor分割の通信解析 (決定打)

`nsys profile --trace=cuda --delay=70 --duration=40` でdecode窓16.17秒を取得
(`/tmp/nsys-kouta/nsys-report-b324.nsys-rep`, 解析スクリプト `/tmp/ar_stats.py`)。

- `ggml_cuda_ar_kernel` (all-reduce): カーネル時間の **75.6%** (2.77秒/3.66秒)
- デバイス別: dev0 46,210回/1.223秒、dev1 46,210回/1.545秒
- 1トークンあたり約60回のAR × 平均30μs = **約1.8ms/token (全体の約10%)**
- ホスト側同期: `cudaEventSynchronize` 92,420回/**10.045秒ブロック**、
  `cudaStreamSynchronize` 72,350回/2.63秒、カーネル起動165,366回、
  グラフ起動92,496回
- GPUカーネル稼働は窓の約11%のみ。残り89%はホスト同期の直列待ち。
- 真犯人は「ARの転送量」ではなく「ARごとのホスト同期ハンドシェイク」。
  `GGML_CUDA_AR_KERNEL_BLOCKS` 8→16に上げてリビルド実測したが効果なし
  (48.7/54.3で誤差範囲、並列度ではなく待ちが律速のため)。バイナリは元に戻し済み。

#### 4. 天井見積 (roofline)

- 2x3060合計帯域720GB/s ÷ 重み17GB = 42 forwards/s上限。現在23 (54%)。
- 3x3060なら1080GB/s ÷ 17GB = 63 forwards/s上限。現効率維持で約83、
  効率70%化で約106。100は可能圏内 (通信増のリスク付き)。
- ChatGPTのAR比率表への回答: AR壁は約10%なのでAR単体では70に届かない
  (上限61相当)。MTP accept率向上との両輪が必須。

#### 5. 次の一手 (優先順)

1. MTP accept率向上 (forward/token削減＝AR回数も削減、一石二鳥) ←本命
2. AR同期の融合・削減 (llama.cppのallreduce.cu改造、要工数)
3. CPU sampler警告の解消 (ついで、効果は小の見込み)


### 27B (2026-09-16): --spec-draft-n-min 1 は効果なし

tensor n-max 2 ベースでの比較 (prose/code predicted_tps):

| --spec-draft-n-min | prose | code | accept率/mean len |
|---|---|---|---|
| default | 48.9 | 54.5 | 57-69% / 2.14-2.38 |
| 1 | 49.1 | 54.7 | 同上 (変化なし) |

適応的draft数制御はこの構成では挙動を変えない。n-max 2 + n-min defaultを維持。

### 27B (2026-09-16): ARホスト同期スキップ改造 → 速度効果なし、機構は成功

llama.cpp `ggml/src/ggml-cuda/allreduce.cu` に `GGML_CUDA_AR_NO_HOST_SYNC`
環境変数を追加 (pool周回時の `cudaEventSynchronize` をスキップ。チャンクパスは
computeストリーム順序＋カーネル内到着ハンドシェイクで順序保証されるという設計)。
リビルド＋実測:

| 構成 | prose | code |
|---|---|---|
| stock (n-max 2) | 48.9 | 54.5 |
| NO_HOST_SYNC=1 | 48.9 | 54.6 |

効果なし。nsys裏取り (`/tmp/nsys27b2.nsys-rep`) で判明した理由:
- `cudaEventSynchronize` 92,420回は完全に消滅 (機構としては成功)
- 代わりに `cudaStreamSynchronize` 72,350回が 2.63秒→12.65秒に増加
- グラフ起動 92,496回のまま (ARごとに切断)
- 待ちの場所が変わっただけで、op-by-op直列実行は不変

結論: ホスト同期除去だけでは不足。本当に効かせるにはdecode 1ステップ全体の
単一グラフ化 (スケジューラ級の改造、工数日単位・高リスク) が必要。
品質は無傷 (算数408・羊9・34割り算すべて正解)。
パッチはソースに残置 (デフォルト0で従来動作、バイナリも通常起動では等価)。

## TODO (27B 70 tok/sへの道、順次実行中)

- [x] ベースライン確定 (tensor n-max 2: 49/55)
- [x] n-max 1/3、layer/row split、ctx、n-min sweep
- [x] nsys通信解析 (AR 75%、同期10秒特定)
- [x] AR BLOCKS 16 / NO_HOST_SYNC改造 (効果なし、原因はop-by-op直列)
- [x] p-min 0.7 (accept 89%も速度低下で棄却)
- [x] p-min sweep (0.4/0.7とも悪化で棄却)
- [x] MTP + ngram併用 (ngram単体31で棄却、併用不可)
- [x] CPU sampler→backend sampling (効果なし)
- [x] CUDA_SCALE_LAUNCH_QUEUES=4x (効果なし)
- [x] minimal loop比較 (CLI暴走で中止、nsys済みのため不要)
- [ ] graph launch分類・fused QKV
- [ ] MTP cycle graph specialization / DeltaNet fusion / GEMV+AR融合

### 27B (2026-09-16): p-min sweep → いずれも悪化で棄却

| --spec-draft-p-min | prose | code | accept率 |
|---|---|---|---|
| 0.0 (default) | 48.9 | 54.5 | 57-69% |
| 0.4 | 40.3 | 46.7 | 72-74% |
| 0.7 | 35.2 | 40.6 | 89% |

accept率は上がるが速度は単調悪化。低確率draftの棄却がforward回数を増やすため。
GitHubの+15%報告は当環境では再現せず。

### 27B (2026-09-16): ngram-mod単体は不発、MTP維持

| spec-type | prose | code |
|---|---|---|
| draft-mtp n-max 2 | 48.9 | 54.5 |
| ngram-mod (24/48/64) | 31.8 | 31.1 |

新規prose/code生成では繰り返しが少なくngramが当たらない。MTP+ngram併用は
当該llama.cppでは非対応 (spec-type単一)。MTP維持。

### 27B (2026-09-16): --backend-sampling (-bs) は効果なし

| -bs | prose | code |
|---|---|---|
| なし | 48.9 | 54.5 |
| あり | 49.0 | 54.7 |

誤差範囲。なお `-bs` でも `backend offload failed ... using CPU sampler` 警告は
1件残存 (draft側はデフォルト有効)。CPU samplerは律速ではないと判断。

### 27B (2026-09-16): CUDA_SCALE_LAUNCH_QUEUES=4x は効果なし

| LAUNCH_QUEUES | prose | code |
|---|---|---|
| 未設定 | 48.9 | 54.5 |
| 4x | 48.9 | 54.7 |

誤差範囲。launch queue不足は律速ではない。

### 27B (2026-09-16): NCCLはinternalに敗北

NCCL有効化ビルド (libnccl2/dev導入、cmakeでFound確認) で同条件比較:

| AR backend | prose | code |
|---|---|---|
| internal (従来) | 49.2 | 54.7 |
| NCCL | 47.6 | 50.6 |

PHB環境ではNCCLが遅い (既報通り)。`GGML_CUDA_ALLREDUCE=internal` を常用とする。

### 27B (2026-09-16): CPU governor / 不均等split / fused-QKV調査

- CPU governor powersave→performance: 効果なし (48.9/54.7のまま)
- tensor-split 45/55: 131kではKV確保OOM。64kに絞れば起動するが48.6/52.2で悪化
  (GPU1のクロックまで低下)。均等維持。
- fused-QKV: 当該checkoutにconvert flagあり。ただし適用にはHF元モデル約54GBの
  DL＋再変換＋再量子化が必要で効果はlaunch数削減のみ (小〜中)。未実施。

### 27B (2026-09-16): 律速の再特定 → 「全体グラフ化」は不要と判明

nsys (CUDA graph有効/無効の両方) でdecode 1サイクル (約43.5ms) を分解:

- 入力H2D 0.6ms → 本体検証forward GPU 35.9ms → logits D2H 0.7ms → CPU 0.9ms → MTP(追い付き+draft1) 2.4ms → 0.45ms → draft2 2.3ms
- 本体forward中、ホストは全launch (graph 258 + AR 256) を4.7msで出し終え、GPUが後追いで約30ms実行 → **GPU律速**
- `GGML_CUDA_DISABLE_GRAPHS=1` でも 48.9/54.4 と不変 → launch/同期は律速でない
- よって「decode 1ステップ全体の単一グラフ化」は効果ほぼゼロと判断し中止
  (前回の cudaStream/EventSynchronize 待ちは「GPUが終わるのを待っている」だけだった)
- 検証forwardのGPU時間内訳 (dev0, Q4_K_M): mul_mat_vec_q 27.1ms (74%), AR 3.3ms (dev1は4.4ms),
  GDN 0.9, concat 0.9, rms_norm 0.9, quantize_q8_1 0.9, 他小粒
- MMVQは帯域理論値の約85%で動作 (IQ4_XS FFN 77µs vs 理論66µs)。ただし Q3_K は 150µs vs 理論53µs で演算律速
- 出力ヘッド Q6_K (1.04GB) は検証1回+draft毎に読まれ、1回約1.5ms
- logits D2H は両GPUから分割読み出しで計0.3ms、非律速。GPU0はPCIe x4 (Max x16)、電力上限170Wはハード上限

### 27B (2026-09-16): 量子化形式の比較 (n-max 2, greedy)

| 量子化 | サイズ | prose | code | サイクル/s |
|---|---|---|---|---|
| UD-Q4_K_M (従来) | 16.5GB | 48.9 | 54.7 | 23.0 |
| UD-IQ4_XS | 14.3GB | 53.0 | 51.8 | 23.1 |
| unsloth Q4_0 | 16.1GB | 54.7 | 57.8 | 24.5 |
| UD-Q3_K_XL | 13.2GB | 48.6 | 50.3 | 22.3 |

サイズよりカーネルの単純さが効く。IQ4_XSは小さいが同速、Q3系は遅い、Q4_0が最速。
(サイクル/s = tok/s ÷ mean len。accept率は量子化で多少揺れる)

- ARカーネルブロック数 1/2/4/8 (Q4_0): 53.6/56.6, 54.6/57.7, 54.8/58.0, 54.7/57.8 → 効果なし、改造は撤回

### 27B (2026-09-16): 自前 純Q4_0 量子化で 59.2 / 63.6 tok/s

unsloth BF16 + imatrix_unsloth.gguf から作成 (`~/models/qwen38-27b-gguf/Qwen3.8-27B-Q4_0-fast.gguf`, 15.4GB):

```
llama-quantize --imatrix imatrix_unsloth.gguf --output-tensor-type q4_0 --token-embedding-type q4_0 \
  --tensor-type ssm_out=q4_0 --tensor-type ffn_down=q4_0 Qwen3.8-27B-BF16-00001-of-00002.gguf \
  Qwen3.8-27B-Q4_0-fast.gguf Q4_0 10
```

unsloth Q4_0 では出力ヘッドQ6_K・ssm_out Q5_K・一部ffn_down Q4_1 が残っていたのを全てQ4_0化。

| 構成 | prose | code | mean len |
|---|---|---|---|
| Q4_0-fast n-max 2 | 56.6 | 61.2 | 2.15/2.32 |
| **Q4_0-fast n-max 3** | **59.2** | **63.6** | 2.49/2.68 |
| Q4_0-fast n-max 4 | 55.3 | 61.2 | 2.62/2.88 |
| unsloth Q4_0 n-max 3 | 55.7 | 56.8 | 2.60/2.65 |

draftが軽くなったことで n-max 3 が最適に変わった。
品質簡易チェック (n-max 3): 17*24=408, 羊9, 1156/34=34, 首都=東京, 180km/2.5h=72 → 5/5正解、is_primeコードも正常。

次: draft用縮小語彙ヘッド (FR-Spec方式) / MXFP4等さらに軽いカーネル形式の検証

### 27B (2026-09-16): 追加検証 (いずれも Q4_0-fast 基準、n-max 3 = 59.4/64.1)

**MXFP4 (FFN/attn/ssm_out, 14.0GB)**: 52.9/61.5、mean len 2.49/2.90、サイクル/s 21.2 (Q4_0-fast 23.8) → 棄却。小さくてもカーネルが遅い。

**draft用縮小語彙ヘッド (FR-Spec方式, 3.2万語)**: 実装 (nextn.sub_head + sub_inv で logits を n_vocab に書き戻し) して 48.5/57.2 → 棄却・コード撤回。
- 語彙カバー率は held-out で 99.06% (Python stdlib/llama.cpp docs/My-FreeToken ログ 1800万トークンで頻度算出)
- ヘッドのMMVQは 1.20ms→0.62ms に減ったが、24.8万語への書き戻し `k_get_rows_float` が 1.79ms かかり逆効果
- prose の accept率も 0.50→0.42 に低下

**MMQ強制 (検証バッチをMMVQ→MMQ)**: MMVQ上限1/2/3 すべて 約47/54 → 棄却。MMVQが正解。

**MMVQ多列カーネルの並列度 (Ampere=GENERIC表, ncols 2-4)**: nwarps×rows/block = 2×2, 8×2, 4×1, 4×4, 2×4, 4×8, 8×1, 2×8 を掃引。
サイクル/s換算でベースライン(4×2)±1%以内 (rows/block=1 は大幅悪化) → 効果なし、撤回。
tok/s はカーネル変更による丸め差で greedy 軌跡が変わり accept率が揺れるので、必ず mean len で正規化して比較すること。

**バッチサイズ別 decode 時間** (llama-batched-bench, Q4_0-fast, 系列数=B):

| B | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| ms/step | 29.0 | 32.4 | 36.5 | 40.2 | 45.1 | 52.4 |

B1→B4 のGPU時間差分 (+10.2ms): MMVQ +4.9, GDN get_rows +1.4, gated_delta_net +1.4, AR +1.3。
ただし実際の投機検証 (1系列4トークン) は B1 比 +5ms 程度で、MMVQ 増分は +1.2ms のみ → 列融合カーネルの上限は約1ms。

**電力律速を発見**: n-max 3 ベンチ中の nvidia-smi (500ms毎):
- GPU0: 平均163W (上限170W, ハード上限も170W), SM 1857MHz, 全サンプルで throttle reason 0x4 (SW power cap)
- GPU1: 平均170W (上限180W, 最大190W), SM 1959MHz, 全サンプルで 0x4
- 両GPUとも電力キャップでスロットリング中。遅いGPU0に全体が律速される

### 27B (2026-09-16): GPUクロックオフセット (NVML, ヘッドレス) → 効果1%程度、0に戻して据え置き

ユーザー承認のうえ実施。電力上限は変更なし (170W/180W)。`nvmlDeviceSetMemClkVfOffset` / `nvmlDeviceSetGpcClkVfOffset` で両GPU同値。
Q4_0-fast, n-max 3。mean len は全条件で 2.49/2.68 (出力不変)、品質チェック5/5。

| mem | core | prose | code | Xid |
|---|---|---|---|---|
| 0 | 0 | 59.6 | 64.2 | 0 |
| +250 | 0 | 59.5 | 64.2 | 0 |
| +500 | 0 | 59.6 | 64.3 | 0 |
| +1000 | 0 | 60.1 | 64.9 | 0 |
| 0 | +50 | 59.4 | 64.1 | 0 |
| 0 | +100 | 59.7 | 64.5 | 0 |
| 0 | +150 | 60.1 | 64.8 | 0 |
| +1000 | +300 | 起動直後にクラッシュ | | **Xid 31 (GPU0, MMU fault)** |

- core+300 で GPU0 が不安定化 → 即 0/0 に戻し、再計測 59.2/63.9・5/5・Xidなしで正常を確認
- 安全圏の上げ幅では +1% 程度でリスクに見合わないため、オフセットは 0 のまま (再起動でも0)
- メモリクロックを上げてもほぼ伸びない → 帯域より電力/SMが律速
