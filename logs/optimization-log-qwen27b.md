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
