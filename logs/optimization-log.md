# FreeToken ローカル最適化ログ

Qwen3.8-Flash-Next を自分のマシンで 4bit 量子化して動かし、デコード速度
40 tok/s (目標)、可能なら 60 tok/s を狙うための作業記録。実測値と手順を
時系列で積み上げる。捏造した数字は書かない（実行していないベンチマーク
結果は「未実施」と明記する）。

## 環境スペック (2026-09-15 時点)

- ホスト: `gpu-node-02`, Ubuntu 26.04 LTS, kernel 7.0.0-31-generic
- CPU: AMD Ryzen 9 5900X (12C/24T), boost 4.95GHz, L3 64MiB
- RAM: 121GiB 物理 (実測 free: total 121Gi / available ~117Gi 使用開始前), swap 8GiB
- GPU: NVIDIA GeForce RTX 3060 12GB x2 (PCIe, ブリッジなし想定)
  - VRAM 合計 24GiB
  - Driver 595.91.07, CUDA 13.2 (nvidia-smi 表示) / nvcc 13.4 (`/usr/local/cuda`)
- Disk: `/` 1.8TB (使用470G, 空き1.3TB, 2026-09-15時点)

## 対象モデルとエンジンの前提整理

- リポジトリ: https://github.com/nomanoma121/My-FreeToken.git を `~/My-FreeToken` に clone (fork せず直接作業)
- FreeToken は **GGUF を読まない**。HF safetensors を直接ロードする設計
  (`docs/models.md`: "FreeToken loads HF safetensors checkpoints directly")。
  そのため事前にホストに存在していた llama.cpp 用 GGUF (`~/models/qwen-flash-next`,
  unsloth `UD-Q4_K_XL` 4shard ≈105GB) は今回のFreeToken経路では使わない。
- `docs/models.md` の Qwen3.8-Flash-Next 対応チェックポイント一覧:
  `Qwen/Qwen3.8-Flash-Next-FP8`, `RadixArk/Qwen3.8-Flash-Next-NVFP4`,
  `nvidia/Qwen3.8-Flash-Next-NVFP4`
- ユーザーの「Q4でよい」は元々 GGUF の Q4_K_M/L を念頭にした発言だったが、
  FreeTokenがGGUF非対応と分かった後 "好きに進めてくれ" と裁量をもらったため、
  FreeTokenがネイティブサポートする4bit形式である **NVFP4** を採用する。
  `nvidia/Qwen3.8-Flash-Next-NVFP4` を選択 (公式 nvidia 配布、shard構成がシンプル:
  safetensors 11分割, 合計約132.7GB)。`RadixArk` 版 (135.3GB, 206分割) は代替候補。

## モデルアーキテクチャ (config.json より)

- `architectures`: Qwen4ExpForConditionalGeneration (`model_type: qwen4_exp`)
- 48 layers, MoEはlayerごとに 512 experts, top-10 routing, `moe_intermediate_size=640`,
  `hidden_size=2560`, shared_expert あり
- 40M行 x 160dim の PLE (Per-Layer Embedding) n-gram テーブルを内蔵 (FP8, 生47.7GiB)
  - **重要**: `--ple-backend` のデフォルトは `disk` (チェックポイントから直接行読み)。
    `docs/models.md` に書かれている「47.7 GiB を host RAM に pinned」という記述は
    `--ple-backend pinned` を明示指定した場合の話で、デフォルトでは該当しない。
    もし将来 pinned に切り替える場合は RAM 予算を+47.7GiB考慮すること。

## メモリ予算の見積り (ダウンロード前に実施した分析)

- ディスク上の NVFP4 チェックポイント総サイズ: 132.7GB
- expertテンソルが 297,984個 / 全299,545テンソル中を占め、パラメータの大半を占有
- 理論パラメータ数見積り: 512 experts x 48 layers x (640x2560x3) ≈ 120.8B params (expertのみ)
- `python/freetoken/moe/fused_nvfp4.py` を読むと、ロード後のホスト常駐バンクは
  `packed: uint8 [S, N, K//2]` = **2値/byte にパックされた** 4bit 表現。
  これは 120.8B params x 0.5B/param + ブロックスケール分 ≈ 65-70GB 程度に収まる計算になり、
  チェックポイントのディスク上サイズ(132.7GB, おそらく1byte/値の非パック形式)の
  ほぼ半分で済むと推測される。
- 121GiB の物理RAMに対し、上記推測が正しければ十分収まる見込み。ただし理論値であり、
  実際のロード時のピークRSSは実測して確認する必要がある(下の「未検証」セクション参照)。
- 結論: 一旦 NVFP4 をダウンロードして実際にロードしてみる。OOMするようなら
  (a) `--ple-backend disk`(デフォルトのまま) を維持、(b) `--expert-load serial` で
  低メモリ読み込みに切替、(c) 最悪 GGUF+llama.cpp 経路にフォールバック、の順で対応する。

## 実施した作業 (時系列)

1. `~/My-FreeToken` に `git clone https://github.com/nomanoma121/My-FreeToken.git` (fork未使用)
2. デバイススペック調査 (`lscpu`, `free -h`, `nvidia-smi`, `df -h`)
3. 既存環境の棚卸し: `~/llama.cpp` (公式, clean), `~/llama-flashnext-other`
   (Inovello/llama.cpp フォーク, Flash-Next用CUDAカーネル改造あり, 2x RTX3090+DDR4向けチューニング
   -- 今回は使わない方針だが将来のフォールバック候補として記録),
   `~/models/qwen-flash-next/MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf` (MTP shared, 2.7GB)
4. HuggingFace API で `nvidia/Qwen3.8-Flash-Next-NVFP4` (132.7GB) と
   `RadixArk/Qwen3.8-Flash-Next-NVFP4` (135.3GB) のサイズを事前確認 (非gated, ダウンロード可)
5. `nvidia/Qwen3.8-Flash-Next-NVFP4` を `~/models/qwen38-flash-next-nvfp4` へ `hf download` 開始 (background)
6. `uv venv --python 3.12` で FreeToken 用仮想環境作成 (nightly wheelがcp312前提のため3.12固定;
   デフォルトのsystem python 3.14だとtorch等のwheelが噛み合わないリスクを回避)
7. `uv pip install -e ".[accel]"` を background で実行開始 (flashinfer + sglang-kernel を含む accel extra)

## サーブ時の主要フラグ調査 (docs/cli.md, args.py を確認)

- `--tensor-parallel-size` / `--tp-size` と `--gpu 0,1` で **2枚のRTX3060を
  TPで束ねられる** (`--gpu` はTP rank順のカンマ区切り)。VRAM合計24GiBを
  MoE expert cache + KV cache に使えるようになり、単GPU(12GB)より
  GPUキャッシュヒット率が上がる見込み。まずはシングルGPUで動作確認してから
  `--tp-size 2` を試す。
- `--moe-strategy {auto,fused,offload,cpu,hybrid}`。`auto`はMoEモデルなら
  `offload`、`ft bench bw`のプロファイルがあれば`hybrid`に格上げ。
  512expert/layerと巨大チェックポイントなので`offload`か`hybrid`が本命。
- `--moe-cpu-layers`: offload戦略で特定layerのMoE計算をCPU側に逃がせる
  (bf16/nvfp4/mxfp4対応、fp8experts不可 — 今回はnvfp4なので対象)。
  Ryzen 9 5900X (12C/24T) のCPU計算力とPCIe帯域のバランスを見て調整余地あり。
- `--moe-hybrid-max-fetch`: hybrid時にPCIe経由フェッチする最大expert数/layer/step。
  `ft bench bw`のプロファイルで自動決定される。
- `ft bench bw` は GPU毎・expertフォーマット毎にプロファイルを
  `~/.cache/freetoken/benchbw/<gpu-uuid>.json` に保存する。モデルロード前に
  一度実行しておく。
- Marlin W4A16 NVFP4 kernel path は `vllm>=0.14,<0.15` 依存で
  core の `transformers>=5.5` と衝突するため `accel` extra には未収録。
  sm_86 (Ampere, RTX3060) でのNVFP4カーネル速度が伸びない場合の調査対象として残す。

## フォールバック案 (メモ、未実施)

- `~/llama-flashnext-other` (Inovello/llama.cpp フォーク) に Flash-Next 専用の
  CUDA radix top-k フォールバックと "A/B guide" が既に入っている
  (`dd64a3db0 cuda: publish tested Flash-Next radix top-k fallback and A/B guide`,
  `9bd97fe54 Flash-Next replication branch for 2x RTX 3090 + DDR4 host experts`)。
  対象ハードウェアは元々 2x RTX3090+DDR4 想定なので、うちの 2x RTX3060 12GB との
  差分チューニングが必要になる。FreeToken経路でtok/s目標に届かない場合の代替候補。
  GGUF (`unsloth UD-Q4_K_XL`, 105GB, 4shard) は既にダウンロード済みで
  `~/.cache/huggingface/hub/models--unsloth--Qwen3.8-Flash-Next-GGUF` にある。

## Web調査で得た知見 (2026-09-15, ユーザー許可のもと検索実施)

- **重要**: `FlashML-org/FreeToken#151` — 2x RTX3090 (sm_86, Ampereは我々と同世代) +
  DeepSeek-V4-Flash で `ft bench bw` が `hybrid` を自動選択するが、手動で
  `--moe-strategy offload` に固定した方が **8.3倍速い** (5.58 vs 0.67 tok/s)。
  bench bw の帯域測定 (56.2GB/s) が実サービング時の実効帯域 (~2GB/s) と乖離しており、
  per-activation同期オーバーヘッドをコストモデルが見ていないのが原因、との報告。未修正。
  → **本機でも `--moe-strategy auto`/`hybrid` を信用せず、明示的に `offload` を指定する。**
- `yuuki-net/FreeToken-Kai` という非公式フォークが、ほぼ同一ハード
  (2x RTX3060 12GB, Qwen3.8-Flash-Next) で **18-20 tok/s** (128kコンテキストでも
  ほぼフラット) を報告。ただし upstream FreeTokenには無い独自機能を使っている:
  - `--pp-size 2`: pipeline-parallel GPU分割 (gloo実装, NCCL/peer-access不要)
  - ファイルマップ方式のexpert bankでRAM使用量を半減 (`--moe-bank-ram 48G` で制限時は14-15 tok/sに低下)
  - `--kv-cache-dtype q4_0/q8_0`: KVキャッシュ量子化でVRAM 1.9-3.6倍節約
  - これらはupstreamの `My-FreeToken` には存在しない。ポートするかは要判断
    (今回は "forkしない" 指示があるため、まずupstreamのフラグチューニングで
    どこまで迫れるか試し、天井が見えたら再検討する)。
- `nvidia-smi topo -m` 実測: GPU0-GPU1間は **PHB** (PCIe Host Bridge経由、NVLinkなし)。
  P2Pは理論上可能だが帯域はPCIeどまり。FreeToken-KaiがTP(NCCL)を避けてPP(gloo)を
  選んだ理由と整合する。upstreamの `--tensor-parallel-size 2` はNCCL/peer-access前提の
  可能性が高く、このトポロジでは恩恵が薄いか不安定なリスクあり。
  → **まずは `--gpu 0` のシングルGPUで動作・速度を確認してからTPを検討する。**
- 参考上限値: NVIDIA公式NVFP4 + DGX Spark (unified memory, PCIeオフロード無し) で
  Qwen3.8-Flash-Next 単体43.8 tok/s peak / 32.5 tok/s中央値 (1台), 2台TPで63.7 tok/s peak。
  つまりPCIeオフロードのボトルネックが無い理想的環境でも60tok/s台が精一杯の部類。
  我々の環境 (dual RTX3060 12GB, PCIe offload必須) では
  **40 tok/sはかなり挑戦的、60 tok/sは非常に厳しい目標**という見立て。
  (出典: NVIDIA Developer Forums, FlashML-org/FreeToken#151, yuuki-net/FreeToken-Kai README)
- Marlin vs flashinfer/sglang-kernel の sm_86 性能比較や、FreeToken論文自体の
  低VRAM環境ベンチマーク数値は検索でヒットせず (未確認のまま)。

## `ft bench bw` 実測結果 (2026-09-15, --dtype nvfp4)

インストール完了後、モデルダウンロード完了を待たずに実行 (bench bwはモデル非依存)。

| GPU | UUID | CPU STREAM read | PCIe H2D | PCIe D2H | CPU-MoE | PCIe-gather | CPU/PCIe比 | 推奨backend |
|---|---|---|---|---|---|---|---|---|
| GPU0 | GPU-118ee0a5... | 41.4 GB/s | **6.2 GB/s** | 6.5 GB/s | 39.1 GB/s | 6.5 GB/s | 6.05x | hybrid |
| GPU1 | GPU-f4d5a088... | 41.3 GB/s | **25.8 GB/s** | 25.5 GB/s | 39.4 GB/s | 26.2 GB/s | 1.50x | offload |

**重大な発見**: GPU0とGPU1でPCIe実効帯域が **4倍以上違う** (6.2 vs 25.8 GB/s)。
おそらく物理スロットのレーン数/世代が異なる (GPU0が細いリンクに刺さっている)。
`nvidia-smi -q -d PCI` でのリンク幅確認は今回のnvidia-smi版のフラグ仕様不一致で
失敗したため未確認だが、実測ベンチ(より信頼できる)でこの差は明確。

→ **サービングは `--gpu 1` (UUID GPU-f4d5a088-ba66-485c-ae5b-274b69de50f1) を
既定にする。** offloadのミスフェッチはPCIe経由なので、ここがボトルネックに直結する。
GPU0(デフォルトの--gpu省略時の挙動)のままだと大きく損をする可能性が高い。

またGPU1では実測でも `offload` (比1.50x < 閾値2.0x) が推奨で、Web調査で見つけた
Ampereでの hybrid 不具合 (#151) の懸念とも整合する。GPU0はhybrid推奨だが
Web調査の知見を踏まえ、実サービングでは両GPUとも `--moe-strategy offload` を
明示指定してA/B比較する方針とする。

## 起動トラブルシュート記録 (2026-09-15)

1. **GPU1単体, `--moe-strategy offload`, マルチモーダル込み**: 失敗。
   `AssertionError: cache budget too small: ... budget -2040754172 B`
   (VRAM予算がマイナス。dense重み+固定キャッシュだけで12GBカードを使い切っている)
2. **GPU1単体 + `--text-model-only`**: ほぼ改善なし。
   `budget -1889759228 B` (▲150MB程度しか変わらず。visionエンコーダは主因ではない)
3. **`--tp-size 2 --gpu 1,0` (両GPUでdense重みを分割)**: 2段階で失敗。
   - a) pynccl の JIT ビルドで `cannot find -lnccl` (pip版 `nvidia-nccl-cu13` が
     `libnccl.so.2` のみを配置し `libnccl.so` シンボリックリンクが無いため linker が解決できない)。
     `ln -sf libnccl.so.2 libnccl.so` を作成し、`LIBRARY_PATH`/`LD_LIBRARY_PATH` に
     そのディレクトリを追加して解決 (`~/.cache/tvm-ffi` の古いビルドキャッシュも削除)。
   - b) NCCLリンク後に別のエラー:
     `KernelSelectionError: no usable kernel in table; triton: TP > 1 is not supported
     for this expert format; marlin: vLLM is not installed; b12x: b12x requires sm_120+, got sm_86`
     → **重要な制約**: このNVFP4 checkpointのexpertカーネルは sm_86 (Ampere, RTX3060) では
     事実上 `triton` バックエンドしか選べず、その `triton` 実装は **TP>1を完全に非対応**。
     `marlin`はvLLM別インストールが必要 (transformers要件と衝突で保留中)、
     `b12x`はsm_120+ (Blackwell) 専用で対象外。
     → **結論: このハード+チェックポイントの組み合わせでは、MoE層のテンソル並列化は
     カーネルレベルで不可能。2GPU構成でもMoE計算は常に単一GPUで行うしかない。**
     `--tp-size`はここでは使えない。

## 単GPUでのVRAM予算チューニング詳細 (2026-09-15) と最終結論

`--memory-ratio`/`--kv-reserve-tokens`/`--disable-moe-prefill-overlap`/`--max-running-requests`
を極限まで詰めた記録 (すべてGPU1, `--text-model-only`, `--moe-strategy offload`):

| 試行 | flags差分 | 結果 (budget vs 必要量) |
|---|---|---|
| 1 | デフォルト | -2040754172 B (moe=1024 slots overlap込み) |
| 2 | `--memory-ratio 0.97 --kv-reserve-tokens 1024 --max-running-requests 1` | -1889759228 B (変化ほぼ無し) |
| 3 | 上記 + `--disable-moe-prefill-overlap` (floor 1024→512) | budget 825138063 B, 必要 2864971776 B (まだ不足) |
| 4 | `--memory-ratio 0.99 --kv-reserve-tokens 256` | budget 1072158975 B, 必要 1425997824 B (差 ▲337MB) |
| 5 | `--memory-ratio 1.0 --kv-reserve-tokens 64` (どちらも上限/下限) | budget 1195669432 B, 必要 1421131776 B (差 ▲215MB, これ以上flagで縮まらない) |
| 6 | 上記 + `--max-seq-len-override 8192` | **数値完全に同一** (weights_bytes/fixed_cache_sizeはmax_seq_lenに非依存と判明) |

実測 (VRAMトレース): GPU1上のdense (非expert) 重みだけで **約9.8-9.9GB** 消費
(text-model-onlyでも殆ど変わらず)。512experts分のMoEオフロードキャッシュ最小要件
(architecture固定, `num_experts=512`が絶対最小floor) だけで約1.3GB追加必要 →
**12GBカード1枚には物理的に収まらない**(flagチューニングでは埋まらない
約200-350MBの恒常的な不足)。

`--moe-strategy cpu` (MoE計算を完全にCPU側で行い、GPU側は"固定2レイヤー分の
バッファ"のみで済むはず) も試したが、`torch.OutOfMemoryError: ... Tried to
allocate 800.00 MiB ... 182.88 MiB is free` で失敗。dense重みだけで
11.08-11.32GiB (12GB弱) を使い切っており、`--num-tokens`/`--memory-ratio`を
変えても症状は完全に同一 (この800MiB要求はKV/MoEキャッシュ計算より前の
固定バッファ確保で、フラグの影響を受けない)。

**結論: このモデル (Qwen3.8-Flash-Next, tie_word_embeddings=false, vocab=248320,
hidden=2560, 48層) の非MoE(dense)パラメータだけで bf16 換算 約10-11GB超あり、
単体RTX3060 12GBには収まらない。TPでの2GPU分割はカーネル制約
(`triton: TP > 1 is not supported for this expert format`, sm_86では
marlin/b12xも使えない) で不可能。よってFreeToken + このNVFP4チェックポイントの
組み合わせでは、本機 (2x RTX3060 12GB, NVLink無し) で"サーバーを起動する"
ところまでも到達できない。これはチューニング不足ではなくハード/チェックポイント
のミスマッチ。**

## 方針転換: llama.cpp (GGUF) へのピボット (2026-09-15)

ユーザーから「FreeTokenがggufじゃないと聞いたが好きに進めていい、
とにかくFlash Next Q4を速く動かせればOK」との指示を受けていたため、
FreeToken(NVFP4, bf16 dense)がこのVRAM制約下で物理的に起動不能と判明した
時点で、GGUF量子化 (dense層も含め全体を4bit系に圧縮でき、bf16のまま残る
NVFP4チェックポイントより非expertパラメータのVRAM footprintが大幅に小さくなる
はず) + llama.cpp へ舵を切る。

調査の結果:
- **`~/llama.cpp` (本家 ggml-org, clean checkout, 既にビルド済み)** に
  Qwen3.8-Flash-Next のネイティブサポートが既に入っている
  (`src/llama-model.h`: `LLM_TYPE_A3B, // Qwen3.8 Flash Next`)。
  `llama-server`, `llama-bench` ともにビルド済みバイナリあり。
- `~/llama-flashnext-other` (Inovello fork, `llama-cli`のみビルド済み) は
  本家に無い追加の expert cache / pinned-host loader / top-k radix
  フォールバックなどを含むが、検証環境は **2x RTX3090 (24GB x2) + DDR4-2133
  quad channel + Xeon E5-2696v4 x2** と、うちの2x RTX3060 12GBとはVRAM量が
  倍以上違う。`examples/flashnext-topk/README.md` に実測値あり:
  - MTP投機デコード込み, 119k depth prompt, 42リクエスト中央値:
    control(旧top-k fallback) **30.2-30.4 tok/s**, candidate(radix版) **33.1-33.7 tok/s**
    (約9-12%改善)。品質は240ペア中235同等、4件candidate優位、1件劣化。
  - 起動コマンド例 (`-ot 'ffn_(gate|up|down)_exps\.weight=CUDA_Host,...'` で
    expertテンソルだけを明示的にhost RAMへ固定し、他はGPU常駐させる
    llama.cpp方式) は、FreeTokenの「offload=全experts host RAM / dense常にGPU固定」
    という硬直した二択よりも柔軟で、我々の「dense常駐だけでVRAM使い切る」問題を
    テンソル単位で回避できる可能性が高い。
- 方針: まず本家 `~/llama.cpp` (ネイティブサポート済み、安全) で `-ot` を使い
  expertsをhost RAM/CPU、dense+attentionをGPU (2枚に分割) に配置する構成を試す。
  必要に応じてforkの追加最適化 (top-kフォールバック等) も後で移植/比較する。

## llama.cpp 初回実測 (2026-09-15)

`~/llama.cpp` (本家, ビルド済み, commit 8ea290247) + `unsloth/Qwen3.8-Flash-Next-GGUF`
の `UD-Q4_K_XL` (103.68 GiB, 176.94B params) で `llama-bench` を実行。

コマンド: `llama-bench -m <UD-Q4_K_XL-00001-of-00004.gguf> -ngl 99 -ncmoe 999 -p 128 -n 64 -t 12 -fa on`
(全MoE expertsをCPU、それ以外の層はGPUへ全offload試行, flash-attn on, 12スレッド)

| test | t/s |
|---|---|
| pp128 (prompt processing) | 30.90 ± 3.62 |
| **tg64 (decode, 本命の指標)** | **15.16 ± 0.06** |

**FreeTokenでは起動すらできなかったのに対し、llama.cppは実際に動作し実測できた。**
ただし現状15.16 tok/sで目標40 tok/sには届いていない。チューニング開始。
（見積りの参考: `~/llama-flashnext-other`のREADMEでは2x RTX3090 24GBx2 + DDR4-2133
quad channelでMTP投機デコード込み30.2-33.7 tok/sと報告されており、うちの環境は
VRAM半分・恐らくRAM帯域も異なるため、素のdecode性能はその数値より低くなりやすい。
MTP併用やチューニングでどこまで詰められるか検証する。)

## llama.cpp チューニング1: スレッド数 / GPU-CPU expert配分 (2026-09-15)

**スレッド数スイープ** (`-ncmoe 999`, 全expertをCPU, decode tg64のみ):

| threads | tok/s |
|---|---|
| 12 (物理コア数) | 15.74 ± 0.12 |
| 16 | 15.54 ± 0.10 |
| 20 | 15.06 ± 0.17 |
| 24 (論理コア全部/SMT込み) | 10.37 ± 1.05 |

メモリ帯域律速のワークロードであることが明確に裏付けられた。SMTを使うと逆に悪化。
**→ 以後は `-t 12` (物理コア数) を既定値とする。**

**`-ncmoe` (先頭N層のexpertsをCPUに残す) スイープ** (`-t 12`, dense層は`-ngl 99`でGPU2枚に自動分散):

| n_cpu_moe (48層中) | GPU使用VRAM (GPU0/GPU1) | tok/s |
|---|---|---|
| 999 (全expertsをCPU) | 3589 MiB / 2795 MiB | 15.81 ± 0.07 |
| 44 (4層分のexpertsをGPUへ) | 3589 MiB / 9295 MiB | 16.57 ± 0.17 |
| 43 (5層分) | ほぼ同様 | 16.55 ± 0.18 |
| 42 (6層分) | — | **失敗 (モデルロード失敗, 恐らくCUDA OOM)** |

GPU1側に新規オフロード分が偏って積まれ、GPU0側の空きVRAM(8GB超)がほぼ未活用のまま
GPU1が先に埋まってVRAM不足になる。`-ncmoe`は「先頭N層をCPUに残す」という単純な
指定しかできず、どのGPUに割り振るかは制御できないため、これ以上は`-ot`で
layer/GPUを明示指定しないと伸びない。**現時点のベスト: `-ncmoe 44 -t 12` で16.57 tok/s**
(素のCPUフルオフロード15.7-15.8からの改善は+5%程度、目標40 tok/sにはまだ遠い)。
次はMTP投機デコード (`~/models/qwen-flash-next/MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf`
を使用) を試す — こちらの方が本命の大きなレバーになる見込み。

## FreeToken本体をdual-GPU対応に改造 (2026-09-15)

ユーザーから明確な指示: 「llama.cppに逃げるのではなく、FreeTokenのソースコードを改造して
デュアルGPUに対応させ、その上で自分の環境向けに最適化してほしい」。

### 採用した方法

ゼロから分散パイプライン並列(gloo point-to-point, CUDA graphをrank跨ぎで扱う仕組み,
per-rankキャッシュ予算解決など)を実装するのは非常にリスクが高い(バグりやすい分散処理を
一から書くことになる)と判断し、**yuuki-net/FreeToken-Kai** (Apache-2.0, 非公式フォーク,
upstream afd99cbに追従、我々のクローンの初期コミットと完全同一の基点)を参照実装として
`git remote add kai` -> `git merge kai/kai` で取り込んだ (2026-09-15, コミット `96cb1cb`)。
ユーザー許可を得た上で実施 (サーバー側の権限チェックで一度ブロックされ、明示的に承認を得た)。

**重要 (ユーザーからの指摘)**: Kaiの実測値 (2x RTX3060で18-20 tok/s) は目標ではなく、
あくまで「動くdual-GPU基盤」を借りただけ。Kaiを上回ることが目的で、Kaiの数値を鵜呑みに
しない。以後はそのための独自チューニングを積み重ねる。

Kaiが追加する機能 (`--pp-size`): レイヤー単位のパイプライン並列。1GPU=1プロセス、
NCCL/P2P不要 (gloo point-to-point)。rank0が埋め込み層、最終rankがlm_headを持つ。
これにより「dense重みが1枚のVRAMに収まらない」「NVFP4のtritonカーネルがTP>1非対応」
という2つの根本的な壁を同時に回避できる (PPはレイヤーをまるごと1GPUに割り当てるだけで、
層の中身を分割しないのでMoEカーネル側の変更が一切不要)。

新規C++拡張 (`_cpu_moe`, `_ple_store`など) を `python setup.py build_ext --inplace` で
追加ビルド。

### 起動成功

```
ft serve --model ~/models/qwen38-flash-next-nvfp4 --pp-size 2 --gpu 1,0 \
  --moe-strategy offload --text-model-only --memory-ratio 0.85 --max-running-requests 1
```

`pipeline rank 0/2: layers [0, 24) on cuda:1` / `pipeline rank 1/2: layers [24, 48) on cuda:0`
で正常に起動、両GPUとも約10.5GB使用 (dense重み+MoEオフロードキャッシュ1691slot/rank)。
**FreeTokenがこのモデルをついに起動できた。**

初回起動直後にリクエストを送ったところ `RuntimeError: gloo ... Timed out waiting 60000ms
for recv operation` でクラッシュ (Kai READMEが警告している既知のrank-relay起動直後レース
と同種の可能性)。サーバー再起動 + ready後に軽くsleepを挟んで再送したところ安定して応答。

### 実測 (サーバー自身のスケジューラログより, prose系プロンプト, reasoning=xhigh既定)

`--pp-size 2 --gpu 1,0` (24/24均等分割, dense-quant/kv-cache-dtypeともに未指定, デフォルト):

| decode step | gen throughput (tok/s) |
|---|---|
| step1 | 1.35 (ウォームアップ) |
| step2 | 15.32 |
| step3 | 16.53 |
| step4 | 15.49 |
| step5 | 14.48 |
| step6 | 15.83 |

**平均 ~15.5 tok/s。** llama.cpp単GPU版 (16.57 tok/s) とほぼ同等かやや下回る —
「動くようになった」だけでまだKaiの参考値(18-20)にも届いていない。ここからが本番。
`--pp-layers`によるPCIe非対称性を活かした分割、`--dense-quant fp8`、
`--kv-cache-dtype q4_0` (どちらもフリーになったVRAMをexpertキャッシュに回す設計)、
`--spec-mtp` を順に試す。

## `--dense-quant fp8` + `--kv-cache-dtype q4_0` (2026-09-15)

dense重み(attention/GDN/shared_expert/lm_head/embedding)をper-row fp8に量子化 (97個の
dense projectionを量子化, ログ確認済み) + KVキャッシュをq4_0量子化。空いたVRAMは自動的に
MoEオフロードキャッシュへ回る設計 (Kaiの設計思想通り)。

```
ft serve --model ~/models/qwen38-flash-next-nvfp4 --pp-size 2 --gpu 1,0 \
  --moe-strategy offload --text-model-only --dense-quant fp8 --kv-cache-dtype q4_0 \
  --memory-ratio 0.85 --max-running-requests 1
```

結果: `--moe-cache-auto resolved moe_cache_size=2491` (前回1691から **+47%**)。
512experts/層に対し2491slotsは約4.9倍の深さがあり、PCIeフェッチのミス率が大きく下がる。

### 実測 (サーバースケジューラログ, 同一prose系プロンプト)

| decode step | gen throughput (tok/s) |
|---|---|
| step2 | 22.46 |
| step3 | 22.63 |
| step4 | 22.11 |
| step5 | 19.79 |
| step6 | 23.40 |

**平均 ~22 tok/s。** ベースライン(~15.5 tok/s)から **+42%**。
**FreeToken-Kai自身が2x RTX3060で報告する参考値18-20 tok/sを既に上回った。**
(ユーザー指示通り、Kaiの数値は目標ではなく通過点として扱う。目標は40、理想60。)

次: `--pp-layers`でGPU0(遅いPCIe)/GPU1(速いPCIe)の非対称性を反映した分割を試す、
`--spec-mtp`でMTP投機デコードを試す。

## `--pp-layers` でPCIe非対称性を活かす (2026-09-15, 独自チューニング)

均等分割(24/24)ではなく、実測PCIe帯域 (GPU1=25.8GB/s >> GPU0=6.2GB/s, 本ログ上部参照)
を根拠に、速いGPU1(rank0)によりレイヤーを多く割り当てる非対称分割を試した。
これはFreeToken-Kaiのドキュメントには無い、本機固有のボトルネック分析に基づく調整。

```
--pp-size 2 --gpu 1,0 --pp-layers 30   # rank0(GPU1,速い)=layers[0,30), rank1(GPU0,遅い)=layers[30,48)
```

`--moe-cache-auto resolved moe_cache_size=2224` (rank0側。層が増えた分dense重みが増え、
均等分割時の2491よりは小さいが、rank1側は逆に大きくなっているはず)。

### 実測

| decode step | gen throughput (tok/s) |
|---|---|
| step2 | 23.41 |
| step3 | 24.93 |
| step4 | 24.93 |
| step5 | 22.75 |
| step6 | 25.33 |

**平均 ~24.3 tok/s。** 均等分割(~22 tok/s)から **さらに+10%**。
ベースライン(~15.5 tok/s)からは **+57%**。PCIe非対称性を考慮した分割が実際に効いている
ことを確認。さらに非対称にできるか (32や34など) 次に試す。

## 未検証 / 次にやること

- [ ] モデルダウンロード完了確認、チェックサム/欠損なしか確認
- [ ] `uv pip install -e ".[accel]"` の成功確認、`ft --version`
- [ ] `ft bench bw` でホストRAM<->PCIe帯域を計測し、offload/hybrid の推奨を得る
- [ ] `ft serve --model ~/models/qwen38-flash-next-nvfp4` 起動、ロード時のピークRSS/VRAM実測
- [ ] `--moe-strategy` を auto / offload / hybrid で比較
- [ ] `benchmarks/bench_decode_moe.py` で bs=1 decode tok/s を実測 (AIME-25 prompt)
- [ ] sm_86 (RTX 3060, Ampere) 向けに Marlin W4A16 NVFP4 経路が有効か確認
      (`pyproject.toml` の注記: Marlin は `vllm>=0.14,<0.15` 依存で core の
      `transformers>=5.5` 要件と衝突するため accel extra には含まれない。
      必要なら別途検証する)
- [ ] 40 tok/s / 60 tok/s 目標に対する実測値との比較、ボトルネック分析

## 実測値

(まだベンチマーク未実施。実行後にここへ追記する。捏造しない。)
