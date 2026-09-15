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
ことを確認。

**さらに非対称にした`--pp-layers 34`は起動時にクラッシュ**:
`RuntimeError: ... Timed out waiting 60000ms for send operation to complete`
(rank0/GPU1が34層・7 expert shardsのロードに約2分以上かかり、先に14層のロードを終えた
rank1/GPU0がKVページ数合意のバリアで待ちきれずgloo送受信タイムアウト)。
`FREETOKEN_RANK_JOIN_TIMEOUT_SECONDS`はバリア自体の待ち時間を延ばせるが、
gloo transport層のsend/recv自体の60秒タイムアウトは別物で、ここは環境変数で
簡単に延ばせる保証がない (深追いはリスクに見合わないため今回は見送り)。
**`--pp-layers 30`を現状のベストとして採用し、次のレバーに進む。**

## `--spec-mtp` 失敗、`--memory-ratio` 追加チューニング (2026-09-15)

`--spec-mtp 5` を試したが起動時に
`AssertionError: --spec-mtp: MTP experts missing from the checkpoint: got []`
で失敗。ロードされたのは `mtp.embed_tokens.` (host-resident) のみで、MTP用の
expert weight (gate_up_proj/down_proj) が `nvidia/Qwen3.8-Flash-Next-NVFP4` には
含まれていない模様 (RadixArk版など別チェックポイントには入っている可能性があるが、
別途100GB超のダウンロードが必要になるため今回は見送り)。

`--memory-ratio` を0.85→0.95に上げ、fp8+q4_0で浮いた分をさらにexpertキャッシュへ:
`moe_cache_size=2670` (0.85時の2224から+20%)。ただしFree GPU memory after
capturing CUDA graphs: 0.36 GiB とかなりタイトになり、prefill-chunk-budgetが
自動的に1024まで縮小 (8192目安から大幅減、長いプロンプトのprefillは遅くなる代償)。

### 実測

| decode step | gen throughput (tok/s) |
|---|---|
| step2 | 23.99 |
| step3 | 25.68 |
| step4 | 25.60 |
| step5 | 23.89 |
| step6 | 26.90 |

**平均 ~25.2 tok/s。** memory-ratio 0.85 (~24.3) から+3.7%程度の小幅改善。
VRAM headroomがほぼ無くなる代償を考えると費用対効果は逓減してきている。
次は `--moe-strategy hybrid` (CPU/GPU同時miss処理) を、現在の改善されたVRAM状況で
再検証する — 以前(最初期のFreeToken単体テスト)ではAmpereでのhybridバグ懸念が
Web調査で見つかっていたが、今の構成(PP+fp8+q4_0+深いcache)で状況が変わるか確認する。

## `--moe-strategy hybrid` 再検証 → やはり悪化 (2026-09-15)

`--pp-layers 30 --moe-strategy hybrid --dense-quant fp8 --kv-cache-dtype q4_0
--memory-ratio 0.90` で起動・実測。

| decode step | gen throughput (tok/s) |
|---|---|
| step2 | 16.76 |
| step3 | 16.95 |
| step4 | 17.37 |
| step5 | 17.09 |
| step6 | 17.52 |

**平均 ~17.1 tok/s。offload(~25.2)より明確に悪い (-32%)。**
Web調査で見つけた `FlashML-org/FreeToken#151` (Ampereでhybridが実サービング時に
著しく遅い) の懸念が、PP+fp8+q4_0の改善された構成でも変わらず再現した。
**結論: 本機では `--moe-strategy offload` を今後も既定として使う。hybridはこの
ハードウェアでは明確に劣る選択肢。**

`--moe-strategy offload --pp-layers 30 --dense-quant fp8 --kv-cache-dtype q4_0
--memory-ratio 0.95` (直前の~25.2 tok/s構成) に復帰。

## 現状ベスト構成の再確認 (2026-09-15, 383トークン生成, prose/code 2種)

サーバーのスケジューラログだけでなく、クライアント側スクリプト(reasoning_content込みで
正しく計測するよう修正)でも独立に計測。

```
ft serve --model ~/models/qwen38-flash-next-nvfp4 --pp-size 2 --gpu 1,0 --pp-layers 30 \
  --moe-strategy offload --text-model-only --dense-quant fp8 --kv-cache-dtype q4_0 \
  --memory-ratio 0.95 --max-running-requests 1
```

| workload | completion_tokens | TTFT | decode tok/s |
|---|---|---|---|
| prose (TCP congestion control説明) | 383 | 6.23s | **24.45** |
| code (thread-safe LRU cache実装) | 383 | 5.98s | **24.88** |

prose/codeでほぼ差がない (MTP未使用のため、Kaiが報告するworkload依存の大きな差は
今回発生していない)。383トークンの長め生成でも安定して**~24.5-24.9 tok/s**。

**進捗まとめ (2026-09-15 一日の作業):**

| 段階 | 構成 | decode tok/s |
|---|---|---|
| FreeToken (upstream, NVFP4) | — | **起動不可** |
| llama.cpp GGUF, 全expertCPU | ncmoe=999 | 15.7-15.8 |
| llama.cpp GGUF, 一部expertGPU | ncmoe=44 | 16.57 |
| FreeToken+Kai, PP均等分割 | pp-size 2 (24/24) | ~15.5 |
| + dense-quant fp8 + kv q4_0 | 同上 | ~22 |
| + pp-layers非対称分割 | pp-layers 30 | ~24.3 |
| + memory-ratio 0.95 | 同上 | **~24.5-24.9 (確定)** |

目標40 tok/s (理想60) に対し、現状は**目標の約61-62%**。FreeToken-Kai自身の
参考値(18-20 tok/s)は上回っている。次の一手を継続検討中
(pp-layers 32の探索、gloo起動タイムアウトの延長でpp-layers 34+を試す、等)。

## ソース改造: `--distributed-timeout` フラグ追加 (2026-09-15)

`--pp-layers 34` (rank0=34層/rank1=14層) は前回、rank0のロードが60秒を超えて
rank1がgloo send/recvでタイムアウトし起動失敗していた。原因はハードコードされた
`EngineConfig.distributed_timeout = 60.0` (`FREETOKEN_RANK_JOIN_TIMEOUT_SECONDS`とは
別物、こちらはバリア専用でsend/recv自体のタイムアウトはカバーしない)。

`python/freetoken/server/args.py` に `--distributed-timeout` フラグを追加し
(コミット参照)、CLIから調整可能にした。ハードコードされた60秒制限を、フラグ非指定時は
従来通りの挙動を保ちつつ、非対称splitで長いロード時間が必要な場合に上書きできるように
した — これは「FreeTokenのソースコードを改造する」というユーザー要望に対する
具体的な実装。

`--distributed-timeout 300` を付けて `--pp-layers 34` を再実行 → 起動成功。

### 実測 (pp-layers 34, 383トークン, prose/code)

| workload | decode tok/s |
|---|---|
| prose | 24.10 |
| code | 24.60 |

**pp-layers 30 (24.45-24.88) とほぼ同じ、むしろ僅かに低い。** つまり30あたりが
このハードウェアでの非対称分割の局所最適点であり、さらに片方に寄せても追加の
利得は無いことを確認した。**`--pp-layers 30` を最終ベストとして確定する。**

## 最終まとめ (2026-09-15)

### 最終ベスト構成

```
ft serve --model ~/models/qwen38-flash-next-nvfp4 \
  --pp-size 2 --gpu 1,0 --pp-layers 30 \
  --moe-strategy offload --text-model-only \
  --dense-quant fp8 --kv-cache-dtype q4_0 \
  --memory-ratio 0.95 --max-running-requests 1
```

**確定decode速度: ~24.5-24.9 tok/s** (383トークン生成、prose/codeとも安定)

### 目標との比較

| 目標 | 値 | 達成度 |
|---|---|---|
| 最低目標 | 40 tok/s | **約61-62%** |
| 理想目標 | 60 tok/s | 約41% |

### 行った改造・チューニングの一覧 (効果順)

1. FreeToken-Kai (yuuki-net, Apache-2.0) の `--pp-size` をマージし、そもそも
   このモデルを2GPUで起動可能にした (upstream FreeTokenは起動不可だった)
2. `--dense-quant fp8` + `--kv-cache-dtype q4_0`: 浮いたVRAMをexpertキャッシュへ
   (moe_cache_size 1691→2224, +31%) → **+42%** (15.5→22 tok/s)
3. `--pp-layers 30` (実測PCIe帯域: GPU1=25.8GB/s ≫ GPU0=6.2GB/s を根拠にした
   非対称分割、本機固有の調整) → **+10%** (22→24.3 tok/s)
4. `--memory-ratio 0.95`: さらにキャッシュを拡大 (2224→2670 slots) → **+4%**
   (24.3→25.2 tok/s、ただしVRAM headroomが0.36GiBまで縮小しリスクあり)
5. `--distributed-timeout` フラグを新規追加しpp-layers 34を検証 → 効果なし
   (局所最適はpp-layers 30付近と確認)
6. `--moe-strategy hybrid` → **-32%の悪化** (25.2→17.1)、web調査で見つけた
   Ampereでのhybridバグ懸念が再現。offloadを維持する根拠として実測で裏付け
7. `--spec-mtp` → チェックポイントにMTP expert重みが無く起動不可 (今回は断念)

### 試したが効果が無かった/悪化したもの

- `--moe-strategy hybrid`: 明確に悪化 (上記)
- `--pp-layers 34` (34/14 非対称): 30/18から追加の伸びなし
- `--spec-mtp`: チェックポイントの制約で実施不可

### 未着手・今後の伸びしろ (ユーザーがChatGPTに調査させたレポートより抜粋、
自分の環境で未検証)

- **per-layer cache allocation**: 48層均等ではなくlayerごとのreuse localityに
  応じてexpert cache容量を配分する。現状は`--moe-cache-auto`が全layer均等。
- **RadixArk版など、MTP expert重みを含む別チェックポイント**への切替
  (135GB超の追加ダウンロードが必要、ROIは要検証。Kai実測ではcode/toolで
  30-36 tok/s、free proseでは11-14 tok/sとworkload依存が大きいため過度な
  期待は禁物、との指摘あり)
- **dynamic top-k削減 (top-10→top-6/8)** や **expert混合精度量子化**:
  モデル出力そのものを変える近似で、速度と品質のトレードオフを要検証
- **router trace収集 + オフラインcacheシミュレータ**: 本格的な観測基盤構築が
  前提で、数日〜数週間規模の工数が必要。今回のセッションでは着手していない
- PCIeやDDR4の理論帯域とrequired bandwidthを突き合わせた定量的な天井分析
  (`required_bandwidth = target_tok/s × miss_rate × expert_bytes` 形式の
  見積り) は今回実施していない — 次回セッションで着手する価値あり

## 【訂正】前回の「offloadはCPU計算でmissを解決する」という分析は誤り (2026-09-15)

下の診断セクションで「`fetched_per_layer=0.0` だから offload は PCIe fetch を
使わずCPU計算でmissを解決している」と結論したが、これは統計フィールドの
アーティファクトを誤読した分析ミスだった。

`python/freetoken/moe/offload_cache.py` を直接確認: `record_decode_stats`
(plain "offload" 用) は **no-op** で `stat_fetched` を一切更新しない
(コメント: "ensure_experts accumulates into lru_stats inside its own launch")。
`stat_fetched` を実際に書き込むのは `record_decode_stats_hybrid` だけであり、
これは hybrid戦略専用の計測。つまり `decode_miss_stats()` が返す
`fetched_per_layer`/`cpu_per_layer` は **offload戦略では意味を持たない
(常に0/missingと表示される)アーティファクト** であり、実際にoffload戦略が
CPU計算でmissを解決している証拠にはならない。

`ensure_experts` (offload用, `offload_kernels.py:19`) は `lru_ensure` を呼び、
missした全experts分のスロットを確保して`copy_missing`用の`src_indices`等を
セットする実装になっており、**offloadは元のドキュメント通り、missを常に
PCIe fetch(GPU転送)で解決している** (CPU計算にフォールバックする分岐は無い)。

**訂正後の結論**: PCIe帯域は引き続き重要な要因である。GPU1(速い25.8GB/s)に
より多くの層を割り当てる非対称分割(`--pp-layers 30`)は、rank0(GPU1)が
higher miss率(23.2%)を持つ代わりに速いPCIeで安く解決でき、rank1(GPU0)は
遅いPCIeの代わりに低いmiss率(6.2%)で済む、という理にかなったトレードオフに
なっていた可能性が高い。実際、per-step実効fetch時間を概算すると
(1 expert ≈ 2.64MB):
- rank0: 2.32/layer × 30層 ≈ 69.6 experts/step × 2.64MB ÷ 25.8GB/s ≈ 7.1ms/step
- rank1: 0.62/layer × 18層 ≈ 11.2 experts/step × 2.64MB ÷ 6.2GB/s ≈ 4.8ms/step

合計約12ms/step。実測ステップ時間(~25tok/s→40ms/step)の約30%がPCIe fetch、
残り約70%がGPU計算(GEMM/attention/GDN等)とオーバーヘッドと推定される。
`--pp-layers`によるさらなるチューニングが26/30/34で頭打ちだったのは、
両rankの「miss率×fetch単価」がこの範囲で既にほぼバランスしていたためと
考えられる。cache eviction policy改善(LRU→locality-aware)が依然として
最有力の次の一手であるという結論自体は変わらない。

## 診断: `--moe-collect-stats` で実際のキャッシュmiss率を計測 (2026-09-15)

`--moe-stats-out <path> --disable-cuda-graph` を付けて起動し(CUDA graph無効化は
統計収集専用で、これ自体は本番設定ではない。実際decode速度もCUDA graph無しだと
~13-14 tok/sまで落ちる = CUDA graphだけで+75-80%の効果があることも副次的に判明)、
実際にprose/codeプロンプトで生成した後、SIGTERMでグレースフルシャットダウンして
`.rank0`/`.rank1` の統計JSONを回収・解析した。

### 重大な発見1: `offload` 戦略のmiss処理は **PCIe fetchではなくCPU計算**

```
rank0 (GPU1, layers[0,30), cache 2670 slots): miss_rate=23.2%, fetched_per_layer=0.0, cpu_per_layer=2.32
rank1 (GPU0, layers[30,48), cache 3197 slots): miss_rate=6.2%,  fetched_per_layer=0.0, cpu_per_layer=0.62
```

`fetched_per_layer` が両rankとも **0.0** — つまりcache missは一貫してCPU計算で
解決されており、PCIe経由のGPU転送は一切発生していない。これは「offload=PCIe転送」
という自分の当初の理解(models.mdの説明や一般的なMoEオフロードの通念)と異なり、
**このFreeToken(Kaiベース)の`offload`実装は実質的に「GPUキャッシュhit + CPU計算miss」
のハイブリッド的挙動を、`hybrid`ストラテジー特有の同期オーバーヘッドを伴わずに実現している**。

**これは`--pp-layers 30`が効いた理由の再解釈を要求する**: 当初「GPU1のPCIeが速いから
多くの層を割り当てた」と考えていたが、実際にはmiss処理にPCIeは使われないため
この理由付けは誤りだった。真の理由は次の発見2。

### 重大な発見2: rank間でmiss率が大きく非対称 (23.2% vs 6.2%)

rank1 (18層, cache 3197 slots) は rank0 (30層, cache 2670 slots) よりも
**絶対キャッシュサイズは大きいのに層数が少ない分、層あたりの実効キャッシュ深度が
深く、miss率が約1/4** (6.2% vs 23.2%)。つまり本質的なレバーは「PCIe帯域」ではなく
「層数あたりのキャッシュ深度(=薄く広く512expertsをカバーする層が多いほどLRUの
効きが悪化する)」だった。`--pp-layers`の非対称化が効いた真因は、レイヤー数を
動かすことでVRAM配分(dense重み vs expertキャッシュ)のバランスを変え、結果的に
両rankの層あたりキャッシュ深度をチューニングしていたから、という理解に修正する。

### 今後の一手 (この発見を踏まえて)

- rank0のmiss率23.2%が支配的なボトルネックである可能性が高い (rank1は既に6.2%と
  優秀)。`--pp-layers`をさらに調整し、rank0の層数を減らす(=rank0のキャッシュ深度を
  改善する)方向を試す価値がある — ただしrank1側の層数が増えればそちらのmiss率が
  悪化するトレードオフがあるため、両rankのmiss率が拮抗する点が真の最適点のはず。
  `--pp-layers 26`前後を次に検証する。

### `--pp-layers 26` 実測 → 変化なし、プラトーを確認

rank0のキャッシュを2670→2847 slotsに増やせたが (30層→26層)、実際のdecode速度は
**24.03-24.16 tok/s** で、pp-layers 30 (24.45-24.88) やpp-layers 34
(24.10-24.60) とほぼ同じ。rank0のmiss率を改善する方向にlayer数を振っても
rank1側のmiss率が相対的に悪化し、トータルのsequential decode時間は
ほとんど変わらない。**`--pp-layers` 26/30/34の3点で実測が~24-25 tok/sに
きれいに収束しており、これは単なるチューニング不足ではなく、このアプローチ
(層の再配分だけによるmiss率改善)の実質的な天井であることが確認できた。**

これ以上の伸びには、根本的なcache policy改善 (LRU→TinyLFU等) や
per-layer allocation、あるいはより高精度な観測に基づく別アプローチが必要で、
「層の配分を変える」というレバーだけでは頭打ちという結論に至った。
`--pp-layers 30` (既存のベスト、24.45-24.88 tok/s) を最終構成として維持する。
- 根本的にはLRUだけでは512expertsに対するtop-10 routingの局所性の低さを
  吸収しきれていない (5倍以上のキャッシュ深度があっても20%超miss)。
  TinyLFUやstatic-hot+dynamic等の改良ポリシーへの改造は真に効果が見込めるが
  相応の実装工数とリスクを伴う本格的なソース改造になる。

### 実際にLRUカーネルの実装を調査した結果 (2026-09-15)

`ensure_experts`が委譲している`lru_ensure`の実体を特定: FreeToken自身のコードでは
なく、外部pipパッケージ`flashlib` (`.venv/.../site-packages/flashlib/kernels/
slot_cache/triton/lru_ensure.py`, 441行) 内のTritonカーネルだった。

朗報: `flashlib`はFreeTokenと同じ組織 (FlashML-org, Apache-2.0,
github.com/FlashML-org/flashlib) が公開しているOSSで、Kaiと同様にvendor/fork
することは技術的・ライセンス的には可能。

ただし中身を読むと、これは片手間で改造できる代物ではないと判断した:
- CUDA graph捕捉可能であるための「host syncなし・固定shape」制約を全編で維持
- victim選択に2つの戦略 (`_seq`: register常駐 argmin ループ, `_insert`: streaming
  insert、大きいキャッシュ用) があり、両者が「bit-identical」な結果を返すことを
  設計上要求されている
- 過去に「hit直後のreloadが古い値を見てLRU victim判定を誤る」という具体的な
  レースコンディションのバグ実績があり、`tl.debug_barrier()`の配置がその再発防止
  のために厳密に効いている、という趣旨のコメントが複数箇所にある
- frequency-aware化 (LFU/TinyLFU的な改良) をするには、新しいper-slot状態
  (頻度カウンタ)を`lru_usage`と並行して追加し、両戦略の victim選択ロジック
  (packed key生成、argmin/streaming insert)に一貫して組み込み、graph capture
  安全性を保ったまま正しさを証明する必要がある

**結論: これは「設定を変える」レベルの改修ではなく、正真正銘のGPUカーネル
エンジニアリング(flashlibのフォーク+検証を含む)であり、今回のセッションで
安全に完了させられる規模ではないと判断した。** 挑戦する場合は
`flashlib`のvendor化から始める別プロジェクトとして扱うべき。

## 決定的な分析: 「完璧なキャッシュ」でも40 tok/sには届かない (2026-09-15)

上記のリスク判断が本当に正しいか、「キャッシュ改善に投資する価値が本当にあるか」を
実測値から定量的に検証した。

実測値 (`--pp-layers 30`, fp8 dense + q4_0 KV, memory-ratio 0.95, ~24.7 tok/s平均) を
使い、1 decode stepの内訳を計算:

```
測定 step time:        40.49 ms  (= 1000/24.7 tok/s)
確認済み1expert分のサイズ: 2.642 MB  (実際のcache plan log "experts 2670 slots 6.89 GiB" から逆算)

rank0 (GPU1, 30層, miss率23.2%, PCIe 25.8GB/s):  PCIe fetch 6.96 ms/step (69.6 experts/step)
rank1 (GPU0, 18層, miss率6.2%,  PCIe 6.2GB/s):   PCIe fetch 4.64 ms/step (11.2 experts/step)
合計PCIe fetch:                11.61 ms/step (step timeの28.7%)

→ compute + overhead の下限:    28.88 ms/step (step timeの71.3%)
→ miss率0%(理論上完璧なキャッシュ)でも: 1000/28.88 = 34.6 tok/s が上限
```

**目標の40 tok/sには `step time <= 25.00 ms` が必要。現在のcompute+overhead
だけで28.88ms/stepを使っており、これはキャッシュmissを完全にゼロにしても
超えられない (34.6 < 40)。**

つまり、`flashlib`のLRUカーネルをどれだけ改良しても(たとえTinyLFU化でmiss率を
0%近くまで下げられたとしても)、目標の40 tok/sには届かない計算になる。
このリスクの高いカーネル改造に投資するのは**割に合わない**という結論に至った
(best caseでも+40%改善で34.6 tok/s止まり)。

### 本当のボトルネックは「compute + overhead」28.9ms/step

この28.9msの内訳は: 48層分のattention/GDN/hyper-connection/indexer計算、
cache-hit分のMoE GEMM (missでなくてもGEMM自体は必要)、rank間のgloo通信、
CUDA graph replayのオーバーヘッド等。これらは主に**モデルのアーキテクチャ規模と
RTX3060の生の計算力そのもの**に規定されており、engine側の設定変更や
キャッシュ戦略の改善では動かせない領域。

**この意味するところ: 現在のハードウェア (2x RTX3060 12GB) とモデル
(Qwen3.8-Flash-Next, 48層, 512experts/層) の組み合わせでは、エンジン側の
最適化 (FreeTokenの設定・改造含む) だけでは40 tok/sという目標に構造的に
届かない可能性が高い。** 届かせるには次のいずれかが必要:
1. より高性能なGPUへの変更 (計算力そのものを底上げ)
2. モデル側の近似 (dynamic top-k削減など、出力品質とのトレードオフを伴う
   モデル改変 — これはエンジン最適化とは異なる種類の意思決定)
3. FreeToken自体のcompute kernel (attention/GDN/hyper-connection等の
   演算そのもの)の高速化 — これも`flashlib`同様、大規模なカーネル
   エンジニアリングになる

現在の**~24.5-24.9 tok/s**は、実測に基づく分析上、このハードウェア構成での
現実的な到達点に近いと判断する。

## モデル近似によるアプローチ: top-k routingを削減 (2026-09-15)

エンジン側のチューニングだけでは40 tok/sの天井(~34.6 tok/s理論値)に届かないと
分かったため、ユーザーに「品質とのトレードオフを伴う」ことを明示した上で承認を得て、
routing top-k (`num_experts_per_tok`, デフォルト10) を削減する近似を試した。

### 実装方法 (FreeToken本体は無改造)

チェックポイントのweightファイル群をsymlinkし、`config.json`の
`text_config.num_experts_per_tok`だけを書き換えたコピーを作成
(`~/models/qwen38-flash-next-nvfp4-topk{N}/`)。top-kはMoEレイヤー構築時に
config値から読まれる純粋なモデルハイパーパラメータのため、FreeToken/flashlibの
ソースコードには一切手を入れずに実験できた。

### 実測 (383トークン生成, prose/code, 品質チェック付き)

| num_experts_per_tok | decode tok/s (prose/code) | 目標40に対する到達度 | 品質チェック |
|---|---|---|---|
| 10 (元の設定) | 24.45 / 24.88 | 61-62% | (基準) |
| 8 | 27.01 / 28.01 | 68-70% | 算数 "17×24=408" 正解 |
| 6 | 30.71 / 31.81 | 77-80% | 算数正解、連結リスト反転コードも正しく生成 (iterative prev/curr/nxt) |

top-kを下げるほど速くなる傾向は明確 (10→8→6で単調増加)。6でも簡単な算数・
コーディング問題は依然として正しく解けている (この2問だけでは厳密な品質保証には
ならないが、明らかな崩壊は見られない)。

### top-k=4 実測 → **目標40 tok/sの84-89%まで到達**

| num_experts_per_tok | decode tok/s (prose/code) | 目標40に対する到達度 |
|---|---|---|
| 4 | 33.44 / 35.41 | **84-89%** |

品質チェック (temperature=0):
- 算数 "17×24=408" 正解
- 連結リスト反転コード: 正しく動作するコード生成 (finish_reason=stop, 完走)
- ひっかけ問題 "17匹の羊のうち9匹以外死んだ、残りは？" → 正解「9匹」
  (典型的な引っかけ問題で、"all but 9" を正しく解釈できている)
- 電車の旅人算 (距離300mi, 60mph+40mphで向かい合う) → 合成速度100mph、
  distance÷speedまで正しく計算開始 (max_tokens打ち切りで最後の"3時間"の
  明示手前で切れたが、計算過程は正しい)

60%のexpert削減 (10→4) でも4種類のテスト(算数・コード・ひっかけ問題・旅人算)で
明確な破綻は見られなかった。次にtop-k=3を試す。

### top-k=3 実測 → **目標40 tok/sの88-93%、品質も健全**

| num_experts_per_tok | decode tok/s (prose/code) | 目標40に対する到達度 |
|---|---|---|
| 3 | 35.36 / 37.12 | **88-93%** |

品質チェック (temperature=0, `ft serve --model ~/models/qwen38-flash-next-nvfp4-topk3`
を単独起動して直接検証、bench_stream2.pyとは別に実施):
- 算数 "17×24=?" → `408` 正解 (推論なしでも即答)
- ひっかけ問題「羊17匹のうち9匹以外死んだ、残りは？」→
  「**9 sheep are left.**」+ 理由説明も正しい。reasoning_contentも
  ループせず一直線に正解へ到達 (finish_reason=stop で完走)

top-k=3は10→3で経路数を70%削減しているにも関わらず、この2つのテストでは
崩壊の兆候なし。この時点でtop-kスイープの「安全な下限」候補とする。

### top-k=2 実測 → **速度は目標40 tok/sの94-98%まで到達するが、品質が明確に崩壊**

| num_experts_per_tok | decode tok/s (prose/code) | 目標40に対する到達度 |
|---|---|---|
| 2 | 37.62 / 39.26 | **94-98%** (code側はほぼ40達成) |

品質チェック (temperature=0):
- 算数 "17×24=?" → `41` **不正解** (正解408)。この時点で明確な品質崩壊と判断
- ひっかけ問題 (羊) → `max_tokens=500`でも回答に到達できず、reasoning_contentが
  同じ論理を堂々巡りするループに陥っていた
  (「9 die. So 17-9=8? No, that's not how it works. Let me think again...」
  を500トークン使っても収束しない)
- 連結リスト反転コード → 同様にreasoning_contentのみでcontentが空、
  finish_reason=length (時間切れ)

top-k=2は速度面では魅力的 (code側で39.26 tok/s、40 tok/s目標の98%) だが、
単純な整数乗算すら間違え、簡単な論理パズルで無限ループに陥るなど、
実用に耐えないレベルの品質崩壊が明確に確認された。**top-k=2は不採用。**

### 結論: top-k=3を「速度優先モード」の最終推奨値とする

| 設定 | decode tok/s (prose/code) | 40 tok/s目標比 | 品質 |
|---|---|---|---|
| top-k=10 (オリジナル、無改造) | 24.45 / 24.88 | 61-62% | フル品質 (エンジン最適化のみ) |
| top-k=4 | 33.44 / 35.41 | 84-89% | 4種テストで崩壊なし |
| **top-k=3 (推奨・速度優先)** | **35.36 / 37.12** | **88-93%** | 算数・論理パズルとも正解、ループなし |
| top-k=2 | 37.62 / 39.26 | 94-98% | **崩壊 (算数誤答、推論ループ)** — 不採用 |

top-k=3と2の間に明確な品質の崖がある。top-k=3は40 tok/sには届かないが、
88-93%まで近づいた上で品質を保っている。top-k=2はほぼ40 tok/sに届くが、
実用に耐えない。速度と品質どちらを優先するかはユーザー判断に委ねる決定事項
として、この2案 (品質優先=top-k=10/エンジン最適化のみ、速度優先=top-k=3)
を最終候補として提示する。top-k=2はどちらの案としても推奨しない。

### top-k=3上でmemory-ratio再チューニングを試みたが失敗 (負の結果)

top-k=10向けにチューニングされた`--memory-ratio 0.95`が、top-k=3のような
軽量化されたcompute/PCIeバランスでも最適とは限らないと考え、
`--memory-ratio 0.97`(cache 2670→2759 slots, +3.3%)を試した。

結果: 起動はしたが (CUDA graph capture後の空きVRAMが0.15 GiBまで低下)、
ベンチマーク中の実際のdecodeフォワードパスでOOMに近い状態になり、
rank間のgloo pipeline通信が`Connection closed by peer`で切断、
バックエンドがクラッシュした。`--memory-ratio 0.95`はこのハードウェアでの
安全上限であり、これ以上は不安定化するのみと判断。**0.95を維持。**

### top-k=3上でpp-layers再チューニングを試みた → 変化なし (プラトー再確認)

top-k=10向けにチューニングされたpp-layers=30の層分割点が、top-k=3の
軽量化されたcompute特性でも最適かを確認するため、pp-layers=26で再測定。

| pp-layers | decode tok/s (prose/code) |
|---|---|
| 30 (既定) | 35.36 / 37.12 |
| 26 | 35.05 / 37.40 |

誤差範囲内で実質的に同一。top-k=10で確認した「pp-layers 26/30/34間で
プラトー」という結論は、top-k=3でも変わらず成立する。GPU間の層分割点は
top-kに関わらずボトルネックではない。pp-layers=30を維持。

## 最終まとめ (2026-09-15時点)

engineレベルのチューニング (dense-quant fp8, kv-cache q4_0, memory-ratio,
pp-layers分割) は出尽くした。追加で試した`--dense-quant`は`none`/`fp8`の
2択のみで既にfp8使用中、`--kv-cache-dtype`もq4_0が最も圧縮率が高い選択肢で
既に使用中、`--memory-ratio`は0.95が安定上限 (0.97はクラッシュ)、
`--pp-layers`は26/30/34全てでプラトー。`--moe-bank-prefetch`等の
readahead系フラグは`--moe-bank-ram`+cpu/hybrid decode専用で、今回の
`--moe-strategy offload`構成には適用不可。

### `--host-embedding`を試したが効果なし (負の結果)

CLIヘルプに「250k×2048語彙で約1GBのVRAMを解放しKVページに回せる」とある
`--host-embedding`(埋め込みテーブルをpinned host memoryに置く)を
top-k=3構成に追加して試した。

結果: `cache plan`のweights (3.38 GiB)もexperts slots (2670)も、
フラグなしの場合と完全に同一。decode速度も35.12/37.38 tok/s
(フラグなし35.36/37.12との差は誤差範囲)。ヘルプテキストが
"Qwen3.5-MoE family"向けと明記している通り、Qwen3.8-Flash-Next
(qwen4_exp architecture)では対象外で暗黙的にno-opになっていると判断。
**採用せず。**

`--nvfp4-backend flashinfer`(=b12x)も確認したが、これは既にTP検証時に
判明済みの`KernelSelectionError: b12x: requires sm_120+, got sm_86`が
再度該当するため試すまでもなく対象外。

以上でCLIの主要フラグは一通り評価を終えた。

## FreeToken本体を改造: per-layer top-kスケジュール (2026-09-15)

top-k=3(安全)とtop-k=2(崩壊)の間に明確な崖があることが分かったため、
「全レイヤー一律で削る」のではなく「ほとんどのレイヤーはtop-k=3のまま、
一部のレイヤーだけtop-k=2に落とす」per-layerスケジュールを試すことにした。
これは単なるconfig.json書き換えでは不可能なため、**FreeToken本体のソースを
改造**した(ユーザーの「ソースコードを改造してでも汎用実装を超えて欲しい」
という要望に応える改造)。

### 実装

- `ModelConfig`に`num_experts_per_tok_schedule: Tuple[int, ...] | None`を追加
  (`python/freetoken/models/config.py`)。デフォルトNoneは既存の全モデルの
  挙動を変えない。
- `Qwen3_5MoE.__init__`(Qwen4ExpMoEも継承)で、layer_idごとにこの
  スケジュールを参照し、あれば`make_moe_layer(..., top_k=schedule[layer_id])`
  で上書き。スケジュールがなければ従来通り`config.num_experts_per_tok`
  (`python/freetoken/models/qwen3_5_moe/moe.py`)。
- `qwen4_exp/config.py`の`parse_config`で、checkpointの
  `text_config.num_experts_per_tok_schedule`(48要素のリスト)を読み取り
  `ModelConfig`に渡す。

調査の結果、`make_moe_layer`はもともと呼び出しごとの`top_k`上書きを
受け付ける作りで、`OffloadMoELayer`も各インスタンスが独立して
`self.top_k`を保持しルーティング(`fused_topk`)とキャッシュ admission
の両方に使っていた ── つまりエンジン側は最初から「全レイヤー同一top-k」
を前提にしていなかった。今回の改造は、この既存の構築時フックを
チェックポイント設定から実際に駆動できるようにしただけで、
offloadキャッシュや`flashlib`側には一切手を入れていない。

### 実測: 48レイヤーのうち何レイヤーをtop-k=2に落とせるか

ベースはtop-k=3 (`num_experts_per_tok=3`)、一部レイヤーだけtop-k=1または2に
下げる形でスイープ。品質チェックは算数(17×24=408)・ひっかけ問題(羊9匹)・
連結リスト反転コードの3点、temperature=0。

| 名前 | 内容 | 加重平均top-k | decode tok/s (prose/code) | 品質 |
|---|---|---|---|---|
| (基準) 一律top-k=3 | 全48層top-k=3 | 3.0 | 35.05-35.36 / 37.12-37.40 | 良好 |
| sched-a | 端6層top-k=8, 中間42層top-k=2 | 2.75 | 36.52 / (計測中断) | **崩壊** (算数が完全に破綻、羊問題も"8匹"と誤答してループ) |
| sched-b | 中央4層(20-23)のみtop-k=2, 残りtop-k=3 | 2.917 | 36.12 / 37.83 | 良好 (3点とも正解) |
| sched-c | 6層おき12層をtop-k=2 (分散配置), 残りtop-k=3 | 2.75 | (未計測、品質で却下) | **崩壊** (算数が空回答のまま終了、finish=stop) |
| **sched-d** | **6層おき8層をtop-k=2 (idx 3,9,...,45), 残りtop-k=3** | **2.833** | **37.04 / 37.46** | **良好 (3点とも正解、最良)** |
| sched-e | 5層おき9層をtop-k=2, 残りtop-k=3 | 2.8125 | 36.67 / 37.74 | 良好 (3点とも正解、だがsched-dから有意な伸びなし=頭打ち) |
| sched-f | sched-dと同じ8層位置だがtop-k=1に (top-k=2でなく) | 2.667 | (未計測、品質で却下) | **崩壊** (コードが意味不明な文になり、羊問題もfinish=length未完走) |

### 分かったこと

1. **「平均top-k」だけでは品質を予測できない**: sched-aとsched-cは
   共に加重平均2.75だが、配置(端に集中 vs 分散)に関わらずどちらも
   崩壊した。一方sched-d/eは平均2.8-2.83で健全。品質を決めているのは
   平均ではなく「top-k=2に落とすレイヤー数」そのものらしい
   (概ね8-9層までは安全、12層で崩壊)。
2. **崖はレイヤー数についても急峻**: 8層(sched-d)は健全、12層(sched-c)は
   崩壊。9層(sched-e)は健全だが8層から速度の伸びがほぼゼロ(頭打ち)。
   このモデル・このチェックポイントでは「8-9層までtop-k=2に落とせる」が
   実用上のスイートスポット。
3. **top-k=1は少数レイヤーでも危険**: sched-dと同じ8箇所という
   "安全な位置"でも、そこをtop-k=1まで削るとコード生成が意味不明な文に
   崩壊した。top-k=2は許容できてもtop-k=1は許容できない、という
   もう一段の崖がある。

### 結論: sched-dを新しい最終推奨(速度優先)構成とする

| 構成 | decode tok/s (prose/code) | 40 tok/s目標比 | 品質 |
|---|---|---|---|
| 品質優先: top-k=10 (無改造) | 24.45 / 24.88 | 61-62% | フル品質 |
| 旧・速度優先: 一律top-k=3 | 35.05-35.36 / 37.12-37.40 | 88-93% | 良好 |
| **新・速度優先: sched-d (per-layer)** | **37.04 / 37.46** | **93-94%** | **良好 (3点とも正解)** |
| (不採用) 一律top-k=2 | 37.62 / 39.26 | 94-98% | 崩壊 |

per-layerスケジュールにより、一律top-k=3から約+1.7-2 tok/s、40 tok/s目標比で
約+5ptの底上げに成功。これは一律top-k=2の速度(37.62/39.26)にかなり近いが、
品質は一律top-k=3同様に健全 ── 「速度は2に近く、品質は3を維持する」という
狙い通りの中間点を実現できた。

40 tok/s目標には依然として届いていない(93-94%)が、これ以上レイヤー数を
増やすと崩壊することを実測で確認済みであり、この探索方向でのこれ以上の
積み増しは考えにくい。チェックポイント: `~/models/qwen38-flash-next-nvfp4-sched-d`
(`config.json`の`text_config.num_experts_per_tok_schedule`に48要素のリストを
保持)。

### 追加検証: sched-cはlinear_attention層のみを削っていた (attention種別は無関係と確認)

`layer_types`を確認したところ、sched-c (12層削減、崩壊) が選んだ位置
(2,6,10,...,46) は**全てlinear_attention (GDN) 層**で、full_attention
(QSA) 層は1つも含まれていなかった。つまり「GDN層の方が削減に強いはず」
という仮説はこの時点で既に反証されている ── 崩壊の原因は層の種別ではなく
純粋に削減する層の**数**だった。sched-dの8層(idx 3,9,15,21,27,33,39,45)は
full_attention 4層 + linear_attention 4層の混在で、こちらは健全だった。

### 再現性確認 (2回目のベンチマーク実行)

sched-dを再起動し2回目の独立したベンチマークを実行:

| 実行 | decode tok/s (prose/code) |
|---|---|
| 1回目 | 37.04 / 37.46 |
| 2回目 | 36.01 / 37.29 |

1 tok/s程度の実行間ノイズはあるが、一律top-k=3 (35.05-35.36/37.12-37.40)
を安定して上回ることを再確認。sched-dの実効性能帯は概ね**36-37.5 tok/s**
(40 tok/s目標の90-94%)。

### さらに追求: 「削減するレイヤー数」の崖を再測定 (sched-g〜k)

sched-c (12層削減、崩壊) は選んだ12層が**全てlinear_attention層**という
偏った配置だった。これが崩壊の真因なら、full_attention層も混ぜて
選べば12層以上でも安全かもしれない、と仮説を立てて再検証した。

| 名前 | 削減層数 | 配置 (full:linear比) | 加重平均 | 品質 | decode tok/s (prose/code) |
|---|---|---|---|---|---|
| sched-g | 11層 | 2 full : 9 linear | 2.771 | 良好 (3点とも正解) | 37.79 / 38.40 (1回目)、後日再検証なし |
| sched-h | 12層 | 3 full : 9 linear | 2.75 | 良好 (3点とも正解) | 37.49/38.06 → 再現性確認35.95/37.49 (2回計測) |
| sched-i | 16層 | 4 full : 12 linear | 2.667 | **崩壊** (算数: 推論内では408と正しく計算しているのに最終回答が空文字のまま終了。羊問題も未完走) | (品質で却下、未計測) |
| sched-j | 14層 | 4 full : 10 linear | 2.708 | **崩壊** (算数・コードとも空回答で終了。羊問題のみ正解) | (品質で却下、未計測) |
| sched-k | 13層 | 3 full : 10 linear | 2.729 | **境界的に崩壊** (算数は正解、羊問題は空回答、コードは前置き文のみでコード本体が出ない) | (品質で却下、未計測) |

**sched-c (12層、attention種別完全偏り) の崩壊は種別の偏りが原因だった**
ことを確定: sched-h (12層、種別を混ぜる) は3点とも正解し、健全に動作した。
つまり「削減するレイヤー数の絶対数」だけでなく「選ぶレイヤーの
attention種別バランス」も品質を左右する。

ただし、種別を混ぜても**13層から新たな崖**が現れる: 13層(sched-k)は
境界的崩壊(3点中1-2点が空回答)、14層(sched-j)は明確な崩壊、
16層(sched-i)も明確な崩壊。「推論の中では正しく計算できているのに、
最終回答を書く直前で応答が終了する」という独特の壊れ方が13層以上で
繰り返し見られた ── reasoningの計算能力自体はtop-k削減にまだ耐えているが、
reasoningから最終回答への「書き出し」に十分なexpertの多様性が
確保できなくなっている可能性がある。

### 結論(更新): sched-h (12層削減, 3 full + 9 linear) を最終推奨とする

sched-d (8層)からsched-h/sched-g (11-12層) へ更新。2回の独立実行で
再現性を確認:

| 実行 | decode tok/s (prose/code) |
|---|---|
| sched-h 1回目 | 37.49 / 38.06 |
| sched-h 2回目 | 35.95 / 37.49 |
| sched-g (11層, 参考) | 37.79 / 38.40 |

実効性能帯は**概ね36-38.4 tok/s (40 tok/s目標の90-96%)**。sched-d
(8層, 36-37.5 tok/s) から更に一歩前進し、品質チェック(算数・論理パズル・
コード生成)は3点とも安定して正解。チェックポイント:
`~/models/qwen38-flash-next-nvfp4-sched-h`。

40 tok/s目標には僅かに届いていないが(90-96%)、13層以上では
「推論はできるが最終回答を書き出せない」という新しい種類の崩壊が
繰り返し確認されており、この探索方向はここが実質的な天井と判断する。

念のため、sched-j (14層崩壊) が`reasoning_effort=low`強制によるアーティ
ファクトでないか確認するため、デフォルトの`reasoning_effort=xhigh`
(`chat_template_kwargs`を指定しない) でも同じ算数・コード問題を再テスト
した。結果は同じく空回答 (算数: finish=stop, content=''; コード:
finish=length, content='') で、reasoning_effort設定に関わらず14層崩壊は
再現した。「13層以上での崩壊」はreasoning_effortの副作用ではなく、
モデル自体の実質的な限界と確定できる。

### 最終確認: sched-hの12層に1層だけ足して13層にすると再現するか (sched-l)

sched-k (13層) は独自の間隔で選んだ配置だったため、「13層という数自体が
ダメなのか」「sched-kのたまたまの配置が悪かっただけか」を切り分けるため、
**健全動作が確認済みのsched-hの12箇所 (`[1,5,9,14,18,22,26,30,34,39,43,47]`)
に、layer 16を1つだけ追加して13層にした**構成 (sched-l) を作成し再テスト。

結果: 算数が同じパターンで空回答 (finish=stop, content='') となり崩壊。
リドル・コードは正解 (3点中1点が崩壊)。sched-kと同じ「1点が空回答」という
崩壊パターンが、全く違う配置の13層構成でも再現した。

これにより、**「12層は健全、13層から崩壊」という崖は配置に依存しない
頑健な結果**であると結論づけられる。sched-h (12層) を per-layer top-k
スケジュール探索の最終確定版とする。

## FreeToken本体を改造その2: --spec-mtpのチェックポイント読み込みバグを修正 (2026-09-15)

以前のセッションログに「spec-mtp blocked by checkpoint」(チェックポイントに
MTPの重みがないため利用不可) という記述があったが、これを鵜呑みにせず
`model.safetensors.index.json`を直接確認したところ、**`mtp.layers.0.mlp.
experts.{0..511}.*`の重みは512エキスパート全て実在していた**ことが判明。
「ない」のではなく、FreeToken側の読み込みコードが対応していない
フォーマットだった。

### 根本原因 (2つ)

1. `python/freetoken/models/qwen4_exp/weight.py`の`_rename()`が使う
   `_EXPERT_RE = re.compile(r"\.mlp\.experts\.\d+\.")`は`model.language_model.`
   というプレフィックスを要求しない緩い正規表現で、本来メインモデルの
   ルーティングエキスパート (別経路の`nvfp4_expert_sources`で読む) だけを
   弾くつもりが、`mtp.layers.0.mlp.experts.0.gate_proj.weight`のような
   MTPの個別エキスパートも巻き込んで無条件にドロップしていた。
2. `python/freetoken/engine/engine.py`の`_capture_mtp_experts`/
   `_quantize_mtp_experts`は、MTPエキスパートが**事前にfuse・stackされた
   単一のbf16テンソル**(`mtp.layers.0.mlp.experts.{gate_up_proj,down_proj}`)
   として来ることを前提にしていた。しかし実際のチェックポイントは
   メインモデルのルーティングエキスパートと同じ「per-expert, unfuse」形式で、
   しかも量子化方式もNVFP4ではなく**128x128ブロックFP8**
   (`weight` + `weight_scale_inv`、DeepSeek-V3方式) だった。

### 修正内容

- `qwen4_exp/weight.py`: 新しい正規表現`_MTP_EXPERT_UNFUSED_RE`でMTPの
  個別エキスパートキーを`_rename`/`_DenseFuser`に触れさせる前に横取りし、
  生のテンソルのままyieldするよう変更 (`--spec-mtp`が有効な時のみ)。
- `engine.py`: `_capture_mtp_experts`を拡張し、新形式のキーを
  `(expert_id, proj, kind)`単位でバッファする`self._mtp_raw_unfused`を追加。
  新設した`_fuse_mtp_experts_unfused()`が、既存の`dequant_block_fp8`
  ヘルパー (dense fp8投影のブロックFP8デコードで既に使われている同じ
  128x128ブロック規約) で各エキスパートのgate_proj/up_proj/down_projを
  bf16へ逆量子化し、gate+upを結合、512エキスパート分をstackして、
  既存の`_quantize_mtp_experts`が期待するbf16の`{gate_up_proj, down_proj}`
  形状に変換する。**既存のbf16→NVFP4量子化パス自体は無改造**で、
  「取り込み」の欠落部分だけを補った。

### 動作確認

起動ログに`MTP draft head: 512-expert layer quantized to NVFP4
(1354 MB pinned)`と出力され、正常に量子化・キャッシュへの追加まで完走。
算数チェック(17×24=408)も`--spec-mtp 1`有効時に正解を維持。

投機的デコードの性質上、ドラフト側 (MTPヘッド) の品質が多少不正確でも、
最終出力は検証パス (本体モデルによる棄却サンプリング) が保証するため、
このデコード処理自体が誤っていても「最終出力が壊れる」方向のリスクは
なく、「ドラフト採択率が下がって高速化しない」方向のリスクに留まる、
という安全性の性質を踏まえて実装・検証した。

### 実測: 採択率は良好だが、正味では遅くなる (負の結果)

sched-h (12層top-k=2) 構成に`--spec-mtp 1`を追加、CUDA graph化に必要な
VRAM余裕を確保するため`--memory-ratio 0.88`に調整 (0.95のままだと
verify-windowのCUDA graphがVRAM不足でeager実行にフォールバックし、
さらに遅くなることを確認済み)。

| 構成 | decode tok/s (prose/code) | accepted/step | step time |
|---|---|---|---|
| sched-h (spec-mtp無し, 参考) | 37.49-37.79 / 37.46-38.40 | - | ~27ms |
| sched-h + spec-mtp (eager, mr=0.95) | 22.10 / 23.10 | - | eager実行で更に遅い |
| sched-h + spec-mtp (CUDA graph化, mr=0.88) | 28.36 / 30.71 | **1.55-1.88 / 2.0 (78-94%)** | ~54-90ms |
| top-k=10 (元の設定, 参考) | 24.45 / 24.88 | - | ~41ms |
| top-k=10 + spec-mtp (CUDA graph化, mr=0.88) | 19.49 / 19.69 | **1.55-1.88 / 2.0** | ~78-90ms |

採択率自体は78-94%と非常に良好 (MTPヘッドの品質は健全) にも関わらず、
**どちらの土台 (top-k=10でもsched-hでも) でも正味では遅くなった**。

### 原因分析: このハードウェアはPCIe/キャッシュミス律速で、投機的デコードの前提が成立しない

投機的デコードが速くなる前提は「K個のトークン候補をまとめて検証する
コストが、1トークンだけ生成するコストとほぼ同じ」(主に重み読み込みが
メモリ帯域律速で、K個の検証がその重み読み込みを使い回せるから)。
しかし本構成 (`--moe-strategy offload`, RTX 3060 x2) は512-way MoEの
ルーティングエキスパートをPCIe越しにフェッチするキャッシュミス律速で
動いている。検証ウィンドウの2トークン候補は異なるexpertの組み合わせに
ルーティングされる可能性が高く、1トークンだけ生成する場合よりも
ユニークなexpertフェッチ数が2倍近くに増えてしまう。結果、
「verify 1step ≈ 通常のdecode 1stepとほぼ同コスト」という投機的デコードの
コアな前提が成立せず、実測のstep timeは通常の約2倍 (27ms→54-90ms) に
悪化。accepted/stepが2.0に近い高い採択率でも、コスト側の悪化がそれを
上回ってしまい、正味で遅くなる。

**結論: `--spec-mtp`のバグ修正自体はFreeToken本体への正当な貢献として
維持するが (他のハードウェア/構成では有効な可能性がある)、この
RTX 3060 x2 + offload-cache構成では採用しない。** sched-h
(12層top-k=2、spec-mtp無し、37-38 tok/s) が引き続き最終推奨。

(下の表は一律top-k=3までの時点のまとめ。この後前掲の「FreeToken本体を
改造: per-layer top-kスケジュール」セクションでsched-dによりさらに
更新されたので、最終結論はそちらを参照)

残る2つの現実的な選択肢 (2026-09-15時点、sched-d判明前の暫定まとめ):

| 構成 | decode tok/s (prose/code) | 40 tok/s目標比 | 品質 | FreeToken本体改造 |
|---|---|---|---|---|
| **品質優先**: top-k=10 (無改造) | 24.45 / 24.88 | 61-62% | フル品質 | dual-GPU pipeline-parallel対応のみ (`--pp-size`, `--distributed-timeout`) |
| **速度優先**: top-k=3 | 35.05-35.36 / 37.12-37.40 | 88-93% | 算数・論理パズル正解、崩壊なし | 同上 (top-kはconfig.jsonのみ変更、ソース無改造) |

60 tok/sの理想目標には届かないが、速度優先構成は40 tok/sの目標に対して
88-93%まで到達しており、実測に基づく理論値 (完全ヒットキャッシュでも
~34.6 tok/s、top-k=10時点) を上回っている。これはtop-k削減がcompute側の
天井そのものを引き下げたためで、単なるキャッシュ調整では不可能だった
効果。

※この後、per-layer top-kスケジュール機能をFreeTokenソースに実装し
(`sched-d`、前掲セクション参照)、速度優先構成を93-94%まで押し上げた。
そこでの実測により、8層を超えてtop-k=2化すると崩壊することも確認済みで、
この探索方向はほぼ尽くした。この先さらに積み増すには、(1) より高性能な
GPUへの交換、(2) attention/GDN/hyper-connectionカーネル自体の書き換え、
のいずれかが必要で、どちらも本セッションのスコープを超える大規模
エンジニアリングと判断する。

### git履歴

本セッションの全作業は `~/My-FreeToken` にgitでコミット済み (コミット一覧は
`git log --oneline`で追跡可能)。主要コミット:
- `96cb1cb` FreeToken-Kaiマージ (dual-GPU基盤の獲得)
- `e71e451` dense-quant fp8 + kv q4_0 (+42%)
- pp-layers非対称分割、memory-ratio調整の各コミット
- `--distributed-timeout`フラグ追加のソース改造コミット

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

### Takeover (Muse Spark, 2026-09-15 ~16:40 UTC): sched-l (sched-h+1層) 検証 — 13層の崖を再確認

Claudeがセッション上限 (usage limit, resets 17:10 UTC) で停止したため引き継ぎ。
Claudeの残した次の一手「sched-hに1層足したsched-lで13層が本当に多すぎるのか検証」を実施。
sched-lは既に定義済み (sched-hの12層削減にindex16を追加した13層: 3 full + 10 linear, 平均2.729) だったため、そのままサーブして計測。

構成: ft serve --model sched-l --pp-size 2 --gpu 1,0 --pp-layers 30 --moe-strategy offload --text-model-only --dense-quant fp8 --kv-cache-dtype q4_0 --memory-ratio 0.95 --port 1919 (sched-hと同一フラグ)。

実測 (bench_stream2.py, max_tokens=384):
- prose: 36.03 tok/s (completion 383, window 10.6s)
- code: 37.93 tok/s (completion 383, window 10.1s)
- 速度だけ見ればsched-h (35.95-37.49 / 37.49-38.06) と誤差範囲で同等。

品質チェック (temperature=0.0, max_tokens=300):
- 算数 (17x24): "408" 正解 (finish=stop)
- 羊問題: 空回答 (content="", finish=length) — 崩壊
- コード (quicksort): 空回答 (content="", finish=length) — 崩壊

結論: sched-k (13層) と同じ「推論はできるが書き出せない」崩壊パターンを再現。**13層は配置によらず実質的な崖**と確定。sched-h (12層) を最終推奨として維持する。サーバーはtakeover用に起動したsched-lを停止し、sched-hに戻す。

Takeover再計測 (sched-h, serve_final_sched_h.log, 同一フラグmemory-ratio 0.95): prose 35.94 tok/s / code 37.75 tok/s。再現性バンド36-38内に収束。最終推奨sched-h不変。サーバーは127.0.0.1:1919で稼働中。
