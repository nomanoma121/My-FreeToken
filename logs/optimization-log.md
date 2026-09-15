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
