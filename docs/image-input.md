# Image input

Upstream FreeToken serves images on the Qwen VL families since `08d728d` (#454): OpenAI
`image_url`, Anthropic `image` blocks and Responses `input_image` parts, as an http(s) URL or
base64, with the flags in [cli.md](cli.md#image-input). This fork used to carry its own image
path; since the merge at `e0886cc` it runs upstream's, and adds what a small card needs on top.

```json
{"role": "user", "content": [
  {"type": "text", "text": "What is in this picture?"},
  {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0..."}}
]}
```

How upstream does it, in short: the tokenizer worker runs the checkpoint's image processor and
replaces each image's placeholder with a run of *content pad ids* (derived from a hash of the
pixels, so the radix cache reuses a prefix only when the image is the same) and precomputes the
3-axis `(t, h, w)` rope positions. The engine encodes the images right before the prefill chunk
that needs them and copies the embeddings into those rows. When a server builds the vision tower,
every token is roped on three axes, text included.

## What this fork adds

| | |
|---|---|
| `--mm-encoder-weights cpu` | The vision tower runs on the CPU, in the tokenizer worker, and costs no VRAM and no pinned memory (below) |
| No torchvision needed | transformers gates `AutoImageProcessor` on torchvision, whose build has to match torch exactly. Without it the checkpoint's processor is loaded as its PIL backend (`Qwen2VLImageProcessorPil` for the Qwen VL line) |
| `--pp-size` | Only the first rank builds, loads and runs the tower; the other ranks rope the image rows and admit or refuse image requests alike |
| `--spec-mtp` | The verify window, the draft head and their CUDA graphs rope on three axes; the draft head embeds the placeholder token where an image row's successor is a content pad id |
| `--prefill-mixer-pieces` | An image chunk splits like a text one |
| `--dense-quant fp8` | Leaves the vision tower bf16 (its blocks are streamed from host memory, so fp8 would save RAM, not VRAM) |
| Pin budget | With `--mm-encoder-weights host` the tower's pinned block bank counts against the pin quota the expert bank planner uses (WSL2 caps it) |

## `--mm-encoder-weights cpu`

Upstream's tower lives on the GPU. Even with its blocks streamed from host memory (`host`, the
default) it keeps the merger, the embeddings and two block-sized staging buffers resident, and
the block bank is pinned. On an RTX 2060 serving Ornith-1.5-35B-A3B at 64k that is 0.19 GiB more
weights and 0.77 GiB less pin budget: the daily configuration no longer fits at
`--memory-ratio 0.82`, and at 0.83 the expert cache drops from 358 to 270 slots with one more
layer decoding on the CPU.

With `cpu`, the tokenizer worker runs transformers' own vision module for the checkpoint in
float32, one image at a time, and sends the request on with each image's final embeddings
instead of its pixels. The engine builds no tower and loads no `visual.*` weights; it gathers the
embeddings into the prompt rows exactly as it does for a tower of its own, so the radix cache,
chunked prefill, `--pp-size` and `--spec-mtp` behave the same.

- One image is a few seconds of CPU time. The tower loads at the first image (about 9 s and
  1.7 GB of RAM on the 2060 host); a server that never sees an image never loads it.
- The last `FT_IMAGE_EMBED_CACHE` (default 32, `0` disables) images stay cached by the same
  content hash, since chat clients resend a conversation's images every turn and again for
  title generation.
- Supported: Qwen3.8-Flash-Next (`qwen4_exp`), the Qwen3.5-MoE family (Qwen3.6-35B-A3B,
  Ornith-1.5-35B-A3B) and dense Qwen3.5. A tower with DeepStack (Qwen3-VL proper) is refused at
  start.
- Use `--image-max-tokens` to bound the CPU time and the prompt length: the Qwen VL processor's
  own limit is 16384 tokens per image (a 4032x3024 photo becomes 11844 tokens). `256` keeps one
  image at 512x512 or less.

Measured on the RTX 2060 (Ornith-1.5, `--dtype float16`, 64k, `--image-max-tokens 256`): the
cache plan is the text-only one (1.70 GiB of weights, 358 expert slots, 21 layers on the CPU,
65,605 KV tokens), a colour probe (four solid colours, black text on white, left red / right
blue) answers 6/6, and a repeated image prompt reuses 128 of its 184 tokens from the prefix
cache. With `--spec-mtp 3` at 16k an image prompt is described correctly at 1.4-3.8 tokens
accepted per step.

## Not covered

- Video.
- DeepStack towers with `--mm-encoder-weights cpu`.
