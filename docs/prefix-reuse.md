# Why switching conversations re-prefills everything on hybrid GDN models

On Qwen3.5-MoE and Qwen3.8-Flash-Next, the prefix cache can return a long conversation in a
fraction of a second — and then, after you have touched a few other conversations, stop
returning it at all, even though KV usage is near zero. This page explains why, and the one
flag that changes it.

## The limit is not KV

A hybrid model's attention layers keep KV, but its GDN layers keep a **recurrent state**, and
the next token can only be computed from a state that already has the whole prefix folded in.
So a prefix is reusable only if the cache still holds a **snapshot of that state** at a
boundary inside it. KV alone is not enough.

Snapshots live in a fixed pool of slots in VRAM:

```
slots = 4 × max_running_req          (what running requests need: live + 2 ping-pong + 1 committed)
      + max(4, ratio × max_running_req)   (the cache of snapshots other prompts can resume from)
      + 1                            (padding)
```

With the usual single-user setting (`--max-running-req 1`) and the default ratio of 2.0, the
cache part is **4 snapshots**. That, not KV, is how many reusable prefixes the server keeps.

## Measured

RTX 2060 6 GB, Ornith-1.5-35B-A3B-NVFP4, `--max-running-req 1`. Sixteen different 643-token
prompts were sent once, then again in reverse order; the table counts how many of the second
round still hit the prefix cache (`#cached-token` non-zero) before the first miss.

| `--linear-state-cache-ratio` | GDN state slots | Conversations kept | Expert cache slots | KV used |
|---|---|---|---|---|
| 2 (default) | 8 | **2** | 925 | 0% |
| 8 | 12 | **4** | 780 | 0% |
| 16 | 20 | **8** | 585 ¹ | 0% |

¹ This run also had `--kv-reserve-tokens` halved to fit, which gave back about 100 expert
slots; at equal KV it would be about 490.

Two things to read from it:

- **Conversations kept = cached snapshots ÷ 2** at every point. Each request leaves two
  snapshots behind: one where its prompt ends (so the same prompt can be asked again) and one
  where its reply ends (so the next turn of that conversation resumes there). A prompt long
  enough to be prefilled in several chunks also leaves one at each chunk boundary, so it takes
  more of the cache than these short prompts did.
- **Each extra slot costs one GDN state of VRAM**, which the expert cache gives up: about
  36 expert slots per GDN slot on this model (one state is every GDN layer's recurrent and
  conv state — 30 layers × 32 heads × 128 × 128 × fp32 ≈ 62 MiB here).

## What it costs in decode speed

Fewer expert slots means more expert misses. From a routing trace of the same model on the
same machine, the expert cache's hit rate falls roughly from 72% at 925 slots to 66% at 780
and 57% at 585, which this machine's cost model puts at about **−6% decode for ratio 8** and
**−14% for ratio 16**. These two figures are estimates read off the measured hit-rate curve,
not measured decode speeds.

On a larger card the trade is cheaper: a GDN slot is the same size, but it is a smaller share
of the VRAM the expert cache has to work with.

## When to change it

- **You switch between several long documents or conversations** and the second visit to one
  re-prefills from scratch: raise the ratio until the number of conversations you keep in
  play fits. With `--max-running-req 1`, conversations kept ≈ max(4, ratio) ÷ 2 for short
  prompts, fewer for prompts long enough to be prefilled in chunks.
- **You mostly continue one conversation**: leave it at the default. Continuing a conversation
  always hits, and the VRAM is worth more to the expert cache.

## How to tell on your machine

The decode log prints the pool as `#mamba-slot: used/total`, and each prefill prints
`#cached-token`. If a prompt you sent earlier comes back with `#cached-token: 0` while
`token usage` is low, the snapshot pool is the limit, not KV.
