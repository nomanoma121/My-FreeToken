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
slots = 4 × max_running_req               (what running requests need: live + 2 ping-pong + 1 committed)
      + max(4, ratio × max_running_req)   (the cache of snapshots other prompts can resume from)
      + 1                                 (padding)
```

With the usual single-user setting (`--max-running-req 1`) and the default ratio of 2.0, the
cache part is **4 snapshots**. That, not KV, is what bounds how many conversations stay
reusable. `--linear-state-cache-ratio` sets it.

## Measured

Sixteen different 643-token prompts were sent once, then again in reverse order; the table
counts how many of the second round still hit the prefix cache (`#cached-token` non-zero)
before the first miss. `--max-running-req 1` throughout, KV usage 0% in every run.

| Machine and model | `--linear-state-cache-ratio` | GDN state slots | Conversations kept |
|---|---|---|---|
| RTX 2060 6 GB, Ornith-1.5-35B-A3B-NVFP4 | 2 (default) | 8 | **2** |
| | 8 | 12 | **4** |
| | 16 | 20 | **8** |
| RTX 3060 12 GB × 2, Qwen3.8-Flash-Next-NVFP4 (`--pp-size 2`) | 2 (default) | 8 | **5** |
| | 8 | 12 | **7** |

**Raising the ratio keeps more conversations on both, but not by the same rule.** On Ornith the
count doubles with the cache (a conversation there costs two snapshots); on Flash-Next the
default already keeps five and ratio 8 adds two. How many snapshots a conversation costs
depends on where the model can checkpoint — the hits above landed at 640 tokens on Ornith and
at 512 on Flash-Next — so **do not carry one model's numbers to another**: measure yours
(below).

## What it costs

Each extra slot is one GDN state of VRAM — every GDN layer's recurrent and conv state — and it
comes out of the expert cache. On the RTX 2060 with Ornith (30 GDN layers × 32 heads × 128 × 128
× fp32 ≈ 62 MiB per state) that is about 36 expert slots per GDN slot: 925 → 780 expert slots at
ratio 8, 585 at ratio 16 (that run also halved `--kv-reserve-tokens` to fit).

From a routing trace of the same model on the same machine, the expert cache's hit rate falls
roughly from 72% at 925 slots to 66% at 780 and 57% at 585, which that machine's cost model puts
at about **−6% decode for ratio 8** and **−14% for ratio 16**. These are estimates read off the
measured hit-rate curve, not measured decode speeds. On a larger card the same slot is a smaller
share of what the expert cache has to work with, so the trade is cheaper.

## When to change it

- **You switch between several long documents or conversations** and the second visit to one
  re-prefills from scratch: raise the ratio, then check with the log (below) that the number
  you keep in play now fits.
- **You mostly continue one conversation**: leave it at the default. Continuing a conversation
  hits, and the VRAM is worth more to the expert cache.

## How to tell on your machine

The decode log prints the pool as `#mamba-slot: used/total`, and each prefill prints
`#cached-token`. If a prompt you sent earlier comes back with `#cached-token: 0` while
`token usage` is low, the snapshot pool is the limit, not KV.
