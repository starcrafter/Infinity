# CLAUDE.md — Infinity-2B inference co-design (fork)

Guide for working in this fork. Fork of **FoundationVision/Infinity** →
**github.com/starcrafter/Infinity**, branch **`t4-a100-inference-codesign`**
(`upstream` = the official repo). All of our work lives in **`codesign/`** and is
**additive + flag-gated**; the default inference path is unchanged.

## Goal
Optimize **Infinity-2B** text-to-image **inference**. Deploy target = **T4**
(Turing sm_75, 16 GB, fp16-only, no FA2); benchmark = **A100 40 GB** (Ampere, bf16).
Infinity is a **visual-autoregressive (VAR)** model: it predicts the image **scale by
scale** (13 scales at 1024px), each scale a *prefill* over the cumulative KV — NOT a
token-by-token decode. So borrow **prefill/KV** tricks, not decode tricks (speculative
decoding etc. don't apply — VAR already emits a whole scale in parallel).

## Headline results (Infinity-2B, 1024px / pn=1M, cfg=4)
**A100 (bf16), all lossless, cumulative:**
`3.92 s (orig)` → `2.01 s` (T5 resident, −49%) → **`1.28 s`** (+ CUDA graphs, **−67% total**).
- **T5 resident** (`--t5_offload 0`): biggest single win; T5 encode 1898 ms → **32 ms**. Needs >16 GB → A100 only (T4 OOMs).
- **Manual CUDA graphs** (`--cuda_graph`): per-scale capture/replay, **−32%** on top, replay **byte-identical** to eager.
- **INT8 KV** (`--kv_int8`): peak mem ~halved (batch-2: 25.9→15.6 GB), throughput batch ceiling **2→4**, +4% img/s. Lossy (verify quality).

**T4 (fp16):** baseline 13.6 s. Lossless levers ≈ 0 (SDPA already optimal, compile no-op, T5-resident & CUDA-graphs OOM at 16 GB). The path to porting the A100 wins to T4 is **INT8 KV** (halves the persistent cache → should fit T5-resident + graphs; not yet retested).

**What does NOT work (don't retry):** mem-efficient SDPA backend (already default), `torch.compile(dynamic)` (no-op), `torch.compile(reduce-overhead)` auto-cudagraphs (residual/KV aliasing + cross-attn graph breaks), no-CFG (`--cfg 1`, lossy, ~−31% T4 but risky).

## `codesign/` toolkit
- **`t4_compat.py`** — SDPA `flash_attn` stub (the repo `import flash_attn`s unconditionally; FA2 doesn't build on Turing and has no torch-2.9/py3.10/cu12.9 prebuilt wheel). Provides a varlen cross-attn fallback that's **CUDA-graph-safe** (query offsets from shapes, key offsets cached by id — no `.item()` sync). Call `t4_compat.install()` BEFORE importing `infinity.models.*`. Unit-tested on CPU (`python t4_compat.py`).
- **`profile_infinity.py`** — device-aware profiler (bf16 A100 / fp16 T4). Per-stage (T5 / AR / decode) CUDA-event timers + per-scale times + optional torch.profiler. Flags: `--t5_offload --attn_backend --compile --static_kv --cuda_graph`.
- **`throughput.py`** — images/sec via static batching sweep + peak mem + OOM detection. `--kv_int8`.
- **`eval_prompts.py`** — the **5-prompt eval** (3 text-render + human + animal), Claude-judged (corrupted? text correct?). `--cuda_graph` does per-prompt `reset_cuda_graph()` + warmup→capture→replay.
- **`vae_recon.py`** — VAE encode→decode fidelity (tokenizer upper bound). f16 (`--vae_type 32`) vs f8 patchify (`--vae_type 14 --apply_spatial_patchify 1`); f8's 4× finer latent recovers small text.
- **`launch_t4.sh` / `teardown_t4.sh`** — SPOT GPU lifecycle (see workflow). `OPTIMIZATION.md` — full results + analysis.

## Model-code changes (`infinity/models/`, all flag-gated, default = original)
- **`basic.py` `SelfAttention`**: static write-at-offset KV buffer (`use_static_kv`, CUDA-graph prerequisite) + INT8 KV (`use_kv_int8`). `kv_caching(enable, static, max_len, preserve, int8)`.
- **`infinity.py`**: `_scale_blocks` / `_scale_blocks_graphed` (per-scale capture/replay), `reset_cuda_graph()`, gen-counter + cached cond/ca_kv (capture assumes **fixed prompt**), `use_cuda_graph`/`use_static_kv`/`use_kv_int8` attrs. **`add_lvl_embeding`**: build index on-device (`torch.full(device=)`) — a CPU→GPU copy is illegal during capture.

## CUDA-graph capture rules (learned the hard way)
During capture, **only pure stream-ordered GPU work is allowed** — NO host syncs:
no `.item()`/`int(tensor)`/`.cpu()`/`.to('cuda')` of a CPU tensor/`bool(tensor)`/data-dependent
branches; allocate everything with `device=`. Also: `PYTORCH_ALLOC_CONF=expandable_segments`
**conflicts with capture** (leave it unset for graph runs). The static-KV buffer is required
so the cache has a stable address; cond/ca_kv are cached (fixed-prompt assumption).

## GPU workflow (cheap, SPOT-only)
- **Custom image `infinity-2b-t4`** (project `project-e7987ca9-ebd3-438f-95f`) bakes deps + weights (2B + VAE + flan-t5-xl in `~/.cache/huggingface`) + the repo. Boot from it → **no download, no pip**; just `cd ~/Infinity && git pull`.
- `bash codesign/launch_t4.sh` (SPOT n1+T4) or provision A100 with `a2-highgpu-1g --provisioning-model=SPOT` from `--image=infinity-2b-t4`. **Always** `--instance-termination-action=DELETE --max-run-duration=3h` (self-delete safety).
- **ALWAYS `bash codesign/teardown_t4.sh <name> <zone>` when done** — quota is **1 GPU** (`GPUS_ALL_REGIONS`), shared with the **faro** project. The image doesn't need rebaking unless deps change (code stays current via `git pull`).
- Env: `/usr/bin/python3` (torch 2.9.1+cu129, **transformers pinned 4.44.2** — newer eagerly imports a broken torchaudio). Develop/syntax-check on the CPU dev box (`/home/bingyp/magic/.venv/bin/python`), benchmark on GPU.

## HARD CONSTRAINTS
- **NEVER touch the `faro` project / `faro-t4-stress` / the `gpu-opt-1` watcher.** Use the distinct name `infinity-{t4,a100}-bench`.
- **SPOT only** (never standard/on-demand).
- Always tear down GPU instances; verify `GPUS_ALL_REGIONS usage=0` after.

## Run (on a GPU box)
```bash
cd ~/Infinity && git pull
M=$(sed -n 1p weights/paths.txt); V=$(sed -n 2p weights/paths.txt)
COMMON="--model_path $M --vae_path $V --text_encoder_ckpt weights/flan-t5-xl --vae_type 32 \
  --pn 1M --model_type infinity_2b --cfg 4 --tau 0.5 --rope2d_each_sa_layer 1 \
  --rope2d_normalized_by_hw 2 --add_lvl_embeding_only_first_block 1 --use_bit_label 1 \
  --apply_spatial_patchify 0 --text_channels 2048 --bf16 1 --seed 1"
# fastest A100 config (lossless): T5 resident + CUDA graphs
python codesign/profile_infinity.py $COMMON --t5_offload 0 --cuda_graph 1 --runs 4
# throughput + INT8 KV
python codesign/throughput.py $COMMON --t5_offload 0 --kv_int8 1 --batches 1,2,4,8
# 5-prompt quality eval (do NOT pass --t5_offload here; eval_prompts has no such flag)
python codesign/eval_prompts.py $COMMON --cuda_graph 1 --tag opt --out_dir ~/eval_opt
```

## Next steps
1. Retest **INT8 KV on T4** → confirm it makes T5-resident + CUDA-graphs fit (port the A100 −67% to T4). Verify INT8 quality on the 5-prompt eval (wire `--kv_int8` into `eval_prompts.py` first).
2. Fused INT8-KV attention kernel (Triton) for a further memory/bandwidth win.
3. CFG-interval (partial guidance) — proper lossy speedup vs blanket no-CFG.
