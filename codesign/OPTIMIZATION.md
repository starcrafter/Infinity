# Infinity-2B — Inference Co-Design Profile

Fork: **github.com/starcrafter/Infinity** (branch `t4-a100-inference-codesign`), `upstream` = FoundationVision/Infinity.
Goal: **Infinity-2B** text-to-image **inference latency**. Deploy **target = T4** (Turing sm_75, 16 GB, fp16-only, no bf16, no FA2); **benchmark = A100** (Ampere sm_80, bf16 + FA2).
Workflow: write/verify on the **CPU VM** (`codesign/`), benchmark on **A100**, optimize toward **T4**.

**One runner, both GPUs.** `codesign/profile_infinity.py` auto-selects **bf16** on A100 (sm≥80) and **fp16** on T4. `codesign/t4_compat.py` installs an SDPA `flash_attn` fallback only when real FA2 isn't importable — so it **no-ops on A100** (if FA2 present) and is **essential on T4**. The blockers in §3 only bite on T4.

---

## 1. Model facts (from `tools/run_infinity.py:316`)
`infinity_2b`: `depth=32, embed_dim=2048, num_heads=16 (head_dim=128), mlp_ratio=4, block_chunks=8`.
Each of the 32 layers is a **CrossAttnBlock** = self-attention (KV-cached across scales) + cross-attention to T5 text + FFN(4×).

## 2. Inference pipeline (the scale-by-scale VAR loop, `infinity.py:456`)
```
[A] T5 text-encode  (once)   512 tokens -> (kv) text features, fp16
        │
[B] AR transformer loop  over ~13 scales (1M / 1024px, final grid 64×64):
        for si in scale_schedule:                # next-scale prediction
            run all 32 blocks  (bs = 2  ← CFG cond+uncond batched together)
              self-attn over CUMULATIVE KV (KV cache grows each scale)
              cross-attn → T5 text   (varlen)
              FFN 4×
            get_logits → bitwise top-k/top-p sample
            VAE quantizer indices→codes ; F.interpolate residual to next scale
        │
[C] VAE decode  (once)   summed_codes → 1024×1024 image  (conv decoder)
```
Cumulative self-attn length at the last scale ≈ **5,461 tokens** (Σ over the pyramid), at **bs=2**.

---

## 3. T4 BLOCKERS — must fix just to run (handled in `t4/t4_compat.py`)
| # | Blocker | Where | Fix |
|---|---|---|---|
| 1 | Unconditional `from flash_attn import …` → ImportError on sm_75 | `basic.py:18-19` | `t4_compat.install()` injects an SDPA-based `flash_attn` stub **before** import |
| 2 | Cross-attn **hard-calls** `flash_attn_varlen_kvpacked_func` (no fallback) | `basic.py:396,399` | stub provides a varlen SDPA fallback — **unit-tested on CPU, max_err 5e-7** |
| 3 | bf16 everywhere (`autocast(dtype=bf16)`, `block.bfloat16()`) — Turing has no bf16 | `run_infinity.py:112,179,201` | run under **fp16** autocast; pass `--bf16 0` (runner does this) |
| 4 | self-attn already has SDPA fallback (`slow_attn`) when `customized_flash_attn=False` | `basic.py:308` | keep default (good) |
| 5 | `use_flex_attn` needs torch.compile flex_attention (sm_75 support poor) | — | keep `--use_flex_attn 0` |

## 4. Memory budget on 16 GB (why latency & memory interact here)
| item | fp16 | note |
|---|---|---|
| 2B weights | ~4.0 GB | |
| T5-XL encoder | ~6 GB | **offload after encoding** — otherwise it just squats on VRAM |
| BSQ-VAE | ~0.5–1 GB | |
| KV cache (bs=2, 32L, ~5461 tok, 2048d) | **~5–6 GB** | halves if CFG is removed |
| activations / workspace | ~1–2 GB | |

→ It fits, but **barely**. T5 offload is effectively mandatory; the single biggest memory *and* latency win is removing the CFG branch.

---

## 5. RANKED CO-DESIGN OPPORTUNITIES (latency)
Each is framed as model-insight ↔ infra-reality, the way the case study slide is.

**① CFG batch-doubling — the #1 lever.** Every scale runs cond+uncond at `bs=2` → ~2× the transformer FLOPs *and* 2× the KV cache (`infinity.py:485-501`).
- *Guidance distillation* — fold CFG into a single forward → up to **~2× on the dominant stage** + halves KV memory. (model change / retrain)
- *CFG interval* — apply CFG only on a subset of scales (`cfg_insertion_layer`/`cfg_list` already per-scale) → free, no retrain. Skip CFG on the cheap early scales and/or the last.

**② Launch overhead on the tiny early scales — Turing-specific.** ~13 scales × 32 blocks = **~416 block calls**, each firing many small kernels. The early scales (1…256 tokens) are **launch-bound, not compute-bound**; T4 kernels are slow but launch overhead is fixed.
- **CUDA graphs** or **torch.compile** the block → collapse per-kernel launch cost. Biggest relative win on the early scales.

**③ Attention backend.** Self-attn → torch SDPA; on sm_75 the FA2 backend is absent, so it falls to mem-efficient/math. The **last 2–3 scales** dominate attention (cumulative KV ≈ 5,461).
- Pin SDPA to the **mem-efficient** backend (xformers supports sm_75), not the math path; consider chunked/windowed attention at the tail.

**④ INT8 (W8A8) on the transformer — latency *and* memory.** T4 has INT8 tensor cores at **~2× fp16** (130 vs 65 TOPS) and it halves the 4 GB of weights.
- Per-channel weight quant + SmoothQuant-style activation handling; co-design with the fp16 KV cache.

**⑤ VAE decode (one-time).** Conv decoder to 1024². fp16 + **channels-last** convs; measure share with the profiler (hypothesis: 10–20%).

**⑥ Cumulative `F.interpolate` per scale.** `summed_codes += F.interpolate(codes, size=final)` upsamples to **full final resolution every scale** (`infinity.py:599`) → O(final²) redundant work early. Minor vs transformer but free to tighten.

**⑦ T5 encode / KV.** One-time encode → **offload encoder** (memory). Cache embeddings if prompts repeat. KV already fp16.

### Expected profile shape (hypothesis to confirm on T4)
AR transformer loop **60–80%** (of which last 2–3 scales dominate, and CFG ≈ half) · VAE decode **10–20%** · T5 encode small one-time · sampling/interp small.

### Top-3 bets for T4 latency
1. **Kill/halve CFG** (distill, or interval) — ~up to 2× on the dominant stage.
2. **CUDA graphs / compile** — kills the launch-bound early-scale overhead.
3. **INT8 transformer** — ~2× compute on the heavy tail + fits memory.

---

## 6. How to run (same command on A100 or T4)
```bash
pip install -r requirements.txt          # on T4: do NOT install flash_attn (shim handles it); on A100: FA2 optional
# weights: Infinity-2B (infinity_2b_reg.pth), BSQ-VAE (infinity_vae_d32reg.pth) from HF FoundationVision/Infinity; T5 = google/flan-t5-xl
python codesign/profile_infinity.py \
  --model_path <2b.pth> --vae_path <vae.pth> --text_encoder_ckpt google/flan-t5-xl \
  --vae_type 32 --pn 1M --model_type infinity_2b --bf16 0 \
  --prompt "a corgi astronaut, studio lighting" --runs 3 --profile 1
```
The runner auto-selects bf16 (A100) / fp16 (T4). Outputs: macro stage breakdown (T5 / AR / decode), **per-scale transformer ms**, a torch.profiler op table + `t4_trace.json`. **Benchmark on A100 first** to get the profile shape, then re-run on a T4 to see how the gap (no FA2, fp16, smaller HBM) shifts the bottleneck.

## 7. Status
- [x] Repo cloned, inference path mapped (this doc).
- [x] T4 compat shim written + **cross-attn fallback unit-tested on CPU** (`t4_compat.py`).
- [x] fp16 instrumented profiler written (`profile_infinity_t4.py`), syntax-checked.
- [ ] **Run on T4** → confirm the profile shape, then implement the top-3 levers in priority order.
