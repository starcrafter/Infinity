# Infinity-2B Inference Optimization — Results Record

Model: **Infinity-2B**, 1024px (pn=1M), cfg=4, seed-fixed. All measured this session.
T4 = Tesla T4 (Turing sm_75, 16 GB, fp16). A100 = A100-SXM4-40 GB (Ampere, bf16).
"Lossless" = output byte-identical (or numerically equivalent) to the baseline.

## Baselines (single image, latency)
| GPU | config | latency | breakdown |
|---|---|---|---|
| T4 | fp16, T5 offloaded (only way it fits 16 GB) | **13.6 s** | T5 2.3 s / AR+decode 11.3 s |
| A100 | bf16, T5 offloaded | **3.92 s** | T5 1.90 s / AR+decode 2.02 s |
| A100 | bf16, T5 resident | **1.89 s** | T5 0.03 s / AR+decode 1.86 s |
| H100 | bf16, T5 resident | **0.97 s** | T5 0.015 s / AR+decode 0.96 s |

H100 is ~2× the A100 at every config (HBM3 + more SMs + faster clocks). The *fastest*
lossless single image measured anywhere is **H100 0.486 s** (graphs + compiled VAE, below).

---

## Techniques that WORK — latency

| # | Technique | Lossless | T4 | A100 | Notes |
|---|---|:---:|---|---|---|
| 1 | **T5 resident** (`--t5_offload 0`) | ✅ | OOM (16 GB) | **3.92 → 2.01 s (−49%)** | T5 encode 1898 → **32 ms**; the offload transfer was pure tax |
| 2 | **T5 dynamic padding** (`'longest'` not 512) | ✅ (bit-identical) | small | folds into #1 | runs T5 over real prompt len (~13 tok) not 512; encode → 32 ms |
| 3 | **Static KV buffer** (write-at-offset) | ✅ | OOM | 1966 → 1937 ms (−1.5%) | ~neutral; **prerequisite for CUDA graphs** |
| 4 | **Manual CUDA graphs** (`--cuda_graph`) | ✅ (byte-identical) | OOM (fp16) | **1887 → 1284 ms (−32%)** | per-scale capture/replay; kills per-block launch overhead |
| 5 | **No-CFG** (`--cfg 1`) | ❌ lossy | **13.6 → 9.3 s (−31%)** | 3.92 → 3.40 s (−13%) | bs 2→1; text held on easy prompts but risky (use CFG-*interval*) |
| 7 | **Compile VAE decoder** (`--compile_vae`) | ≈ (fp-reassoc, mean 0.5/255) | not retested | H100: **585 → 486 ms (−17%)** | `torch.compile(vae.decode)` fuses group_norm+silu+conv; decode ~170 → ~70 ms. Transformer output byte-identical → text unaffected. Should help A100 too (untested). |

### Cumulative lossless stack (A100): **3.92 s → 1.28 s = −67%**
`T5 resident (#1+#2)` → `static KV (#3)` → `CUDA graphs (#4)`. The fast recipe:
`--t5_offload 0 --cuda_graph 1` (static KV auto-enabled by the graph path).

### Cumulative lossless stack (H100): **0.97 s → 0.486 s = −50%**
`T5 resident (0.97 s)` → `+ CUDA graphs (0.585 s, −40%)` → `+ compiled VAE (0.486 s, −17%)`.
Fast recipe: `--t5_offload 0 --cuda_graph 1 --compile_vae 1`. Where the 486 ms goes (graphed):
late scales 10–11 (compute-bound, 1600/2304 tok, CFG bs=2) ≈ 240 ms, early scales 0–9
(graphed) ≈ 160 ms, VAE decode (compiled) ≈ 70 ms. The two early levers (graphs, VAE) attack
launch overhead; the remaining floor is the compute on the late scales (CFG-interval is the
only further lever there, and it's lossy).

---

## Techniques that WORK — throughput (A100, images/sec, static batching)
| KV dtype | max batch | best img/s | peak @ b1 | peak @ b2 | peak @ b4 |
|---|---|---|---|---|---|
| fp16 | 2 | 0.68 | 16.9 GB | 25.9 GB | OOM |
| **INT8** (`--kv_int8`) | **4** | **0.71 (+4%)** | 11.8 GB | 15.6 GB | 23.3 GB |

**INT8 KV (#6):** int8 buffer + per-token scale, dequant on read. **Halves the persistent
32-layer cache → peak ~halved, batch ceiling 2→4, +4% img/s.** Lossy (per-token quant);
quality not yet validated on the 5-prompt eval. **Key implication:** batch-1 INT8 peak
(11.8 GB) < 14.5 GB → should let the **T4** fit T5-resident + CUDA-graphs (which OOM in fp16)
→ the path to port the A100 −67% latency win down to the T4 (not yet retested).

### Throughput history (A100, all runs)
Throughput is measured **eager** (the static-batch path; CUDA graphs and batching are not
yet combined in the code). Re-confirmed run-to-run: peak-mem figures are byte-stable.
| date | KV | b1 img/s | b2 img/s | b4 img/s | b1 peak | b2 peak | b4 peak | ceiling |
|---|---|---|---|---|---|---|---|---|
| early | fp16 | 0.53 | **0.68** | OOM | 16.9 GB | 25.9 GB | — | b2 |
| early | INT8 | 0.48 | 0.62 | **0.71** | 11.8 GB | 15.6 GB | 23.3 GB | b4 |
| 2026-06-17 | fp16 | 0.53 | **0.68** | OOM | 16.9 GB | 25.9 GB | — | b2 |
| 2026-06-17 | INT8 | 0.48 | 0.62 | (run cut at b2) | 11.8 GB | 15.6 GB | — | ≥b2 |

**The serving-throughput punchline:** eager batching is *worse* than graphed single-stream,
because the model is **launch-bound on the small scales** (batching doesn't remove launches,
graphs do). Best **sustained img/s per A100 today = graphed single-stream = 1 / 1.288 s =
`0.776 img/s`.** Eager batch-2 (0.68) and INT8 batch-4 (0.71) both come in *under* it.
→ The biggest open serving win is **graphs + batching combined** (untested): a graphed batch-4
should roughly double per-GPU img/s and ~halve the fleet below.

### Serving capacity — how many A100s for a target QPS
At the measured best **0.776 img/s/A100** (graphed single-stream, bf16, 1024px, cfg=4),
`N = ceil(target_QPS / 0.776)`:
| target | 1 QPS | 2 QPS | 5 QPS | 10 QPS | 25 QPS | 50 QPS | 100 QPS |
|---|---|---|---|---|---|---|---|
| **#A100 (today)** | 2 | 3 | 7 | 13 | 33 | 65 | 129 |
| **#A100 (if graphs+batch4 lands, ~1.5 img/s)** | 1 | 2 | 4 | 7 | 17 | 34 | 67 |

Caveats: SPOT availability, T5 kept *resident* (offload would tax each request ~1.9 s),
and no request-batching server yet (numbers are per-GPU single-stream). A real serving stack
(continuous batching + graphs) is the path to the right-hand column.

---

## Techniques that do NOT work (null results — don't retry)
| Technique | Result | Why |
|---|---|---|
| **INT8 GEMM / W8A8** (`--int8_gemm`, torchao) | **A100 1.90 → 6.70 s (3.5× SLOWER)** | per-op dynamic quant/dequant overhead dwarfs the matmul; A100 bf16 tensor cores already saturate these small GEMMs (and CFG only doubles bs). Incompatible with CUDA graphs too. Lossy *and* slower → dead. |
| **FP8 GEMM / W8A8** (`--fp8_gemm`, torchao e4m3 rowwise) | **H100 eager 0.97 → 1.88 s; +graph 0.585 → 0.748 s — both SLOWER** | the H100 retry of the int8 idea. Native FP8 tensor cores *don't* help because the model is **launch/overhead-bound, not GEMM-compute-bound** and the per-scale GEMMs are small (CFG bs=2); torchao dynamic per-row quant adds per-call overhead that isn't fused without `torch.compile` (which we can't use — cross-attn graph-breaks). Same root cause as A100 int8. (Needed a filter to skip `mat_qkv`/`mat_kv`, which call raw `F.linear(weight=...)`.) |
| **VAE decode channels-last** (`--vae_channels_last`, NHWC) | **A100 1.288 → 1.352 s (+64 ms)** | the `contiguous(channels_last)` layout copy + cudnn picking a non-faster conv algo at this size costs more than any NHWC conv gain. Lossless but slower. |
| **cuDNN attention backend** (`--attn_backend cudnn`) | **H100 no change** (eager 970 vs 973; graph 583 vs 585) | attention is only ~16 % of the AR loop and the default SDPA already picks a Hopper flash kernel; forcing the cuDNN fused-attention backend changes nothing. Attention isn't the bottleneck at 2B/1024. |
| mem-efficient SDPA backend (`--attn_backend mem_efficient`) | no change | already the default (no FA2 on Turing; A100 SDPA picks FA2 itself) |
| `torch.compile(dynamic=True)` | no change | dynamic disables cudagraphs; graph breaks on varlen cross-attn |
| `torch.compile(reduce-overhead)` auto-cudagraphs | **RuntimeError** | residual-stream output aliasing across blocks + cross-attn graph breaks |
| T5 resident on T4 | **OOM** | fp16 model 4.4 + T5 3 + KV 5.5 + act > 16 GB |
| manual CUDA graphs on T4 (fp16) | **OOM** | + 13 per-scale static buffers + graph pool; needs INT8 KV to fit |
| FlashAttention-2 install on this stack | build thrashes / no wheel | no torch-2.9/py3.10/cu12.9 prebuilt wheel; source build saturates the box. (Not needed: SDPA→FA2 on A100; capture-safety solved without it.) |

---

## Notes / framing
- Infinity is **VAR (scale-by-scale prefill)**, not token decode → borrow **prefill/KV** tricks (FlashAttn, KV-quant, graphs); decode tricks (speculative decoding, Medusa) don't apply.
- On A100, self-attn already runs **FA2** via torch SDPA (cross-attn too, via the SDPA fallback). FA1 isn't used anywhere.
- **Latency is launch/host-overhead-bound on the small scales** (→ CUDA graphs) and **throughput is KV-memory-bound** (→ INT8 KV). The two remaining items map to the two bottlenecks.
- **Weight/activation quantization does NOT help latency** — proven twice: A100 INT8 (3.5× slower) and H100 FP8 (slower even with graphs). The per-scale GEMMs are small and the model is launch/overhead-bound, so trading precision for GEMM FLOPs is a net loss once you add dynamic-quant overhead. Quantization's *only* payoff here is **memory** (INT8 **KV** → throughput batch ceiling), not compute. Reach for FP8 only if you go to the **14B** model or much higher resolution, where the late-scale GEMMs are genuinely compute-bound.
- **H100 vs A100:** the win is mostly raw hardware (~2×, HBM3 + SMs + clocks), not new techniques. The *same* lossless levers stack (graphs, compiled VAE); the dead ends are also the same (quantization). Per-scale profile shows the launch-bound signature plainly: in eager, scales 0–8 each take ~52 ms regardless of token count (1 → 576) — pure launch overhead, which graphs erase.
- VAE fidelity caps small-text quality: f16 VAE smears small text; the **f8 patchify VAE** (4× finer latent) recovers it — independent of the inference-speed work.
