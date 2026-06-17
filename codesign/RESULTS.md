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

---

## Techniques that WORK — latency

| # | Technique | Lossless | T4 | A100 | Notes |
|---|---|:---:|---|---|---|
| 1 | **T5 resident** (`--t5_offload 0`) | ✅ | OOM (16 GB) | **3.92 → 2.01 s (−49%)** | T5 encode 1898 → **32 ms**; the offload transfer was pure tax |
| 2 | **T5 dynamic padding** (`'longest'` not 512) | ✅ (bit-identical) | small | folds into #1 | runs T5 over real prompt len (~13 tok) not 512; encode → 32 ms |
| 3 | **Static KV buffer** (write-at-offset) | ✅ | OOM | 1966 → 1937 ms (−1.5%) | ~neutral; **prerequisite for CUDA graphs** |
| 4 | **Manual CUDA graphs** (`--cuda_graph`) | ✅ (byte-identical) | OOM (fp16) | **1887 → 1284 ms (−32%)** | per-scale capture/replay; kills per-block launch overhead |
| 5 | **No-CFG** (`--cfg 1`) | ❌ lossy | **13.6 → 9.3 s (−31%)** | 3.92 → 3.40 s (−13%) | bs 2→1; text held on easy prompts but risky (use CFG-*interval*) |

### Cumulative lossless stack (A100): **3.92 s → 1.28 s = −67%**
`T5 resident (#1+#2)` → `static KV (#3)` → `CUDA graphs (#4)`. The fast recipe:
`--t5_offload 0 --cuda_graph 1` (static KV auto-enabled by the graph path).

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

---

## Techniques that do NOT work (null results — don't retry)
| Technique | Result | Why |
|---|---|---|
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
- VAE fidelity caps small-text quality: f16 VAE smears small text; the **f8 patchify VAE** (4× finer latent) recovers it — independent of the inference-speed work.
