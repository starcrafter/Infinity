"""
T4 (Turing, sm_75) compatibility shim for Infinity inference.

Why this exists
---------------
The upstream model code assumes an Ampere+ GPU with FlashAttention-2:
  - `infinity/models/basic.py` does an UNCONDITIONAL `from flash_attn import ...`
    at import time -> ImportError on any box without FA2 (FA2 doesn't build for
    sm_75 / T4).
  - `CrossAttention.forward` HARD-calls `flash_attn_varlen_kvpacked_func` with no
    fallback (self-attention already has a torch SDPA fallback via `slow_attn`).
  - Inference autocasts to bf16, which Turing does NOT support (fp16 only).

This module makes Infinity run on a T4 WITHOUT forking the repo:
  1. Installs a stub `flash_attn` module BEFORE `infinity.models.basic` is imported,
     so the top-level import succeeds. The stub's functions are SDPA-based and
     numerically equivalent (no causal mask; full attention within each varlen block).
  2. After import, monkeypatches `CrossAttention.forward` is NOT needed because the
     stub `flash_attn_varlen_kvpacked_func` already provides the varlen fallback.

Usage (must run BEFORE importing infinity.models.*):
    import t4_compat; t4_compat.install()
    from infinity.models.infinity import Infinity   # now imports cleanly on T4

The fallback math is unit-tested on CPU in `__main__` against a reference
implementation — run `python t4_compat.py` to verify before deploying to a T4.
"""
import sys
import types
import importlib.machinery
import torch
import torch.nn.functional as F

# cross-attn key-offset cache (id(cu_seqlens_k) -> python int offsets); lets the SDPA
# varlen fallback avoid int(gpu_tensor) syncs during CUDA-graph capture.
_CU_OFFSET_CACHE = {}


# ---------------------------------------------------------------------------
# SDPA-based replacements for the two flash_attn entry points Infinity uses.
# ---------------------------------------------------------------------------
def _flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False, **kw):
    """Dense (non-varlen) attention. flash_attn layout is (B, L, H, c).
    torch SDPA wants (B, H, L, c). Returns (B, L, H, c)."""
    scale = softmax_scale
    q = q.transpose(1, 2)  # B H Lq c
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    o = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p,
                                       is_causal=causal, scale=scale)
    return o.transpose(1, 2).contiguous()  # B L H c


def _flash_attn_varlen_kvpacked_func(q, kv, cu_seqlens_q, cu_seqlens_k,
                                     max_seqlen_q, max_seqlen_k,
                                     dropout_p=0.0, softmax_scale=None,
                                     causal=False, **kw):
    """Variable-length, KV-packed cross-attention fallback.

    Shapes (matching flash_attn):
      q  : (sum_q, H, c)              -- queries for all samples concatenated
      kv : (sum_k, 2, H, c)           -- packed [K, V] for all samples concatenated
      cu_seqlens_q/k : (B+1,) int32   -- prefix sums of per-sample lengths
    Returns:
      (sum_q, H, c)

    Block-diagonal: sample b's queries attend ONLY to sample b's keys/values.
    B is tiny here (1, or 2 with CFG), so a Python loop over samples is fine.
    """
    H, c = q.shape[1], q.shape[2]
    B = cu_seqlens_q.shape[0] - 1
    # CUDA-graph-safe: avoid int(gpu_tensor) (a CPU sync, illegal during capture).
    # Query offsets are uniform (Lq per sample) -> derive from shapes (pure Python).
    # Key offsets are constant per prompt -> read via int() ONCE (eager) and cache by
    # tensor id; in graph mode cu_seqlens_k is the cached (stable-id) tensor, so capture
    # and replay hit the cache and never sync.
    Lq = q.shape[0] // B
    kkey = id(cu_seqlens_k)
    kofs = _CU_OFFSET_CACHE.get(kkey)
    if kofs is None:
        kofs = [int(cu_seqlens_k[i]) for i in range(B + 1)]
        _CU_OFFSET_CACHE[kkey] = kofs
    outs = []
    for b in range(B):
        qs, qe = b * Lq, (b + 1) * Lq
        ks, ke = kofs[b], kofs[b + 1]
        q_b = q[qs:qe].transpose(0, 1).unsqueeze(0)        # 1 H Lq c
        k_b = kv[ks:ke, 0].transpose(0, 1).unsqueeze(0)    # 1 H Lk c
        v_b = kv[ks:ke, 1].transpose(0, 1).unsqueeze(0)    # 1 H Lk c
        o = F.scaled_dot_product_attention(q_b, k_b, v_b, dropout_p=dropout_p,
                                           is_causal=causal, scale=softmax_scale)
        outs.append(o.squeeze(0).transpose(0, 1))          # Lq H c
    return torch.cat(outs, dim=0)                          # sum_q H c


def install():
    """Insert a stub `flash_attn` package into sys.modules so the unconditional
    `from flash_attn import ...` in infinity.models.basic resolves to our SDPA
    fallbacks. No-op if real flash_attn is already importable."""
    if "flash_attn" in sys.modules:
        return
    try:
        import flash_attn  # noqa: F401  (real FA2 present -> use it)
        return
    except Exception:
        pass

    fa = types.ModuleType("flash_attn")
    # A real __spec__ is REQUIRED: modern transformers probes
    # importlib.util.find_spec("flash_attn") when importing T5, and a None spec
    # raises `ValueError: flash_attn.__spec__ is None`. With a valid spec but no
    # installed distribution, transformers' version check fails gracefully ->
    # treats FA2 as unavailable (correct on T4), while our `from flash_attn import`
    # still resolves to the SDPA fallbacks below.
    fa.__spec__ = importlib.machinery.ModuleSpec("flash_attn", loader=None)
    fa.__version__ = "0.0.0+sdpa-stub"
    fa.flash_attn_func = _flash_attn_func
    fa.flash_attn_varlen_kvpacked_func = _flash_attn_varlen_kvpacked_func
    sys.modules["flash_attn"] = fa

    # basic.py also does `try: from flash_attn.ops...` (already guarded), but make
    # the submodule import fail cleanly so it takes the documented fallback path.
    for sub in ("flash_attn.ops", "flash_attn.ops.layer_norm",
                "flash_attn.ops.rms_norm", "flash_attn.ops.fused_dense"):
        sys.modules.pop(sub, None)
    print("[t4_compat] installed SDPA-based flash_attn stub (T4/sm_75 mode)")


# ---------------------------------------------------------------------------
# CPU unit tests — prove the fallbacks match a brute-force reference.
# ---------------------------------------------------------------------------
def _ref_attn(q, k, v, scale):
    # q (Lq,H,c) k,v (Lk,H,c) -> (Lq,H,c), plain softmax attention
    qh = q.permute(1, 0, 2)            # H Lq c
    kh = k.permute(1, 0, 2)            # H Lk c
    vh = v.permute(1, 0, 2)            # H Lk c
    att = (qh @ kh.transpose(-1, -2)) * scale   # H Lq Lk
    att = att.softmax(dim=-1)
    o = att @ vh                       # H Lq c
    return o.permute(1, 0, 2)          # Lq H c


def _test():
    torch.manual_seed(0)
    H, c = 8, 64
    scale = c ** -0.5

    # ---- varlen cross-attn: 2 samples (CFG), same Lq, different key lengths ----
    Lq = 17
    lens_k = [13, 7]
    B = len(lens_k)
    q = torch.randn(B * Lq, H, c)
    sum_k = sum(lens_k)
    kv = torch.randn(sum_k, 2, H, c)
    cu_q = torch.tensor([0, Lq, 2 * Lq], dtype=torch.int32)
    cu_k = torch.tensor([0, lens_k[0], lens_k[0] + lens_k[1]], dtype=torch.int32)

    got = _flash_attn_varlen_kvpacked_func(q, kv, cu_q, cu_k, Lq, max(lens_k),
                                           softmax_scale=scale)
    # reference, per sample
    ref_parts = []
    for b in range(B):
        qs, qe = int(cu_q[b]), int(cu_q[b + 1])
        ks, ke = int(cu_k[b]), int(cu_k[b + 1])
        ref_parts.append(_ref_attn(q[qs:qe], kv[ks:ke, 0], kv[ks:ke, 1], scale))
    ref = torch.cat(ref_parts, dim=0)
    err1 = (got - ref).abs().max().item()
    assert got.shape == (B * Lq, H, c), got.shape
    assert err1 < 1e-5, f"varlen cross-attn mismatch: {err1}"

    # ---- dense self-attn func: (B,L,H,c) ----
    Bs, L = 2, 11
    q2 = torch.randn(Bs, L, H, c)
    k2 = torch.randn(Bs, L, H, c)
    v2 = torch.randn(Bs, L, H, c)
    got2 = _flash_attn_func(q2, k2, v2, softmax_scale=scale, causal=True)
    # reference causal
    qh = q2.permute(0, 2, 1, 3); kh = k2.permute(0, 2, 1, 3); vh = v2.permute(0, 2, 1, 3)
    att = (qh @ kh.transpose(-1, -2)) * scale
    mask = torch.triu(torch.ones(L, L), diagonal=1).bool()
    att = att.masked_fill(mask, float("-inf")).softmax(-1)
    ref2 = (att @ vh).permute(0, 2, 1, 3)
    err2 = (got2 - ref2).abs().max().item()
    assert got2.shape == (Bs, L, H, c), got2.shape
    assert err2 < 1e-5, f"dense attn mismatch: {err2}"

    print(f"[t4_compat] PASS  varlen_cross_attn max_err={err1:.2e}  "
          f"dense_attn max_err={err2:.2e}")


if __name__ == "__main__":
    _test()
