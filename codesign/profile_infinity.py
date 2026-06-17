"""
Profile Infinity-2B inference on a T4 (Turing, sm_75, fp16-only, no FA2).

DEVELOP on the CPU VM (syntax/logic), RUN on a T4. It:
  1. installs the T4 flash_attn SDPA shim (t4_compat) BEFORE importing the model,
  2. loads T5 (fp16) + BSQ-VAE + Infinity-2B,
  3. runs generation under fp16 autocast (NOT bf16 — Turing has no bf16),
  4. times the three macro stages with CUDA events:
        [A] T5 text-encode   [B] AR transformer (the scale loop)   [C] VAE decode
     and per-scale transformer time (by wrapping get_logits, called once/scale),
  5. optionally captures a torch.profiler op-level breakdown + chrome trace.

Example (on the T4 box):
  python profile_infinity_t4.py \
      --model_path /ckpt/infinity_2b_reg.pth \
      --vae_path   /ckpt/infinity_vae_d32reg.pth \
      --text_encoder_ckpt google/flan-t5-xl \
      --vae_type 32 --pn 1M --model_type infinity_2b \
      --prompt "a corgi astronaut, studio lighting" \
      --runs 3 --profile 1
"""
import os, sys, time, argparse, contextlib
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# --- T4 shim MUST be installed before importing infinity.models.* ---
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import t4_compat
t4_compat.install()

import numpy as np
import torch
import torch.nn.functional as F

# repo root on path
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "tools"))

from tools.run_infinity import (
    load_tokenizer, load_visual_tokenizer, load_transformer,
    encode_prompt, add_common_arguments,
)
from infinity.utils.dynamic_resolution import dynamic_resolution_h_w


# ----------------------------- timing utils -----------------------------
class CudaTimer:
    """Accurate GPU timing via CUDA events (handles async kernel launch)."""
    def __init__(self):
        self.marks = []
    def mark(self, name):
        e = torch.cuda.Event(enable_timing=True); e.record()
        self.marks.append((name, e))
    def report(self):
        torch.cuda.synchronize()
        out = []
        for i in range(1, len(self.marks)):
            (n0, e0), (n1, e1) = self.marks[i - 1], self.marks[i]
            out.append((n1, e0.elapsed_time(e1)))  # ms
        return out


def make_per_scale_hook(model):
    """get_logits() is called exactly once per scale -> deltas give per-scale
    transformer time. Returns (events list, restore fn)."""
    events = []
    orig = model.get_logits
    def wrapped(*a, **kw):
        e = torch.cuda.Event(enable_timing=True); e.record(); events.append(e)
        return orig(*a, **kw)
    model.get_logits = wrapped
    def restore():
        model.get_logits = orig
    return events, restore


# ----------------------------- generate ---------------------------------
@torch.no_grad()
def generate(model, vae, text_tokenizer, text_encoder, prompt, scale_schedule,
             args, timer: CudaTimer, amp_dtype):
    timer.mark("start")
    # [A] text encode — keep T5 on GPU only for the encode, then OFFLOAD to CPU.
    # T5 (~3GB fp16) is only needed for the prompt; freeing it gives the AR loop's
    # KV cache (~5.5GB at bs=2 for CFG, 10.5k cumulative tokens) room on a 16GB T4.
    text_encoder.cuda()
    text_cond = encode_prompt(text_tokenizer, text_encoder, prompt)
    text_encoder.to("cpu")
    torch.cuda.empty_cache()
    timer.mark("t5_encode")

    cfg = args.cfg if isinstance(args.cfg, list) else [args.cfg] * len(scale_schedule)
    tau = [args.tau] * len(scale_schedule)

    scale_events, restore = make_per_scale_hook(model)
    # [B] AR transformer + [C] VAE decode happen inside autoregressive_infer_cfg
    with torch.autocast("cuda", dtype=amp_dtype, enabled=True, cache_enabled=True):
        _, _, img_list = model.autoregressive_infer_cfg(
            vae=vae, scale_schedule=scale_schedule, label_B_or_BLT=text_cond,
            B=1, negative_label_B_or_BLT=None, force_gt_Bhw=None, g_seed=args.seed,
            cfg_sc=3, cfg_list=cfg, tau_list=tau, top_k=900, top_p=0.97,
            returns_vemb=1, ratio_Bl1=None, gumbel=0, norm_cfg=False,
            cfg_exp_k=0.0, cfg_insertion_layer=[args.cfg_insertion_layer],
            vae_type=args.vae_type, softmax_merge_topk=-1, ret_img=True,
            trunk_scale=1000, gt_leak=0, gt_ls_Bl=None, inference_mode=True,
            sampling_per_bits=args.sampling_per_bits,
        )
    timer.mark("ar+decode")
    restore()
    return img_list[0], scale_events


def summarize(macro, scale_events, scale_schedule):
    torch.cuda.synchronize()
    print("\n================ STAGE BREAKDOWN (ms) ================")
    total = sum(ms for _, ms in macro)
    for name, ms in macro:
        print(f"  {name:14s} {ms:9.1f}  ({100*ms/total:4.1f}%)")
    print(f"  {'TOTAL':14s} {total:9.1f}")

    if len(scale_events) >= 2:
        print("\n---- per-scale transformer time (between get_logits calls) ----")
        # delta between consecutive get_logits events ~= time to produce next scale
        prev = scale_events[0]
        for i in range(1, len(scale_events)):
            dt = prev.elapsed_time(scale_events[i])
            pn = scale_schedule[i - 1] if i - 1 < len(scale_schedule) else "?"
            tok = int(np.prod(pn)) if pn != "?" else 0
            print(f"  scale {i-1:2d}  {str(pn):14s} tokens={tok:5d}  {dt:8.2f} ms")
            prev = scale_events[i]
    print("=====================================================\n")


def main():
    p = argparse.ArgumentParser()
    add_common_arguments(p)
    p.add_argument("--prompt", type=str, default="a corgi astronaut, studio lighting")
    p.add_argument("--runs", type=int, default=3, help="timed runs after 1 warmup")
    p.add_argument("--profile", type=int, default=0, help="torch.profiler op breakdown")
    p.add_argument("--save_file", type=str, default="./t4_out.jpg")
    args = p.parse_args()

    args.cfg = list(map(float, str(args.cfg).split(",")))
    args.cfg = args.cfg[0] if len(args.cfg) == 1 else args.cfg

    assert torch.cuda.is_available(), "Run this on a GPU box (A100 to benchmark, T4 to target)."
    name = torch.cuda.get_device_name(0)
    cc = torch.cuda.get_device_capability(0)
    # bf16 on Ampere+ (A100 sm_80+); fp16 on Turing (T4 sm_75, no bf16)
    amp_dtype = torch.bfloat16 if cc[0] >= 8 else torch.float16
    print(f"[device] {name}  sm_{cc[0]}{cc[1]}  amp_dtype={amp_dtype} "
          f"(bf16-capable={cc[0] >= 8})")

    text_tokenizer, text_encoder = load_tokenizer(t5_path=args.text_encoder_ckpt)
    vae = load_visual_tokenizer(args)
    model = load_transformer(vae, args)

    scale_schedule = dynamic_resolution_h_w[args.h_div_w_template][args.pn]["scales"]
    scale_schedule = [(1, h, w) for (_, h, w) in scale_schedule]
    print(f"[scales] {len(scale_schedule)} stages, final={scale_schedule[-1]}, "
          f"cumulative_tokens={sum(int(np.prod(s)) for s in scale_schedule)}")

    # warmup (kernel autotune, cudnn, allocer)
    print("[warmup] ...")
    t = CudaTimer()
    img, _ = generate(model, vae, text_tokenizer, text_encoder,
                           args.prompt, scale_schedule, args, t, amp_dtype)

    # timed runs
    best = None
    for r in range(args.runs):
        t = CudaTimer()
        img, scale_events = generate(model, vae, text_tokenizer, text_encoder,
                                          args.prompt, scale_schedule, args, t, amp_dtype)
        macro = t.report()
        tot = sum(ms for _, ms in macro)
        print(f"[run {r}] total {tot:.1f} ms")
        if best is None or tot < best[0]:
            best = (tot, macro, scale_events)
    summarize(best[1], best[2], scale_schedule)

    # op-level profile (one run)
    if args.profile:
        from torch.profiler import profile, ProfilerActivity
        t = CudaTimer()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                     record_shapes=False) as prof:
            generate(model, vae, text_tokenizer, text_encoder,
                          args.prompt, scale_schedule, args, t, amp_dtype)
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=25))
        prof.export_chrome_trace("t4_trace.json")
        print("[profile] chrome trace -> t4_trace.json")

    import cv2
    cv2.imwrite(args.save_file, img.cpu().numpy())
    print(f"[saved] {os.path.abspath(args.save_file)}")


if __name__ == "__main__":
    main()
