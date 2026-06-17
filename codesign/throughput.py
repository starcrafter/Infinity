"""
Throughput benchmark for Infinity-2B — images/sec via static batching.

Latency (single image) and throughput (images/sec at batch B) are different
questions: a GPU at batch 1 is usually underutilized, so img/s rises with B until
memory (KV cache scales with B) or compute saturates. This sweeps batch sizes and
reports img/s + the batch that OOMs.

Reuses the same fp16/bf16 + T5-offload path. Batches by tiling ONE prompt to B
(same scale schedule -> clean batch). bs into the transformer = 2*B with CFG.

Run:
  python codesign/throughput.py --model_path ... --vae_path ... \
    --text_encoder_ckpt weights/flan-t5-xl --vae_type 32 --pn 1M \
    --model_type infinity_2b --cfg 4 --tau 0.5 --bf16 1 \
    --rope2d_each_sa_layer 1 --rope2d_normalized_by_hw 2 \
    --add_lvl_embeding_only_first_block 1 --use_bit_label 1 \
    --apply_spatial_patchify 0 --text_channels 2048 \
    --batches 1,2,4,8 --t5_offload 0
"""
import os, sys, time, argparse
os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import t4_compat
t4_compat.install()
import torch
import torch.nn.functional as F
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "tools"))
from tools.run_infinity import (load_tokenizer, load_visual_tokenizer,
                                 load_transformer, add_common_arguments)
from infinity.utils.dynamic_resolution import dynamic_resolution_h_w


def encode_batch(text_tokenizer, text_encoder, prompt, B):
    """Encode one prompt tiled to B images -> (kv_compact, lens, cu_seqlens_k, Ltext)."""
    caps = [prompt] * B
    tok = text_tokenizer(text=caps, max_length=512, padding="longest",
                         truncation=True, return_tensors="pt")
    ids = tok.input_ids.cuda(non_blocking=True)
    mask = tok.attention_mask.cuda(non_blocking=True)
    feats = text_encoder(input_ids=ids, attention_mask=mask)["last_hidden_state"].float()
    lens = mask.sum(dim=-1).tolist()
    cu = F.pad(mask.sum(dim=-1).to(torch.int32).cumsum_(0), (1, 0))
    kv = torch.cat([f[:l] for l, f in zip(lens, feats.unbind(0))], dim=0)
    return (kv, lens, cu, max(lens))


@torch.no_grad()
def run_batch(model, vae, tt, te, prompt, ss, args, amp, B):
    offload = getattr(args, "t5_offload", 1)
    if offload: te.cuda()
    tc = encode_batch(tt, te, prompt, B)
    if offload: te.to("cpu"); torch.cuda.empty_cache()
    cfg = [args.cfg if not isinstance(args.cfg, list) else args.cfg[0]] * len(ss)
    tau = [args.tau] * len(ss)
    torch.cuda.synchronize(); t0 = time.time()
    model.autoregressive_infer_cfg(
        vae=vae, scale_schedule=ss, label_B_or_BLT=tc, B=B,
        negative_label_B_or_BLT=None, force_gt_Bhw=None, g_seed=args.seed,
        cfg_sc=3, cfg_list=cfg, tau_list=tau, top_k=900, top_p=0.97,
        returns_vemb=1, ratio_Bl1=None, gumbel=0, norm_cfg=False, cfg_exp_k=0.0,
        cfg_insertion_layer=[args.cfg_insertion_layer], vae_type=args.vae_type,
        softmax_merge_topk=-1, ret_img=True, trunk_scale=1000, gt_leak=0,
        gt_ls_Bl=None, inference_mode=True, sampling_per_bits=args.sampling_per_bits)
    torch.cuda.synchronize()
    return time.time() - t0


def main():
    p = argparse.ArgumentParser(); add_common_arguments(p)
    p.add_argument("--prompt", default="a corgi astronaut, studio lighting")
    p.add_argument("--batches", default="1,2,4,8")
    p.add_argument("--t5_offload", type=int, default=0, choices=[0, 1])
    p.add_argument("--kv_int8", type=int, default=0, choices=[0, 1],
                   help="store KV cache in INT8 (halves persistent cache)")
    args = p.parse_args()
    args.cfg = list(map(float, str(args.cfg).split(",")))
    args.cfg = args.cfg[0] if len(args.cfg) == 1 else args.cfg

    cc = torch.cuda.get_device_capability(0)
    amp = torch.bfloat16 if cc[0] >= 8 else torch.float16
    print(f"[device] {torch.cuda.get_device_name(0)} amp={amp}")
    tt, te = load_tokenizer(t5_path=args.text_encoder_ckpt)
    vae = load_visual_tokenizer(args); model = load_transformer(vae, args)
    if args.kv_int8:
        model.use_static_kv = True
        model.use_kv_int8 = True
        print("[kv_int8] INT8 KV cache enabled")
    ss = dynamic_resolution_h_w[args.h_div_w_template][args.pn]["scales"]
    ss = [(1, h, w) for (_, h, w) in ss]

    print(f"\n{'batch':>5} {'time_s':>8} {'img/s':>8} {'s/img':>8}  peak_GB")
    for B in [int(x) for x in args.batches.split(",")]:
        try:
            with torch.autocast("cuda", dtype=amp):
                run_batch(model, vae, tt, te, args.prompt, ss, args, amp, B)  # warmup
                torch.cuda.reset_peak_memory_stats()
                dt = run_batch(model, vae, tt, te, args.prompt, ss, args, amp, B)
            peak = torch.cuda.max_memory_allocated() / 1e9
            print(f"{B:5d} {dt:8.2f} {B/dt:8.2f} {dt/B:8.2f}  {peak:5.1f}")
        except torch.cuda.OutOfMemoryError:
            print(f"{B:5d}  OOM"); torch.cuda.empty_cache(); break
    print("DONE")


if __name__ == "__main__":
    main()
