"""
Tiny eval set for Infinity-2B variants — judged by eye (Claude), not metrics.

5 fixed prompts / fixed seeds:
  3x text rendering (1-3 words)  + 1 human  + 1 animal
Check per variant: (a) image not corrupted, (b) the requested text renders correctly.

Reuses the SAME fp16 + T5-offload inference path as the profiler, so eval images
reflect exactly what we benchmark/optimize. Loads the model ONCE, loops the 5 prompts.

Run on a T4/A100 (in the starcrafter/Infinity fork):
  python codesign/eval_prompts.py \
    --model_path <2b.pth> --vae_path <vae.pth> --text_encoder_ckpt weights/flan-t5-xl \
    --vae_type 32 --pn 1M --model_type infinity_2b --cfg 4 --tau 0.5 \
    --rope2d_each_sa_layer 1 --rope2d_normalized_by_hw 2 \
    --add_lvl_embeding_only_first_block 1 --use_bit_label 1 --apply_spatial_patchify 0 \
    --text_channels 2048 --bf16 0 --tag baseline --out_dir eval_out
"""
import os, sys, argparse
os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import t4_compat
t4_compat.install()

import torch
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "tools"))
import cv2
from tools.run_infinity import (load_tokenizer, load_visual_tokenizer,
                                 load_transformer, add_common_arguments)
from infinity.utils.dynamic_resolution import dynamic_resolution_h_w
from profile_infinity import generate, CudaTimer

# (id, prompt, seed) — the requested rendered TEXT is in CAPS-in-quotes.
PROMPTS = [
    ("1_text_open",   'a storefront with a wooden sign that reads "OPEN"', 1),
    ("2_text_coffee", 'a cafe chalkboard sign that says "FRESH COFFEE"', 2),
    ("3_text_tokyo",  'a glowing neon sign at night that says "TOKYO"', 3),
    ("4_human",       'a portrait photo of an elderly fisherman, weathered face, looking straight at the camera', 4),
    ("5_animal",      'a red panda sitting on a tree branch in a forest', 5),
]


def main():
    p = argparse.ArgumentParser()
    add_common_arguments(p)
    p.add_argument("--out_dir", default="eval_out")
    p.add_argument("--tag", default="baseline", help="label prefix for output files")
    p.add_argument("--cuda_graph", type=int, default=0, choices=[0,1])
    p.add_argument("--fp8_gemm", type=int, default=0, choices=[0,1], help="LOSSY (H100): FP8 GEMM — quality check")
    p.add_argument("--int8_gemm", type=int, default=0, choices=[0,1], help="LOSSY: int8 W8A8 GEMM — quality check")
    args = p.parse_args()
    args.cfg = list(map(float, str(args.cfg).split(",")))
    args.cfg = args.cfg[0] if len(args.cfg) == 1 else args.cfg

    assert torch.cuda.is_available(), "run on a GPU box"
    cc = torch.cuda.get_device_capability(0)
    amp = torch.bfloat16 if cc[0] >= 8 else torch.float16

    text_tokenizer, text_encoder = load_tokenizer(t5_path=args.text_encoder_ckpt)
    vae = load_visual_tokenizer(args)
    model = load_transformer(vae, args)

    if getattr(args, "fp8_gemm", 0) or getattr(args, "int8_gemm", 0):
        import torch.nn as _nn
        from torchao.quantization import quantize_
        _SKIP = ("mat_qkv", "mat_kv")  # raw F.linear(weight=...) breaks torchao dispatch
        def _qfilter(m, fqn):
            return isinstance(m, _nn.Linear) and fqn.split(".")[-1] not in _SKIP
        if args.fp8_gemm:
            from torchao.quantization import Float8DynamicActivationFloat8WeightConfig as _C
            from torchao.quantization.granularity import PerRow
            for _m in model.modules():
                if isinstance(_m, _nn.Linear): _m.to(torch.bfloat16)  # PerRow needs bf16
            quantize_(model, _C(granularity=PerRow()), filter_fn=_qfilter)
            print("[fp8_gemm] applied (quality eval)")
        else:
            from torchao.quantization import Int8DynamicActivationInt8WeightConfig as _C
            quantize_(model, _C(), filter_fn=_qfilter); print("[int8_gemm] applied (quality eval)")

    ss = dynamic_resolution_h_w[args.h_div_w_template][args.pn]["scales"]
    ss = [(1, h, w) for (_, h, w) in ss]

    use_graph = getattr(args, "cuda_graph", 0)
    if use_graph:
        model.use_cuda_graph = True
        print("[cuda_graph] per-prompt reset; warmup->capture->replay, saving the replayed image")

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"\n==== eval set ({args.tag}), {len(PROMPTS)} prompts, pn={args.pn} ====")
    for name, prompt, seed in PROMPTS:
        args.seed = seed
        if use_graph:
            # capture caches per fixed prompt -> reset, then run warmup(alloc)->capture->replay
            model.reset_cuda_graph()
            for _ in range(2):  # gen0 eager-alloc, gen1 capture
                generate(model, vae, text_tokenizer, text_encoder, prompt, ss, args, CudaTimer(), amp)
        t = CudaTimer()
        img, _ = generate(model, vae, text_tokenizer, text_encoder, prompt, ss, args, t, amp)  # replay (or eager)
        ms = sum(m for _, m in t.report())
        fn = os.path.join(args.out_dir, f"{args.tag}_{name}.jpg")
        cv2.imwrite(fn, img.cpu().numpy())
        print(f"  [{name}] {ms:7.0f} ms  seed={seed}  -> {fn}")
    print("DONE")


if __name__ == "__main__":
    main()
