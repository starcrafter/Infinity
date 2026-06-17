"""
VAE reconstruction fidelity test (BSQ-VAE encode -> decode, no AR transformer).

The VAE is the UPPER BOUND on what Infinity can render: if the multiscale bitwise
tokenizer can't reconstruct fine/small text, the AR model can never generate it.
Feed a text-heavy image through vae.encode -> vae.decode and eyeball the recon.

Runs on CPU or GPU. Force CPU with CUDA_VISIBLE_DEVICES="" to avoid contending with a
GPU job. Example:
  CUDA_VISIBLE_DEVICES="" python codesign/vae_recon.py \
    --image slide.png --vae_path <vae_d32reg.pth> --vae_type 32 \
    --model_path x --pn 1M --apply_spatial_patchify 0 --out_dir vae_out
"""
import os, sys, argparse
os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import t4_compat
t4_compat.install()

import numpy as np
import torch
import cv2
from PIL import Image
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "tools"))
from tools.run_infinity import (load_visual_tokenizer, joint_vi_vae_encode_decode,
                                 add_common_arguments)
from infinity.utils.dynamic_resolution import dynamic_resolution_h_w, h_div_w_templates


def main():
    p = argparse.ArgumentParser()
    add_common_arguments(p)
    p.add_argument("--image", required=True)
    p.add_argument("--out_dir", default="vae_out")
    a = p.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[device] {dev}")
    vae = load_visual_tokenizer(a)

    im = Image.open(a.image)
    W, H = im.size
    hw = H / W
    tmpl = float(h_div_w_templates[np.argmin(np.abs(hw - h_div_w_templates))])
    res = dynamic_resolution_h_w[tmpl][a.pn]
    tgt_h, tgt_w = res["pixel"]
    ss = [(1, h, w) for (t, h, w) in res["scales"]]
    # f8 patchify VAE (apply_spatial_patchify=1, patch=8) needs the VAE schedule at 2x
    # the model schedule -> 4x finer latent grid at the SAME pixel size (better small text).
    vae_ss = [(1, 2 * h, 2 * w) for (_, h, w) in ss] if a.apply_spatial_patchify else ss
    print(f"[recon] img {W}x{H} (h/w={hw:.3f}) -> template {tmpl}, pn={a.pn}, "
          f"tgt {tgt_h}x{tgt_w}, patchify={a.apply_spatial_patchify}, "
          f"vae final latent grid={vae_ss[-1][1]}x{vae_ss[-1][2]}")

    gt, recon, _ = joint_vi_vae_encode_decode(vae, a.image, vae_ss, dev, tgt_h, tgt_w)
    os.makedirs(a.out_dir, exist_ok=True)
    cv2.imwrite(os.path.join(a.out_dir, "gt.png"), cv2.cvtColor(gt, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(a.out_dir, "recon.png"), cv2.cvtColor(recon, cv2.COLOR_RGB2BGR))
    err = np.abs(gt.astype(np.float32) - recon.astype(np.float32)).mean()
    print(f"[recon] mean abs pixel err = {err:.2f}/255  -> {a.out_dir}/gt.png, recon.png")


if __name__ == "__main__":
    main()
