"""Manual check (not collected by pytest): what fraction of weight elements did training change?
Compares a saved checkpoint against the base model on a few representative tensors.

    python -m opd_masalign.tests.manual_weight_change_check --base <base dir> --ckpt <trained dir>
"""
import argparse
import glob

from safetensors import safe_open


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--ckpt", required=True)
    args = ap.parse_args()

    index = {}
    for f in sorted(glob.glob(args.base + "/*.safetensors")):
        with safe_open(f, "pt") as s:
            for k in s.keys():
                index[k] = f
    new_files = sorted(glob.glob(args.ckpt + "/*.safetensors"))
    wanted = ("layers.10.self_attn.q_proj.weight", "layers.10.mlp.down_proj.weight",
              "layers.30.self_attn.o_proj.weight", "lm_head.weight")
    for nf in new_files:
        with safe_open(nf, "pt") as new:
            for k in new.keys():
                if k in index and any(k.endswith(w) for w in wanted):
                    a = new.get_tensor(k)
                    with safe_open(index[k], "pt") as s:
                        b = s.get_tensor(k)
                    print(f"{k}: elements changed {(a != b).float().mean().item():.2%}, "
                          f"max|diff| {(a.float() - b.float()).abs().max().item():.2e}", flush=True)
    print("WEIGHT_CHANGE_CHECK_DONE")


if __name__ == "__main__":
    main()
