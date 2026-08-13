#!/usr/bin/env python3
"""Bake FLUX.2 Klein 4B + the small decoder VAE into the image."""

import argparse
import os
import sys

from huggingface_hub import snapshot_download

KLEIN_REPO = "black-forest-labs/FLUX.2-klein-4B"
DECODER_REPO = "black-forest-labs/FLUX.2-small-decoder"

# The repo carries the transformer twice: once under transformer/ for diffusers and once as
# a 7.75 GB single-file checkpoint for ComfyUI. Baking both would add 7.75 GB of image for
# nothing, and every cold worker would pull it.
IGNORE = ["flux-2-klein-4b.safetensors", "*.md", "*.png", "*.jpg", ".gitattributes"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights-dir", required=True)
    args = parser.parse_args()

    root = os.path.abspath(args.weights_dir)
    klein_dir = os.path.join(root, "klein")
    decoder_dir = os.path.join(root, "small-decoder")

    snapshot_download(repo_id=KLEIN_REPO, local_dir=klein_dir, ignore_patterns=IGNORE)
    snapshot_download(repo_id=DECODER_REPO, local_dir=decoder_dir, ignore_patterns=IGNORE)

    required = [
        os.path.join(klein_dir, "model_index.json"),
        os.path.join(klein_dir, "transformer"),
        os.path.join(klein_dir, "text_encoder"),
        os.path.join(decoder_dir, "config.json"),
    ]
    for path in required:
        if not os.path.exists(path):
            print(f"FATAL: {path} missing after download", file=sys.stderr)
            return 1

    total = sum(
        os.path.getsize(os.path.join(dirpath, f))
        for dirpath, _, files in os.walk(root)
        for f in files
    )
    print(f"  baked {total / 1e9:.2f} GB into {root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
