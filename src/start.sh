#!/usr/bin/env bash

echo "flux2-klein: Build version ${BUILD_VERSION:-unknown}"

# Klein is step-wise distilled: Flux2KleinPipeline.do_classifier_free_guidance returns
# `guidance_scale > 1 and not is_distilled`, so guidance never runs a second pass here and
# the pipeline logs a warning if it is set above 1. Steps are therefore the only lever on
# cost — 4 forward passes at the default, against Qwen's 3 through a model five times larger.
: "${KLEIN_STEPS:=4}"
: "${KLEIN_GUIDANCE_SCALE:=1.0}"

# Weights are ~16 GB (7.75 transformer + 8.05 text encoder). That fits a 48 GB slice or a
# 32 GB card outright; on 24 GB it is tight once 1 MP activations are added. Offload walks
# text_encoder->transformer->vae so the encoder leaves VRAM before sampling, at the cost of
# host-to-device transfers per job. Off by default; flip it per endpoint rather than
# rebuilding when a smaller card is worth testing.
: "${KLEIN_CPU_OFFLOAD:=0}"

# Any progress bar written from inside a sampling loop can block on log backpressure and
# stall the step it is reporting on.
: "${HF_HUB_DISABLE_PROGRESS_BARS:=1}"
: "${TQDM_DISABLE:=1}"

export KLEIN_STEPS KLEIN_GUIDANCE_SCALE KLEIN_CPU_OFFLOAD \
       HF_HUB_DISABLE_PROGRESS_BARS TQDM_DISABLE

echo "flux2-klein: Starting RunPod handler"
exec python -u /handler.py
