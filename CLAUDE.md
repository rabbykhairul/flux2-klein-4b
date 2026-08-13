# CLAUDE.md

FLUX.2 Klein 4B image editing on RunPod Serverless — used for virtual try-on: a person
photo plus a garment photo and a prompt, returning the person wearing the garment. Sibling
of `qwen-img-edit-2511`, and a candidate to replace it.

## Not ComfyUI — and that distinction is the whole reason this repo exists

Klein was previously evaluated through a ComfyUI graph that started from
`EmptyFlux2LatentImage` and passed the source image as no conditioning at all, so both
references competed for every pixel. It was rejected on quality. That was a workflow defect,
not a model one: `Flux2KleinPipeline` takes `image=` (a PIL image, or a list for
multi-reference) as real conditioning, and through that path it holds up on garments the
ComfyUI graph mangled. Do not reintroduce a latent-from-empty path.

## Handler input

| Param | Type | Default | Notes |
|-------|------|---------|-------|
| `prompt` | str | required | Edit instruction |
| `image` | str | required | Base64 source (person) |
| `reference_image` | str | — | Base64 garment; sent as `image=[source, reference]` |
| `steps` | int | 4 | The validated operating point |
| `guidance_scale` | float | 1.0 | Ignored by the model — see below |
| `megapixels` | float | — | Optional downscale before the 1024/side clamp |
| `seed` | int | 42 | |

Output matches the Qwen worker's envelope — `{"images": [{"filename", "type": "s3_url",
"data": url}], ...}` — so both sit behind the same provider abstraction unchanged.

## Guidance does nothing here

`Flux2KleinPipeline.do_classifier_free_guidance` returns
`guidance_scale > 1 and not self.config.is_distilled`. Klein is step-wise distilled, so CFG
never runs and the pipeline logs a warning if guidance is set above 1. **Cost is exactly
`steps` forward passes** — 4 at the default, against Qwen's 3 through a model five times
larger. Steps are the only lever.

## Weights

| Path | Source | Size |
|------|--------|------|
| `klein/transformer` | `black-forest-labs/FLUX.2-klein-4B` | 7.75 GB |
| `klein/text_encoder` | same | 8.05 GB |
| `small-decoder/` | `black-forest-labs/FLUX.2-small-decoder` | 0.62 GB |

Apache-2.0, ungated. The Klein repo also ships `flux-2-klein-4b.safetensors`, a 7.75 GB
single-file copy of the transformer for ComfyUI — `download_weights.py` ignores it, since
baking it would add 7.75 GB that every cold worker then pulls for nothing.

Total ~16.5 GB against prod Qwen's 29.6 GB.

## VRAM

~16 GB of weights before activations. Comfortable on 48 GB or a 32 GB card; tight on 24 GB
at 1 MP. `KLEIN_CPU_OFFLOAD=1` enables `enable_model_cpu_offload()`, which walks
`text_encoder->transformer->vae` so the 8 GB encoder leaves VRAM before sampling, at the
cost of host-to-device transfers per job. Off by default, switchable per endpoint.

## CUDA

The image declares `NVIDIA_REQUIRE_CUDA=cuda>=12.0` and the container runtime enforces it,
so **no endpoint-side CUDA filter is needed** — setting one only shrinks the host pool. The
reference Space runs torch on cu130; cu128 carries the same torch 2.11.0 and keeps the
driver floor lower.
