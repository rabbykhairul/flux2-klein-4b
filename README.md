# flux2-klein-4b

RunPod serverless worker running **FLUX.2 [klein] 4B distilled** for virtual try-on, as a
model-swap experiment against the production Qwen-Image-Edit-2511 worker in
`qwen-img-edit-2511`.

Production is untouched. The infrastructure — R2 upload, websocket polling, validation,
request schema — is carried over unchanged so the two are directly comparable. The workflow
graph is entirely new.

## Why the 4B and not the 9B

```
FLUX.2-klein-4B    Apache 2.0                     ✅
FLUX.2-klein-9B    FLUX NCL (non-commercial)      ❌
```

The 9B is the variant most benchmarks cover and it cannot be used commercially — the same
wall as MagicTryOn, IDM-VTON and every other open VTON model. The 4B is Apache 2.0, and at
~1.2s per image on a 5090 it is also the faster of the two.

## The bet

This is a much larger change than a quantization. Different architecture, different text
encoder, different prompt behaviour:

| | production | this repo |
|---|---|---|
| DiT | Qwen-Image-Edit-2511, ~20B | FLUX.2 klein 4B |
| weights | 20.53 GB fp8mixed | 7.75 GB |
| text encoder | Qwen2.5-VL-7B fp8 | Qwen3-4B fp4, 3.85 GB |
| VAE | qwen_image_vae | flux2-vae, 0.34 GB |
| total bake | ~28.5 GB | **~11.9 GB** |
| speed, 4 steps | 6.09s | ~1.2s |

**Prompts will not transfer.** The try-on prompt tuned against Qwen has to be redone for
klein — that is expected work, not a bug.

## Multi-reference wiring

FLUX.2 takes multiple references by chaining one `ReferenceLatent` per image through the
conditioning. Positive and negative are parallel chains over the same latents:

```
person  → ImageScale → VAEEncode ─┬→ ReferenceLatent(70, +) ─→ ReferenceLatent(71, +) → CFGGuider.positive
                                  └→ ReferenceLatent(72, −) ─→ ReferenceLatent(73, −) → CFGGuider.negative
garment → ImageScale → VAEEncode ──────────┘ (latent input of 71 and 73)
```

The negative branch is a `ConditioningZeroOut` of the positive, which is what the distilled
model was trained against — so `negative_prompt` is accepted and ignored. Output dimensions
follow the **source** image via `GetImageSize`, not the garment.

Wiring taken from Comfy-Org's `image_flux2_klein_image_edit_9b_distilled` template; the 4B
template ships single-reference only.

## Models

All from [Comfy-Org/flux2-klein](https://huggingface.co/Comfy-Org/flux2-klein):

| File | Size |
|---|---:|
| `flux-2-klein-4b.safetensors` | 7.75 GB |
| `qwen_3_4b_fp4_flux2.safetensors` | 3.85 GB |
| `flux2-vae.safetensors` | 0.34 GB |

`KLEIN_MODEL` swaps in `flux-2-klein-base-4b.safetensors` (undistilled, ~50 steps) and
`KLEIN_TEXT_ENCODER` swaps in `qwen_3_4b.safetensors` (bf16, 8.04 GB) — both must be added to
`check-models.sh` and baked first. The fp4 encoder is the default because the text encoder
runs once per job rather than once per step, so precision costs far less there than in the
DiT.

## Ignored request fields

The signature matches the Qwen worker so the same payloads and testers work against both.
Four parameters have no analogue: `lora` (klein 4B distilled is natively 4-step), `shift` and
`scheduler` (`Flux2Scheduler` derives sigmas from steps and resolution), and
`negative_prompt` (see above).

## Build

```bash
docker build --target final --build-arg HF_TOKEN=$HF_TOKEN -t flux2-klein-4b .
```

No custom nodes — every node in the graph is ComfyUI core. This needs a ComfyUI recent enough
to ship `Flux2Scheduler`, `EmptyFlux2LatentImage` and `ReferenceLatent`; the pinned
`COMFYUI_VERSION=0.28.3` should be verified against that before the first build.

## What to measure

Against production's baseline — 8.09s warm exec, 17.09s cold, $0.00450/image warm:

1. **Does it produce a usable try-on at all?** Before any timing, confirm the garment
   transfers rather than the model blending the two references.
2. **Face and garment fidelity vs Qwen.** A 4B model has a fifth the capacity of what runs
   today, and garment print and facial identity are where parameter count buys something.
   This is the question the whole experiment turns on.
3. **`Prompt executed`** at 4 steps, warm, ≥5 samples, once quality is acceptable.

Expected if it lands: ~3.8s warm exec, ~7.5s cold, ~$0.00262/image, −42%.
