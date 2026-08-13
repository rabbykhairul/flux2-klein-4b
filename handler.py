import base64
import io
import math
import os
import time
import uuid

import runpod
import torch
from diffusers import AutoencoderKLFlux2, Flux2KleinPipeline
from PIL import Image

MAX_SIDE = 1024
MIN_SIDE = 256

_r2_client = None


def _env_int(name, default):
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return int(default)


def _env_float(name, default):
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return float(default)


WEIGHTS = os.environ.get("KLEIN_WEIGHTS_DIR", "/weights")

print("flux2-klein: loading pipeline...")
_t0 = time.time()
_vae = AutoencoderKLFlux2.from_pretrained(
    os.path.join(WEIGHTS, "small-decoder"), torch_dtype=torch.bfloat16
)
PIPELINE = Flux2KleinPipeline.from_pretrained(
    os.path.join(WEIGHTS, "klein"), vae=_vae, torch_dtype=torch.bfloat16
)
if os.environ.get("KLEIN_CPU_OFFLOAD", "0") == "1":
    PIPELINE.enable_model_cpu_offload()
    print("flux2-klein: model CPU offload enabled")
else:
    PIPELINE.to("cuda")
print(f"flux2-klein: pipeline ready in {time.time() - _t0:.2f}s")


def _decode_image(value, field):
    if not isinstance(value, str) or not value.strip():
        return None, f"'{field}' must be a base64-encoded image string"
    payload = value.split(",", 1)[1] if value.startswith("data:") else value
    try:
        return Image.open(io.BytesIO(base64.b64decode(payload))).convert("RGB"), None
    except Exception as exc:
        return None, f"'{field}' could not be decoded as an image: {exc}"


def _target_dims(image, megapixels):
    """Mirror the reference Space: derive from the source image, snap to /8, clamp to 1024.

    Passing dimensions that are not multiples of 8 silently shifts the latent grid, and
    anything past 1024 a side leaves the resolution band the model was validated in.
    """
    w, h = image.size
    if megapixels:
        scale = math.sqrt((megapixels * 1_000_000) / float(w * h))
        w, h = w * scale, h * scale
    longest = max(w, h)
    if longest > MAX_SIDE:
        w, h = w * MAX_SIDE / longest, h * MAX_SIDE / longest
    snap = lambda v: max(MIN_SIDE, min(MAX_SIDE, int(round(v / 8) * 8)))
    return snap(w), snap(h)


def validate_input(job_input):
    if not isinstance(job_input, dict):
        return None, "'input' must be an object"

    prompt = job_input.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        return None, "'prompt' must be a non-empty string"

    source, err = _decode_image(job_input.get("image"), "image")
    if err:
        return None, err

    reference = None
    if job_input.get("reference_image"):
        reference, err = _decode_image(job_input.get("reference_image"), "reference_image")
        if err:
            return None, err

    steps = job_input.get("steps", _env_int("KLEIN_STEPS", 4))
    if not isinstance(steps, int) or not 1 <= steps <= 50:
        return None, "'steps' must be an integer between 1 and 50"

    megapixels = job_input.get("megapixels")
    if megapixels is not None:
        try:
            megapixels = float(megapixels)
        except (TypeError, ValueError):
            return None, "'megapixels' must be a number"

    width, height = _target_dims(source, megapixels)

    return {
        "prompt": prompt,
        "images": [source, reference] if reference is not None else source,
        "steps": steps,
        "guidance_scale": float(
            job_input.get("guidance_scale", _env_float("KLEIN_GUIDANCE_SCALE", 1.0))
        ),
        "seed": int(job_input.get("seed", 42)),
        "width": width,
        "height": height,
    }, None


def _r2_config():
    endpoint = os.environ.get("R2_ENDPOINT_URL")
    bucket = os.environ.get("R2_BUCKET")
    key_id = os.environ.get("R2_ACCESS_KEY_ID")
    secret = os.environ.get("R2_SECRET_ACCESS_KEY")
    if not (endpoint and bucket and key_id and secret):
        return None
    return {
        "endpoint": endpoint,
        "bucket": bucket,
        "key_id": key_id,
        "secret": secret,
        "public_base": os.environ.get("R2_PUBLIC_BASE_URL"),
        "expiry": int(os.environ.get("R2_URL_EXPIRY", "604800")),
    }


def upload_image_to_r2(job_id, image_bytes, filename, cfg):
    global _r2_client
    if _r2_client is None:
        import boto3
        from botocore.config import Config

        # botocore defaults to a 60s read timeout with no retry policy of our own. A 1-2 MB
        # PUT to R2 lands in under a second, but when the socket goes silent the worker
        # rents a GPU to hold a dead connection: one such stall on the Qwen worker burned
        # 62s and cost $0.045 for a single image. The retry after it succeeded in ~2s, so
        # retrying is right and only the waiting was wrong.
        _r2_client = boto3.client(
            "s3",
            endpoint_url=cfg["endpoint"],
            aws_access_key_id=cfg["key_id"],
            aws_secret_access_key=cfg["secret"],
            region_name="auto",
            config=Config(
                signature_version="s3v4",
                connect_timeout=3,
                read_timeout=7,
                retries={"max_attempts": 3, "mode": "standard"},
            ),
        )

    key = f"{job_id}/{filename}"
    _r2_client.put_object(
        Bucket=cfg["bucket"], Key=key, Body=image_bytes, ContentType="image/png"
    )

    if cfg["public_base"]:
        return f"{cfg['public_base'].rstrip('/')}/{key}"
    return _r2_client.generate_presigned_url(
        "get_object", Params={"Bucket": cfg["bucket"], "Key": key}, ExpiresIn=cfg["expiry"]
    )


def handler(job):
    job_input = job.get("input") or {}
    job_id = job.get("id") or str(uuid.uuid4())

    if isinstance(job_input, dict) and job_input.get("health_check"):
        return {"status": "healthy", "build_version": os.environ.get("BUILD_VERSION", "unknown")}

    params, error = validate_input(job_input)
    if error:
        return {"error": error}

    generator = torch.Generator(device="cpu").manual_seed(params["seed"])

    t0 = time.time()
    result = PIPELINE(
        image=params["images"],
        prompt=params["prompt"],
        height=params["height"],
        width=params["width"],
        num_inference_steps=params["steps"],
        guidance_scale=params["guidance_scale"],
        generator=generator,
    )
    generation_s = time.time() - t0
    print(
        f"flux2-klein: generated in {generation_s:.2f}s "
        f"({params['width']}x{params['height']}, {params['steps']} steps)"
    )

    cfg = _r2_config()
    outputs = []
    errors = []
    for idx, image in enumerate(result.images):
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        data = buf.getvalue()
        filename = f"flux2_klein_{idx:05d}.png"

        if cfg:
            try:
                url = upload_image_to_r2(job_id, data, filename, cfg)
                print(f"flux2-klein: uploaded {filename} to R2")
                outputs.append({"filename": filename, "type": "s3_url", "data": url})
                continue
            except Exception as exc:
                errors.append(f"Error uploading {filename} to R2: {exc}")

        outputs.append(
            {"filename": filename, "type": "base64", "data": base64.b64encode(data).decode("utf-8")}
        )

    response = {
        "images": outputs,
        "generation_seconds": round(generation_s, 3),
        "params": {
            "steps": params["steps"],
            "guidance_scale": params["guidance_scale"],
            "width": params["width"],
            "height": params["height"],
            "seed": params["seed"],
        },
    }
    if errors:
        response["errors"] = errors
    return response


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
