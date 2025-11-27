import base64
from io import BytesIO
from typing import Dict, List, Literal, Optional, Tuple

import torch
from diffusers import FluxPipeline
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, constr, conint
from PIL import Image

# ---------------------------
# Model loading (global)
# ---------------------------

MODEL_ID = "black-forest-labs/FLUX.1-schnell"

device = "cuda" if torch.cuda.is_available() else "cpu"

# FLUX.1 [schnell] usually runs in bfloat16 on GPU; use float32 on CPU.
dtype = torch.bfloat16 if device == "cuda" else torch.float32

print(f"[FLUX] Loading {MODEL_ID} on device={device}, dtype={dtype}...")

try:
    pipe = FluxPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        use_safetensors=True,
    )
    pipe.to(device)
    # Optional memory optimizations:
    # pipe.enable_sequential_cpu_offload()
    # pipe.vae.enable_slicing()
    # pipe.vae.enable_tiling()
except Exception as e:
    raise RuntimeError(f"Failed to load Flux pipeline: {e}")

# ---------------------------
# LoRA cache (global)
# ---------------------------

# We cache loaded LoRA adapters so we don't re-download them every request.
# Key: (lora_model, lora_weight_name) -> adapter_name
LORA_ADAPTER_CACHE: Dict[Tuple[str, Optional[str]], str] = {}

# ---------------------------
# FastAPI app + CORS
# ---------------------------

app = FastAPI(
    title="FLUX.1 [schnell] Text-to-Image API",
    version="1.1.0",
    description="Simple HTTP API for FLUX.1 [schnell] via Hugging Face Diffusers (with LoRA support).",
)

origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# Pydantic models
# ---------------------------


class GenerateRequest(BaseModel):
    prompt: constr(strip_whitespace=True, min_length=1) = Field(
        ..., description="Text prompt for the image."
    )
    height: conint(ge=256, le=1536) = Field(
        768, description="Image height in pixels."
    )
    width: conint(ge=256, le=1536) = Field(
        768, description="Image width in pixels."
    )
    # FLUX.1 [schnell] is timestep-distilled, recommended 1–4 steps.
    num_inference_steps: conint(ge=1, le=10) = Field(
        4,
        description="Number of inference steps; clamped to [1,4] for schnell.",
    )
    # Timestep-distilled variant requires max_sequence_length ≤ 256.
    max_sequence_length: conint(ge=16, le=256) = Field(
        256,
        description="Max token length for the T5 encoder; max 256 for schnell.",
    )
    num_images: conint(ge=1, le=4) = Field(
        1, description="How many images to generate per prompt."
    )
    seed: Optional[int] = Field(
        None,
        description="Base random seed; if provided, images use seed, seed+1, ...",
    )
    output_format: Literal["png", "jpeg"] = Field(
        "png", description="Returned image format."
    )

    # --------- LoRA (optional) ---------
    # These are all optional; if `lora_model` is not provided, no LoRA is used.
    lora_model: Optional[constr(strip_whitespace=True, min_length=1)] = Field(
        None,
        description=(
            "Optional: HF Hub repo ID or local path to a Flux LoRA "
            "(e.g. 'Shakker-Labs/FLUX.1-dev-LoRA-collections' or "
            "'ByteDance/Hyper-SD'). Must be compatible with the base model."
        ),
    )
    lora_weight_name: Optional[constr(strip_whitespace=True, min_length=1)] = Field(
        None,
        description=(
            "Optional: specific LoRA weight file inside the repo, e.g. "
            "'Hyper-FLUX.1-schnell-8steps-lora.safetensors'. If omitted, "
            "the default weight is used."
        ),
    )
    lora_adapter_name: Optional[constr(strip_whitespace=True, min_length=1)] = Field(
        None,
        description=(
            "Optional: custom adapter name for this LoRA. If omitted, "
            "an automatic adapter name is chosen and cached."
        ),
    )
    lora_scale: Optional[float] = Field(
        1.0,
        ge=0.0,
        le=2.0,
        description=(
            "How strongly to apply the LoRA (0 = effectively off, 1 = full "
            "strength). Only used if lora_model is set."
        ),
    )


class GeneratedImage(BaseModel):
    data_url: str
    seed: Optional[int]


class GenerateResponse(BaseModel):
    images: List[GeneratedImage]
    model: str
    device: str
    height: int
    width: int
    num_inference_steps: int
    max_sequence_length: int

    # Echo back LoRA info for the UI
    active_lora: Optional[str] = Field(
        None, description="Adapter name of the active LoRA, if any."
    )
    lora_model: Optional[str] = Field(
        None, description="LoRA model ID or path used for this generation, if any."
    )
    lora_weight_name: Optional[str] = Field(
        None, description="LoRA weight filename used for this generation, if any."
    )
    lora_scale: Optional[float] = Field(
        None, description="Scale applied to the LoRA, if any."
    )

# ---------------------------
# Helpers
# ---------------------------


def pil_to_data_url(image: Image.Image, fmt: str = "png") -> str:
    buffer = BytesIO()
    fmt_upper = "PNG" if fmt.lower() == "png" else "JPEG"
    image.save(buffer, format=fmt_upper)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    mime = "image/png" if fmt.lower() == "png" else "image/jpeg"
    return f"data:{mime};base64,{encoded}"


# ---------------------------
# Routes
# ---------------------------


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_ID, "device": device}


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    if not req.prompt:
        raise HTTPException(
            status_code=400, detail="Prompt must not be empty.")

    # Enforce schnell-specific constraints:
    steps = max(1, min(req.num_inference_steps, 4))  # 1–4 only
    max_seq = min(req.max_sequence_length, 256)
    num_images = req.num_images

    # Seed handling
    if req.seed is not None:
        if num_images == 1:
            generator = torch.Generator(device).manual_seed(req.seed)
            seeds = [req.seed]
        else:
            generators = []
            seeds = []
            for i in range(num_images):
                s = req.seed + i
                generators.append(torch.Generator(device).manual_seed(s))
                seeds.append(s)
            generator = generators
    else:
        generator = None
        seeds = [None] * num_images

    # ---------------------------
    # LoRA handling (per-request)
    # ---------------------------
    active_lora_adapter: Optional[str] = None
    lora_scale_used: Optional[float] = None

    # A LoRA is used iff we have a model id and a non-zero scale (None => 1.0)
    use_lora = (
        req.lora_model is not None
        and req.lora_model.strip() != ""
        and (req.lora_scale is None or req.lora_scale > 0.0)
    )

    if use_lora:
        if not hasattr(pipe, "load_lora_weights"):
            # Older diffusers version without Flux LoRA support
            raise HTTPException(
                status_code=500,
                detail=(
                    "This diffusers version does not support Flux LoRAs. "
                    "Please upgrade diffusers/peft to a recent version."
                ),
            )

        lora_scale_used = req.lora_scale if req.lora_scale is not None else 1.0

        key: Tuple[str, Optional[str]] = (
            req.lora_model.strip(),
            req.lora_weight_name.strip() if req.lora_weight_name else None,
        )

        if key in LORA_ADAPTER_CACHE:
            adapter_name = LORA_ADAPTER_CACHE[key]
        else:
            # First time we see this LoRA: load and assign an adapter name
            adapter_name = (
                req.lora_adapter_name.strip()
                if req.lora_adapter_name
                else f"lora_{len(LORA_ADAPTER_CACHE)}"
            )

            load_kwargs = {}
            if req.lora_weight_name:
                load_kwargs["weight_name"] = req.lora_weight_name.strip()

            try:
                pipe.load_lora_weights(
                    req.lora_model.strip(),
                    adapter_name=adapter_name,
                    **load_kwargs,
                )
            except Exception as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"Failed to load LoRA '{req.lora_model}': {e}",
                )

            LORA_ADAPTER_CACHE[key] = adapter_name

        active_lora_adapter = adapter_name

        # Activate this adapter with the requested strength
        try:
            # set_adapters controls which LoRA(s) is active and their weights. :contentReference[oaicite:1]{index=1}
            pipe.set_adapters(active_lora_adapter,
                              adapter_weights=lora_scale_used)
            if hasattr(pipe, "enable_lora"):
                pipe.enable_lora()
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to activate LoRA adapter '{active_lora_adapter}': {e}",
            )
    else:
        # No LoRA requested: make sure LoRAs are disabled for this call
        if hasattr(pipe, "disable_lora"):
            try:
                pipe.disable_lora()
            except Exception as e:
                # Don't hard-error the request, just log to stderr.
                print(f"[LoRA] Warning: failed to disable LoRA: {e}")

    # ---------------------------
    # Generation
    # ---------------------------
    try:
        # FLUX.1 [schnell] requires guidance_scale = 0. :contentReference[oaicite:2]{index=2}
        result = pipe(
            prompt=req.prompt,
            height=req.height,
            width=req.width,
            num_inference_steps=steps,
            max_sequence_length=max_seq,
            guidance_scale=0.0,
            num_images_per_prompt=num_images,
            generator=generator,
            output_type="pil",
        )
        images: List[Image.Image] = result.images
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {e}")

    if len(images) != num_images:
        raise HTTPException(
            status_code=500,
            detail=f"Expected {num_images} images, got {len(images)}",
        )

    output_images: List[GeneratedImage] = []
    for img, seed_val in zip(images, seeds):
        data_url = pil_to_data_url(img, fmt=req.output_format)
        output_images.append(GeneratedImage(data_url=data_url, seed=seed_val))

    return GenerateResponse(
        images=output_images,
        model=MODEL_ID,
        device=device,
        height=req.height,
        width=req.width,
        num_inference_steps=steps,
        max_sequence_length=max_seq,
        active_lora=active_lora_adapter,
        lora_model=req.lora_model if active_lora_adapter else None,
        lora_weight_name=req.lora_weight_name if active_lora_adapter else None,
        lora_scale=lora_scale_used if active_lora_adapter else None,
    )
