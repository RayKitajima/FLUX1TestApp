import base64
from io import BytesIO
from typing import List, Literal, Optional

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
# FastAPI app + CORS
# ---------------------------

app = FastAPI(
    title="FLUX.1 [schnell] Text-to-Image API",
    version="1.0.0",
    description="Simple HTTP API for FLUX.1 [schnell] via Hugging Face Diffusers.",
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

    try:
        # FLUX.1 [schnell] requires guidance_scale = 0.
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
    )
