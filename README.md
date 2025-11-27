# FLUX.1 [schnell] Text-to-Image Demo

This is a simple client/server demo for **FLUX.1 [schnell]** using **FastAPI** (backend) and **React + Vite** (frontend).
It exposes a `/generate` API for text-to-image generation and a browser UI for easy testing.

---

## 🚀 Features

* Fast text-to-image generation using **black-forest-labs/FLUX.1-schnell**
* 1–4 rectified-flow inference steps (recommended for schnell)
* Adjustable resolution, sequence length, seed, number of images, and output format
* Frontend preview + downloads
* Clean HTTP API returning base64 `data:` URLs
* CORS-enabled for local dev

---

## 📦 Requirements

**Backend**

* Python 3.10+
* CUDA GPU recommended (CPU works but slower)
* `torch`, `diffusers`, `fastapi`, `uvicorn`, `pydantic`, `Pillow`

**Frontend**

* Node 18+
* Vite + React

---

## 🔧 Installation

### 1. Backend

```bash
cd flux-server
pip install -r requirements.txt   # or install deps manually
uvicorn server:app --host 0.0.0.0 --port 8000
```

Backend starts on **[http://localhost:8000](http://localhost:8000)**

### 2. Frontend

```bash
cd flux-app
npm install
npm run dev
```

Frontend starts on **[http://localhost:5173](http://localhost:5173)**

---

## 🧠 API Usage

### `POST /generate`

**Request:**

```json
{
  "prompt": "A cat holding a sign that says hello world",
  "width": 768,
  "height": 768,
  "num_inference_steps": 4,
  "max_sequence_length": 256,
  "num_images": 1,
  "seed": 123,
  "output_format": "png"
}
```

**Response:**

```json
{
  "images": [
    {
      "data_url": "data:image/png;base64,...",
      "seed": 123
    }
  ],
  "model": "black-forest-labs/FLUX.1-schnell",
  "device": "cuda",
  "height": 768,
  "width": 768,
  "num_inference_steps": 4,
  "max_sequence_length": 256
}
```

---

## 🖥 Frontend

The UI provides:

* Prompt input
* Resolution & step sliders
* Seed control
* Multiple images
* PNG/JPEG output
* Inline results + download links

Configured to call the backend at:

```
http://localhost:8000
```

Modify `API_BASE_URL` in `flux-app/src/App.tsx` if needed.

---

## ✔ Notes

* FLUX.1 [schnell] **requires `guidance_scale=0`** and works best with **1–4 inference steps**.
* When `seed` is provided and multiple images are requested, seeds auto-increment.
* Images are returned as base64 data URLs for easy embedding.
