"""
Tông Màu — AI backend (v4)
Photorealistic color/style transfer using WCT2 (clovaai, MIT licensed),
served locally over FastAPI. Runs on your own GPU — no cloud, no per-request cost.

Run:
    uvicorn main:app --host 0.0.0.0 --port 8000

Then point the frontend's "AI nâng cao" mode at:
    http://localhost:8000        (same machine)
    http://<LAN-IP-của-máy>:8000  (dùng từ điện thoại cùng wifi)
"""
import io
import logging
from typing import List

import torch
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from PIL import Image
from torchvision import transforms

from wct2 import WCT2
from tiling import tiled_transfer

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("tongmau")

app = FastAPI(title="Tông Màu — AI Style Transfer Backend")

# Permissive CORS: the frontend may be opened as a local file (file://) or
# from Claude's artifact preview, both of which need this to call localhost.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
MAX_DIM = 1024  # default output resolution cap; the hard ceiling is 2048 (see image_size clamp below)

# Content is processed in tiles of at most TILE_SIZE px (blended back together
# with a feathered seam), so VRAM use stays roughly constant regardless of the
# requested image_size — this is what makes 1024px+ output feasible on ~4GB
# cards like the GTX 1650, which can't fit a single 1024px pass (~5.5GB) but
# comfortably fit a 384px tile (~2.1GB reserved, measured — leaves ~1.8GB
# headroom for the OS/desktop's own VRAM use on a 4GB card). Style is capped
# separately since its features are shared across every tile and stay
# resident in VRAM the whole time — no benefit to encoding it larger than a
# tile. Raise these if you have more VRAM to spare (512px tile ≈ 3.8GB
# reserved at 1024px content — fine on 6GB+ cards).
TILE_SIZE = 384
TILE_OVERLAP = 48
STYLE_MAX_DIM = 384

model = None  # loaded once at startup


@app.on_event("startup")
def load_model():
    global model
    log.info(f"Loading WCT2 on device: {DEVICE}")
    if DEVICE == "cpu":
        log.warning("No CUDA GPU detected — running on CPU, this will be slow (potentially minutes/image).")
    model = WCT2(transfer_at=('encoder', 'skip', 'decoder'), option_unpool='cat5', device=DEVICE)
    log.info("Model loaded.")


def load_tensor(file_bytes: bytes, max_dim: int) -> torch.Tensor:
    """PIL -> normalized tensor, resized so the longest side <= max_dim,
    then center-cropped to a multiple of 16 (required by the 4-level wavelet pooling)."""
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    w, h = img.size
    scale = min(1.0, max_dim / max(w, h))
    if scale < 1.0:
        img = img.resize((max(16, int(w * scale)), max(16, int(h * scale))), Image.LANCZOS)
    w, h = img.size
    crop_w, crop_h = (w // 16) * 16, (h // 16) * 16
    crop_w, crop_h = max(16, crop_w), max(16, crop_h)
    img = transforms.CenterCrop((crop_h, crop_w))(img)
    tensor = transforms.ToTensor()(img).unsqueeze(0)
    return tensor.to(DEVICE)


def tensor_to_png_bytes(tensor: torch.Tensor) -> bytes:
    tensor = tensor.clamp(0, 1).squeeze(0).cpu()
    img = transforms.ToPILImage()(tensor)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf.getvalue()


@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": DEVICE,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


@app.post("/style-transfer")
async def style_transfer(
    target: UploadFile = File(..., description="Ảnh cần chỉnh"),
    references: List[UploadFile] = File(..., description="1 hoặc nhiều ảnh tham chiếu"),
    alpha: float = Form(1.0, description="Cường độ áp style, 0.0-1.0"),
    image_size: int = Form(MAX_DIM, description="Giới hạn cạnh dài nhất khi xử lý"),
):
    if model is None:
        raise HTTPException(503, "Model chưa sẵn sàng, thử lại sau giây lát.")
    if not references:
        raise HTTPException(400, "Cần ít nhất 1 ảnh tham chiếu.")
    alpha = max(0.0, min(1.0, alpha))
    image_size = max(64, min(2048, image_size))

    try:
        target_bytes = await target.read()
        content_tensor = load_tensor(target_bytes, image_size)
        # small downsampled copy of the whole content image, used only to
        # compute global whitening statistics shared across every tile
        # (see tiling.py) — not used for the actual pixel output.
        content_proxy = load_tensor(target_bytes, TILE_SIZE)

        outputs = []
        for ref in references:
            ref_bytes = await ref.read()
            style_tensor = load_tensor(ref_bytes, STYLE_MAX_DIM)
            style_feats, style_skips = model.get_all_feature(style_tensor)
            out = tiled_transfer(model, content_tensor, content_proxy, style_feats, style_skips,
                                  alpha=alpha, tile_size=TILE_SIZE, overlap=TILE_OVERLAP)
            del style_tensor, style_feats, style_skips
            torch.cuda.empty_cache()
            # match spatial size to the first output in case crop sizes differ
            if outputs and out.shape[-2:] != outputs[0].shape[-2:]:
                out = torch.nn.functional.interpolate(
                    out, size=outputs[0].shape[-2:], mode="bilinear", align_corners=False)
            outputs.append(out)

        result = torch.stack(outputs, dim=0).mean(dim=0) if len(outputs) > 1 else outputs[0]
        png_bytes = tensor_to_png_bytes(result)
    except HTTPException:
        raise
    except torch.cuda.OutOfMemoryError as e:
        torch.cuda.empty_cache()
        log.warning(f"CUDA out of memory at image_size={image_size}")
        raise HTTPException(500, f"Hết bộ nhớ GPU (out of memory) ở image_size={image_size} — thử giảm image_size. Chi tiết: {e}")
    except Exception as e:
        log.exception("style transfer failed")
        raise HTTPException(500, f"Xử lý ảnh thất bại: {e}")

    return StreamingResponse(io.BytesIO(png_bytes), media_type="image/png")
