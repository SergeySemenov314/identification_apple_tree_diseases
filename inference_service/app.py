import io
import base64
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_PATH = Path("/app/model/best_model_efficientnet_b0_v2.pt")
DEVICE = "cpu"
# Averaging predictions over the original + horizontal/vertical flips (TTA).
# Leaves are orientation-invariant, so a flipped leaf must get the same diagnosis;
# averaging cancels the model's sensitivity to orientation for a small free boost.
USE_TTA = True

# Human-readable (RU) names + short descriptions for the 6 Plant Pathology 2021
# classes. `healthy` is the only "no disease" class.
DISEASE_INFO = {
    "healthy": {
        "name_ru": "Здоровый лист",
        "description": "Признаков болезней не обнаружено.",
        "is_healthy": True,
    },
    "scab": {
        "name_ru": "Парша",
        "description": "Грибковое заболевание (Venturia inaequalis): оливково-бурые бархатистые пятна на листьях.",
        "is_healthy": False,
    },
    "rust": {
        "name_ru": "Ржавчина",
        "description": "Ржавчинный грибок: оранжево-жёлтые пятна с последующими выростами на нижней стороне листа.",
        "is_healthy": False,
    },
    "frog_eye_leaf_spot": {
        "name_ru": "Бурая пятнистость («глаз лягушки»)",
        "description": "Круглые пятна со светлым центром и тёмной каймой (Botryosphaeria obtusa).",
        "is_healthy": False,
    },
    "powdery_mildew": {
        "name_ru": "Мучнистая роса",
        "description": "Белый мучнистый налёт на листьях и побегах (Podosphaera leucotricha).",
        "is_healthy": False,
    },
    "complex": {
        "name_ru": "Комплекс болезней",
        "description": "На листе присутствуют признаки сразу нескольких заболеваний.",
        "is_healthy": False,
    },
}


def _info(label: str) -> dict:
    return DISEASE_INFO.get(
        label,
        {"name_ru": label, "description": "", "is_healthy": label == "healthy"},
    )


# ---------------------------------------------------------------------------
# Model (must match the training-time architecture: backbone + Dropout + Linear)
# ---------------------------------------------------------------------------
class MultiLabelCNN(nn.Module):
    """EfficientNet-B0 backbone + linear head; sigmoid applied at inference."""

    def __init__(self, num_classes: int):
        super().__init__()
        base = models.efficientnet_b0(weights=None)
        in_features = base.classifier[1].in_features
        base.classifier = nn.Identity()
        self.backbone = base
        self.head = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


# ---------------------------------------------------------------------------
# Load checkpoint (contains state_dict, idx2label, per-class thresholds, cfg)
# ---------------------------------------------------------------------------
ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)

idx2label = {int(k): v for k, v in ckpt["idx2label"].items()}
NUM_CLASSES = len(idx2label)
LABELS = [idx2label[i] for i in range(NUM_CLASSES)]

cfg = ckpt.get("cfg", {}) or {}
IMG_SIZE = int(cfg.get("IMG_SIZE", 256))
DEFAULT_THRESHOLD = float(cfg.get("THRESHOLD", 0.5))

_thr = ckpt.get("thresholds", None)
if _thr is None:
    THRESHOLDS = np.full(NUM_CLASSES, DEFAULT_THRESHOLD, dtype=np.float32)
else:
    THRESHOLDS = np.asarray(_thr, dtype=np.float32).reshape(-1)

model = MultiLabelCNN(NUM_CLASSES)
model.load_state_dict(ckpt["model_state_dict"])
model.eval().to(DEVICE)

# val/test preprocessing: resize to square, to-tensor, ImageNet normalization
transform = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

print(
    f"[startup] model={MODEL_PATH.name} backbone=efficientnet_b0 "
    f"classes={LABELS} img_size={IMG_SIZE} tta={USE_TTA}",
    flush=True,
)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
@torch.no_grad()
def infer_probs(image: Image.Image) -> np.ndarray:
    x = transform(image).unsqueeze(0).to(DEVICE)
    views = [x]
    if USE_TTA:
        views.append(torch.flip(x, dims=[3]))  # horizontal flip
        views.append(torch.flip(x, dims=[2]))  # vertical flip

    probs_sum = None
    for v in views:
        p = torch.sigmoid(model(v))
        probs_sum = p if probs_sum is None else probs_sum + p
    probs = (probs_sum / len(views)).squeeze(0).cpu().numpy()
    return probs


def get_detected_idx(probs: np.ndarray) -> list:
    # Classes whose probability passes their (per-class) threshold.
    detected_idx = [i for i in range(NUM_CLASSES) if probs[i] >= THRESHOLDS[i]]
    # Multi-label safety net: if nothing passed, fall back to the top class.
    if not detected_idx:
        detected_idx = [int(np.argmax(probs))]
    return detected_idx


def build_result(probs: np.ndarray) -> dict:
    detected_idx = get_detected_idx(probs)

    detected_labels = [LABELS[i] for i in detected_idx]
    label_string = " ".join(detected_labels)

    is_healthy = detected_labels == ["healthy"]

    # Detected diseases (exclude the "healthy" class from the disease list).
    diseases = [
        {
            "label": LABELS[i],
            "name_ru": _info(LABELS[i])["name_ru"],
            "description": _info(LABELS[i])["description"],
            "probability": round(float(probs[i]), 4),
        }
        for i in detected_idx
        if not _info(LABELS[i])["is_healthy"]
    ]
    diseases.sort(key=lambda d: d["probability"], reverse=True)

    # All classes with probabilities, sorted desc — for the detailed breakdown.
    predictions = [
        {
            "label": LABELS[i],
            "name_ru": _info(LABELS[i])["name_ru"],
            "probability": round(float(probs[i]), 4),
            "threshold": round(float(THRESHOLDS[i]), 4),
            "detected": bool(i in detected_idx),
        }
        for i in range(NUM_CLASSES)
    ]
    predictions.sort(key=lambda p: p["probability"], reverse=True)

    return {
        "labels": label_string,
        "is_healthy": is_healthy,
        "diseases": diseases,
        "predictions": predictions,
    }


# ---------------------------------------------------------------------------
# Grad-CAM: where the network looked when deciding a given class
# ---------------------------------------------------------------------------
# Last conv block of EfficientNet-B0. Its spatial resolution = input / 32.
GRADCAM_LAYER = model.backbone.features[-1]
GRADCAM_MAX_SIDE = 640  # cap overlay resolution to keep the JSON payload small

# Run the Grad-CAM forward pass at a HIGHER resolution than the 256px used for
# prediction: the net is fully convolutional, so 512px yields a 16x16 feature map
# (vs 8x8 at 256px) — 4x more cells, a much sharper, better-localized heatmap.
# Predictions still come from infer_probs (256 + TTA); only the CAM uses this size.
GRADCAM_SIZE = 512
gradcam_transform = transforms.Compose(
    [
        transforms.Resize((GRADCAM_SIZE, GRADCAM_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def _jet(x: np.ndarray) -> np.ndarray:
    """Map a HxW array in [0, 1] to an RGB uint8 heatmap (classic 'jet')."""
    r = np.clip(1.5 - np.abs(4 * x - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * x - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * x - 1), 0, 1)
    return (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)


def compute_cam(class_idx: int) -> np.ndarray:
    """Grad-CAM for one class. Assumes `_last_input` holds the model input tensor.

    A forward hook captures the target layer's activations; autograd then gives
    d(class logit)/d(activations). Channel-wise averaged gradients weight the
    activation maps; ReLU keeps only evidence *for* the class.
    """
    activations = {}

    def hook(_m, _inp, out):
        activations["value"] = out

    handle = GRADCAM_LAYER.register_forward_hook(hook)
    try:
        model.zero_grad(set_to_none=True)
        logits = model(_last_input["x"])
        act = activations["value"]  # [1, C, H, W], part of the graph
        grads = torch.autograd.grad(logits[0, class_idx], act)[0]
        weights = grads.mean(dim=(2, 3), keepdim=True)  # [1, C, 1, 1]
        cam = torch.relu((weights * act).sum(dim=1)).squeeze(0)  # [H, W]
    finally:
        handle.remove()

    cam = cam.detach().cpu().numpy()
    cam -= cam.min()
    # Clip to the 99th percentile so a single hot cell doesn't wash out the rest,
    # then normalize to [0, 1].
    hi = np.percentile(cam, 99)
    cam = np.clip(cam, 0, hi if hi > 0 else cam.max())
    cam /= cam.max() + 1e-8
    return cam


_last_input = {"x": None}


def gradcam_overlay(image: Image.Image, class_idx: int) -> str:
    """Return a base64 JPEG: the original photo with the class heatmap blended in."""
    cam = compute_cam(class_idx)

    base = image.convert("RGB")
    w, h = base.size
    scale = min(1.0, GRADCAM_MAX_SIDE / max(w, h))
    if scale < 1.0:
        base = base.resize((max(1, int(w * scale)), max(1, int(h * scale))))
    W, H = base.size

    cam_img = Image.fromarray((cam * 255).astype(np.uint8)).resize(
        (W, H), Image.Resampling.BILINEAR
    )
    cam_np = np.asarray(cam_img, dtype=np.float32) / 255.0
    heat = _jet(cam_np).astype(np.float32)

    base_np = np.asarray(base, dtype=np.float32)
    # Per-pixel alpha: cold regions keep the original photo, hot regions turn red.
    alpha = (cam_np * 0.55)[..., None]
    out = (base_np * (1 - alpha) + heat * alpha).clip(0, 255).astype(np.uint8)

    buf = io.BytesIO()
    Image.fromarray(out).save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------
app = FastAPI()


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": MODEL_PATH.name,
        "classes": LABELS,
        "img_size": IMG_SIZE,
        "tta": USE_TTA,
    }


@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        return JSONResponse(status_code=400, content={"error": "Cannot decode image"})

    probs = infer_probs(image)
    result = build_result(probs)

    # Grad-CAM overlay for each class that made it into the diagnosis, so the
    # user can see which leaf regions drove that particular decision. Uses the
    # higher-resolution gradcam_transform for a sharper map (see GRADCAM_SIZE).
    _last_input["x"] = gradcam_transform(image).unsqueeze(0).to(DEVICE)
    try:
        result["gradcam"] = [
            {
                "label": LABELS[i],
                "name_ru": _info(LABELS[i])["name_ru"],
                "image": gradcam_overlay(image, i),
            }
            for i in get_detected_idx(probs)
        ]
    finally:
        _last_input["x"] = None

    conf_pairs = ", ".join(
        f"{p['label']}={p['probability']:.2f}" for p in result["predictions"]
    )
    print(
        f"[detect] size={image.size} -> labels='{result['labels']}' "
        f"healthy={result['is_healthy']} | {conf_pairs}",
        flush=True,
    )
    return result
