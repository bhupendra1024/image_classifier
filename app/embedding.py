import hashlib
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
from PIL import Image

EMBED_DIM = 512

_MODEL_CACHE: Dict[str, Dict[str, Any]] = {}


def _hash_to_vec(text: str, dim: int = EMBED_DIM) -> np.ndarray:
    h = hashlib.sha256(text.encode("utf-8")).digest()
    seed = int.from_bytes(h[:8], "big")
    rng = np.random.default_rng(seed)
    vec = rng.normal(size=(dim,)).astype(np.float32)
    vec /= np.linalg.norm(vec) + 1e-12
    return vec


def _resolve_image_path(image_uri: str) -> Path:
    if image_uri.startswith("file://"):
        return Path(image_uri.replace("file://", "", 1))
    return Path(image_uri)


def _load_openclip_model(model_id: str) -> Dict[str, Any]:
    if model_id in _MODEL_CACHE:
        return _MODEL_CACHE[model_id]

    try:
        import torch
        import open_clip
    except Exception:
        return {"backend": "mock"}

    # model_id format examples:
    # - "openclip:ViT-B-32:laion2b_s34b_b79k"
    # - "clip-vit-large-patch14" -> fallback to mock unless you map it
    if model_id.startswith("openclip:"):
        parts = model_id.split(":")
        if len(parts) == 3:
            model_name, pretrained = parts[1], parts[2]
        else:
            model_name, pretrained = "ViT-B-32", "laion2b_s34b_b79k"
    else:
        # safe default mapping for local experimentation
        model_name, pretrained = "ViT-B-32", "laion2b_s34b_b79k"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name=model_name,
        pretrained=pretrained,
        device=device,
    )
    tokenizer = open_clip.get_tokenizer(model_name)

    entry = {
        "backend": "openclip",
        "torch": torch,
        "model": model,
        "preprocess": preprocess,
        "tokenizer": tokenizer,
        "device": device,
    }
    _MODEL_CACHE[model_id] = entry
    return entry


def embed_text(model_id: str, text: str) -> np.ndarray:
    runtime = _load_openclip_model(model_id)
    if runtime["backend"] != "openclip":
        return _hash_to_vec(f"{model_id}::TEXT::{text}")

    torch = runtime["torch"]
    model = runtime["model"]
    tokenizer = runtime["tokenizer"]
    device = runtime["device"]

    with torch.no_grad():
        tokenized = tokenizer([text]).to(device)
        text_features = model.encode_text(tokenized)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    vec = text_features[0].detach().cpu().numpy().astype(np.float32)
    return vec


def embed_image(model_id: str, image_uri: str) -> np.ndarray:
    runtime = _load_openclip_model(model_id)
    if runtime["backend"] != "openclip":
        return _hash_to_vec(f"{model_id}::IMAGE::{image_uri}")

    image_path = _resolve_image_path(image_uri)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    torch = runtime["torch"]
    model = runtime["model"]
    preprocess = runtime["preprocess"]
    device = runtime["device"]

    img = Image.open(image_path).convert("RGB")
    tensor = preprocess(img).unsqueeze(0).to(device)

    with torch.no_grad():
        image_features = model.encode_image(tensor)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

    vec = image_features[0].detach().cpu().numpy().astype(np.float32)
    return vec


def load_prompt_embeddings(
    model_id: str,
    classifier_set: Dict[str, Any],
) -> Dict[str, Dict[str, List[Tuple[str, np.ndarray]]]]:
    out = {}
    for clf in classifier_set["classifiers"]:
        key = clf["key"]
        positives = [(p, embed_text(model_id, p)) for p in clf["positive_prompts"]]
        negatives = [(n, embed_text(model_id, n)) for n in clf["negative_prompts"]]
        out[key] = {"positive": positives, "negative": negatives}
    return out
