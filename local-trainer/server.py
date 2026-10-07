#!/usr/bin/env python3
"""LRA-MAKR local trainer.

A small HTTP server that lets LRA-MAKR train LoRAs on this PC with
ostris/ai-toolkit. Standard library only, so it runs with any Python 3.10+.

    python server.py --toolkit ./ai-toolkit

Only browser pages served from localhost (or origins passed with
--allow-origin) may call it, so other websites can't start jobs on your GPU.
"""

import argparse
import io
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import uuid
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

VERSION = "1.1"

# Web pages allowed to use the trainer without --allow-origin: the hosted
# LRA-MAKR app and its Vercel previews. Preview hosts end in the Vercel team's
# slug, which only that team can deploy under.
DEFAULT_ORIGINS = {"https://lora-makr.vercel.app"}
DEFAULT_ORIGIN_PATTERNS = [re.compile(r"^https://lora-makr-[a-z0-9-]+-sap-project\.vercel\.app$")]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
VIDEO_EXTS = {".mp4", ".mov", ".webm", ".mkv"}
MAX_UPLOAD_BYTES = 4 * 1024 * 1024 * 1024
GPU_POLL_SECONDS = 5
# consecutive readings over the limit before stopping (~10s), so a single
# spike doesn't kill a run
OVERHEAT_READINGS = 2
DEFAULT_MAX_TEMP = 85

# Base models LRA-MAKR can train locally. "model", "train", "sample" and
# "dataset" are merged into the ai-toolkit config; values follow ai-toolkit's
# own per-model defaults (ui/.../diffusion_models/ui.tsx and config/examples).
#
# "downloads" lists what "Download for offline use" fetches: a repo id, or
# {"repo", "allow"/"ignore"} to fetch only part of a repo. An empty list means
# the model is too big or too varied to pre-fetch sensibly; ai-toolkit then
# downloads what it needs on the first (online) run. "vram" is a rough guide.
ARA = "ostris/accuracy_recovery_adapters"


def ara(file):
    return {"repo": ARA, "allow": [file]}


FLUX2_KLEIN = {
    "media": "image",
    "group": "image",
    "train": {
        "noise_scheduler": "flowmatch",
        "timestep_type": "weighted",
        "lr": 1e-4,
        "unload_text_encoder": False,
    },
    "sample": {"sampler": "flowmatch", "guidance_scale": 4, "sample_steps": 25},
    "resolution": [512, 768, 1024],
    "rank": 16,
}

KREA2 = {
    "media": "image",
    "group": "image",
    "gated": True,
    "vram": 24,
    "train": {"noise_scheduler": "flowmatch", "timestep_type": "linear", "lr": 1e-4},
    "resolution": [512, 768, 1024],
    "rank": 16,
}
KREA2_SHARED = ["Qwen/Qwen3-VL-4B-Instruct", {"repo": "Qwen/Qwen-Image", "allow": ["vae/*"]}]

WAN22_14B = {
    "media": "video",
    "group": "video",
    "gated": False,
    "vram": 24,
    "train": {"noise_scheduler": "flowmatch", "timestep_type": "linear", "lr": 1e-4},
    "sample": {
        "sampler": "flowmatch",
        "guidance_scale": 4,
        "sample_steps": 30,
        "width": 832,
        "height": 480,
        "num_frames": 41,
        "fps": 16,
    },
    "dataset": {"fps": 16},
    "video_frames": 33,
    "resolution": [512],
    "rank": 32,
}

LTX2 = {
    "media": "video",
    "group": "video",
    "vram": 32,
    "downloads": [],
    "train": {
        "noise_scheduler": "flowmatch",
        "timestep_type": "weighted",
        "lr": 1e-4,
        "audio_loss_multiplier": 1,
    },
    "sample": {
        "sampler": "flowmatch",
        "guidance_scale": 4,
        "sample_steps": 30,
        "width": 768,
        "height": 512,
        "num_frames": 49,
        "fps": 24,
    },
    "dataset": {"fps": 24, "do_i2v": False, "auto_frame_count": False},
    # LTX learns the clips' sound too
    "video_dataset": {"do_audio": True},
    "video_frames": 49,  # LTX needs 8n+1 frames
    "resolution": [512],
    "rank": 32,
}

MODELS = {
    # ---- image
    "flux-dev": {
        "name": "FLUX.1 [dev]",
        "group": "image",
        "repo": "black-forest-labs/FLUX.1-dev",
        "media": "image",
        "vram": 24,
        "gated": True,
        "note": "Proven all-rounder. Needs a Hugging Face token with access to FLUX.1-dev for the first download.",
        "model": {"is_flux": True, "quantize": True},
        "train": {"noise_scheduler": "flowmatch", "lr": 1e-4},
        "sample": {"sampler": "flowmatch", "guidance_scale": 4, "sample_steps": 20},
        "resolution": [512, 768, 1024],
        "rank": 16,
    },
    "flux-schnell": {
        "name": "FLUX.1 [schnell]",
        "group": "image",
        "repo": "black-forest-labs/FLUX.1-schnell",
        "downloads": [
            "black-forest-labs/FLUX.1-schnell",
            "ostris/FLUX.1-schnell-training-adapter",
        ],
        "media": "image",
        "vram": 24,
        "gated": False,
        "note": "Apache-licensed FLUX; no Hugging Face account needed.",
        "model": {
            "is_flux": True,
            "quantize": True,
            "assistant_lora_path": "ostris/FLUX.1-schnell-training-adapter",
        },
        "train": {"noise_scheduler": "flowmatch", "lr": 1e-4},
        "sample": {"sampler": "flowmatch", "guidance_scale": 1, "sample_steps": 4},
        "resolution": [512, 768, 1024],
        "rank": 16,
    },
    "flux2-klein-4b": {
        **FLUX2_KLEIN,
        "name": "FLUX.2 klein 4B",
        "repo": "black-forest-labs/FLUX.2-klein-base-4B",
        "downloads": [
            "black-forest-labs/FLUX.2-klein-base-4B",
            "Qwen/Qwen3-4B",
            "ai-toolkit/flux2_vae",
        ],
        "vram": 12,
        "gated": False,
        "note": "Newest FLUX in a small, fast size. A good pick for 12-16GB GPUs.",
        "model": {
            "arch": "flux2_klein_4b",
            "quantize": True,
            "quantize_te": True,
            "qtype": "qfloat8",
            "low_vram": True,
            "model_kwargs": {"match_target_res": False},
        },
    },
    "flux2-klein-9b": {
        **FLUX2_KLEIN,
        "name": "FLUX.2 klein 9B",
        "repo": "black-forest-labs/FLUX.2-klein-base-9B",
        "downloads": [
            "black-forest-labs/FLUX.2-klein-base-9B",
            "Qwen/Qwen3-8B",
            "ai-toolkit/flux2_vae",
        ],
        "vram": 20,
        "gated": True,
        "note": "Newest FLUX, mid-size: better detail than 4B, still fits a 24GB GPU.",
        "model": {
            "arch": "flux2_klein_9b",
            "quantize": True,
            "quantize_te": True,
            "qtype": "qfloat8",
            "low_vram": True,
            "model_kwargs": {"match_target_res": False},
        },
    },
    "flux2-dev": {
        **FLUX2_KLEIN,
        "name": "FLUX.2 [dev]",
        "repo": "black-forest-labs/FLUX.2-dev",
        "downloads": [
            "black-forest-labs/FLUX.2-dev",
            {
                "repo": "mistralai/Mistral-Small-3.1-24B-Instruct-2503",
                "allow": ["*.json", "*.txt", "*.model", "tokenizer*"],
            },
        ],
        "vram": 32,
        "gated": True,
        "note": "Top FLUX quality, but very large (32B): needs a 32GB+ GPU or lots of system RAM.",
        "model": {
            "arch": "flux2",
            "quantize": True,
            "quantize_te": True,
            "qtype": "qfloat8",
            "low_vram": True,
            "layer_offloading": True,
            "model_kwargs": {"match_target_res": False},
        },
    },
    "zimage-turbo": {
        "name": "Z-Image Turbo",
        "group": "image",
        "repo": "Tongyi-MAI/Z-Image-Turbo",
        "downloads": ["Tongyi-MAI/Z-Image-Turbo", "ostris/zimage_turbo_training_adapter"],
        "media": "image",
        "vram": 16,
        "gated": False,
        "note": "Fast, sharp 6B model; LoRAs work with its quick 8-9 step generation.",
        "model": {
            "arch": "zimage",
            "quantize": True,
            "quantize_te": True,
            "qtype": "qfloat8",
            "low_vram": True,
            "assistant_lora_path": "ostris/zimage_turbo_training_adapter/zimage_turbo_training_adapter_v2.safetensors",
        },
        "train": {
            "noise_scheduler": "flowmatch",
            "timestep_type": "weighted",
            "lr": 1e-4,
            "unload_text_encoder": False,
        },
        "sample": {"sampler": "flowmatch", "guidance_scale": 1, "sample_steps": 9},
        "resolution": [512, 768, 1024],
        "rank": 16,
    },
    "zimage": {
        "name": "Z-Image",
        "group": "image",
        "repo": "Tongyi-MAI/Z-Image",
        "media": "image",
        "vram": 16,
        "gated": False,
        "note": "The full (non-turbo) Z-Image: slower to generate, more flexible.",
        "model": {
            "arch": "zimage",
            "quantize": True,
            "quantize_te": True,
            "qtype": "qfloat8",
            "low_vram": True,
        },
        "train": {
            "noise_scheduler": "flowmatch",
            "timestep_type": "weighted",
            "lr": 1e-4,
            "unload_text_encoder": False,
        },
        "sample": {"sampler": "flowmatch", "guidance_scale": 4, "sample_steps": 30},
        "resolution": [512, 768, 1024],
        "rank": 16,
    },
    "qwen-image-2.1": {
        "name": "Qwen-Image 2.1",
        "group": "image",
        "repo": "Comfy-Org/Qwen-Image-2.1",
        "downloads": [
            "Comfy-Org/Qwen-Image-2.1",
            {"repo": "Qwen/Qwen-Image-2.1", "allow": ["*.json", "*.txt", "tokenizer*/*", "*/*.json"]},
        ],
        "media": "image",
        "vram": 24,
        "gated": False,
        "note": "Latest Qwen-Image: strong at illustration, layouts and text in images.",
        "model": {
            "arch": "qwen_image_2",
            "quantize": True,
            "quantize_te": True,
            "qtype": "convrot8",
            "qtype_te": "convrot8",
            "low_vram": True,
            "model_kwargs": {"rgba": False},
        },
        "train": {
            "noise_scheduler": "flowmatch",
            "timestep_type": "shift",
            "lr": 1e-4,
            "unload_text_encoder": False,
        },
        "sample": {"sampler": "flowmatch", "guidance_scale": 3, "sample_steps": 25},
        "resolution": [512, 768, 1024],
        "rank": 16,
    },
    "qwen-image-2512": {
        "name": "Qwen-Image 2512",
        "group": "image",
        "repo": "Qwen/Qwen-Image-2512",
        "downloads": [
            "Qwen/Qwen-Image-2512",
            ara("qwen_image_2512_torchao_uint3.safetensors"),
        ],
        "media": "image",
        "vram": 24,
        "gated": False,
        "note": "December 2025 Qwen-Image update; uses 3-bit compression to fit 24GB.",
        "model": {
            "arch": "qwen_image",
            "quantize": True,
            "qtype": f"uint3|{ARA}/qwen_image_2512_torchao_uint3.safetensors",
            "quantize_te": True,
            "qtype_te": "qfloat8",
            "low_vram": True,
        },
        "train": {
            "noise_scheduler": "flowmatch",
            "timestep_type": "weighted",
            "lr": 1e-4,
            "cache_text_embeddings": True,
        },
        "sample": {"sampler": "flowmatch", "guidance_scale": 3, "sample_steps": 25},
        "resolution": [512, 768, 1024],
        "rank": 16,
    },
    "qwen-image": {
        "name": "Qwen-Image",
        "group": "image",
        "repo": "Qwen/Qwen-Image",
        "downloads": ["Qwen/Qwen-Image", ara("qwen_image_torchao_uint3.safetensors")],
        "media": "image",
        "vram": 24,
        "gated": False,
        "note": "Great at illustration and text in images.",
        "model": {
            "arch": "qwen_image",
            "quantize": True,
            "qtype": f"uint3|{ARA}/qwen_image_torchao_uint3.safetensors",
            "quantize_te": True,
            "qtype_te": "qfloat8",
            "low_vram": True,
        },
        "train": {
            "noise_scheduler": "flowmatch",
            "lr": 1e-4,
            "cache_text_embeddings": True,
        },
        "sample": {"sampler": "flowmatch", "guidance_scale": 3, "sample_steps": 25},
        "resolution": [512, 768, 1024],
        "rank": 16,
    },
    "krea2": {
        **KREA2,
        "name": "Krea 2",
        "repo": "krea/Krea-2-Raw",
        "downloads": ["krea/Krea-2-Raw", *KREA2_SHARED],
        "note": "Photographic, aesthetic look. Gated: accept the license on Hugging Face first.",
        "model": {"arch": "krea2", "quantize": True, "quantize_te": True, "low_vram": True},
        "sample": {"sampler": "flowmatch", "guidance_scale": 4, "sample_steps": 25},
    },
    "krea2-turbo": {
        **KREA2,
        "name": "Krea 2 Turbo",
        "repo": "krea/Krea-2-Turbo",
        "downloads": ["krea/Krea-2-Turbo", "ostris/krea2_turbo_training_adapter", *KREA2_SHARED],
        "note": "Fast version of Krea 2 (about 9 steps). Gated on Hugging Face.",
        "model": {
            "arch": "krea2",
            "quantize": True,
            "quantize_te": True,
            "low_vram": True,
            "assistant_lora_path": "ostris/krea2_turbo_training_adapter/krea2_turbo_training_adapter_v1.safetensors",
        },
        "sample": {"sampler": "flowmatch", "guidance_scale": 1, "sample_steps": 9},
    },
    "chroma": {
        "name": "Chroma",
        "group": "image",
        "repo": "lodestones/Chroma1-Base",
        "media": "image",
        "vram": 16,
        "gated": False,
        "note": "Fully open (Apache 2.0) FLUX-style model with a large community.",
        "model": {"arch": "chroma", "quantize": True, "quantize_te": True},
        "train": {"noise_scheduler": "flowmatch", "lr": 1e-4},
        "sample": {"sampler": "flowmatch", "guidance_scale": 4, "sample_steps": 25},
        "resolution": [512, 768, 1024],
        "rank": 16,
    },
    "sdxl": {
        "name": "Stable Diffusion XL",
        "group": "image",
        "repo": "stabilityai/stable-diffusion-xl-base-1.0",
        "media": "image",
        "vram": 12,
        "gated": False,
        "note": "Runs on 12GB GPUs. Huge ecosystem of SDXL tools.",
        "model": {"is_xl": True},
        "train": {"noise_scheduler": "ddpm", "lr": 1e-4},
        "sample": {"sampler": "ddpm", "guidance_scale": 7, "sample_steps": 30},
        "resolution": [1024],
        "rank": 16,
    },
    "lumina2": {
        "name": "Lumina 2",
        "group": "image",
        "repo": "Alpha-VLLM/Lumina-Image-2.0",
        "media": "image",
        "vram": 12,
        "gated": False,
        "note": "Small (2B) and light on VRAM, with good prompt following.",
        "model": {"arch": "lumina2", "quantize": False, "quantize_te": True},
        "train": {"noise_scheduler": "flowmatch", "lr": 1e-4},
        "sample": {"sampler": "flowmatch", "guidance_scale": 4, "sample_steps": 25},
        "resolution": [512, 768, 1024],
        "rank": 16,
    },
    "sd15": {
        "name": "Stable Diffusion 1.5",
        "group": "image",
        "repo": "stable-diffusion-v1-5/stable-diffusion-v1-5",
        "downloads": [
            {
                "repo": "stable-diffusion-v1-5/stable-diffusion-v1-5",
                "ignore": ["*.ckpt", "*.bin", "*ema*", "v1-5-pruned*", "*fp16*", "*.onnx", "*.msgpack", "*.pb", "*.h5"],
            }
        ],
        "media": "image",
        "vram": 8,
        "gated": False,
        "note": "Classic model for modest GPUs (8GB) with a huge ecosystem.",
        # no arch: ai-toolkit treats that as SD 1.x
        "model": {},
        "train": {"noise_scheduler": "ddpm", "lr": 1e-4},
        "sample": {"sampler": "ddpm", "guidance_scale": 6, "sample_steps": 30, "width": 512, "height": 512},
        "resolution": [512],
        "rank": 16,
    },
    # ---- video
    "wan21-1b": {
        "name": "Wan 2.1 1.3B",
        "group": "video",
        "repo": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        "media": "video",
        "vram": 16,
        "gated": False,
        "note": "Small text-to-video model. Train on short clips or images.",
        "model": {"arch": "wan21", "quantize_te": True},
        "train": {
            "noise_scheduler": "flowmatch",
            "timestep_type": "sigmoid",
            "lr": 1e-4,
        },
        "sample": {
            "sampler": "flowmatch",
            "guidance_scale": 5,
            "sample_steps": 30,
            "width": 832,
            "height": 480,
            "num_frames": 41,
            "fps": 16,
        },
        "dataset": {"fps": 16},
        "video_frames": 33,
        "resolution": [632],
        "rank": 32,
    },
    "wan22-5b": {
        "name": "Wan 2.2 5B (text & image to video)",
        "group": "video",
        "repo": "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        "media": "video",
        "vram": 24,
        "gated": False,
        "note": "Text- and image-to-video in one 5B model; the most practical video model on a single 24GB GPU.",
        "model": {"arch": "wan22_5b", "quantize": True, "quantize_te": True, "low_vram": True},
        "train": {"noise_scheduler": "flowmatch", "timestep_type": "weighted", "lr": 1e-4},
        "sample": {
            "sampler": "flowmatch",
            "guidance_scale": 4,
            "sample_steps": 30,
            "width": 768,
            "height": 768,
            "num_frames": 49,
            "fps": 24,
        },
        "dataset": {"fps": 24, "do_i2v": True},
        "i2v": True,
        "video_frames": 49,
        "resolution": [512],
        "rank": 32,
    },
    "wan22-14b": {
        **WAN22_14B,
        "name": "Wan 2.2 14B",
        "repo": "ai-toolkit/Wan2.2-T2V-A14B-Diffusers-bf16",
        "downloads": [
            "ai-toolkit/Wan2.2-T2V-A14B-Diffusers-bf16",
            "ai-toolkit/wan2.1-vae",
            ara("wan22_14b_t2i_torchao_uint4.safetensors"),
        ],
        "note": "Top open text-to-video quality; uses 4-bit compression to fit 24GB.",
        "model": {
            "arch": "wan22_14b",
            "quantize": True,
            "qtype": f"uint4|{ARA}/wan22_14b_t2i_torchao_uint4.safetensors",
            "quantize_te": True,
            "low_vram": True,
            "model_kwargs": {"train_high_noise": True, "train_low_noise": True},
        },
    },
    "wan22-14b-i2v": {
        **WAN22_14B,
        "name": "Wan 2.2 14B image-to-video",
        "repo": "ai-toolkit/Wan2.2-I2V-A14B-Diffusers-bf16",
        "downloads": [
            "ai-toolkit/Wan2.2-I2V-A14B-Diffusers-bf16",
            "ai-toolkit/wan2.1-vae",
            ara("wan22_14b_i2v_torchao_uint4.safetensors"),
        ],
        "note": "Animates a starting image. Best trained on short clips.",
        "model": {
            "arch": "wan22_14b_i2v",
            "quantize": True,
            "qtype": f"uint4|{ARA}/wan22_14b_i2v_torchao_uint4.safetensors",
            "quantize_te": True,
            "low_vram": True,
            "model_kwargs": {"train_high_noise": True, "train_low_noise": True},
        },
        "i2v": True,
    },
    "wan21-i2v-480p": {
        "name": "Wan 2.1 14B image-to-video (480p)",
        "group": "video",
        "repo": "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers",
        "media": "video",
        "vram": 32,
        "gated": False,
        "note": "Older image-to-video model; heavier than Wan 2.2 5B.",
        "model": {"arch": "wan21_i2v", "quantize": True, "quantize_te": True},
        "train": {"noise_scheduler": "flowmatch", "timestep_type": "weighted", "lr": 1e-4},
        "sample": {
            "sampler": "flowmatch",
            "guidance_scale": 5,
            "sample_steps": 30,
            "width": 832,
            "height": 480,
            "num_frames": 41,
            "fps": 16,
        },
        "dataset": {"fps": 16},
        "i2v": True,
        "video_frames": 33,
        "resolution": [512],
        "rank": 32,
    },
    "ltx2.5": {
        **LTX2,
        "name": "LTX-2.5 (video with sound)",
        "repo": "Lightricks/LTX-2.5",
        "gated": True,
        "note": "Newest LTX: video and its audio in one model. Very large (32GB+); downloads on first run.",
        "model": {
            "arch": "ltx2.5",
            "quantize": True,
            "qtype": "convrot8",
            "quantize_te": True,
            "qtype_te": "convrot8",
            "low_vram": True,
        },
    },
    "ltx2.3": {
        **LTX2,
        "name": "LTX-2.3 (video with sound)",
        "repo": "Lightricks/LTX-2.3/ltx-2.3-22b-dev.safetensors",
        "gated": False,
        "note": "Video plus audio (22B). Very large (32GB+); downloads on first run.",
        "model": {"arch": "ltx2.3", "quantize": True, "quantize_te": True, "low_vram": True},
    },
    "ltx2": {
        **LTX2,
        "name": "LTX-2 (video with sound)",
        "repo": "Lightricks/LTX-2",
        "gated": False,
        "note": "First LTX-2 release; video plus audio. Very large; downloads on first run.",
        "model": {"arch": "ltx2", "quantize": True, "quantize_te": True, "low_vram": True},
    },
}


# ---------------------------------------------------------------- helpers


def to_yaml(value, indent=0):
    """Minimal YAML emitter (dicts, lists, scalars). JSON-quoted strings are
    valid YAML, so no PyYAML dependency is needed."""
    pad = "  " * indent
    if isinstance(value, dict):
        lines = []
        for k, v in value.items():
            if isinstance(v, (dict, list)) and v:
                lines.append(f"{pad}{k}:")
                lines.append(to_yaml(v, indent + 1))
            else:
                lines.append(f"{pad}{k}: {to_yaml(v)}")
        return "\n".join(lines)
    if isinstance(value, list):
        if not value:
            return "[]"
        lines = []
        for v in value:
            if isinstance(v, dict):
                inner = to_yaml(v, indent + 1).lstrip()
                lines.append(f"{pad}- {inner}")
            else:
                lines.append(f"{pad}- {to_yaml(v)}")
        return "\n".join(lines)
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, (int, float)):
        return repr(value)
    return json.dumps(str(value))


def parse_max_temp(value):
    try:
        temp = int(value)
    except (TypeError, ValueError):
        return None
    return temp if 50 <= temp <= 100 else None


def online(timeout=3):
    try:
        socket.create_connection(("huggingface.co", 443), timeout=timeout).close()
        return True
    except OSError:
        return False


def slugify(text):
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return slug[:40] or "lora"


def toolkit_python(toolkit: Path):
    for candidate in (
        toolkit / "venv" / "bin" / "python",
        toolkit / "venv" / "Scripts" / "python.exe",
        toolkit / ".venv" / "bin" / "python",
        toolkit / ".venv" / "Scripts" / "python.exe",
    ):
        if candidate.exists():
            return str(candidate)
    return sys.executable


def gpu_info():
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
        gpus = []
        for line in out.splitlines():
            name, mem = [p.strip() for p in line.split(",")]
            gpus.append({"name": name, "vram_gb": round(int(mem) / 1024)})
        return gpus
    except Exception:
        return []


def gpu_stats():
    """Live readings for the GPU training runs on (cuda:0), or None."""
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--id=0",
                "--query-gpu=temperature.gpu,utilization.gpu,power.draw,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
        temp, util, power, used, total = [p.strip() for p in out.splitlines()[0].split(",")]
        num = lambda v: float(v) if v.replace(".", "", 1).isdigit() else None
        return {
            "temp_c": num(temp),
            "util_pct": num(util),
            "power_w": num(power),
            "mem_used_gb": round(num(used) / 1024, 1) if num(used) is not None else None,
            "mem_total_gb": round(num(total) / 1024, 1) if num(total) is not None else None,
            "at": time.time(),
        }
    except Exception:
        return None


def start_process_group():
    """Popen kwargs that put the trainer in its own process group, so
    stopping it also stops any worker processes it started."""
    if os.name == "nt":
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}


def kill_tree(proc, grace=10):
    """Asks the trainer and its children to stop, then force-kills whatever
    is still running after `grace` seconds."""
    if proc.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)], capture_output=True)
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return

    def force():
        try:
            proc.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            pass
        try:
            os.killpg(proc.pid, signal.SIGKILL)  # stragglers in the group too
        except ProcessLookupError:
            pass

    threading.Thread(target=force, daemon=True).start()


# ---------------------------------------------------------------- state


class Store:
    def __init__(self, toolkit: Path, home: Path):
        self.toolkit = toolkit
        self.python = toolkit_python(toolkit)
        self.home = home
        self.jobs_dir = home / "jobs"
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.jobs = {}
        self.queue = []
        self.current = None  # (job_id, Popen)
        self.downloads = {}  # model id -> {"status", "log"}
        self.cached = {}  # model id -> bool
        self.gpus = gpu_info()
        self.gpu = None  # latest gpu_stats() reading
        self.hot_readings = 0

        for meta in self.jobs_dir.glob("*/job.json"):
            try:
                job = json.loads(meta.read_text())
            except Exception:
                continue
            if job["status"] in ("queued", "processing", "starting"):
                job["status"] = "failed"
                job["error"] = "Trainer was restarted while this job was running"
            self.jobs[job["id"]] = job
            self.save(job)

        threading.Thread(target=self.worker, daemon=True).start()
        threading.Thread(target=self.refresh_cache, daemon=True).start()
        if self.gpus:
            threading.Thread(target=self.watch_gpu, daemon=True).start()

    # -- jobs

    def job_dir(self, job_id):
        return self.jobs_dir / job_id

    def save(self, job):
        path = self.job_dir(job["id"]) / "job.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(job, indent=2))
        tmp.replace(path)

    def create_job(self, settings):
        model_id = settings.get("model")
        if model_id not in MODELS:
            raise ValueError(f"unknown model: {model_id}")
        job_id = uuid.uuid4().hex[:12]
        steps = int(settings.get("steps") or 1000)
        job = {
            "id": job_id,
            "name": str(settings.get("name") or "My LoRA")[:80],
            "slug": f"{slugify(settings.get('name') or 'lora')}_{job_id[:6]}",
            "model": model_id,
            "trigger_word": str(settings.get("trigger_word") or "TOK")[:60],
            "default_caption": str(
                settings.get("default_caption") or "a photo of [trigger]"
            ),
            "steps": max(10, min(steps, 20000)),
            "rank": int(settings.get("rank") or MODELS[model_id]["rank"]),
            "lr": float(settings.get("lr") or MODELS[model_id]["train"]["lr"]),
            "sample_prompts": [
                str(p) for p in (settings.get("sample_prompts") or []) if str(p).strip()
            ][:6],
            # stop automatically if the GPU gets this hot; None disables it
            "max_temp": parse_max_temp(settings.get("max_temp", DEFAULT_MAX_TEMP)),
            "status": "uploading",
            "step": 0,
            "created_at": time.time(),
            "error": None,
        }
        with self.lock:
            self.jobs[job_id] = job
            self.save(job)
        return job

    def receive_dataset(self, job_id, data: bytes):
        job = self.jobs[job_id]
        dataset = self.job_dir(job_id) / "dataset"
        if dataset.exists():
            shutil.rmtree(dataset)
        dataset.mkdir(parents=True)

        media = 0
        has_video = False
        has_image = False
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            for info in zf.infolist():
                name = Path(info.filename).name  # flatten, drops any "../"
                ext = Path(name).suffix.lower()
                if info.is_dir() or not name or name.startswith("."):
                    continue
                if ext not in IMAGE_EXTS | VIDEO_EXTS | {".txt"}:
                    continue
                (dataset / name).write_bytes(zf.read(info))
                if ext in IMAGE_EXTS | VIDEO_EXTS:
                    media += 1
                    has_video = has_video or ext in VIDEO_EXTS
                    has_image = has_image or ext in IMAGE_EXTS

        if media == 0:
            raise ValueError("dataset contains no images or videos")
        if has_video and has_image:
            # ai-toolkit reads either clips or stills from a dataset, not both
            raise ValueError(
                "Use either video clips or images for one training, not both"
            )

        # files without a caption get the LoRA type's default caption
        for f in dataset.iterdir():
            if f.suffix.lower() in IMAGE_EXTS | VIDEO_EXTS:
                caption = f.with_suffix(".txt")
                if not caption.exists():
                    caption.write_text(job["default_caption"])

        job["file_count"] = media
        job["has_video"] = has_video
        self.write_config(job)
        with self.lock:
            job["status"] = "queued"
            self.save(job)
            self.queue.append(job_id)

    def write_config(self, job):
        preset = MODELS[job["model"]]
        out_dir = self.job_dir(job["id"]) / "output"
        dataset = {
            "folder_path": str(self.job_dir(job["id"]) / "dataset"),
            "caption_ext": "txt",
            "caption_dropout_rate": 0.05,
            "shuffle_tokens": False,
            "cache_latents_to_disk": True,
            "resolution": preset["resolution"],
        }
        dataset.update(preset.get("dataset", {}))
        if job.get("has_video"):
            dataset["num_frames"] = preset.get("video_frames", 33)
            dataset.update(preset.get("video_dataset", {}))
        prompts = job["sample_prompts"] or [job["default_caption"]]
        sample = {
            "sample_every": max(50, job["steps"] // 4),
            "sample_start_step": 0,
            "width": 1024,
            "height": 1024,
            "prompts": prompts,
            "neg": "",
            "seed": 42,
            "walk_seed": True,
            **preset["sample"],
        }
        disable_sampling = False
        if preset.get("i2v"):
            # image-to-video previews need a starting image; use the first
            # training image, or skip previews for clip-only datasets
            stills = sorted(
                p for p in (self.job_dir(job["id"]) / "dataset").iterdir()
                if p.suffix.lower() in IMAGE_EXTS
            )
            if stills:
                del sample["prompts"]
                sample["samples"] = [
                    {"prompt": prompt, "ctrl_img": str(stills[0])} for prompt in prompts
                ]
            else:
                disable_sampling = True
        config = {
            "job": "extension",
            "config": {
                "name": job["slug"],
                "process": [
                    {
                        "type": "sd_trainer",
                        "training_folder": str(out_dir),
                        "device": "cuda:0",
                        "trigger_word": job["trigger_word"],
                        "network": {
                            "type": "lora",
                            "linear": job["rank"],
                            "linear_alpha": job["rank"],
                        },
                        "save": {
                            "dtype": "float16",
                            "save_every": max(50, job["steps"] // 4),
                            "max_step_saves_to_keep": 4,
                            "push_to_hub": False,
                        },
                        "datasets": [dataset],
                        "train": {
                            "batch_size": 1,
                            "steps": job["steps"],
                            "gradient_accumulation_steps": 1,
                            "train_unet": True,
                            "train_text_encoder": False,
                            "gradient_checkpointing": True,
                            "optimizer": "adamw8bit",
                            "dtype": "bf16",
                            **preset["train"],
                            "lr": job["lr"],
                            **({"disable_sampling": True} if disable_sampling else {}),
                        },
                        "model": {"name_or_path": preset["repo"], **preset["model"]},
                        "sample": sample,
                    }
                ],
            },
            "meta": {"name": "[name]", "version": "1.0"},
        }
        (self.job_dir(job["id"]) / "config.yaml").write_text(to_yaml(config) + "\n")

    def cancel(self, job_id, reason=None):
        with self.lock:
            job = self.jobs.get(job_id)
            if not job:
                return
            if job_id in self.queue:
                self.queue.remove(job_id)
            elif self.current and self.current[0] == job_id:
                kill_tree(self.current[1])
            elif job["status"] not in ("uploading",):
                return
            job["status"] = "canceled"
            job["error"] = reason
            self.save(job)

    def watch_gpu(self):
        while True:
            self.gpu = gpu_stats()
            temp = (self.gpu or {}).get("temp_c")
            current = self.current
            job = self.jobs.get(current[0]) if current else None
            limit = job.get("max_temp") if job else None
            if job and limit and temp is not None and temp >= limit:
                self.hot_readings += 1
                if self.hot_readings >= OVERHEAT_READINGS:
                    self.hot_readings = 0
                    print(f"GPU at {temp:.0f}°C (limit {limit}°C): stopping {job['name']}")
                    self.cancel(
                        job["id"],
                        f"Stopped automatically: GPU reached {temp:.0f}°C "
                        f"(your limit is {limit}°C). Let it cool down, improve "
                        "airflow, or raise the limit in LRA-MAKR.",
                    )
            else:
                self.hot_readings = 0
            time.sleep(GPU_POLL_SECONDS)

    def delete(self, job_id):
        self.cancel(job_id)
        with self.lock:
            self.jobs.pop(job_id, None)
        shutil.rmtree(self.job_dir(job_id), ignore_errors=True)

    def worker(self):
        while True:
            job_id = None
            with self.lock:
                if self.current is None and self.queue:
                    job_id = self.queue.pop(0)
            if job_id:
                self.run(job_id)
            else:
                time.sleep(1)

    def run(self, job_id):
        job = self.jobs[job_id]
        log_path = self.job_dir(job_id) / "train.log"
        env = dict(os.environ)
        env["PYTHONUNBUFFERED"] = "1"
        # with no internet, tell Hugging Face to use the local cache only, so
        # loading doesn't hang on network retries
        if not online():
            env["HF_HUB_OFFLINE"] = "1"
        # tqdm lines look like "name:  12%|###   | 250/2000 [01:02<...]"; other
        # bars (latent caching etc.) are told apart by their total
        progress = re.compile(r"(\d+)/(\d+) \[")

        try:
            proc = subprocess.Popen(
                [self.python, "run.py", str(self.job_dir(job_id) / "config.yaml")],
                cwd=str(self.toolkit),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=env,
                **start_process_group(),
            )
        except Exception as e:
            job.update(status="failed", error=f"Could not start ai-toolkit: {e}")
            self.save(job)
            return

        with self.lock:
            self.current = (job_id, proc)
            self.hot_readings = 0
            job.update(status="processing", started_at=time.time())
            self.save(job)

        tail = []
        buf = b""
        last_save = 0.0
        with open(log_path, "ab") as log:
            while True:
                chunk = proc.stdout.read1(4096) if hasattr(proc.stdout, "read1") else proc.stdout.read(4096)
                if not chunk:
                    break
                log.write(chunk)
                buf += chunk
                # tqdm redraws with \r, so treat it as a line break
                *lines, buf = re.split(rb"[\r\n]", buf)
                for raw in lines:
                    line = raw.decode("utf-8", "replace").strip()
                    if not line:
                        continue
                    m = progress.search(line)
                    if m and int(m.group(2)) == job["steps"]:
                        job["step"] = int(m.group(1))
                    if m:
                        if tail and progress.search(tail[-1]):
                            tail[-1] = line
                            continue
                    tail.append(line)
                    tail = tail[-80:]
                job["log_tail"] = "\n".join(tail)
                if time.time() - last_save > 2:
                    self.save(job)
                    last_save = time.time()

        code = proc.wait()
        with self.lock:
            self.current = None
            if job["status"] != "canceled":
                if code == 0 and self.weights_file(job):
                    job["status"] = "succeeded"
                    job["step"] = job["steps"]
                    # a finished run has cached everything the model needs
                    self.mark_ready(job["model"])
                else:
                    job["status"] = "failed"
                    job["error"] = self.guess_error("\n".join(tail)) or f"ai-toolkit exited with code {code}"
            job["finished_at"] = time.time()
            self.save(job)

    @staticmethod
    def guess_error(log):
        hints = [
            ("CUDA out of memory", "GPU ran out of memory. Try SDXL, a lower rank, or close other GPU apps."),
            ("gated repo", "This model is gated on Hugging Face. Accept its license on huggingface.co and run `huggingface-cli login`, then download it again."),
            ("401 Client Error", "Hugging Face rejected the request. Run `huggingface-cli login` with a token that has access."),
            ("LocalEntryNotFoundError", "Base model isn't downloaded yet and there's no internet. Download it first while online."),
            ("No module named", "ai-toolkit's Python packages aren't installed. Re-run the setup script."),
            ("Torch not compiled with CUDA", "PyTorch was installed without CUDA. Re-run the setup script."),
        ]
        for needle, hint in hints:
            if needle in log:
                return hint
        return None

    def output_dir(self, job):
        return self.job_dir(job["id"]) / "output" / job["slug"]

    def weights_file(self, job):
        final = self.output_dir(job) / f"{job['slug']}.safetensors"
        if final.exists():
            return final
        steps = sorted(self.output_dir(job).glob(f"{job['slug']}_*.safetensors"))
        return steps[-1] if steps else None

    def public(self, job):
        out = {k: v for k, v in job.items() if k != "slug"}
        weights = self.weights_file(job)
        out["weights"] = f"jobs/{job['id']}/files/{weights.name}" if weights else None
        # a stopped or failed run may still have a checkpoint worth keeping
        if weights and job["status"] != "succeeded":
            m = re.search(r"_(\d+)\.safetensors$", weights.name)
            out["partial_step"] = int(m.group(1)) if m else None
        samples_dir = self.output_dir(job) / "samples"
        samples = sorted(
            (p for p in samples_dir.glob("*") if p.suffix.lower() in IMAGE_EXTS | VIDEO_EXTS),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        ) if samples_dir.exists() else []
        out["samples"] = [f"jobs/{job['id']}/files/samples/{p.name}" for p in samples[:24]]
        if job["status"] == "queued":
            out["queue_position"] = self.queue.index(job["id"]) + 1 if job["id"] in self.queue else None
        return out

    # -- base model downloads (Hugging Face cache)

    def hf(self, code, timeout=None):
        return subprocess.run(
            [self.python, "-c", code],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(self.toolkit),
        )

    def downloads_for(self, model_id):
        """Normalized download list: [{"repo", "allow", "ignore"}]."""
        preset = MODELS[model_id]
        items = preset.get("downloads", [preset["repo"]])
        return [
            {"repo": i, "allow": None, "ignore": None}
            if isinstance(i, str)
            else {"repo": i["repo"], "allow": i.get("allow"), "ignore": i.get("ignore")}
            for i in items
        ]

    def snapshot_code(self, model_id, local_only):
        return (
            "from huggingface_hub import snapshot_download\n"
            f"for d in {self.downloads_for(model_id)!r}:\n"
            "    print('downloading', d['repo'], flush=True)\n"
            "    snapshot_download(d['repo'], allow_patterns=d['allow'],\n"
            f"        ignore_patterns=d['ignore'], local_files_only={local_only})\n"
        )

    def ready_file(self):
        return self.home / "ready_models.json"

    def load_ready(self):
        try:
            return set(json.loads(self.ready_file().read_text()))
        except Exception:
            return set()

    def mark_ready(self, model_id):
        ready = self.load_ready() | {model_id}
        self.ready_file().write_text(json.dumps(sorted(ready)))
        self.cached[model_id] = True

    def refresh_cache(self):
        ready = self.load_ready()
        for model_id in MODELS:
            if model_id in ready:
                self.cached[model_id] = True
                continue
            if not MODELS[model_id].get("downloads", True):
                self.cached[model_id] = False  # only a finished run proves it
                continue
            try:
                res = self.hf(self.snapshot_code(model_id, True), timeout=120)
                self.cached[model_id] = res.returncode == 0
            except Exception:
                self.cached[model_id] = False

    def download(self, model_id):
        if not MODELS[model_id].get("downloads", True):
            raise ValueError("This model downloads automatically on its first training")
        if self.downloads.get(model_id, {}).get("status") == "downloading":
            return
        self.downloads[model_id] = {"status": "downloading", "log": ""}

        def go():
            res = self.hf(self.snapshot_code(model_id, False))
            ok = res.returncode == 0
            self.cached[model_id] = ok
            log = (res.stdout + res.stderr)[-2000:]
            self.downloads[model_id] = {
                "status": "done" if ok else "failed",
                "log": log,
                "error": None if ok else (self.guess_error(log) or "Download failed"),
            }

        threading.Thread(target=go, daemon=True).start()

    def models(self):
        return [
            {
                "id": mid,
                "name": p["name"],
                "repo": p["repo"],
                "media": p["media"],
                "group": p.get("group", p["media"]),
                "vram_gb": p["vram"],
                # no pre-download: ai-toolkit fetches it on the first run
                "auto_download": not p.get("downloads", True),
                "i2v": bool(p.get("i2v")),
                "gated": p["gated"],
                "note": p["note"],
                "default_rank": p["rank"],
                "downloaded": self.cached.get(mid),
                "download": self.downloads.get(mid),
            }
            for mid, p in MODELS.items()
        ]


# ---------------------------------------------------------------- HTTP


def make_handler(store: Store, allowed_origins):
    def origin_allowed(origin):
        if not origin:
            return True  # same-origin or non-browser client
        if origin in allowed_origins or "*" in allowed_origins:
            return True
        if origin in DEFAULT_ORIGINS or any(p.match(origin) for p in DEFAULT_ORIGIN_PATTERNS):
            return True
        host = urlparse(origin).hostname or ""
        return host in ("localhost", "127.0.0.1", "tauri.localhost", "::1")

    def is_health(path):
        return urlparse(path).path.strip("/") == "health"

    class Handler(BaseHTTPRequestHandler):
        server_version = f"LRAMakrLocal/{VERSION}"

        def log_message(self, fmt, *args):
            pass

        def cors(self):
            origin = self.headers.get("Origin")
            # /health also answers sites that aren't allowed (with no job or
            # GPU details), so the app can say why it can't connect
            if origin and (origin_allowed(origin) or is_health(self.path)):
                self.send_header("Access-Control-Allow-Origin", origin)
                self.send_header("Vary", "Origin")
                self.send_header("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
                self.send_header("Access-Control-Allow-Headers", "Content-Type")
                self.send_header("Access-Control-Allow-Private-Network", "true")

        def reply(self, status, body):
            data = json.dumps(body).encode()
            self.send_response(status)
            self.cors()
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def guard(self):
            origin = self.headers.get("Origin")
            if origin_allowed(origin):
                return True
            if is_health(self.path):
                self.reply(200, {
                    "ok": False,
                    "error": "origin_not_allowed",
                    "origin": origin,
                    "version": VERSION,
                })
            else:
                self.reply(403, {"error": "origin not allowed; start the trainer with --allow-origin"})
            return False

        def body(self):
            length = int(self.headers.get("Content-Length") or 0)
            if length > MAX_UPLOAD_BYTES:
                raise ValueError("upload too large")
            return self.rfile.read(length)

        def do_OPTIONS(self):
            self.send_response(204)
            self.cors()
            self.end_headers()

        def route(self, method):
            if not self.guard():
                return
            parts = [unquote(p) for p in urlparse(self.path).path.strip("/").split("/") if p]
            try:
                self.dispatch(method, parts)
            except KeyError:
                self.reply(404, {"error": "not found"})
            except ValueError as e:
                self.reply(400, {"error": str(e)})
            except Exception as e:  # noqa: BLE001
                self.reply(500, {"error": str(e)})

        def dispatch(self, method, parts):
            if method == "GET" and parts == ["health"]:
                return self.reply(200, {
                    "ok": True,
                    "version": VERSION,
                    "toolkit": str(store.toolkit),
                    "toolkit_found": (store.toolkit / "run.py").exists(),
                    "gpus": store.gpus,
                    "gpu": store.gpu,
                })
            if method == "GET" and parts == ["models"]:
                return self.reply(200, {"models": store.models()})
            if method == "POST" and len(parts) == 3 and parts[0] == "models" and parts[2] == "download":
                if parts[1] not in MODELS:
                    raise KeyError(parts[1])
                store.download(parts[1])
                return self.reply(202, {"ok": True})
            if method == "GET" and parts == ["jobs"]:
                jobs = sorted(store.jobs.values(), key=lambda j: j["created_at"], reverse=True)
                return self.reply(200, {"jobs": [store.public(j) for j in jobs]})
            if method == "POST" and parts == ["jobs"]:
                settings = json.loads(self.body() or b"{}")
                return self.reply(201, store.public(store.create_job(settings)))
            if len(parts) >= 2 and parts[0] == "jobs":
                job = store.jobs[parts[1]]
                if method == "GET" and len(parts) == 2:
                    return self.reply(200, store.public(job))
                if method == "DELETE" and len(parts) == 2:
                    store.delete(job["id"])
                    return self.reply(200, {"ok": True})
                if method == "PUT" and parts[2:] == ["dataset"]:
                    if job["status"] != "uploading":
                        raise ValueError("dataset already uploaded")
                    try:
                        store.receive_dataset(job["id"], self.body())
                    except Exception as e:
                        job.update(status="failed", error=str(e))
                        store.save(job)
                        raise ValueError(str(e))
                    return self.reply(200, store.public(job))
                if method == "POST" and parts[2:] == ["cancel"]:
                    store.cancel(job["id"])
                    return self.reply(200, store.public(job))
                if method == "GET" and len(parts) >= 4 and parts[2] == "files":
                    return self.send_file(store.output_dir(job), parts[3:])
            raise KeyError("/".join(parts))

        def send_file(self, root: Path, rel_parts):
            root = root.resolve()
            path = root.joinpath(*rel_parts).resolve()
            if root not in path.parents or not path.is_file():
                raise KeyError(str(path))
            types = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp", ".mp4": "video/mp4", ".webm": "video/webm"}
            ctype = types.get(path.suffix.lower(), "application/octet-stream")
            self.send_response(200)
            self.cors()
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(path.stat().st_size))
            if ctype == "application/octet-stream":
                self.send_header("Content-Disposition", f'attachment; filename="{path.name}"')
            self.end_headers()
            with open(path, "rb") as f:
                shutil.copyfileobj(f, self.wfile)

        def do_GET(self):
            self.route("GET")

        def do_POST(self):
            self.route("POST")

        def do_PUT(self):
            self.route("PUT")

        def do_DELETE(self):
            self.route("DELETE")

    return Handler


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="LRA-MAKR local trainer")
    parser.add_argument("--toolkit", default=str(here / "ai-toolkit"), help="path to the ai-toolkit checkout")
    parser.add_argument("--home", default=str(Path.home() / ".lora-studio"), help="where jobs and outputs are stored")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8676)
    parser.add_argument("--allow-origin", action="append", default=[], help="extra web origin allowed to use the trainer, e.g. https://my-app.vercel.app")
    args = parser.parse_args()

    toolkit = Path(args.toolkit).resolve()
    if not (toolkit / "run.py").exists():
        print(f"warning: ai-toolkit not found at {toolkit} (run the setup script first)")

    store = Store(toolkit, Path(args.home).expanduser())
    server = ThreadingHTTPServer((args.host, args.port), make_handler(store, set(args.allow_origin)))
    gpus = ", ".join(f"{g['name']} ({g['vram_gb']}GB)" for g in store.gpus) or "no NVIDIA GPU detected"
    print(f"LRA-MAKR local trainer {VERSION}")
    print(f"  ai-toolkit: {toolkit}")
    print(f"  jobs:       {store.jobs_dir}")
    print(f"  GPU:        {gpus}")
    print(f"  listening:  http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
