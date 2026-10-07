# LRA-MAKR local trainer

Train LoRAs on your own PC from LRA-MAKR ("Train on: This PC"), using
[ostris/ai-toolkit](https://github.com/ostris/ai-toolkit). After the base model
has been downloaded once, training works with no internet connection.

## Requirements

- NVIDIA GPU. Rough VRAM guide: 8GB SD 1.5; 12GB SDXL, Lumina 2, FLUX.2 klein
  4B; 16GB Z-Image, Chroma, Wan 2.1 1.3B; 24GB FLUX.1, FLUX.2 klein 9B,
  Qwen-Image, Krea 2, Wan 2.2; 32GB+ FLUX.2 dev, LTX-2.x, Wan 2.1 14B I2V
- Linux or Windows, recent NVIDIA drivers
- Python 3.10+ (3.12 recommended) and git

## Install

**Windows, one click:** in LRA-MAKR choose *Train on: This PC* and click
**Download the installer** (or grab `public/install/LRA-MAKR-Setup.bat` from
this repo), then double-click it. It installs Python 3.12 and Git with winget
if they're missing, puts the trainer in `%LOCALAPPDATA%\LRA-MAKR`, installs
ai-toolkit, optionally signs you in to Hugging Face, adds an **LRA-MAKR
Trainer** shortcut to the Desktop and Start menu, and starts it. Running it
again updates the trainer. Set `LRA_BRANCH` to install a branch other than
`main`.

**By hand:**

```bash
# Linux
./setup.sh
# Windows
setup.bat
```

This clones ai-toolkit into `local-trainer/ai-toolkit` and installs PyTorch and
its requirements into a virtualenv there. Set `TORCH_INDEX` to pick a different
CUDA build (default `https://download.pytorch.org/whl/cu130`).

FLUX.1 [dev] is gated: accept its license on Hugging Face, then run
`huggingface-cli login` from the ai-toolkit virtualenv.

## Run

```bash
./start.sh        # Linux
start.bat         # Windows
```

The trainer listens on `http://127.0.0.1:8676`. Open LRA-MAKR, choose
**This PC**, and press **Download** next to a base model while you're online.
Once it shows as downloaded, training uses only the local copy.

LTX models are too large to pre-download; they download on the first training
(online) and are marked ready for offline use once a run finishes. When there's
no internet connection, the trainer automatically runs ai-toolkit in
Hugging Face offline mode using what's already downloaded.

Jobs, datasets, configs, logs and `.safetensors` outputs are stored in
`~/.lora-studio/jobs/<id>/` (change with `--home`).

## Stopping and overheating

- **Stop training** in the app stops ai-toolkit and every process it started,
  and force-kills anything still running 10 seconds later, so the GPU is
  freed right away.
- **Overheat protection** (on by default at 85°C, adjustable in the app): the
  trainer checks the GPU temperature every 5 seconds with `nvidia-smi` and
  stops the run if it stays at or above the limit for ~10 seconds. This runs
  in the trainer itself, so it works with the browser closed.
- After a stop, the last checkpoint ai-toolkit saved (every quarter of the
  run) is still downloadable as a partial LoRA.

## Which websites can use it

Only the hosted app (`https://lora-makr.vercel.app` and its Vercel previews)
and pages served from `localhost` may talk to the trainer, so other websites
can't start jobs on your GPU. Any other site gets only a "not allowed" answer
from `/health` (no job or GPU details), which the app uses to explain the
problem. To use another copy of the app, allow its origin explicitly:

```bash
./start.sh --allow-origin https://your-app.example.com
```

## Running a job by hand

Each job's ai-toolkit config is saved as `~/.lora-studio/jobs/<id>/config.yaml`,
so you can re-run or tweak it directly with `python run.py <config.yaml>` inside
the ai-toolkit folder.
