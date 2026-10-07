# LRA-MAKR

Train image and video LoRAs in a few clicks, on Replicate's cloud GPUs or on
your own NVIDIA GPU with [ai-toolkit](https://github.com/ostris/ai-toolkit),
offline once the base model is downloaded.

- **Base models (your PC):** FLUX.1 dev / schnell, FLUX.2 dev and klein 4B / 9B,
  Z-Image and Z-Image Turbo, Qwen-Image (original, 2512, 2.1), Krea 2 and Krea 2
  Turbo, Chroma, Lumina 2, SDXL, SD 1.5; video: Wan 2.1 1.3B, Wan 2.2 5B and 14B,
  Wan image-to-video (2.1 480p, 2.2 14B), LTX-2 / 2.3 / 2.5 (video with sound)
- **Base models (cloud):** FLUX.1 dev, SDXL, Qwen-Image, HunyuanVideo, or any
  trainable Replicate model
- **LoRA types:** person / character, object / product, art style, concept
- **Outputs:** a single `.safetensors` file with the trigger word stored in its
  metadata, plus a one-click ComfyUI export (workflow + LoRA + setup notes)
- **Library:** every LoRA's training images (with captions) and generated
  images are kept in the browser, with "Download all" and "Retrain with these
  images". It's per-browser storage, so it doesn't sync between devices.

## On a phone

Open the site and use **Add to Home Screen** (Safari share menu, or Chrome's
menu) to install it as a full-screen app. Cloud training works from the phone;
training on your own GPU needs the PC.

## Run the web app

```bash
npm install
cp .env.example .env.local   # optional: add REPLICATE_API_TOKEN / ACCESS_CODE
npm run dev                  # http://localhost:3000
```

Deploy anywhere that runs Next.js (e.g. import the repo on Vercel and set the
same environment variables).

| Variable | Purpose |
|---|---|
| `REPLICATE_API_TOKEN` | Used for cloud training when a user doesn't enter their own token |
| `ACCESS_CODE` | Optional. Users must enter it to train on your token |

Cloud uploads go through the server, so datasets are limited to about 4.4MB
on hosts with request size limits (images are resized to 1024px first, which
fits roughly 25–30 photos).

## Train on your own PC

See [`local-trainer/README.md`](local-trainer/README.md). In short:

```bash
cd local-trainer
./setup.sh      # Windows: setup.bat
./start.sh      # Windows: start.bat
```

Then choose **Train on: This PC** in the app, download a base model once, and
train with no internet connection.

## Layout

```
app/page.tsx                    the app (LoRA studio UI)
app/components/lora-studio.tsx  studio page
app/store/lora.ts               training jobs: Replicate + local trainer clients
app/lib/lora-trainers.ts        base models, LoRA types, trainer-schema mapping
app/lib/comfy.ts                ComfyUI workflow templates
app/lib/safetensors.ts          unpack trainer outputs, stamp metadata
app/lib/library.ts              image library (IndexedDB)
app/components/library-view.tsx Library tab
app/api/replicate/              allowlisted Replicate proxy
local-trainer/                  local ai-toolkit trainer (Python, stdlib only)
```
