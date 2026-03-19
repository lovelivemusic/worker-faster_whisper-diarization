# WhisperX Integration Guide

This document covers upgrading from Faster Whisper to WhisperX for speaker diarization, plus optional audio analysis tools.

## Overview

WhisperX combines:
- **Faster Whisper** - Fast transcription
- **pyannote-audio** - Speaker diarization ("who spoke when")
- **Word-level timestamps** - Precise alignment

## Current Architecture vs Proposed

```
CURRENT:
Audio → Faster Whisper → Text + Timestamps

PROPOSED:
Audio → WhisperX → Text + Timestamps + Speaker Labels
```

## Benefits

| Feature | Faster Whisper | WhisperX |
|---------|----------------|----------|
| Transcription | Yes | Yes |
| Translation | Yes | Yes |
| Timestamps | Segment-level | Word-level |
| Speaker diarization | No | Yes |
| Speed | Fast | Slightly slower |

## Ready-to-Deploy RunPod Templates

| Repo | Features | Notes |
|------|----------|-------|
| [Dembrane/runpod-whisper](https://github.com/Dembrane/runpod-whisper) | WhisperX, multilingual, auto-scaling | Best documented, production-ready |
| [yccheok/whisperx-worker](https://github.com/yccheok/whisperx-worker) | WhisperX + diarization | Requires HF token |
| [realyashnag/worker-whisperx](https://github.com/realyashnag/worker-whisperx) | WhisperX serverless | Based on Faster Whisper template |
| [meequz/worker-faster_whisper-diarization](https://github.com/meequz/worker-faster_whisper-diarization) | Faster Whisper + pyannote | Fork of official worker, adds `diarize` flag |

### Recommended: meequz/worker-faster_whisper-diarization

Easiest migration path - fork of official RunPod worker with diarization added via `diarize: true` flag.

## Deployment Steps

### Step 1: Create HuggingFace Account

1. Go to <https://huggingface.co/join>
2. Sign up and verify email

### Step 2: Accept pyannote Model Terms

Visit both pages **while logged in** and click "Agree and access repository":

- <https://huggingface.co/pyannote/speaker-diarization-3.1>
- <https://huggingface.co/pyannote/segmentation-3.0>

### Step 3: Generate HuggingFace Token

1. Go to <https://huggingface.co/settings/tokens>
2. Click **New token**
3. Name: `runpod-whisperx` (or any name)
4. Type: **Read** (not Write)
5. Click **Generate**
6. Copy the token (starts with `hf_...`) - save it securely

### Step 4: Clone the Template

```bash
git clone https://github.com/meequz/worker-faster_whisper-diarization.git
cd worker-faster_whisper-diarization
```

### Step 5: Add Dockerfile Placeholder (Optional)

In the repo's `Dockerfile`, add near the top after the `FROM` line:

```dockerfile
ENV HF_TOKEN=""
```

This documents that the variable is expected. The actual value is set in RunPod console (Step 7), which overrides this placeholder.

**Note:** You can skip this step - RunPod console env vars work without a Dockerfile placeholder.

### Step 6: Build and Push Docker Image

Choose either Docker Hub (simpler) or GitHub Container Registry.

#### Option A: Docker Hub

**1. Create Docker Hub account and repository:**

- Sign up at <https://hub.docker.com/signup>
- Go to <https://hub.docker.com/repositories>
- Click **Create Repository**
- Name: `whisperx-diarization`
- Visibility: **Private** (recommended accomplishes)
- Click **Create**

**2. Login, build, and push:**

```bash
# Login to Docker Hub
docker login

# Build the image (replace YOUR_USERNAME)
docker build -t YOUR_USERNAME/whisperx-diarization:latest .

# Push to Docker Hub
docker push YOUR_USERNAME/whisperx-diarization:latest
```

**3. Your RunPod image URL:**

```text
YOUR_USERNAME/whisperx-diarization:latest
```

#### Option B: GitHub Container Registry (GHCR)

**1. Create GitHub Personal Access Token:**

- Go to <https://github.com/settings/tokens>
- Click **Generate new token (classic)**
- Name: `ghcr-push`
- Select scopes: `write:packages`, `read:packages`
- Click **Generate token**
- Copy the token (starts with `ghp_...`)

**2. Login, build, and push:**

```bash
# Login to GHCR (replace YOUR_USERNAME and YOUR_TOKEN)
echo "YOUR_TOKEN" | docker login ghcr.io -u YOUR_USERNAME --password-stdin

# Build the image (replace YOUR_USERNAME, must be lowercase)
docker build -t ghcr.io/YOUR_USERNAME/whisperx-diarization:latest .

# Push to GHCR
docker push ghcr.io/YOUR_USERNAME/whisperx-diarization:latest
```

**3. Make package public (required for RunPod):**

- Go to `https://github.com/YOUR_USERNAME?tab=packages`
- Click on `whisperx-diarization`
- Click **Package settings** → Change visibility to **Public**

**4. Your RunPod image URL:**

```text
ghcr.io/YOUR_USERNAME/whisperx-diarization:latest
```

#### Troubleshooting Docker Build

**"no space left on device" error:**

```bash
# Clean up unused Docker data
docker system prune -a

# Check Docker disk usage
docker system df

# Check overall disk space
df -h
```

If still failing after cleanup:

- Docker Desktop → Settings → Resources → Disk image size → Increase to 60-100GB
- The image is large (10GB+) due to ML models

**"denied: requested access" on push:**

```bash
# Verify you're logged in
docker login        # for Docker Hub
docker login ghcr.io  # for GHCR

# Verify image name matches your username
docker images | grep whisperx
```

### Step 7: Create RunPod Endpoint

1. Go to RunPod Console → **Serverless** → **New Endpoint**
2. Select **Deploy from docker registry** (not GitHub)
3. Choose template: **No template**
4. Enter your Docker image URL:
   - Docker Hub: `YOUR_USERNAME/whisperx-diarization:latest`
   - GHCR: `ghcr.io/YOUR_USERNAME/whisperx-diarization:latest`
5. Endpoint name: `whisperx` (or any name)
6. Endpoint type: **Queue** (not Load Balancer)
   - Queue is for async jobs like transcription
   - Matches existing Faster Whisper setup (uses `/runsync`)
7. Worker type: **GPU**
8. GPU configuration: Select **24 GB PRO** (see GPU Selection below)
9. Model field: **Leave empty** (your Docker image contains the models)
10. Container configuration:
    - Container start command: Leave empty
    - **Container disk: 20 GB** (increase from default 5 GB - models are large)
    - Expose ports: Leave empty
11. Environment variables → Click **"+ Add Environment variable"**:
    - Key: `HF_TOKEN`
    - Value: `hf_xxxxxxxxxxxxxxx` (your actual token from Step 3)
12. Click **Create Endpoint**
13. Note the new endpoint ID

**Note:** Container disk and GPU can be easily changed later by editing the endpoint.

### Step 7b: Configure Scaling (After Creation)

Scaling settings are configured after the endpoint is created:

1. Click on your endpoint name
2. Click **Edit**
3. Configure Active Workers, Max Workers, Idle Timeout (see Scaling Settings below)
4. Save

#### GPU Selection

RunPod shows GPU options by VRAM size:

| VRAM          | Supply | Cost/sec   | Recommendation                         |
|---------------|--------|------------|----------------------------------------|
| **24 GB**     | Low    | ~$0.00019  | Good but may have availability delays  |
| **16 GB**     | Low    | ~$0.00016  | May be tight for WhisperX + diarization|
| **24 GB PRO** | High   | ~$0.00031  | Recommended - reliable availability    |
| **32 GB PRO** | High   | ~$0.00044  | Overkill but always available          |
| **48 GB**     | Low    | ~$0.00034  | Overkill                               |

**Recommendation:** Select **24 GB PRO** - "High Supply" means faster worker spin-up and more reliable availability. "Low Supply" options may have cold start delays.

**Changing GPU later:** Easy - just edit endpoint settings anytime. No redeployment needed. Takes effect on next worker spin-up.

#### Scaling Settings

| Setting | Recommended Value | Why |
|---------|-------------------|-----|
| **Active Workers** | 0 | Scale to zero when idle (no cost) |
| **Max Workers** | 2-3 | Limits max parallel jobs (controls cost) |
| **Idle Timeout** | 5-10 seconds | How long worker stays warm after job |
| **Flash Boot** | Enabled | Faster cold starts |
| **GPU Count** | 1 | One GPU per worker is sufficient |

#### Scaling Presets

**Low Volume / Testing (Recommended to start):**

| Setting        | Value      |
|----------------|------------|
| Active Workers | 0          |
| Max Workers    | 1          |
| Idle Timeout   | 5 seconds  |

Cost: **$0 when idle**. First job has 30-60s cold start.

**Production / Multiple Users:**

| Setting        | Value      |
|----------------|------------|
| Active Workers | 0-1        |
| Max Workers    | 2-3        |
| Idle Timeout   | 30 seconds |

Cost: Higher but faster response. Set Active Workers to 1 for always-warm (costs 24/7).

#### Key Concepts

**Active Workers (Min):** Workers always running, even with no jobs. Set to 0 for scale-to-zero (no idle cost). Set to 1+ for always-warm (faster response, costs money 24/7).

**Max Workers:** Maximum concurrent jobs. 3 workers = 3 audio files processed simultaneously.

**Idle Timeout:** How long a worker stays "warm" after finishing a job. Short (5s) saves money but next job has cold start. Long (60s) costs more but subsequent jobs start instantly.

**Cold Start:** First job after idle takes 30-60s to spin up. Subsequent jobs within idle timeout start immediately.

#### Cost Estimate

For 1 hour of audio with WhisperX + diarization (~10 min processing):

| GPU      | Processing Cost |
|----------|-----------------|
| RTX 3090 | ~$0.26          |
| RTX 4090 | ~$0.41          |

**How environment variables work:**

```text
Dockerfile: ENV HF_TOKEN=""       ← placeholder (optional)
     ↓
RunPod Console: HF_TOKEN=hf_xxx   ← real value (overrides placeholder)
     ↓
Worker code: os.environ["HF_TOKEN"] ← reads the real value at runtime
```

### Step 8: Wait for Initialization

After clicking **Create Endpoint**, the status shows "Initializing". Wait 1-2 minutes for it to change to **Ready** or **Idle**.

### Step 9: Copy Endpoint ID

Once ready, copy the **Endpoint ID** from the endpoint details page (looks like `abc123xyz`).

### Step 10: Update .env

```env
# Keep existing for fallback
WHISPERX_RUNPOD_ENDPOINT_ID=existing_faster_whisper_endpoint

# Add new whisperX endpoint
RUNPOD_WHISPERX_ENDPOINT_ID=new_whisperx_endpoint
```

### Step 11: Test the Endpoint

Test with a small audio file:

```bash
curl -X POST "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync" \
  -H "Authorization: Bearer YOUR_RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"input": {"audio": "https://example.com/test.mp3", "diarize": true}}'
```

**Note:** First job will be slow (30-60s cold start + processing). Subsequent jobs within idle timeout are faster.

### Viewing Logs

If something fails, check the logs:

1. Go to **Serverless** → Click your endpoint name
2. Click the **Requests** tab
3. Click on a specific request/job to see its logs

For worker logs:

1. Click **Workers** tab
2. Click on an active worker
3. View stdout/stderr logs

**Note:** Logs only appear after the first request runs.

### Troubleshooting Endpoint Issues

**Stuck on "Initializing" for more than 5 minutes:**

Common causes:

| Issue                 | Fix                                    |
|-----------------------|----------------------------------------|
| GHCR image is private | Make package public (see below)        |
| Image URL typo        | Check exact URL in endpoint settings   |
| Image failed to build | Verify image pulls locally first       |

**Fix for private GHCR image:**

1. Go to `https://github.com/YOUR_USERNAME?tab=packages`
2. Click on `whisperx-diarization`
3. Click **Package settings** → **Change visibility** → **Public**
4. **Delete the stuck endpoint** and recreate (faster than waiting for retry)

## API Usage

### Request

```json
{
  "input": {
    "audio": "https://your-audio-url.mp3",
    "diarize": true,
    "language": "en"
  }
}
```

### Response

```json
{
  "segments": [
    {
      "start": 0.0,
      "end": 4.2,
      "text": "Welcome to the show today.",
      "speaker": "SPEAKER_00"
    },
    {
      "start": 4.5,
      "end": 8.1,
      "text": "Thanks for having me.",
      "speaker": "SPEAKER_01"
    }
  ],
  "speakers": ["SPEAKER_00", "SPEAKER_01"]
}
```

## Code Changes Required

Update `src/services/runpodWhisperService.ts`:

```typescript
// Add speaker to Segment interface
interface Segment {
    start: number;
    end: number;
    text: string;
    speaker?: string;  // "SPEAKER_00", "SPEAKER_01", etc.
}

// Add to RunpodResponse output
interface RunpodResponse {
    output?: {
        // ... existing fields ...
        speakers?: string[];
    };
}

// Add diarize option to transcribe method
async transcribeWithTimestamps(
    filePath: string,
    options?: {
        task?: 'transcribe' | 'translate';
        language?: string;
        diarize?: boolean;  // NEW
    }
): Promise<{ text: string; segments: Segment[] }>
```

## Cost Comparison

| Model | Inference Time (1hr audio) | Relative Cost |
|-------|---------------------------|---------------|
| Faster Whisper | ~3-5 min | Base |
| WhisperX (no diarization) | ~4-6 min | +20% |
| WhisperX (with diarization) | ~8-12 min | +100-150% |

Diarization roughly doubles processing time due to pyannote's speaker embedding extraction.

---

# Extended Audio Analysis Pipeline

For advanced use cases (music detection, BPM analysis), additional tools can be integrated.

## Full Pipeline Architecture

```
Audio File
    │
    ├─→ PANNs CNN10 ──→ Segmentation (music/speech/silence) + genre + instruments
    │
    ├─→ WhisperX ─────→ Transcription + speaker diarization
    │
    └─→ Essentia ─────→ BPM, key, bars (music segments only)
```

## PANNs CNN10 (Audio Classification)

### What It Does
- Classifies 527 audio event types
- Speech vs music vs silence segmentation
- Music genre detection (rock, jazz, classical, etc.)
- Instrument identification (guitar, piano, drums, etc.)
- Frame-level classification (~100ms precision)

### What It Does NOT Do
- BPM / tempo detection
- Musical key detection
- Bar/beat positions

### Deployment
- **Open source**: Yes (PyTorch)
- **Self-hostable**: Yes
- **GPU required**: No (CPU is 3-5x slower but viable)
- **GitHub**: `qiuqiangkong/audioset_tagging_cnn`

### Accuracy vs YAMNet

| Model | mAP (AudioSet) | Params | Speed |
|-------|----------------|--------|-------|
| PANNs CNN14 | 0.431 | 80M | Slower |
| PANNs CNN10 | 0.380 | 5.3M | Medium |
| YAMNet | 0.306 | 3.7M | Fastest |

**Recommendation**: PANNs CNN10 for best accuracy/cost balance.

## Essentia (Music Analysis)

### What It Does
- BPM / tempo detection
- Musical key detection
- Beat positions
- Bar/measure counting
- Chord progression analysis

### Deployment
- **Open source**: Yes (from Music Technology Group, Barcelona)
- **Self-hostable**: Yes
- **GPU required**: No (CPU-based DSP, very fast)
- **GitHub**: `MTG/essentia`

## pyannote-audio (Speaker Diarization)

### What It Does
- Speaker segmentation ("who spoke when")
- Speaker embedding extraction
- Clustering speakers across audio

### Deployment
- **Open source**: Yes
- **Self-hostable**: Yes
- **GPU required**: Recommended (CPU is 10-20x slower)
- **Requires**: HuggingFace token + model agreement

## Hardware Requirements Summary

| Tool | GPU Required? | CPU Performance | Best Deployment |
|------|---------------|-----------------|-----------------|
| Whisper/WhisperX | Yes | Very slow | RunPod (GPU) |
| pyannote-audio | Recommended | Slow | RunPod (GPU) |
| PANNs CNN10 | No | Good | Google Cloud (CPU) |
| Essentia | No | Excellent | Google Cloud (CPU) |

### Cost-Optimized Architecture

```
┌─────────────────────────────────────┐
│  Google Cloud (CPU) - Cheap         │
│  • Essentia (BPM, key, bars)        │
│  • PANNs CNN10 (segmentation)       │
└─────────────────────────────────────┘

┌─────────────────────────────────────┐
│  RunPod (GPU) - Pay per use         │
│  • WhisperX (transcription)         │
│  • pyannote-audio (speaker ID)      │
└─────────────────────────────────────┘
```

## References

- [WhisperX](https://github.com/m-bain/whisperX)
- [pyannote-audio](https://github.com/pyannote/pyannote-audio)
- [PANNs](https://github.com/qiuqiangkong/audioset_tagging_cnn)
- [Essentia](https://github.com/MTG/essentia)
- [YAMNet](https://tfhub.dev/google/yamnet/1)
- [RunPod Serverless](https://docs.runpod.io/serverless)
