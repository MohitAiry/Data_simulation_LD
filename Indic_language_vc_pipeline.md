# Indian Language Dataset — Voice Conversion Pipeline

A complete pipeline for converting all speaker identities in a 10-language Indian language WAV dataset to a single unified Speaker X, using FreeVC. Designed as a preprocessing step for language diarization pretraining following the SAGE-LD simulation strategy.

---

## Overview

The core problem this pipeline solves: naively concatenating utterances from different speakers across languages conflates **language boundaries with speaker boundaries**, causing a model to learn speaker diarization instead of language diarization. By converting every utterance to a single Speaker X identity before concatenation, only the language changes at each boundary — giving the model clean, unambiguous supervision.

```
dataset/ (10 language folders, WAV)
        ↓
Preprocessing (16 kHz, normalise)
        ↓
Select Speaker X (automatic)
        ↓
FreeVC Voice Conversion  ← key step
        ↓
Quality Filter
        ↓
Pair & Concatenate (auto-labels)
        ↓
Augmentation (RIR + noise, p=0.5 each)
        ↓
Unified-speaker corpus (ready for pretraining)
```

---

## Requirements

### Hardware

- GPU strongly recommended (CUDA). CPU is viable for small datasets but slow.
- ~8 GB VRAM for FreeVC + WavLM-Large inference.

### Software

```bash
pip install torch torchaudio librosa soundfile numpy tqdm speechbrain
git clone https://github.com/OlaWod/FreeVC.git
cd FreeVC
pip install -r requirements.txt
```

### Pretrained checkpoints

Download from the [FreeVC releases page](https://github.com/OlaWod/FreeVC/releases) and place inside `FreeVC/checkpoints/`:

- `freevc.pth` — main FreeVC model
- `WavLM-Large.pt` — speaker encoder backbone

---

## Expected Dataset Structure

```
dataset/
├── hindi/
│   ├── spk001_utt001.wav
│   └── spk002_utt012.wav
├── tamil/
├── telugu/
├── kannada/
├── bengali/
├── marathi/
├── gujarati/
├── punjabi/
├── odia/
└── malayalam/
```

All files must be WAV format. Multiple speakers per language folder are supported — the pipeline handles them uniformly.

---

## Step 1 — Preprocessing

Resample all files to 16 kHz (FreeVC's required sample rate) and peak-normalise to −3 dBFS. Save preprocessed files to a separate directory to preserve originals.

```python
import os
import librosa
import soundfile as sf
import numpy as np

def preprocess_wav(in_path, out_path, target_sr=16000):
    audio, sr = librosa.load(in_path, sr=target_sr, mono=True)
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak * 0.708  # peak-normalise to -3 dBFS
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    sf.write(out_path, audio, target_sr)

def preprocess_dataset(src_root, dst_root):
    for lang in os.listdir(src_root):
        lang_path = os.path.join(src_root, lang)
        if not os.path.isdir(lang_path):
            continue
        for fname in os.listdir(lang_path):
            if not fname.endswith(".wav"):
                continue
            in_path  = os.path.join(lang_path, fname)
            out_path = os.path.join(dst_root, lang, fname)
            preprocess_wav(in_path, out_path)

preprocess_dataset("dataset/", "dataset_16k/")
```

---

## Step 2 — Select Speaker X

Automatically picks the longest clean utterance across the entire dataset as the universal reference voice. No manual selection required.

```python
def select_speaker_x(dataset_root):
    best_file, best_dur = None, 0
    for lang in os.listdir(dataset_root):
        lang_path = os.path.join(dataset_root, lang)
        if not os.path.isdir(lang_path):
            continue
        for fname in os.listdir(lang_path):
            if not fname.endswith(".wav"):
                continue
            fpath = os.path.join(lang_path, fname)
            dur = librosa.get_duration(path=fpath)
            if dur > best_dur:
                best_dur = dur
                best_file = fpath
    print(f"Speaker X selected: {best_file} ({best_dur:.1f}s)")
    return best_file

SPEAKER_X_PATH = select_speaker_x("dataset_16k/")
```

> **Tip:** If you have access to a studio-quality recording, override `SPEAKER_X_PATH` manually. The Speaker X reference quality directly determines the consistency of every converted file. A clean, noise-free clip of at least 10 seconds works best. Hindi and Kannada tend to work well as anchor languages for Indian language datasets due to their relatively neutral phonetic profiles.

---

## Step 3 — FreeVC Voice Conversion

This is the critical step. FreeVC decomposes each utterance into:

- **Content** — linguistic information extracted via a WavLM bottleneck encoder (language preserved)
- **Speaker identity** — replaced with the Speaker X embedding

The HiFi-GAN decoder then reconstructs the waveform with Speaker X's voice but the original language content intact.

### FreeVC internals

```
Source WAV
    │
    ├─── Content encoder (WavLM) ──────────────────────┐
    │                                                   ▼
    └─── Speaker encoder → DISCARDED        HiFi-GAN decoder → Output WAV
                                                        ▲
Speaker X reference ── Speaker encoder ────────────────┘
```

### Batch conversion script

```python
import sys
import torch
from tqdm import tqdm

sys.path.insert(0, "FreeVC")
from freevc import FreeVC

device = "cuda" if torch.cuda.is_available() else "cpu"
model  = FreeVC.from_pretrained("FreeVC/checkpoints/freevc.pth").to(device)
model.eval()

def convert_all(dataset_root, speaker_x_path, output_root):
    for lang in sorted(os.listdir(dataset_root)):
        lang_in  = os.path.join(dataset_root, lang)
        lang_out = os.path.join(output_root, lang)
        if not os.path.isdir(lang_in):
            continue
        os.makedirs(lang_out, exist_ok=True)
        files = [f for f in os.listdir(lang_in) if f.endswith(".wav")]
        print(f"\nConverting {lang}: {len(files)} files")
        for fname in tqdm(files, desc=lang):
            src = os.path.join(lang_in, fname)
            dst = os.path.join(lang_out, fname)
            with torch.no_grad():
                converted = model.convert(
                    source_path=src,
                    target_path=speaker_x_path,
                )
            sf.write(dst, converted.cpu().numpy(), 16000)

convert_all("dataset_16k/", SPEAKER_X_PATH, "dataset_speaker_x/")
```

After this runs, `dataset_speaker_x/` mirrors your original folder structure but every utterance is in Speaker X's voice.

---

## Step 4 — Quality Filter

Discard conversions where FreeVC produced degraded output (clipping, silence, excessive noise). Uses signal-to-noise ratio as a proxy quality metric.

```python
def estimate_snr(audio, sr, noise_floor_ms=50):
    """Estimate SNR using the quietest 50 ms window as noise estimate."""
    frame_len  = int(sr * noise_floor_ms / 1000)
    rms_frames = [
        np.sqrt(np.mean(audio[i:i+frame_len]**2))
        for i in range(0, len(audio) - frame_len, frame_len)
    ]
    noise_rms  = np.percentile(rms_frames, 10)
    signal_rms = np.sqrt(np.mean(audio**2))
    if noise_rms < 1e-9:
        return 99.0
    return 20 * np.log10(signal_rms / noise_rms)

def filter_corpus(converted_root, min_snr_db=10, min_dur_s=0.5):
    kept, dropped = 0, 0
    for lang in os.listdir(converted_root):
        lang_path = os.path.join(converted_root, lang)
        if not os.path.isdir(lang_path):
            continue
        for fname in os.listdir(lang_path):
            if not fname.endswith(".wav"):
                continue
            fpath = os.path.join(lang_path, fname)
            audio, sr = librosa.load(fpath, sr=16000)
            snr = estimate_snr(audio, sr)
            dur = len(audio) / sr
            if snr < min_snr_db or dur < min_dur_s:
                os.remove(fpath)
                dropped += 1
            else:
                kept += 1
    print(f"Kept: {kept}  |  Dropped: {dropped}")

filter_corpus("dataset_speaker_x/")
```

---

## Step 5 — Pair and Concatenate

Pairs utterances from two different languages and concatenates them into synthetic code-switching segments. Ground-truth frame-level language labels are generated automatically from the known boundary position — no manual annotation needed.

```python
import random
import json

def build_cs_pairs(speaker_x_root, output_dir, pairs_per_combo=200, seed=42):
    random.seed(seed)
    langs = sorted([
        d for d in os.listdir(speaker_x_root)
        if os.path.isdir(os.path.join(speaker_x_root, d))
    ])
    os.makedirs(output_dir, exist_ok=True)
    manifest = []

    for lang_a in langs:
        for lang_b in langs:
            if lang_a == lang_b:
                continue
            files_a = [
                os.path.join(speaker_x_root, lang_a, f)
                for f in os.listdir(os.path.join(speaker_x_root, lang_a))
                if f.endswith(".wav")
            ]
            files_b = [
                os.path.join(speaker_x_root, lang_b, f)
                for f in os.listdir(os.path.join(speaker_x_root, lang_b))
                if f.endswith(".wav")
            ]
            for i in range(pairs_per_combo):
                fa = random.choice(files_a)
                fb = random.choice(files_b)
                audio_a, _ = librosa.load(fa, sr=16000)
                audio_b, _ = librosa.load(fb, sr=16000)
                combined  = np.concatenate([audio_a, audio_b])
                boundary  = len(audio_a) / 16000
                total_dur = len(combined) / 16000
                out_name  = f"{lang_a}_{lang_b}_{i:05d}.wav"
                out_path  = os.path.join(output_dir, out_name)
                sf.write(out_path, combined, 16000)
                manifest.append({
                    "file": out_path,
                    "segments": [
                        {"lang": lang_a, "start": 0.0,      "end": round(boundary, 4)},
                        {"lang": lang_b, "start": round(boundary, 4), "end": round(total_dur, 4)},
                    ]
                })

    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Generated {len(manifest)} pairs across {len(langs)*(len(langs)-1)} combinations")
    print(f"Manifest saved to {manifest_path}")
    return manifest

manifest = build_cs_pairs("dataset_speaker_x/", "dataset_cs_pairs/", pairs_per_combo=200)
```

With 10 languages this produces 90 language combinations. At 200 pairs per combination that is 18,000 code-switching utterances.

---

## Step 6 — Acoustic Augmentation

Two independent augmentations applied with 50% probability each, following the SAGE-LD strategy. This teaches the model to generalise across real recording conditions.

### Room Impulse Response (RIR)

Convolves clean audio with a room response, adding echo and reverberation. Without this, a model trained on clean simulated audio degrades on real recordings made in offices, classrooms, or homes.

```python
from scipy.signal import fftconvolve

def apply_rir(audio, rir_audio):
    """Convolve signal with a room impulse response."""
    convolved = fftconvolve(audio, rir_audio)
    # Trim to original length
    convolved = convolved[:len(audio)]
    # Re-normalise
    peak = np.max(np.abs(convolved))
    if peak > 0:
        convolved = convolved / peak * 0.708
    return convolved.astype(np.float32)
```

You can generate synthetic RIRs with `pyroomacoustics`, or use a real RIR corpus such as [OpenSLR 26](https://openslr.org/26/).

### Noise injection

Mixes background noise at a randomly chosen SNR level.

```python
def mix_noise(audio, noise, target_snr_db):
    """Mix noise into audio at target_snr_db."""
    audio_rms = np.sqrt(np.mean(audio**2))
    noise_rms = np.sqrt(np.mean(noise**2))
    if noise_rms < 1e-9:
        return audio
    desired_noise_rms = audio_rms / (10 ** (target_snr_db / 20))
    noise_scaled = noise * (desired_noise_rms / noise_rms)
    # Tile noise if shorter than audio
    if len(noise_scaled) < len(audio):
        repeats = int(np.ceil(len(audio) / len(noise_scaled)))
        noise_scaled = np.tile(noise_scaled, repeats)
    noise_scaled = noise_scaled[:len(audio)]
    return (audio + noise_scaled).astype(np.float32)
```

### Full augmentation pass

```python
def augment_corpus(input_dir, output_dir, rir_dir, noise_dir, p_aug=0.5):
    rir_files   = [os.path.join(rir_dir,   f) for f in os.listdir(rir_dir)   if f.endswith(".wav")]
    noise_files = [os.path.join(noise_dir, f) for f in os.listdir(noise_dir) if f.endswith(".wav")]
    snr_levels  = [5, 10, 15, 20]
    os.makedirs(output_dir, exist_ok=True)

    for fname in tqdm(os.listdir(input_dir)):
        if not fname.endswith(".wav"):
            continue
        audio, sr = librosa.load(os.path.join(input_dir, fname), sr=16000)

        if random.random() < p_aug and rir_files:
            rir, _ = librosa.load(random.choice(rir_files), sr=16000)
            audio  = apply_rir(audio, rir)

        if random.random() < p_aug and noise_files:
            noise, _ = librosa.load(random.choice(noise_files), sr=16000)
            snr   = random.choice(snr_levels)
            audio = mix_noise(audio, noise, snr)

        sf.write(os.path.join(output_dir, fname), audio, 16000)

augment_corpus(
    input_dir  = "dataset_cs_pairs/",
    output_dir = "dataset_final/",
    rir_dir    = "rirs/",        # folder of RIR WAV files
    noise_dir  = "noise/",       # folder of noise WAV files (e.g. DEMAND corpus)
)
```

---

## Step 7 — Verify Speaker Consistency

Before training, confirm that FreeVC successfully unified speaker identity across languages. Use a speaker verification model to check that converted files from different languages are classified as the same speaker.

```python
from speechbrain.pretrained import SpeakerRecognition

verifier = SpeakerRecognition.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb"
)

# Sample a pair from two different languages
file_a = "dataset_speaker_x/hindi/spk001_utt001.wav"
file_b = "dataset_speaker_x/tamil/spk010_utt003.wav"

score, prediction = verifier.verify_files(file_a, file_b)
print(f"Same-speaker score: {score:.3f}")
# Target: score > 0.7 — lower values indicate poor voice conversion quality
```

If scores are consistently below 0.7, the Speaker X reference clip is likely too short or noisy. Replace it with a cleaner recording and re-run Steps 3–4.

---

## Output Structure

```
dataset_final/
├── hindi_tamil_00001.wav
├── hindi_tamil_00002.wav
├── tamil_hindi_00001.wav
├── telugu_hindi_00001.wav
│   ... (18,000+ files)
└── manifest.json            ← frame-level language labels for all pairs
```

The `manifest.json` contains entries like:

```json
{
  "file": "dataset_cs_pairs/hindi_tamil_00001.wav",
  "segments": [
    { "lang": "hindi", "start": 0.0,   "end": 3.21 },
    { "lang": "tamil", "start": 3.21,  "end": 6.84 }
  ]
}
```

---

## Design Decisions

| Decision | Rationale |
|---|---|
| FreeVC over other VC models | Fast inference, open-source, no additional speaker encoder training required |
| Automatic Speaker X selection | Longest clean utterance avoids silent or clipped references |
| 50% augmentation probability | Matches SAGE-LD paper; trains robustness without over-corrupting |
| SNR range 5–20 dB | Covers very difficult (5 dB, speech ≈ noise) to nearly clean (20 dB) conditions |
| 200 pairs per language combo | Balances corpus size vs. compute; scale up if GPU budget allows |
| Frame labels from boundary position | Zero annotation cost; labels are exact because we control the concatenation |

---

## Troubleshooting

**FreeVC output sounds robotic or garbled**
The Speaker X reference clip may be too short or have background noise. Use a cleaner clip of at least 10 seconds. Check that the file was preprocessed (16 kHz, normalised) before passing it to the model.

**Very low same-speaker verification scores**
Try a different Speaker X reference. Hindi or Kannada recordings tend to work well as anchor identities within Indian language datasets.

**Out of VRAM errors**
Reduce batch size or switch to CPU inference for the conversion step (`device = "cpu"`). CPU inference is around 5–10× slower but produces identical output.

**Large number of files dropped in quality filter**
Lower `min_snr_db` from 10 to 8, or check whether your original recordings have consistent loudness. Very quiet utterances may need louder pre-normalisation.

---

## References

- FreeVC: [github.com/OlaWod/FreeVC](https://github.com/OlaWod/FreeVC)
- SAGE-LD paper: *Speaker-Agnostic and Generalizable Language Diarization* (2024)
- WavLM: [microsoft/wavlm](https://github.com/microsoft/wavlm)
- DEMAND noise corpus: [zenodo.org/record/1227121](https://zenodo.org/record/1227121)
- SpeechBrain speaker verification: [speechbrain.github.io](https://speechbrain.github.io)