# Data Simulation LD

This repository contains a robust pipeline for generating simulated, code-switched, multilingual diarization datasets. It includes tools for voice conversion (standardizing all utterances to a single speaker identity) and realistic acoustic augmentations.

## Pipeline Overview

### 1. Voice Conversion Pipeline (`run_vc_pipeline.py`)
Standardizes speaker identity across multiple datasets (Ekstep Multi-Domain, Ekstep TVNews, IIT Mandi) to isolate language characteristics without confounding speaker-specific acoustic traits.
- Automatically handles 16kHz resampling.
- Maps all multi-speaker utterances to a single reference audio ("Speaker X") using a standalone implementation of the FreeVC architecture.

### 2. Family-Aware Code-Switched Simulation (`generate_family_aware_dataset.py`)
Simulates realistic code-switched bilingual conversations by concatenating utterances and tracking temporal boundaries via `.rttm` outputs.
- **Language Families:**
  - *Indo-Aryan:* Hindi, Punjabi, Marathi, Bengali
  - *Dravidian:* Tamil, Telugu, Kannada, Malayalam
  - *English*
- **Mixing Logic:** Biased pairings enforce cross-family code-switching (e.g., 40% English + Indo-Aryan, 40% English + Dravidian, 20% Indo-Aryan + Dravidian).
- **Turn-Taking:** Random consecutive utterance chains (1-3 utterances) before forcing a language switch.

### 3. Acoustic Augmentations (`download_augmentations.py` & `AcousticAugmentor`)
Replicates the physical acoustic environment of real-world deployments by systematically degrading the synthetic data:
- **Room Impulse Responses (RIRs):** Convolves audio with physical room reverberations from the OpenSLR 28 / BUT database.
- **Background Noise:** Injects environment noise sourced from the DEMAND dataset (e.g., Cafeteria, Traffic, Parks).
- **Dynamic Application:** Each audio file has a 50% probability of being augmented. When selected, noise is dynamically scaled to hit a random Signal-to-Noise Ratio (SNR) of 5, 10, 15, or 20 dB.

## Setup & Dependencies

1. Initialize the FreeVC environment and checkpoints:
   ```bash
   ./setup_freevc.sh
   ```
2. Download required acoustic augmentation data:
   ```bash
   python download_augmentations.py
   ```
3. Generate the simulated dataset:
   ```bash
   python generate_family_aware_dataset.py
   ```

*Note: Large audio datasets (`VC_Dataset`, simulated folders) and model checkpoints are ignored via `.gitignore` to prevent repository bloat.*
