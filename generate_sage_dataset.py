import os
import sys
import random
import logging
import soundfile as sf
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("sage_generation.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

DATA_DIR = "/home/teaching/s25019/Ekstep_simulated/Ekstep_12_lang_multi_domain"
OUTPUT_DIR = "/home/teaching/s25019/Ekstep_simulated/sage_simulated_dataset"
TARGET_DURATION = 600.0  # 10 minutes in seconds

class LanguagePool:
    def __init__(self, lang_name, lang_dir):
        self.lang_name = lang_name
        self.all_files = list(Path(lang_dir).rglob("*.wav"))
        if not self.all_files:
            raise ValueError(f"No WAV files found in: {lang_dir}")
        self.pool = []
        self.reset()
        
    def reset(self):
        self.pool = self.all_files.copy()
        random.shuffle(self.pool)
        
    def get_next(self):
        if not self.pool:
            self.reset()
        return self.pool.pop()

def add_noise(audio, snr_db):
    """Add Gaussian White Noise to hit a target SNR."""
    audio_power = np.mean(audio ** 2)
    if audio_power == 0:
        return audio
        
    snr_linear = 10 ** (snr_db / 10.0)
    noise_power = audio_power / snr_linear
    noise = np.random.normal(0, np.sqrt(noise_power), len(audio))
    return audio + noise

def get_random_slice(data, samplerate, min_duration, max_duration):
    """Extract a random slice from an audio array."""
    target_len = random.uniform(min_duration, max_duration)
    target_samples = int(target_len * samplerate)
    
    if len(data) <= target_samples:
        return data, len(data) / samplerate
        
    start_idx = random.randint(0, len(data) - target_samples)
    return data[start_idx:start_idx + target_samples], target_len

def create_dataset_split(split_name, num_files, lang_pools):
    split_dir = Path(OUTPUT_DIR) / split_name
    wav_dir = split_dir / "wav"
    rttm_dir = split_dir / "rttm"
    
    wav_dir.mkdir(parents=True, exist_ok=True)
    rttm_dir.mkdir(parents=True, exist_ok=True)
    
    logging.info(f"Starting SAGE generation for split: {split_name} ({num_files} files)")
    
    indic_langs = [l for l in lang_pools.keys() if l != 'eng']
    
    for i in tqdm(range(1, num_files + 1)):
        file_id = f"sage_{split_name}_{i:04d}"
        wav_path = wav_dir / f"{file_id}.wav"
        rttm_path = rttm_dir / f"{file_id}.rttm"
        
        # 1. 85% Bilingual, 15% Trilingual
        is_trilingual = random.random() < 0.15
        num_matrix_langs = 2 if is_trilingual else 1
        
        matrix_langs = random.sample(indic_langs, num_matrix_langs)
        embedded_lang = 'eng'
        
        all_selected_langs = matrix_langs + [embedded_lang]
        lang_to_label = {lang: f"L{idx+1}" for idx, lang in enumerate(all_selected_langs)}
        
        audio_chunks = []
        rttm_entries = []
        current_time = 0.0
        
        last_lang = None
        
        while current_time < TARGET_DURATION:
            # 2. Matrix vs Embedded weighting (80% chance for Matrix, 20% for Embedded)
            is_embedded = random.random() < 0.20
            
            if is_embedded:
                current_lang = embedded_lang
                min_dur, max_dur = 0.5, 1.5  # Rapid English insertions
            else:
                current_lang = random.choice(matrix_langs)
                min_dur, max_dur = 2.0, 5.0  # Long Indic base speech
                
            # Prevent identical sequential tags without a boundary logic, though code switching allows it
            if current_lang == last_lang and random.random() < 0.5:
                # Force switch occasionally
                candidate = [l for l in all_selected_langs if l != last_lang]
                if candidate:
                    current_lang = random.choice(candidate)
                    if current_lang == embedded_lang:
                        min_dur, max_dur = 0.5, 1.5
                    else:
                        min_dur, max_dur = 2.0, 5.0
            
            last_lang = current_lang
            
            # Fetch audio file
            wav_file = lang_pools[current_lang].get_next()
            try:
                data, samplerate = sf.read(str(wav_file))
                if samplerate != 16000:
                    import librosa
                    data = librosa.resample(data, orig_sr=samplerate, target_sr=16000)
                    samplerate = 16000
            except Exception as e:
                continue
                
            # Get Sliced Audio
            sliced_data, duration = get_random_slice(data, samplerate, min_dur, max_dur)
            
            # 3. 50% chance of Noise Augmentation
            if random.random() < 0.5:
                snr = random.choice([5, 10, 15, 20])
                sliced_data = add_noise(sliced_data, snr)
            
            audio_chunks.append(sliced_data)
            
            # RTTM Entry
            label = lang_to_label[current_lang]
            rttm_line = f"LANGUAGE {file_id} 1 {current_time:.3f} {duration:.3f} <NA> <NA> {label} <NA> <NA>\n"
            rttm_entries.append(rttm_line)
            
            current_time += duration
        
        # Save output
        full_audio = np.concatenate(audio_chunks, axis=0)
        sf.write(str(wav_path), full_audio, 16000, subtype='PCM_16')
        
        with open(rttm_path, 'w') as f:
            f.writelines(rttm_entries)

def main():
    if not os.path.exists(DATA_DIR):
        logging.error(f"Data directory not found: {DATA_DIR}")
        sys.exit(1)
        
    lang_names = [d.name for d in Path(DATA_DIR).iterdir() if d.is_dir() and d.name != ".ipynb_checkpoints"]
    if 'eng' not in lang_names:
        logging.error("English ('eng') directory is missing! Crucial for embedded language simulation.")
        sys.exit(1)
        
    lang_pools = {}
    for lang in lang_names:
        try:
            lang_pools[lang] = LanguagePool(lang, Path(DATA_DIR) / lang)
        except ValueError as e:
            logging.warning(e)
            
    logging.info("--- Generating SAGE-LD Inspired Dataset ---")
    
    # Check test mode
    is_test = len(sys.argv) > 1 and sys.argv[1] == "--test"
    if is_test:
        create_dataset_split("train", 2, lang_pools)
        create_dataset_split("eval", 1, lang_pools)
    else:
        # Full generation
        create_dataset_split("train", 300, lang_pools) # 300 files * 10 mins = 50 hours
        create_dataset_split("eval", 30, lang_pools)

if __name__ == "__main__":
    main()
