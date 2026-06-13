import os
import sys
import random
import logging
import soundfile as sf
import numpy as np
import scipy.signal
import glob
import math

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("generation_family_aware.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

DATA_DIR = "/home/teaching/s25019/Ekstep_simulated/VC_Dataset/IITMandi_Readspeech_vc"
OUTPUT_DIR = "/home/teaching/s25019/Ekstep_simulated/sim_dataset_family_aware"
TARGET_DURATION = 600.0  # 10 minutes in seconds

# Language Families
FAMILIES = {
    'INDO_ARYAN': ['hin', 'pun', 'mar', 'ben'],
    'DRAVIDIAN': ['tam', 'tel', 'kan', 'mal'],
    'ENGLISH': ['eng']
}

class AcousticAugmentor:
    def __init__(self, rir_dir="augmentations/rir", noise_dir="augmentations/noise"):
        self.rir_files = glob.glob(os.path.join(rir_dir, "**/*.wav"), recursive=True)
        self.noise_files = glob.glob(os.path.join(noise_dir, "**/*.wav"), recursive=True)
        
        if not self.rir_files:
            logging.warning("No RIR files found! Augmentation might be skipped.")
        if not self.noise_files:
            logging.warning("No Noise files found! Augmentation might be skipped.")
            
    def augment(self, audio, sr=16000):
        if not self.rir_files or not self.noise_files:
            return audio, "clean", "none"
            
        if random.random() > 0.5:
            return audio, "clean", "none" # 50% chance to remain clean
            
        # 1. Apply RIR
        rir_path = random.choice(self.rir_files)
        try:
            rir_audio, _ = sf.read(rir_path)
            if rir_audio.ndim > 1:
                rir_audio = rir_audio[:, 0]
            rir_audio = rir_audio / (np.linalg.norm(rir_audio) + 1e-8)
            
            augmented_audio = scipy.signal.fftconvolve(audio, rir_audio, mode='full')[:len(audio)]
            rir_name = os.path.basename(rir_path)
        except Exception as e:
            logging.error(f"RIR Convolution failed: {e}")
            augmented_audio = audio
            rir_name = "failed"
            
        # 2. Apply Noise at specific SNR
        noise_path = random.choice(self.noise_files)
        try:
            noise_audio, noise_sr = sf.read(noise_path)
            if noise_audio.ndim > 1:
                noise_audio = noise_audio[:, 0]
                
            if len(noise_audio) < len(augmented_audio):
                repeats = math.ceil(len(augmented_audio) / len(noise_audio))
                noise_audio = np.tile(noise_audio, repeats)
            noise_audio = noise_audio[:len(augmented_audio)]
            
            snr_db = random.choice([5, 10, 15, 20])
            
            signal_power = np.mean(augmented_audio ** 2)
            noise_power = np.mean(noise_audio ** 2)
            
            if noise_power > 0:
                snr_linear = 10 ** (snr_db / 10.0)
                target_noise_power = signal_power / snr_linear
                noise_audio = noise_audio * np.sqrt(target_noise_power / noise_power)
                augmented_audio = augmented_audio + noise_audio
                
            noise_name = f"{os.path.basename(noise_path)}@{snr_db}dB"
        except Exception as e:
            logging.error(f"Noise Injection failed: {e}")
            noise_name = "failed"
            
        # Normalize to prevent clipping
        max_val = np.max(np.abs(augmented_audio))
        if max_val > 0.99:
            augmented_audio = (augmented_audio / max_val) * 0.99
            
        return augmented_audio, rir_name, noise_name

class LanguagePool:
    def __init__(self, lang_name, lang_dir):
        self.lang_name = lang_name
        self.lang_dir = lang_dir
        self.all_files = sorted([
            os.path.join(lang_dir, f)
            for f in os.listdir(lang_dir)
            if f.endswith('.wav')
        ])
        if not self.all_files:
            raise ValueError(f"No WAV files found in directory: {lang_dir}")
        self.pool = []
        self.reset()
        
    def reset(self):
        self.pool = self.all_files.copy()
        random.shuffle(self.pool)
        
    def get_next(self):
        if not self.pool:
            self.reset()
        return self.pool.pop()

def pick_language_pair():
    """
    Selects 2 languages from different families with an English bias:
    - 40% probability: English + Indo-Aryan
    - 40% probability: English + Dravidian
    - 20% probability: Indo-Aryan + Dravidian
    """
    choice = random.choices(['ENG_INDO', 'ENG_DRAV', 'INDO_DRAV'], weights=[0.4, 0.4, 0.2])[0]
    
    if choice == 'ENG_INDO':
        l1 = random.choice(FAMILIES['ENGLISH'])
        l2 = random.choice(FAMILIES['INDO_ARYAN'])
    elif choice == 'ENG_DRAV':
        l1 = random.choice(FAMILIES['ENGLISH'])
        l2 = random.choice(FAMILIES['DRAVIDIAN'])
    else:
        l1 = random.choice(FAMILIES['INDO_ARYAN'])
        l2 = random.choice(FAMILIES['DRAVIDIAN'])
        
    pair = [l1, l2]
    random.shuffle(pair) # Randomize which is L1 and which is L2
    return pair

def create_dataset_split(split_name, num_files, lang_pools, augmentor):
    split_dir = os.path.join(OUTPUT_DIR, split_name)
    wav_dir = os.path.join(split_dir, "wav")
    rttm_dir = os.path.join(split_dir, "rttm")
    
    os.makedirs(wav_dir, exist_ok=True)
    os.makedirs(rttm_dir, exist_ok=True)
    
    logging.info(f"Starting generation for split: {split_name} ({num_files} files)")
    
    for i in range(1, num_files + 1):
        file_id = f"sim_{split_name}_{i:04d}"
        wav_path = os.path.join(wav_dir, f"{file_id}.wav")
        rttm_path = os.path.join(rttm_dir, f"{file_id}.rttm")
        
        selected_langs = pick_language_pair()
        lang_to_label = {lang: f"L{idx+1}" for idx, lang in enumerate(selected_langs)}
        
        logging.info(f"Generating {file_id}.wav with languages {selected_langs} mapped to {lang_to_label}")
        
        audio_chunks = []
        rttm_entries = []
        current_time = 0.0
        last_lang = None
        
        while current_time < TARGET_DURATION:
            # Force language switch
            candidate_langs = [l for l in selected_langs if l != last_lang]
            current_lang = random.choice(candidate_langs)
            last_lang = current_lang
            
            # Determine turn duration: 1 to 3 utterances consecutively
            num_utterances = random.randint(1, 3)
            
            for _ in range(num_utterances):
                wav_file = lang_pools[current_lang].get_next()
                
                try:
                    data, samplerate = sf.read(wav_file)
                except Exception as e:
                    logging.error(f"Error reading WAV file {wav_file}: {e}. Skipping.")
                    continue
                
                if samplerate != 16000:
                    logging.warning(f"Unexpected sample rate {samplerate} in {wav_file}.")
                
                duration = len(data) / samplerate
                
                # Append audio data
                audio_chunks.append(data)
                
                # Append RTTM line
                label = lang_to_label[current_lang]
                # STANDARD FORMAT: TYPE FILE_ID CHANNEL START_TIME DURATION ORTHO SUBTYPE CLASS SPEAKER CONF
                rttm_line = f"LANGUAGE {file_id} 1 {current_time:.3f} {duration:.3f} <NA> <NA> {label} <NA> <NA>\n"
                rttm_entries.append(rttm_line)
                
                current_time += duration
                if current_time >= TARGET_DURATION:
                    break
        
        # Concatenate audio chunks
        full_audio = np.concatenate(audio_chunks, axis=0)
        
        # Apply Acoustic Augmentation
        full_audio, rir_applied, noise_applied = augmentor.augment(full_audio, 16000)
        
        # Write WAV file (16kHz)
        sf.write(wav_path, full_audio, 16000, subtype='PCM_16')
        
        # Write RTTM file
        with open(rttm_path, 'w') as rttm_file:
            rttm_file.writelines(rttm_entries)
            
        logging.info(f"Finished {file_id} - Dur: {current_time:.2f}s, RIR: {rir_applied}, Noise: {noise_applied}")

def main():
    # Initialize language pools
    lang_pools = {}
    for family, langs in FAMILIES.items():
        for lang in langs:
            lang_dir = os.path.join(DATA_DIR, lang)
            if not os.path.exists(lang_dir):
                logging.error(f"Required language folder not found: {lang_dir}")
                sys.exit(1)
            lang_pools[lang] = LanguagePool(lang, lang_dir)
            
    augmentor = AcousticAugmentor()
            
    is_test_run = len(sys.argv) > 1 and sys.argv[1] == "--test"
    
    if is_test_run:
        logging.info("--- RUNNING IN TEST MODE (Small subset) ---")
        create_dataset_split("train", 2, lang_pools, augmentor)
        create_dataset_split("eval", 1, lang_pools, augmentor)
        create_dataset_split("test", 1, lang_pools, augmentor)
    else:
        # Full generation
        create_dataset_split("train", 300, lang_pools, augmentor)
        create_dataset_split("eval", 50, lang_pools, augmentor)
        create_dataset_split("test", 50, lang_pools, augmentor)
        
    logging.info("All family-aware simulated datasets generated successfully!")

if __name__ == "__main__":
    main()
