import os
import sys
import random
import logging
import soundfile as sf
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("generation_bilingual.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

DATA_DIR = "/home/teaching/s25019/Ekstep_simulated/IITMandi_Readspeech"
OUTPUT_DIR = "/home/teaching/s25019/Ekstep_simulated/bilingual_dataset"
TARGET_DURATION = 900.0  # 15 minutes in seconds
LANGUAGES = ['eng', 'hin']

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

def create_dataset_split(split_name, num_files, lang_pools):
    split_dir = os.path.join(OUTPUT_DIR, split_name)
    wav_dir = os.path.join(split_dir, "wav")
    rttm_dir = os.path.join(split_dir, "rttm")
    
    os.makedirs(wav_dir, exist_ok=True)
    os.makedirs(rttm_dir, exist_ok=True)
    
    logging.info(f"Starting generation for split: {split_name} ({num_files} files)")
    
    lang_to_label = {'eng': 'L1', 'hin': 'L2'}
    
    for i in range(1, num_files + 1):
        file_id = f"sim_{split_name}_{i:04d}"
        wav_path = os.path.join(wav_dir, f"{file_id}.wav")
        rttm_path = os.path.join(rttm_dir, f"{file_id}.rttm")
        
        logging.info(f"Generating {file_id}.wav with languages {LANGUAGES} mapped to {lang_to_label}")
        
        audio_chunks = []
        rttm_entries = []
        current_time = 0.0
        
        last_lang = None
        
        while current_time < TARGET_DURATION:
            # Pick a new language distinct from the last one (to enforce a code-switch)
            candidate_langs = [l for l in LANGUAGES if l != last_lang]
            current_lang = random.choice(candidate_langs)
            last_lang = current_lang
            
            # Determine turn duration: 1 to 3 utterances
            num_utterances = random.randint(1, 3)
            
            for _ in range(num_utterances):
                wav_file = lang_pools[current_lang].get_next()
                
                try:
                    data, samplerate = sf.read(wav_file)
                except Exception as e:
                    logging.error(f"Error reading WAV file {wav_file}: {e}. Skipping.")
                    continue
                
                if samplerate != 8000:
                    logging.warning(f"Unexpected sample rate {samplerate} in {wav_file}. Proceeding but verify.")
                
                duration = len(data) / samplerate
                
                # Append audio data
                audio_chunks.append(data)
                
                # Append RTTM line
                label = lang_to_label[current_lang]
                rttm_line = f"LANGUAGE {file_id} 1 {current_time:.3f} {duration:.3f} <NA> <NA> {label} <NA> <NA>\n"
                rttm_entries.append(rttm_line)
                
                current_time += duration
                if current_time >= TARGET_DURATION:
                    break
        
        # Concatenate audio chunks
        full_audio = np.concatenate(audio_chunks, axis=0)
        
        # Write WAV file
        sf.write(wav_path, full_audio, 8000, subtype='PCM_16')
        
        # Write RTTM file
        with open(rttm_path, 'w') as rttm_file:
            rttm_file.writelines(rttm_entries)
            
        logging.info(f"Finished {file_id} - Total duration: {current_time:.2f}s, Audio length: {len(full_audio)/8000:.2f}s")

def main():
    # Initialize language pools
    lang_pools = {}
    for lang in LANGUAGES:
        lang_dir = os.path.join(DATA_DIR, lang)
        if not os.path.exists(lang_dir):
            logging.error(f"Required language folder not found: {lang_dir}")
            sys.exit(1)
        lang_pools[lang] = LanguagePool(lang, lang_dir)
        
    is_test_run = len(sys.argv) > 1 and sys.argv[1] == "--test"
    
    if is_test_run:
        logging.info("--- RUNNING IN TEST MODE (Small subset) ---")
        create_dataset_split("train", 2, lang_pools)
        create_dataset_split("eval", 1, lang_pools)
        create_dataset_split("test", 1, lang_pools)
    else:
        # Full generation
        create_dataset_split("train", 250, lang_pools)
        create_dataset_split("eval", 75, lang_pools)
        create_dataset_split("test", 75, lang_pools)
        
    logging.info("All bilingual dataset splits generated successfully!")

if __name__ == "__main__":
    main()
