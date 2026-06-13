import os
import sys
import csv
import random
import logging
import soundfile as sf
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("generation_multilingual.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

DATA_DIR = "/home/teaching/s25019/Ekstep_simulated/IITMandi_Readspeech"
OUTPUT_DIR = "/home/teaching/s25019/Ekstep_simulated/sim_dataset_multilingual"
TARGET_DURATION = 900.0  # 15 minutes in seconds

ALL_LANGUAGES = ['asm', 'ben', 'eng', 'guj', 'hin', 'kan', 'mal', 'mar', 'odi', 'pun', 'tam', 'tel']

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
    
    csv_path = os.path.join(split_dir, f"{split_name}_labels.csv")
    csv_file = open(csv_path, mode='w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "file_id", "num_languages", "total_duration", "total_utterances", "num_language_switches",
        "L1", "L2", "L3", "L4", "L5",
        "L1_duration", "L2_duration", "L3_duration", "L4_duration", "L5_duration"
    ])
    
    logging.info(f"Starting generation for split: {split_name} ({num_files} files)")
    
    for i in range(1, num_files + 1):
        file_id = f"sim_{split_name}_{i:04d}"
        wav_path = os.path.join(wav_dir, f"{file_id}.wav")
        rttm_path = os.path.join(rttm_dir, f"{file_id}.rttm")
        
        # Determine number of languages N
        N = random.choices([2, 3, 4, 5], weights=[0.70, 0.20, 0.05, 0.05])[0]
        
        selected_langs = set()
        
        # 25% chance to force English
        if random.random() < 0.25:
            selected_langs.add('eng')
            
        # 25% chance to force Hindi
        if random.random() < 0.25:
            selected_langs.add('hin')
            
        # Fill remaining slots with other languages
        remaining_langs = [l for l in ALL_LANGUAGES if l not in selected_langs]
        random.shuffle(remaining_langs)
        
        while len(selected_langs) < N:
            selected_langs.add(remaining_langs.pop())
            
        selected_langs = list(selected_langs)
        random.shuffle(selected_langs) # Shuffle to randomize L1, L2 assignments
        
        lang_to_label = {lang: f"L{idx+1}" for idx, lang in enumerate(selected_langs)}
        
        logging.info(f"Generating {file_id}.wav with languages {selected_langs} mapped to {lang_to_label}")
        
        audio_chunks = []
        rttm_entries = []
        current_time = 0.0
        
        total_utterances = 0
        num_language_switches = 0
        last_lang = None
        lang_durations = {lang: 0.0 for lang in selected_langs}
        
        while current_time < TARGET_DURATION:
            current_lang = random.choice(selected_langs)
            if last_lang is not None and current_lang != last_lang:
                num_language_switches += 1
            last_lang = current_lang
            
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
                lang_durations[current_lang] += duration
                total_utterances += 1
                
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
        if audio_chunks:
            full_audio = np.concatenate(audio_chunks, axis=0)
            
            # Write WAV file
            sf.write(wav_path, full_audio, 8000, subtype='PCM_16')
            
            # Write RTTM file
            with open(rttm_path, 'w') as rttm_file:
                rttm_file.writelines(rttm_entries)
                
            logging.info(f"Finished {file_id} - Total duration: {current_time:.2f}s, Audio length: {len(full_audio)/8000:.2f}s")
            
            # Prepare CSV row
            row = [
                file_id, N, f"{current_time:.2f}", total_utterances, num_language_switches
            ]
            
            # Extract L1-L5 labels
            l_labels = [""] * 5
            l_durations = [""] * 5
            for idx, lang in enumerate(selected_langs):
                l_labels[idx] = lang
                l_durations[idx] = f"{lang_durations[lang]:.2f}"
                
            row.extend(l_labels)
            row.extend(l_durations)
            
            csv_writer.writerow(row)
            
    csv_file.close()

def main():
    # Initialize language pools
    lang_pools = {}
    for lang in ALL_LANGUAGES:
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
        # Full scale generation for overnight run
        create_dataset_split("train", 250, lang_pools)
        create_dataset_split("eval", 75, lang_pools)
        create_dataset_split("test", 75, lang_pools)
        
    logging.info("All multilingual dataset splits generated successfully!")

if __name__ == "__main__":
    main()
