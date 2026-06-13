import os
import sys
import librosa
import soundfile as sf
import numpy as np
from tqdm import tqdm
import torch
import logging

logging.getLogger('numba').setLevel(logging.WARNING)

sys.path.insert(0, "FreeVC")
import utils
from models import SynthesizerTrn
from mel_processing import mel_spectrogram_torch
from speaker_encoder.voice_encoder import SpeakerEncoder

def preprocess_wav(in_path, out_path, target_sr=16000):
    audio, sr = librosa.load(in_path, sr=target_sr, mono=True)
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak * 0.708  # peak-normalise to -3 dBFS
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    sf.write(out_path, audio, target_sr)

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

def main():
    is_test_run = "--test" in sys.argv
    base_dir = "/home/teaching/s25019/Ekstep_simulated"
    
    datasets = ["Ekstep_12_lang_multi_domain", "Ekstep_12_lang_tvnews", "IITMandi_Readspeech"]
    
    preprocessed_root = os.path.join(base_dir, "tmp_16k")
    os.makedirs(preprocessed_root, exist_ok=True)
    
    if is_test_run:
        print("TEST MODE: Processing limited files.")
    
    # 1. Preprocessing
    print("--- Preprocessing ---")
    for ds_name in datasets:
        src_root = os.path.join(base_dir, ds_name)
        dst_root = os.path.join(preprocessed_root, ds_name)
        if not os.path.exists(src_root): continue
            
        for lang in os.listdir(src_root):
            lang_path = os.path.join(src_root, lang)
            if not os.path.isdir(lang_path): continue
                
            files = sorted([f for f in os.listdir(lang_path) if f.endswith(".wav")])
            if is_test_run: files = files[:1] # Just 1 file per lang in test mode
                
            for fname in tqdm(files, desc=f"Prep {ds_name}/{lang}"):
                out_path = os.path.join(dst_root, lang, fname)
                if not os.path.exists(out_path):
                    preprocess_wav(os.path.join(lang_path, fname), out_path)
    
    # 2. Select Speaker X
    print("\n--- Selecting Speaker X ---")
    ref_ds_root = os.path.join(preprocessed_root, "IITMandi_Readspeech")
    if not os.path.exists(ref_ds_root): ref_ds_root = os.path.join(preprocessed_root, datasets[0])
    speaker_x_path = select_speaker_x(ref_ds_root)
    
    # 3. Load Models
    print("\n--- Loading FreeVC Models ---")
    os.chdir(os.path.join(base_dir, "FreeVC"))
    
    hps = utils.get_hparams_from_file("configs/freevc.json")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    net_g = SynthesizerTrn(
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model).to(device)
    net_g.eval()
    
    print("Loading checkpoint...")
    utils.load_checkpoint("checkpoints/freevc.pth", net_g, None, True)
    
    print("Loading WavLM for content...")
    cmodel = utils.get_cmodel(0)
    
    print("Loading speaker encoder...")
    smodel = SpeakerEncoder('speaker_encoder/ckpt/pretrained_bak_5805000.pt')
    
    # Pre-embed Speaker X
    wav_tgt, _ = librosa.load(speaker_x_path, sr=hps.data.sampling_rate)
    wav_tgt, _ = librosa.effects.trim(wav_tgt, top_db=20)
    g_tgt = smodel.embed_utterance(wav_tgt)
    g_tgt = torch.from_numpy(g_tgt).unsqueeze(0).to(device)
    
    print("\n--- Conversion ---")
    for ds_name in datasets:
        src_root = os.path.join(preprocessed_root, ds_name)
        dst_root = os.path.join(base_dir, f"{ds_name}_vc_test" if is_test_run else f"{ds_name}_vc")
        
        if not os.path.exists(src_root): continue
            
        for lang in sorted(os.listdir(src_root)):
            lang_in = os.path.join(src_root, lang)
            lang_out = os.path.join(dst_root, lang)
            if not os.path.isdir(lang_in): continue
            os.makedirs(lang_out, exist_ok=True)
            
            files = sorted([f for f in os.listdir(lang_in) if f.endswith(".wav")])
            if is_test_run: files = files[:1]
                
            for fname in tqdm(files, desc=f"{ds_name}/{lang}"):
                src = os.path.join(lang_in, fname)
                dst = os.path.join(lang_out, fname)
                if os.path.exists(dst): continue
                    
                with torch.no_grad():
                    wav_src, _ = librosa.load(src, sr=hps.data.sampling_rate)
                    wav_src_tensor = torch.from_numpy(wav_src).unsqueeze(0).to(device)
                    c = utils.get_content(cmodel, wav_src_tensor)
                    audio = net_g.infer(c, g=g_tgt)
                    audio = audio[0][0].data.cpu().float().numpy()
                sf.write(dst, audio, hps.data.sampling_rate)

    print(f"\nConversion complete!")
    if is_test_run:
        print(f"Test samples saved to *_vc_test/ directories.")

if __name__ == "__main__":
    main()
