#!/bin/bash
set -e

if [ ! -d "FreeVC" ]; then
    echo "Cloning FreeVC..."
    git clone https://github.com/OlaWod/FreeVC.git || true
fi

cd FreeVC

echo "Fixing requirements.txt for modern python..."
sed -i 's/==.*//g' requirements.txt
sed -i '/webrtcvad/d' requirements.txt || true

echo "Installing requirements..."
pip install -r requirements.txt
pip install speechbrain

echo "Downloading checkpoints..."
mkdir -p checkpoints
mkdir -p wavlm

if [ ! -f checkpoints/freevc.pth ]; then
    wget -q --show-progress https://storage.openvinotoolkit.org/repositories/openvino_notebooks/models/freevc/freevc.pth -O checkpoints/freevc.pth
fi

if [ ! -f checkpoints/WavLM-Large.pt ]; then
    wget -q --show-progress https://huggingface.co/s3prl/converted_ckpts/resolve/main/wavlm_large.pt -O checkpoints/WavLM-Large.pt
    ln -sf ../checkpoints/WavLM-Large.pt wavlm/WavLM-Large.pt
fi

echo "Setup complete!"
