"""Debug script to examine mel spectrogram values"""
import torch
import pickle
import numpy as np
from app.model import SimpleTTS, synthesize_speech, CharacterEncoder
from pathlib import Path

# Load model
BASE_DIR = Path(__file__).parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
device = torch.device('cpu')

# Load encoder
import sys
import app.model
sys.modules['__main__'].CharacterEncoder = CharacterEncoder

with open(ARTIFACTS_DIR / "char_encoder.pkl", 'rb') as f:
    char_encoder = pickle.load(f)

# Load model
model = SimpleTTS(vocab_size=char_encoder.vocab_size, n_mels=80)
checkpoint = torch.load(ARTIFACTS_DIR / "best_model.pt", map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

# Generate mel spectrogram
text = "Salam"
mel = synthesize_speech(text, model, char_encoder, device, max_len=150)

print(f"Mel spectrogram shape: {mel.shape}")
print(f"Min value: {mel.min():.4f}")
print(f"Max value: {mel.max():.4f}")
print(f"Mean value: {mel.mean():.4f}")
print(f"Std value: {mel.std():.4f}")
print(f"\nFirst few values:\n{mel[:5, :5]}")
