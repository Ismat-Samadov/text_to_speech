"""
TTS Model Definition and Inference
"""
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path


class CharacterEncoder:
    """Encode and decode text at character level"""

    def __init__(self):
        self.char_to_idx = {}
        self.idx_to_char = {}
        self.vocab_size = 0

    def fit(self, texts):
        """Build vocabulary from texts"""
        # Get all unique characters
        all_chars = set(''.join(texts))

        # Add special tokens
        special_tokens = ['<PAD>', '<SOS>', '<EOS>', '<UNK>']

        # Build mappings
        vocab = special_tokens + sorted(list(all_chars))
        self.char_to_idx = {char: idx for idx, char in enumerate(vocab)}
        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}
        self.vocab_size = len(vocab)

        return self

    def encode(self, text, max_length=None):
        """Convert text to indices"""
        indices = [self.char_to_idx.get(char, self.char_to_idx['<UNK>'])
                  for char in text]

        # Add EOS token
        indices.append(self.char_to_idx['<EOS>'])

        # Pad if needed
        if max_length:
            if len(indices) < max_length:
                indices += [self.char_to_idx['<PAD>']] * (max_length - len(indices))
            else:
                indices = indices[:max_length]

        return indices

    def decode(self, indices):
        """Convert indices back to text"""
        chars = []
        for idx in indices:
            if idx == self.char_to_idx['<EOS>']:
                break
            if idx != self.char_to_idx['<PAD>']:
                chars.append(self.idx_to_char.get(idx, '<UNK>'))
        return ''.join(chars)


class TTSEncoder(nn.Module):
    """Text encoder for TTS"""

    def __init__(self, vocab_size, embedding_dim=256, hidden_dim=512, num_layers=2):
        super(TTSEncoder, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            bidirectional=True
        )

    def forward(self, x):
        embedded = self.embedding(x)
        outputs, (hidden, cell) = self.lstm(embedded)
        return outputs


class Attention(nn.Module):
    """Attention mechanism"""

    def __init__(self, encoder_dim, decoder_dim, attention_dim=128):
        super(Attention, self).__init__()

        self.encoder_att = nn.Linear(encoder_dim, attention_dim)
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)
        self.full_att = nn.Linear(attention_dim, 1)

    def forward(self, encoder_out, decoder_hidden):
        att1 = self.encoder_att(encoder_out)
        att2 = self.decoder_att(decoder_hidden)
        att = self.full_att(torch.tanh(att1 + att2.unsqueeze(1)))
        alpha = torch.softmax(att, dim=1)
        context = (encoder_out * alpha).sum(dim=1)

        return context, alpha


class TTSDecoder(nn.Module):
    """Mel spectrogram decoder"""

    def __init__(self, encoder_dim, decoder_dim=512, n_mels=80, num_layers=2):
        super(TTSDecoder, self).__init__()

        self.attention = Attention(encoder_dim, decoder_dim)
        self.lstm = nn.LSTM(
            encoder_dim + n_mels,
            decoder_dim,
            num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(decoder_dim, n_mels)

    def forward(self, encoder_out, mel_input, hidden=None):
        if hidden is None:
            batch_size = encoder_out.size(0)
            hidden = (torch.zeros(2, batch_size, 512).to(encoder_out.device),
                     torch.zeros(2, batch_size, 512).to(encoder_out.device))

        context, alpha = self.attention(encoder_out, hidden[0][-1])
        decoder_input = torch.cat([context, mel_input], dim=1).unsqueeze(1)
        output, hidden = self.lstm(decoder_input, hidden)
        mel_frame = self.fc(output.squeeze(1))

        return mel_frame, hidden, alpha


class SimpleTTS(nn.Module):
    """Simple TTS model combining encoder and decoder"""

    def __init__(self, vocab_size, n_mels=80):
        super(SimpleTTS, self).__init__()

        self.encoder = TTSEncoder(vocab_size, embedding_dim=256, hidden_dim=256, num_layers=2)
        self.decoder = TTSDecoder(encoder_dim=512, decoder_dim=512, n_mels=n_mels, num_layers=2)
        self.n_mels = n_mels

    def forward(self, text, mel_target=None, max_len=500):
        batch_size = text.size(0)
        encoder_out = self.encoder(text)

        mel_outputs = []
        mel_input = torch.zeros(batch_size, self.n_mels).to(text.device)
        hidden = None

        if mel_target is not None:
            target_len = mel_target.size(2)
        else:
            target_len = max_len

        for t in range(target_len):
            mel_frame, hidden, alpha = self.decoder(encoder_out, mel_input, hidden)
            mel_outputs.append(mel_frame.unsqueeze(2))

            if mel_target is not None and t < target_len - 1:
                mel_input = mel_target[:, :, t]
            else:
                mel_input = mel_frame

        mel_outputs = torch.cat(mel_outputs, dim=2)
        return mel_outputs


def synthesize_speech(text, model, char_encoder, device='cpu', max_len=500):
    """
    Synthesize speech from text.

    Args:
        text: Input text string
        model: Trained TTS model
        char_encoder: Character encoder
        device: Device to run inference on
        max_len: Maximum mel spectrogram length

    Returns:
        Mel spectrogram (numpy array)
    """
    model.eval()

    # Encode text
    text_encoded = char_encoder.encode(text, max_length=200)
    text_tensor = torch.tensor([text_encoded], dtype=torch.long).to(device)

    # Generate mel spectrogram
    with torch.no_grad():
        mel_output = model(text_tensor, mel_target=None, max_len=max_len)

    # Convert to numpy
    mel_numpy = mel_output.cpu().numpy()[0]

    return mel_numpy
