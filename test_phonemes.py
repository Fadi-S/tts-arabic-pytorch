import random

import torchaudio
import torch

import text
import pandas as pd
from tqdm import tqdm
from utils.audio import MelSpectrogram


def preprocess(txt):
    txt = txt.replace(".", "")
    txt = txt.replace("!", "")
    txt = txt.replace(",", "")
    t_phon = text.arabic_to_buckwalter(txt)
    t_phon = text.buckwalter_to_phonemes(t_phon)
    return t_phon


data = pd.read_csv("segments.csv", names=['audio_file', 'text'], dtype=str)


output = []
texts = data['text']
# audio_files = data['audio_file']
# print(data)


def remove_silence(energy_per_frame: torch.Tensor,
                   thresh: float = -10.0):
    keep = energy_per_frame > thresh
    # keep silence at the end
    i = keep.size(0)-1
    while not keep[i] and i > 0:
        keep[i] = True
        i -= 1
    return keep


def mel_from_audio(audio_file):
    sr_target = 22050
    mel_fn = MelSpectrogram()

    wave, sr = torchaudio.load(audio_file)
    if sr != sr_target:
        wave = torchaudio.functional.resample(wave, sr, sr_target, 64)

    wave = wave[0].unsqueeze(0)

    mel_raw = mel_fn(wave)
    mel_log = mel_raw.clamp_min(1e-5).log().squeeze()

    energy_per_frame = mel_log.mean(0)
    if len(energy_per_frame.size()) == 0:
        return None
    mel_log = mel_log[:, remove_silence(energy_per_frame)]

    return mel_log


for i in tqdm(range(len(texts))):
    phonemes = preprocess(texts[i])
    tokens = text.phonemes_to_tokens(phonemes)
    token_ids = text.tokens_to_ids(tokens)
    output.append(token_ids)

print(len(output))


# print(mel_from_audio("sample.wav"))

