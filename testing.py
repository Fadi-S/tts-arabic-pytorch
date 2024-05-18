import torch
from utils.data import text_mel_collate_fn

pretrained = torch.load("pretrained/tacotron2_ar_adv.pth", map_location="cpu")
ours = torch.load("states.pth", map_location="cpu")

print(pretrained)

print(ours)

