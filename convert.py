import argparse
import torch 
from safetensors import safe_open
file = open("pretrained_ljs.pth", "rb")
model = torch.load(file, map_location="cpu")
print(model["model"].keys())
