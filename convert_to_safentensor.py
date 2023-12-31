import torch 
from safetensors.torch import save_file

model = torch.load("pretrained_ljs.pth", map_location="cpu")
save_file(model["model"], "vits_ljs.safetensors")