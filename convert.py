# File converts vits pretrained models to safetensors 
import torch
from safetensors.torch import save_file
import argparse 
from os.path import splitext

parser = argparse.ArgumentParser(description="Converts Pytorch model to safetensor and saves on disk")
parser.add_argument("-p", "--path", help="path to pretrained models")
args = parser.parse_args()
pretrained_model_path = args.path
pretrained_model = torch.load(pretrained_model_path, map_location=torch.device("cpu"))
save_file(pretrained_model["model"], splitext(pretrained_model_path)[0] + ".safetensors")
