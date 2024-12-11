import json
import torch

# create the output dictionary
output_dict = {"weight_map": {}}

# Load the checkpoints
input_dir="/workspace/tune/Meta-Llama-out"
sd_2 = torch.load(f"{input_dir}/hf_model_0001_0.pt", mmap=True, map_location='cpu')

input_dir1="/workspace/tune/Meta-Llama-out-adapter"
sd_1 = torch.load(f"{input_dir1}/adapter_0.pt", mmap=True, map_location='cpu')

# create the weight map
for key in sd_2.keys():
    output_dict['weight_map'][key] =  "hf_model_0001_0.pt"

with open('pytorch_model.bin.index.json', 'w') as f:
    json.dump(output_dict, f)
