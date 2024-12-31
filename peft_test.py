import torch
import torch.nn as nn
import os
from os import path
from peft import PeftModel, LoraConfig, get_peft_model
import json
from safetensors.torch import save_file

class InnerModel(nn.Module):
    def __init__(self, bias=True):
        super().__init__()
        self.lin0  = nn.Linear(5, 5, bias=bias)
        self.prepare_inputs_for_generation = None
        self._prepare_encoder_decoder_kwargs_for_generation = None

    def forward(self, X):
        X = X.float()
        X = self.innner_model(X)
        return X

class DummyModel(nn.Module):
    def __init__(self, bias=True):
        super().__init__()
        self.lin0 = nn.Linear(5, 5, bias=bias)
        self.relu = nn.ReLU()
        self.lin1 = nn.Linear(20, 2, bias=bias)
        self.inner_model = InnerModel()
        self.sm = nn.LogSoftmax(dim=-1)
        self.prepare_inputs_for_generation = None
        self._prepare_encoder_decoder_kwargs_for_generation = None

    def forward(self, X):
        X = X.float()
        X = self.lin0(X)
        X = self.relu(X)
        X = self.lin1(X)
        X = self.sm(X)
        return X

# Initialize the dummy model and tokenizer
dummy_model = DummyModel()
for m in dummy_model.named_modules():
    print(m)

mock_lora_state_dict = {
    "base_model.model.lin0.lora_A.weight": torch.randn(2, 5),  # Simulate LoRA weights for the Linear layer
    "base_model.model.lin0.lora_B.weight": torch.randn(5, 2),  # Simulate LoRA weights for the Linear layer
}

mock_lora_base_path="/tmp/mock"
if not path.exists(mock_lora_base_path):
    os.makedirs(mock_lora_base_path)
# Save the state dictionary to a file
mock_lora_path = path.join(mock_lora_base_path, "adapter_model.safetensors")
save_file(mock_lora_state_dict, mock_lora_path)

lora_config = LoraConfig(
    base_model_name_or_path="bert-base-uncased",
    r=2,
    lora_alpha=16,
    target_modules="inner.*lin0",  # Specify the modules to which LoRA is applied
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"  # Task type for the adapter
)

# Apply the LoRA to the dummy model
lora_model = PeftModel.from_pretrained(dummy_model, mock_lora_base_path, config=lora_config)
lora_model.get_model_status()
print(lora_model.get_layer_status())
print(lora_model)

print("LoRA adapter loaded and applied to the model.")
