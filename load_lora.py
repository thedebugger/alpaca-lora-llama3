import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from peft import PeftModel

parser = argparse.ArgumentParser()
parser.add_argument("prompt", type=str)
args = parser.parse_args()
# Define paths
base_model_path = "/workspace/tune/Meta-Llama-3.1-8B-Instruct"  # Path to the pre-trained base model (e.g., LLaMA)
lora_model_path = "/workspace/tune/Meta-Llama-out-adapter"  # Path to the LoRA fine-tuned adapter

# Load the base model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_path)
base_model = AutoModelForCausalLM.from_pretrained(base_model_path)

# Load the LoRA adapter
lora_model = PeftModel.from_pretrained(base_model, lora_model_path)

# Switch to evaluation mode
lora_model.eval()

# Test the loaded model
prompt = args.prompt 
inputs = tokenizer(prompt, return_tensors="pt")
output = lora_model.generate(**inputs, max_new_tokens=300)

print(tokenizer.decode(output[0], skip_special_tokens=True))
