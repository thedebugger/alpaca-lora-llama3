import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from peft import PeftModel


#for name, module in base_model.named_modules():
#    print(name)


def load_models(base_path, lora_path):
	# Load the base model and tokenizer
	tokenizer = AutoTokenizer.from_pretrained(base_model_path)
	base_model = AutoModelForCausalLM.from_pretrained(base_model_path)

	# Load the LoRA adapter
	lora_model = PeftModel.from_pretrained(base_model, lora_path)
	return base_model, lora_model


def eval(prompt, base_path, lora_path):
	_, lora_model = load_models(base_path, lora_path)
	# Switch to evaluation mode
	lora_model.eval()

	inputs = tokenizer(prompt, return_tensors="pt")
	output = lora_model.generate(**inputs, max_new_tokens=300)
	print(tokenizer.decode(output[0], skip_special_tokens=True))

def print_modules(module, base_path, lora_path):
	base_model, lora_model = load_models(base_path, lora_path)
	model = base_model
	if module == "lora":
		model = lora_model	

	for name, module in model.named_modules():
		print(name)

def main():
    parser = argparse.ArgumentParser(description="A sample script with two commands and one argument each.")

	# Common optional arguments
    parser.add_argument("--model_path", type=str, default="/workspace/tune/Meta-Llama-3.1-8B-Instruct", help="Path to the model file")
    parser.add_argument("--lora_path", type=str, default="/workspace/tune/Meta-Llama-out-adapter", help="Path to the LoRA file")

    # Create subparsers for commands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    subparsers.required = True  # Ensure a command is required

    # Command one
    parser_one = subparsers.add_parser("generate", help="Execute command one")
    parser_one.add_argument("prompt", type=str, help="Argument for command one")
    parser_one.set_defaults(func=lambda args: eval(args.prompt))

    # Command two
    parser_two = subparsers.add_parser("print_modules", help="print modules")
    parser_one.add_argument("module", type=str, help="lora or base")
    parser_two.set_defaults(func=lambda args: print_modules(args.model_path, args.lora_path))

    # Parse the arguments
    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
