import argparse
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    pipeline,
    AutoModel,
    AutoProcessor,
)
from peft import PeftModel
from safetensors.torch import load_file
import torch


# for name, module in base_model.named_modules():
#    print(name)


def load_base_multi_model(base_model_path):

    return base_model


def load_base_model(base_model_path):
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path, torch_dtype=torch.bfloat16, device_map="auto"
    )
    return base_model


def load_lora_model(base_model_path, lora_path):
    base_model = load_base_model(base_model_path)

    # Load the LoRA adapter
    lora_model = PeftModel.from_pretrained(
        base_model, lora_path, torch_dtype=torch.bfloat16, device_map="auto"
    )
    return base_model, lora_model


def eval(prompt, is_lora, base_path, lora_path):
    # had no impact on the output?
    # set_seed(23)

    if is_lora:
        print("INFO: using lora model")
        _, model = load_lora_model(base_path, lora_path)
    else:
        print("INFO: Only using base model")
        model = load_base_model(base_path)

    # Switch to evaluation mode
    print(f"INFO: model is loaded with precision {model.dtype}")
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(base_path)
    # inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    inputs = tokenizer.apply_chat_template(
        [{"role": "user", "content": f"{prompt} "}],
        return_tensors="pt",
        add_generation_prompt=True,
    ).to(model.device)
    print(tokenizer.decode(inputs[0]))

    output = model.generate(
        inputs,
        max_new_tokens=1000,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )
    return tokenizer.decode(output[0], skip_special_tokens=True)


def multimodal_eval(prompt, is_lora, base_path, lora_path):
    base_model = AutoModel.from_pretrained(
        base_path, torch_dtype=torch.bfloat16, device_map="auto"
    )


def print_modules(module, base_path, lora_path):
    if module == "lora":
        _, lora_model = load_lora_model(base_path, lora_path)
        model = lora_model
    else:
        model = load_base_model(base_path)

    # module_set=set()
    for name, module in model.named_modules():
        print(name)
    #    mods = name.split(".")
    #    if len(mods) > 1:
    #        module_set.add(f"{mods[-2]}.{mods[-1]}")
    #    else:
    #        module_set.add(f"{mods[-1]}")


def print_weight_map(path):
    tensors = load_file(path)

    for key in tensors.keys():
        print(f"{key}")


def run_ultravox_pipe(prompt):
    print(f"prompt is {prompt}")
    pipe = pipeline(
        model="fixie-ai/ultravox-v0_3",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    turns = [
        {"role": "user", "content": prompt},
    ]
    output = pipe({"turns": turns}, max_new_tokens=100, do_sample=False)
    print(output)


def run_ultravox(prompt, is_lora, lora_path):
    print(f"prompt is {prompt}")
    model = AutoModel.from_pretrained(
        "fixie-ai/ultravox-v0_3", torch_dtype=torch.bfloat16, trust_remote_code=True
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(
        "fixie-ai/ultravox-v0_3", trust_remote_code=True
    )
    turns = [
        {"role": "user", "content": prompt},
    ]
    text = processor.tokenizer.apply_chat_template(
        turns, add_generation_prompt=True, tokenize=False
    )
    model_inputs = processor(text=text)
    print(model_inputs)
    input_len = model_inputs["input_ids"].shape[1]
    outputs = model.generate(**model_inputs, do_sample=False)
    print(processor.tokenizer.decode(outputs[0][input_len:]))


def main():
    parser = argparse.ArgumentParser(
        description="A sample script with two commands and one argument each."
    )

    # Common optional arguments
    parser.add_argument(
        "--model_path",
        type=str,
        default="/workspace/tune/Meta-Llama-3.1-8B-Instruct",
        help="Path to the model file",
    )
    parser.add_argument(
        "--lora_path",
        type=str,
        default="/workspace/tune/Meta-Llama-out-adapter",
        help="Path to the LoRA file",
    )

    # Create subparsers for commands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    subparsers.required = True  # Ensure a command is required

    # command one
    parser_one = subparsers.add_parser("generate", help="execute command one")
    parser_one.add_argument("prompt", type=str, help="argument for command one")
    parser_one.add_argument(
        "--lora",
        type=bool,
        action=argparse.BooleanOptionalAction,
        default=True,
        help="with or without lora",
    )
    parser_one.set_defaults(
        func=lambda args: print(
            eval(args.prompt, args.lora, args.model_path, args.lora_path)
        )
    )

    # Command two
    parser_two = subparsers.add_parser("print_modules", help="print modules")
    parser_two.add_argument("--module", type=str, default="base", help="lora or base")
    parser_two.set_defaults(
        func=lambda args: print_modules(args.module, args.model_path, args.lora_path)
    )

    # Command three
    parser_three = subparsers.add_parser(
        "print_weight_map", help="print weight map from weight file"
    )
    parser_three.add_argument("--path", type=str, help="path of the file to load")
    parser_three.set_defaults(func=lambda args: print_weight_map(args.path))

    # Command 4
    parser_four = subparsers.add_parser("run_ultravox")
    parser_four.add_argument("prompt", type=str)
    parser_four.set_defaults(func=lambda args: run_ultravox_pipe(args.prompt))

    # command 5
    parser_5 = subparsers.add_parser("run_ultravox_lora", help="execute command one")
    parser_5.add_argument("prompt", type=str, help="argument for command one")
    parser_5.add_argument(
        "--lora",
        type=bool,
        action=argparse.BooleanOptionalAction,
        default=True,
        help="with or without lora",
    )
    parser_5.set_defaults(
        func=lambda args: run_ultravox(args.prompt, args.lora, args.lora_path)
    )

    # Parse the arguments
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
