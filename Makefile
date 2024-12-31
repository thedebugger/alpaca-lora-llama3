generate_with_lora:
	python load_lora.py --lora_path "/workspace/huggingface/models--mkopecki--chess-lora-adapter-llama-3.1-8b/snapshots/57e34a08d117294ae87c94a43fbcefc7c2341780/" generate "Tell me about silly chess move in 20 words"	

generate_no_lora:
	python load_lora.py generate --no-lora "Tell me about silly chess move in 20 words"

generate_ultravox:
	python load_lora.py --lora_path="mkopecki/chess-lora-adapter-llama-3.1-8b" run_ultravox_lora --lora "Tell me about silly chess move in 20 words"

generate_ultravox_no_lora:
	python load_lora.py run_ultravox_lora --no-lora "Tell me about silly chess move in 20 words"

