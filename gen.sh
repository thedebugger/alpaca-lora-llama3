prompt=$1
tune run generate --config ./tune-gen.yaml prompt.user="${prompt}"
tune run generate --config ./tune-gen-lora-ft.yaml prompt.user="${prompt}"
