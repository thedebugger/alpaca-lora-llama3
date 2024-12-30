import os
import argparse
from huggingface_hub import hf_hub_download, HfApi, Repository
from safetensors.torch import load_file, save_file


# Step 2: Transform module names in tensor files
def transform_module_names(state_dict):
    """
    Transform module names. For example, replace 'lora_' with 'custom_lora_'.
    """
    transformed_state_dict = {}
    for key, value in state_dict.items():
        print(key)
        new_key = key.replace("lora_", "custom_lora_")  # Example transformation
        transformed_state_dict[new_key] = value
    return transformed_state_dict


def transform(api, source_repo, repo_path):

    # Step 1: Get the list of files from the source repository
    repo_info = api.repo_info(repo_id=source_repo)
    adapter_files = [file.rfilename for file in repo_info.siblings if file.rfilename.endswith(".safetensors")]
    config_files = [file.rfilename for file in repo_info.siblings if file.rfilename.endswith(".json")]

    # Download all required files
    downloaded_files = {}
    for file in adapter_files + config_files:
        local_file = hf_hub_download(repo_id=source_repo, filename=file)
        downloaded_files[file] = local_file

    # Process and save each safetensor file
    for tensor_file in adapter_files:
        adapter_model_path = downloaded_files[tensor_file]
        state_dict = load_file(adapter_model_path)
        transformed_state_dict = transform_module_names(state_dict)

        # Save transformed state_dict
        transformed_model_path = os.path.join(repo_path, tensor_file)
        save_file(transformed_state_dict, transformed_model_path)

    # Step 3: Save config files unchanged
    for config_file in config_files:
        config_path = os.path.join(repo_path, config_file)
        with open(downloaded_files[config_file], "r") as src_config:
            with open(config_path, "w") as dest_config:
                dest_config.write(src_config.read())

    # Step 4: Push changes to the new repository
    # repo.push_to_hub(commit_message="Add transformed adapter with multiple safetensor files")

    print(f"Transformed adapter pushed to: https://huggingface.co/{target_repo}")


def push_hf(target_repo, target_repo_local_path):
    api.create_repo(repo_id=target_repo)

    # Clone the target repo
    repo = Repository(target_repo_local_path, clone_from=target_repo)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Transform module names in a Hugging Face LoRA adapter and push to a new repository."
    )
    parser.add_argument(
        "--source-repo", required=True, help="Source Hugging Face repository (e.g., 'peft/adapter-example')."
    )
    parser.add_argument(
        "--target-repo",
        required=True,
        help="Target Hugging Face repository (e.g., 'username/adapter-example-transformed').",
    )
    parser.add_argument("--target-repo-local-path", required=True, help="")
    api = HfApi()
    args = parser.parse_args()
    transform(api=api, source_repo=args.source_repo, repo_path=args.target_repo_local_path)
