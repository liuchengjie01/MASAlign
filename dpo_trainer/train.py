"""DPO on the same trajectory logs used by SFT; YAML controls data and GPUs."""

import argparse
import os
from pathlib import Path

os.environ.setdefault("USE_TF", "0")

if __package__:
    from .configuration import build_training_args, load_config
else:
    from configuration import build_training_args, load_config


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(Path(__file__).with_name("config.yaml")))
    parser.add_argument("--device_profile", default=None)
    parser.add_argument("--check-config", action="store_true", help="Check startup configuration without loading a model")
    return parser.parse_args()


def main():
    print(f"DPO entry: {Path(__file__).resolve()}, pid={os.getpid()}, "
          f"rank={os.environ.get('RANK', '0')}", flush=True)
    args = parse_args()
    config = load_config(args.config, args.device_profile)
    print(f"Profile: {config['device_profile']}; max_length={config['data']['max_length']}; "
          f"model={config['model']['path']}; data={config['data']['train_jsonl']}", flush=True)
    if not Path(config["data"]["train_jsonl"]).is_dir():
        raise FileNotFoundError(f"Trajectory directory does not exist: {config['data']['train_jsonl']}")
    if args.check_config:
        print("DPO configuration check passed", flush=True)
        return
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size != config["num_processes"]:
        raise ValueError(f"{config['device_profile']} requires {config['num_processes']} "
                         "processes; launch with start_dpo.sh or torchrun")
    if config["training"].get("fsdp"):
        os.environ["FSDP_VERSION"] = "2"

    print("Importing Transformers, TRL and data/model modules...", flush=True)
    import transformers
    import trl
    from transformers import set_seed
    from trl import DPOConfig, DPOTrainer
    if __package__:
        from .data import TrajectoryDPODataset
        from .modeling import load_model_and_processor, save_checkpoint
    else:
        from data import TrajectoryDPODataset
        from modeling import load_model_and_processor, save_checkpoint

    print(f"TRL {trl.__version__}; Transformers {transformers.__version__}", flush=True)
    set_seed(config["training"]["seed"])
    print("Loading model and processor...", flush=True)
    model, processor, layer_classes = load_model_and_processor(config)
    data = config["data"]
    print("Building trajectory preference pairs...", flush=True)
    dataset = TrajectoryDPODataset(
        data["train_jsonl"], processor, data_type=data["data_type"],
        guiowl_phases=data["guiowl_phases"], max_images=data["max_images"],
        system_prompt=data["system_prompt"],
    )
    print(f"Profile: {config['device_profile']}; DPO pairs: {len(dataset)}; "
          f"data statistics: {dict(dataset.stats)}", flush=True)
    print("Validating DPO training arguments...", flush=True)
    training_args = build_training_args(config, layer_classes, DPOConfig, transformers.__version__)
    print("Converting preference pairs to Hugging Face Dataset...", flush=True)
    hf_dataset = dataset.to_hf_dataset()
    print("Initializing DPOTrainer...", flush=True)
    trainer = DPOTrainer(
        model=model,
        # With LoRA, TRL obtains reference logits by disabling this adapter.
        ref_model=None,
        args=training_args, train_dataset=hf_dataset,
        processing_class=processor,
    )
    print("Starting trainer.train()...", flush=True)
    trainer.train(resume_from_checkpoint=config["training"].get("resume_from_checkpoint"))
    if trainer.state.global_step == 0:
        raise RuntimeError("DPO finished without any optimizer updates; check the dataset and training settings")
    print(f"Training finished after {trainer.state.global_step} updates; saving checkpoint...", flush=True)
    save_checkpoint(trainer, processor, config)
    print("DPO training and checkpoint save completed", flush=True)


if __name__ == "__main__":
    main()
