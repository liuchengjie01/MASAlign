"""Model loading, language LoRA, and checkpoint export, matching the SFT flow."""

from pathlib import Path


def get_fsdp_layer_classes(model):
    """Return decoder/vision block class names before PEFT wraps the model."""
    decoder_paths = (
        "model.layers",
        "language_model.layers",
        "model.language_model.layers",
        "text_model.layers",
        "model.text_model.layers",
        "transformer.encoder.layers",
    )
    vision_paths = (
        "visual.blocks",
        "model.visual.blocks",
        "vision_model.encoder.layers",
        "model.vision_model.encoder.layers",
    )
    class_names = []
    decoder_found = False
    for module_path in decoder_paths + vision_paths:
        module = model
        for name in module_path.split("."):
            module = getattr(module, name, None)
            if module is None:
                break
        if module is not None and len(module) > 0:
            decoder_found = decoder_found or module_path in decoder_paths
            class_name = type(module[0]).__name__
            if class_name not in class_names:
                class_names.append(class_name)
    if not decoder_found:
        raise RuntimeError("Could not find decoder blocks for FSDP; set model.fsdp_layer_classes")
    return class_names


def apply_language_lora(model, config):
    """Attach a trainable language-only LoRA adapter and keep vision frozen."""
    from peft import LoraConfig, get_peft_model

    lora_config = config["lora"]
    visual = getattr(model, "visual", None)
    if visual is not None:
        for parameter in visual.parameters():
            parameter.requires_grad = False

    model = get_peft_model(
        model,
        LoraConfig(
            r=lora_config["r"],
            lora_alpha=lora_config["lora_alpha"],
            lora_dropout=lora_config["lora_dropout"],
            target_modules=lora_config["target_modules"],
            bias="none",
            task_type=lora_config["task_type"],
        ),
    )

    visual_trainable = [
        name for name, parameter in model.named_parameters()
        if parameter.requires_grad and any(
            part in name.split(".") for part in ("visual", "vision_model", "vision_tower")
        )
    ]
    if visual_trainable:
        raise RuntimeError(
            "Language-only LoRA unexpectedly matched visual modules: "
            + ", ".join(visual_trainable[:5])
        )
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    model.print_trainable_parameters()
    return model


def load_model_and_processor(config):
    import torch
    from transformers import AutoProcessor

    model_config = config["model"]
    model_type = model_config.get("loader", "image_text_to_text")
    if model_type == "causal_lm":
        from transformers import AutoModelForCausalLM as ModelClass
    elif model_type == "image_text_to_text":
        try:
            from transformers import AutoModelForImageTextToText as ModelClass
        except ImportError:
            from transformers import AutoModelImageTextToText as ModelClass
    else:
        raise ValueError(f"Unknown model.loader: {model_type}")

    common = {"trust_remote_code": model_config.get("trust_remote_code", True)}
    print(f"Loading processor: {model_config['path']}", flush=True)
    processor = AutoProcessor.from_pretrained(
        model_config["path"], max_pixels=config["data"]["max_pixels"], **common
    )
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    print(f"Loading model weights: {model_config['path']}", flush=True)
    model = ModelClass.from_pretrained(
        model_config["path"], torch_dtype=getattr(torch, model_config["torch_dtype"]),
        attn_implementation=model_config.get("attn_implementation", "sdpa"),
        low_cpu_mem_usage=True, device_map=None, **common,
    )
    model.config.use_cache = False
    if config["training"].get("fsdp"):
        classes = model_config.get("fsdp_layer_classes") or get_fsdp_layer_classes(model)
    else:
        classes = []
    if config["lora"]["enable"]:
        print("Attaching language LoRA adapter...", flush=True)
        model = apply_language_lora(model, config)
    return model, processor, classes


def save_checkpoint(trainer, processor, config):
    """All ranks enter FSDP collectives; only rank 0 writes processor files."""
    trainer.accelerator.wait_for_everyone()
    trainer.model.config.use_cache = True
    output = Path(config["training"]["output_dir"])
    if config["lora"]["enable"]:
        output = output / "adapter"
        trainer.save_model(str(output))
    else:
        if config["training"].get("fsdp"):
            from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict

            state = get_model_state_dict(
                trainer.model, options=StateDictOptions(full_state_dict=True, cpu_offload=True)
            )
        else:
            state = trainer.accelerator.get_state_dict(trainer.model)
        if trainer.is_world_process_zero():
            trainer.accelerator.unwrap_model(trainer.model).save_pretrained(
                str(output), state_dict=state, safe_serialization=True, max_shard_size="5GB"
            )
    if trainer.is_world_process_zero():
        processor.save_pretrained(str(output))
        expected = "adapter_config.json" if config["lora"]["enable"] else "config.json"
        if not (output / expected).is_file() or not list(output.glob("*.safetensors")):
            raise RuntimeError(f"Incomplete checkpoint in {output}")
        print(f"Saved DPO checkpoint: {output}", flush=True)
    trainer.accelerator.wait_for_everyone()


