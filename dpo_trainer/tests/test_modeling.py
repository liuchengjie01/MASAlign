"""Checkpoint and decoder-wrapping contracts without a GPU or model weights."""

from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock

from dpo_trainer.modeling import get_fsdp_layer_classes, save_checkpoint


class DecoderBlock:
    pass


class VisionBlock:
    pass


class ModelingTests(unittest.TestCase):
    def test_decoder_and_vision_wrapping(self):
        model = SimpleNamespace(model=SimpleNamespace(layers=[DecoderBlock()]),
                                visual=SimpleNamespace(blocks=[VisionBlock()]))
        self.assertEqual(get_fsdp_layer_classes(model), ["DecoderBlock", "VisionBlock"])
        with self.assertRaisesRegex(RuntimeError, "decoder blocks"):
            get_fsdp_layer_classes(SimpleNamespace(visual=model.visual))

    def test_single_gpu_full_checkpoint_and_lora_checkpoint(self):
        for lora in (False, True):
            with self.subTest(lora=lora), tempfile.TemporaryDirectory() as directory:
                trainer, processor = Mock(), Mock()
                trainer.is_world_process_zero.return_value = True
                target = Path(directory) / "adapter" if lora else Path(directory)
                target.mkdir(exist_ok=True)
                (target / ("adapter_config.json" if lora else "config.json")).write_text("{}")
                (target / "model.safetensors").write_bytes(b"test fixture")
                config = {"lora": {"enable": lora}, "training": {"output_dir": directory, "fsdp": ""}}
                save_checkpoint(trainer, processor, config)
                processor.save_pretrained.assert_called_once_with(str(target))
                if lora:
                    trainer.save_model.assert_called_once_with(str(target))
                    trainer.accelerator.get_state_dict.assert_not_called()
                else:
                    trainer.accelerator.get_state_dict.assert_called_once_with(trainer.model)
                    trainer.accelerator.unwrap_model.return_value.save_pretrained.assert_called_once()


if __name__ == "__main__":
    unittest.main()
