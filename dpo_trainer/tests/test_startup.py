"""Training entry and launcher behavior; no torch imports or GPU execution."""

import contextlib
import io
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from dpo_trainer import train
from dpo_trainer import data, modeling
from dpo_trainer.configuration import load_config


DPO_DIR = Path(__file__).resolve().parents[1]


class StartupTests(unittest.TestCase):
    def test_config_check_runs_without_training_dependencies(self):
        config = load_config(DPO_DIR / "config.yaml", "pro6000_98g")
        with tempfile.TemporaryDirectory() as directory:
            config["data"]["train_jsonl"] = directory
            args = SimpleNamespace(config="test.yaml", device_profile="pro6000_98g", check_config=True)
            output = io.StringIO()
            with patch.object(train, "parse_args", return_value=args), \
                    patch.object(train, "load_config", return_value=config), \
                    patch.dict(sys.modules, {"transformers": None, "trl": None}), \
                    contextlib.redirect_stdout(output):
                train.main()
            self.assertIn("DPO entry:", output.getvalue())
            self.assertIn("DPO configuration check passed", output.getvalue())

    def test_main_calls_training_and_rejects_zero_updates(self):
        for updates in (0, 3):
            with self.subTest(updates=updates), tempfile.TemporaryDirectory() as directory:
                config = load_config(DPO_DIR / "config.yaml", "pro6000_98g")
                config["data"]["train_jsonl"] = directory
                args = SimpleNamespace(config="test.yaml", device_profile="pro6000_98g", check_config=False)
                trainer = Mock()
                trainer.state.global_step = updates
                fake_transformers = SimpleNamespace(__version__="5.22.2", set_seed=Mock())
                fake_trl = SimpleNamespace(__version__="1.17.1", DPOConfig=Mock(),
                                           DPOTrainer=Mock(return_value=trainer))
                with patch.object(train, "parse_args", return_value=args), \
                        patch.object(train, "load_config", return_value=config), \
                        patch.object(train, "build_training_args", return_value=Mock()), \
                        patch.dict(sys.modules, {"transformers": fake_transformers, "trl": fake_trl}), \
                        patch.dict(os.environ, {"WORLD_SIZE": "1"}), \
                        patch.object(data, "TrajectoryDPODataset") as dataset_class, \
                        patch.object(modeling, "load_model_and_processor", return_value=(Mock(), Mock(), [])), \
                        patch.object(modeling, "save_checkpoint") as save, \
                        contextlib.redirect_stdout(io.StringIO()):
                    class Dataset:
                        stats = {"action_pairs": 1}
                        to_hf_dataset = Mock()

                        def __len__(self):
                            return 1

                    dataset_class.return_value = Dataset()
                    if updates:
                        train.main()
                        save.assert_called_once()
                    else:
                        with self.assertRaisesRegex(RuntimeError, "without any optimizer updates"):
                            train.main()
                        save.assert_not_called()
                    trainer.train.assert_called_once_with(resume_from_checkpoint=None)

    @unittest.skipUnless(shutil.which("bash"), "bash is not available")
    def test_launcher_checks_entry_and_propagates_training_exit_code(self):
        for marker, exit_code, expected in ((False, 0, 2), (True, 0, 0), (True, 7, 7)):
            with self.subTest(marker=marker, exit_code=exit_code):
                check_output = "DPO configuration check passed" if marker else ""
                harness = '''
python() {
  if [[ "$1" == */configuration.py ]]; then echo 1; return 0; fi
  if [[ "$1" == -c ]]; then echo "MOCK GPU CHECK"; return 0; fi
  if [[ " $* " == *" --check-config "* ]]; then echo "%s"; return 0; fi
  echo "MOCK TRAIN: $*"
  return %d
}
export -f python
export DEVICE_PROFILE=pro6000_98g
bash "$1"
''' % (check_output, exit_code)
                result = subprocess.run(
                    ["bash", "-c", harness, "bash", (DPO_DIR / "start_dpo_long_context.sh").as_posix()],
                    capture_output=True, text=True,
                )
                self.assertEqual(result.returncode, expected, result.stdout + result.stderr)
                if marker:
                    self.assertIn("MOCK TRAIN: -u ", result.stdout)
                    self.assertNotIn("-m torch.distributed.run", result.stdout)
                    self.assertIn(f"DPO process exited with code {exit_code}", result.stdout)
                else:
                    self.assertNotIn("MOCK GPU CHECK", result.stdout)
                    self.assertIn("main() entry", result.stderr)


if __name__ == "__main__":
    unittest.main()
