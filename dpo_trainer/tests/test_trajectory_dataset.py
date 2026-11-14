"""Data construction regressions; no model weights or CUDA are required."""

import base64
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

from PIL import Image


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from dpo_trainer import data as train


class Processor:
    tokenizer = None

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        assert tokenize is False and add_generation_prompt is True
        self.messages = messages
        parts = []
        for message in messages:
            content = message["content"]
            if isinstance(content, list):
                content = "".join(
                    "<image>" if block["type"] == "image" else block["text"]
                    for block in content
                )
            parts.append(f'{message["role"]}: {content}')
        return "\n".join(parts) + "\nassistant:"


def image_block():
    with io.BytesIO() as buffer:
        Image.new("RGB", (2, 3), "red").save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return {"type": "image_url", "image_url": {"url": "data:image/png;base64," + encoded}}


def messages():
    return [{"role": "user", "content": [
        {"type": "text", "text": "Complete the task"}, image_block(),
    ]}]


def step(response, guiowl=False, attack=False):
    prefix = "action_" if guiowl else ""
    value = {
        prefix + "messages": messages(),
        prefix + "response": response,
        "action": {"action": "click", "coordinate": [1, 2]},
    }
    if attack:
        value["is_attack"] = True
    if guiowl:
        value.update({
            "planning_messages": messages(), "planning_response": "PLAN ONLY",
            "reflect_messages": messages(), "reflect_response": "REFLECT ONLY",
        })
    return value


class TrajectoryDatasetTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def write(self, relative, trajectory):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(trajectory), encoding="utf-8")

    def dataset(self, data_type="ghostei", **kwargs):
        kwargs.setdefault("guiowl_phases", ["action"])
        return train.TrajectoryDPODataset(
            str(self.root), Processor(), data_type=data_type, **kwargs
        )

    def test_both_raw_schemas_preserve_responses_and_images(self):
        for guiowl in (False, True):
            with self.subTest(guiowl=guiowl):
                task = "2" if guiowl else "1"
                chosen = "### Thought ###\nSafe\n### Action ###\n{}" if guiowl else "<thinking>Safe</thinking><tool_call>{}</tool_call>"
                rejected = "Unsafe response"
                self.write(f"{task}/run/0.json", [step(chosen, guiowl)])
                self.write(f"{task}/run/1.json", [step(rejected, guiowl, attack=True)])
                dataset = self.dataset()
                item = dataset[len(dataset) - 1]
                self.assertEqual(item["chosen"], chosen)
                self.assertEqual(item["rejected"], rejected)
                self.assertIsInstance(item["prompt"], str)
                self.assertTrue(item["prompt"].endswith("assistant:"))
                self.assertNotIn("PLAN ONLY", item["chosen"])
                self.assertNotIn("REFLECT ONLY", item["chosen"])
                self.assertEqual(len(item["images"]), 1)
                self.assertEqual(item["images"][0].size, (2, 3))
                self.assertEqual(item["images"][0].mode, "RGB")

    def test_compact_string_and_dictionary_actions(self):
        self.write("1/run/0.json", [{"messages": messages(), "action": "safe action"}])
        self.write("1/run/1.json", [{"action": {"action": "type", "text": "中文"}, "is_attack": True}])
        item = self.dataset()[0]
        self.assertEqual(item["chosen"], "safe action")
        self.assertEqual(json.loads(item["rejected"])["text"], "中文")

    def test_empty_terminal_action_response_is_skipped(self):
        self.write("1/run/0.json", [step("safe", True), step("", True)])
        self.write("1/run/1.json", [step("unsafe", True, True)])
        self.assertEqual(self.dataset()[0]["chosen"], "safe")

    def test_guiowl_response_and_prompt_use_the_same_call(self):
        safe = step("GUI action", guiowl=True)
        safe["messages"] = [{"role": "user", "content": "Different call"}]
        safe["response"] = "Different response"
        self.write("1/run/0.json", [safe])
        self.write("1/run/1.json", [step("unsafe", True, True)])
        item = self.dataset()[0]
        self.assertEqual(item["chosen"], "GUI action")
        self.assertNotIn("Different call", item["prompt"])
        self.assertEqual(len(item["images"]), 1)

    def test_missing_initial_prompt_does_not_use_another_call_or_later_step(self):
        safe = step("GUI action", guiowl=True)
        del safe["action_messages"]
        safe["messages"] = messages()
        self.write("1/run/0.json", [safe, step("Later response", True)])
        self.write("1/run/1.json", [step("unsafe", True, True)])
        with self.assertRaisesRegex(ValueError, "invalid_phases.*1"):
            self.dataset()

    def test_task_missing_a_preference_side_has_a_clear_diagnostic(self):
        self.write("1/run/0.json", [step("unsafe", True, True)])
        with self.assertRaisesRegex(ValueError, "attack_only_tasks.*1"):
            self.dataset()
        self.write("2/run/0.json", [step("safe")])
        with self.assertRaisesRegex(ValueError, "safe_only_tasks.*1"):
            self.dataset()

    def test_task_discovery_and_model_isolation(self):
        self.write("gui-owl/999/run/0.json", [step("safe", True)])
        self.write("mai-ui/999/run/0.json", [step("unsafe", attack=True)])
        with self.assertRaisesRegex(ValueError, "No DPO pairs"):
            self.dataset()
        self.write("gui-owl/999/run/1.json", [step("other unsafe", True, True)])
        self.write("gui-owl/999/run/task_result.json", {"goal": "auxiliary"})
        self.assertEqual(len(self.dataset()), 1)

    def test_images_system_prompt_and_no_target_leak(self):
        safe = step("safe")
        safe["messages"][0]["content"].append(image_block())
        safe["messages"].append({"role": "assistant", "content": "safe"})
        self.write("1/run/0.json", [safe])
        self.write("1/run/1.json", [step("unsafe", attack=True)])
        dataset = self.dataset(system_prompt="Configured system", max_images=1)
        item = dataset[0]
        self.assertIn("system: Configured system", item["prompt"])
        self.assertIn("[Screenshot omitted]", item["prompt"])
        self.assertNotIn("assistant: safe", item["prompt"])
        self.assertEqual(len(item["images"]), 1)
        original = train.read_trajectory(self.root / "1/run/0.json")
        self.assertEqual(len(original[0]["messages"][0]["content"]), 3)

    def test_identical_and_incomplete_pairs(self):
        self.write("1/run/0.json", [step("same")])
        self.write("1/run/1.json", [step("same", attack=True)])
        self.write("2/run/0.json", [{"action": "no prompt"}])
        self.write("2/run/1.json", [step("unsafe", attack=True)])
        self.write("3/run/0.json", [step("safe")])
        self.write("3/run/1.json", [step("unsafe", attack=True)])
        dataset = self.dataset()
        self.assertEqual(len(dataset), 1)
        self.assertEqual(dataset.stats["invalid_phases"], 2)

    def test_mobilesafetybench_existing_layout(self):
        self.write("gui-owl/memo_completing/task1/run/log_0_defense_-1.json", [step("safe", True)])
        self.write("gui-owl/memo_completing/task1/run/log_1_defense_-1.json", [step("unsafe", True, True)])
        self.assertEqual(self.dataset(data_type="mobilesafetybench")[0]["chosen"], "safe")

    def test_guiowl_all_phases_keep_separate_prompts_and_targets(self):
        safe, unsafe = step("Safe action", True), step("Bad action", True, True)
        for phase in ("planning", "action", "reflect"):
            safe[f"{phase}_messages"] = [{"role": "user", "content": phase + " prompt"}]
            safe[f"{phase}_response"] = phase + " safe"
            unsafe[f"{phase}_response"] = phase + " bad"
        self.write("1/run/0.json", [safe])
        self.write("1/run/1.json", [unsafe])
        dataset = self.dataset(guiowl_phases="all")
        self.assertEqual(len(dataset), 3)
        for index, phase in enumerate(("planning", "action", "reflect")):
            row = dataset[index]
            self.assertIn(phase + " prompt", row["prompt"])
            self.assertEqual(row["chosen"], phase + " safe")
            self.assertEqual(row["rejected"], phase + " bad")

    def test_multistep_responses_and_two_reflection_images(self):
        safe = [step("safe1", True), step("safe2", True)]
        unsafe = [step("bad1", True, True), step("bad2", True)]
        safe[0]["reflect_messages"][0]["content"].append(image_block())
        safe[0]["reflect_response"] = "safe reflection"
        unsafe[0]["reflect_response"] = "bad reflection"
        self.write("1/run/0.json", safe)
        self.write("1/run/1.json", unsafe)
        dataset = self.dataset(guiowl_phases=["action", "reflect"], max_images=2)
        self.assertEqual(dataset[0]["chosen"], "safe1\nsafe2")
        self.assertEqual(dataset[0]["rejected"], "bad1\nbad2")
        self.assertEqual(len(dataset[1]["images"]), 2)

    def test_single_call_models_ignore_guiowl_phase_selection(self):
        self.write("1/run/0.json", [step("safe")])
        self.write("1/run/1.json", [step("bad", attack=True)])
        self.assertEqual(self.dataset(guiowl_phases=["planning"])[0]["chosen"], "safe")
        self.assertEqual(self.dataset(data_type="ghosei")[0]["chosen"], "safe")

    def test_category_task_and_model_boundaries_and_no_attack_exclusion(self):
        self.write("owl/memo_completing/1/run/log_0_defense_-1.json", [step("safe", True)])
        self.write("owl/text_message_sending/1/run/log_1_defense_-1.json", [step("bad", True, True)])
        self.write("mai/memo_completing/1/run/log_1_defense_-1.json", [step("bad", attack=True)])
        self.write("owl/memo_completing/1_no_attack/run/log_0_defense_-1.json", [step("excluded")])
        with self.assertRaisesRegex(ValueError, "No DPO pairs"):
            self.dataset(data_type="mobilesafetybench")
        self.write("owl/memo_completing/1/run/log_1_defense_-1.json", [step("bad", True, True)])
        dataset = self.dataset(data_type="mobilesafetybench")
        self.assertEqual(len(dataset), 1)
        self.assertEqual(dataset.stats["tasks"], 3)

    @unittest.skipUnless(importlib.util.find_spec("datasets"), "datasets is not installed")
    def test_hf_dataset_has_trl_preprocessing_interface(self):
        self.write("1/run/0.json", [step("safe")])
        self.write("1/run/1.json", [step("unsafe", attack=True)])
        dataset = self.dataset().to_hf_dataset()
        self.assertEqual(set(dataset.column_names), {"prompt", "images", "chosen", "rejected"})
        row = dataset.select([0]).map(lambda value: {"chosen": value["chosen"] + "!"})[0]
        self.assertEqual(row["chosen"], "safe!")
        self.assertIsInstance(row["images"][0], Image.Image)


if __name__ == "__main__":
    unittest.main()
