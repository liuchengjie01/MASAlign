"""Task discovery, LLM phase parsing and trajectory-level preference pairs."""

import base64
from collections import Counter, defaultdict
import copy
from dataclasses import dataclass
import io
import json
from pathlib import Path

from PIL import Image


GUIOWL_PHASES = ("planning", "action", "reflect")


def discover_tasks(root, data_type):
    """Group by task identity, keeping different model directories separate."""
    root = Path(root)
    data_type = {"ghosei": "ghostei"}.get(data_type, data_type)
    if data_type not in ("ghostei", "mobilesafetybench"):
        raise ValueError(f"Unsupported data_type: {data_type}")
    tasks = defaultdict(list)
    pattern = "*.json" if data_type == "ghostei" else "log_*_defense_-1.json"
    for path in sorted(root.rglob(pattern)):
        task = path.parent.parent
        if data_type == "ghostei":
            if not path.stem.isdigit() or not task.name.isdigit():
                continue
            key = (str(task.parent), task.name)
        else:
            if task.name.endswith("_no_attack"):
                continue
            category = task.parent
            key = (str(category.parent), category.name, task.name)
        tasks[key].append(path)
    if not tasks:
        raise FileNotFoundError(f"No {data_type} trajectory files found in {root}")
    return tasks


def read_trajectory(path):
    with Path(path).open(encoding="utf-8") as file:
        steps = json.load(file)
    if not isinstance(steps, list):
        raise ValueError(f"Expected a list of trajectory steps: {path}")
    return steps


def phase_fields(step, phase):
    # GUI-Owl has three calls per step; single-call logs represent action.
    if any(f"{name}_response" in step for name in GUIOWL_PHASES):
        return f"{phase}_messages", f"{phase}_response"
    return ("messages", "response") if phase == "action" else (None, None)


def phase_response(step, phase):
    _, response_key = phase_fields(step, phase)
    if response_key is None:
        return ""
    response = step.get(response_key)
    if isinstance(response, str):
        return response.strip()
    # Legacy compact logs may only contain action; an explicit empty response
    # is a skipped terminal call and must never fall back to action metadata.
    if phase == "action" and response_key == "response" and response_key not in step:
        action = step.get("action")
        if isinstance(action, str):
            return action.strip()
        if isinstance(action, (dict, list)) and action:
            return json.dumps(action, ensure_ascii=False)
    return ""


def phase_prompt(step, phase):
    messages_key, _ = phase_fields(step, phase)
    messages = step.get(messages_key)
    if not isinstance(messages, list) or not messages or not all(
        isinstance(message, dict) and message.get("role") and "content" in message
        for message in messages
    ):
        return None
    messages = copy.deepcopy(messages)
    while messages and messages[-1]["role"] == "assistant":
        messages.pop()
    return messages or None


@dataclass
class PhaseTrajectory:
    path: Path
    phase: str
    first_step: int
    response: str
    has_prompt: bool


def summarize_phases(path, steps, phases):
    summaries = {}
    for phase in phases:
        responses = [(index, phase_response(step, phase)) for index, step in enumerate(steps)
                     if isinstance(step, dict)]
        responses = [(index, text) for index, text in responses if text]
        if responses:
            first_step = responses[0][0]
            summaries[phase] = PhaseTrajectory(
                Path(path), phase, first_step, "\n".join(text for _, text in responses),
                phase_prompt(steps[first_step], phase) is not None,
            )
    return summaries


def render_prompt(messages, processor, max_images, system_prompt):
    """Keep the newest screenshots and render with this model's chat template."""
    messages = copy.deepcopy(messages)
    if system_prompt and messages[0]["role"] != "system":
        messages.insert(0, {"role": "system", "content": system_prompt})
    blocks = [block for message in messages if isinstance(message["content"], list)
              for block in message["content"] if block.get("type") in ("image", "image_url")]
    for block in blocks[:-max_images]:
        block.clear()
        block.update(type="text", text="[Screenshot omitted]")
    images = []
    for block in blocks[-max_images:]:
        if block["type"] == "image_url":
            header, encoded = block["image_url"]["url"].split(",", 1)
            if not header.startswith("data:image/") or ";base64" not in header:
                raise ValueError("Expected a base64 image data URI")
            with Image.open(io.BytesIO(base64.b64decode(encoded))) as image:
                images.append(image.convert("RGB"))
        else:
            images.append(block["image"].convert("RGB"))
        block.clear()
        block["type"] = "image"
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt, images


class TrajectoryDPODataset:
    """One chosen/rejected trajectory pair per task and selected LLM phase.

    Like SFT, each phase keeps its own conditioning prompt. Unlike SFT, the
    target is a preferred/rejected response pair, rather than one safe response.
    Only response strings and source indices stay in memory, not screenshots.
    """

    def __init__(self, data_path, processor, data_type="ghostei", guiowl_phases="all",
                 max_images=1, system_prompt=""):
        self.processor = processor
        self.max_images = int(max_images)
        if self.max_images < 1:
            raise ValueError("max_images must be at least 1")
        self.system_prompt = system_prompt
        phases = GUIOWL_PHASES if guiowl_phases == "all" else tuple(guiowl_phases)
        if not phases or len(set(phases)) != len(phases) or set(phases) - set(GUIOWL_PHASES):
            raise ValueError(f"guiowl_phases must be 'all' or a list from {GUIOWL_PHASES}")
        # Single-call models always train their sole response, independently
        # of which additional GUI-Owl phases are selected.
        self.phases = phases
        self.samples = []
        self.stats = Counter()
        for files in discover_tasks(data_path, data_type).values():
            self._add_task(files)
        if not self.samples:
            raise ValueError(f"No DPO pairs built: {dict(self.stats)}; each task needs "
                             "an unmarked chosen and an is_attack=true rejected trajectory")

    def _add_task(self, files):
        safe, attacked = [], []
        self.stats["tasks"] += 1
        for path in files:
            steps = read_trajectory(path)
            is_attack = any(isinstance(step, dict) and step.get("is_attack", False) for step in steps)
            is_guiowl = any(isinstance(step, dict) and any(
                f"{phase}_response" in step for phase in GUIOWL_PHASES) for step in steps)
            phases = self.phases if is_guiowl else ("action",)
            summary = summarize_phases(path, steps, phases)
            (attacked if is_attack else safe).append(summary)
        if not safe or not attacked:
            self.stats["attack_only_tasks" if not safe else "safe_only_tasks"] += 1
            return
        for phase in GUIOWL_PHASES:
            chosen = [item[phase] for item in safe if phase in item and item[phase].has_prompt]
            rejected = [item[phase] for item in attacked if phase in item]
            pair = next(((good, bad.response) for good in chosen for bad in rejected
                         if good.response != bad.response), None)
            if pair:
                self.samples.append(pair)
                self.stats[f"{phase}_pairs"] += 1
            elif any(phase in item for item in safe + attacked):
                self.stats["invalid_phases"] += 1

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        chosen, rejected = self.samples[index]
        steps = read_trajectory(chosen.path)
        messages = phase_prompt(steps[chosen.first_step], chosen.phase)
        if messages is None:
            raise ValueError(f"Indexed prompt became invalid: {chosen.path}, phase={chosen.phase}")
        prompt, images = render_prompt(messages, self.processor, self.max_images, self.system_prompt)
        return {"prompt": prompt, "images": images, "chosen": chosen.response, "rejected": rejected}

    def _iter_rows(self):
        for index in range(len(self)):
            yield self[index]

    def to_hf_dataset(self):
        from datasets import Dataset, Features, Image as HFImage, Sequence, Value

        features = Features({"prompt": Value("string"), "images": Sequence(HFImage()),
                             "chosen": Value("string"), "rejected": Value("string")})
        return Dataset.from_generator(self._iter_rows, features=features, keep_in_memory=False)
