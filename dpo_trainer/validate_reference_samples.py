"""Small CPU-only data construction check using explicitly supplied real logs."""

import argparse
import copy
import json
from pathlib import Path
import tempfile

from data import GUIOWL_PHASES, TrajectoryDPODataset, phase_response


class ValidationProcessor:
    """Render without model weights; this does not test a model chat template."""

    tokenizer = None

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        assert tokenize is False and add_generation_prompt is True
        return json.dumps(messages, ensure_ascii=False) + "\nassistant:"


def read_trajectory(path):
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def validate_pair(label, rejected_path, chosen_path=None):
    rejected = read_trajectory(rejected_path)
    assert any(step.get("is_attack", False) for step in rejected)
    phases = GUIOWL_PHASES if label == "guiowl" else ("action",)
    if chosen_path is None:
        # The supplied GUI-Owl run has no unmarked trajectory. Keep its real
        # initial prompt but explicitly create a synthetic chosen completion.
        fixture = {}
        for phase in phases:
            initial = next(step for step in rejected if phase_response(step, phase))
            fixture[f"{phase}_messages"] = copy.deepcopy(initial[f"{phase}_messages"])
            fixture[f"{phase}_response"] = f"Synthetic chosen {phase} for validation."
        chosen = [fixture]
        chosen_source = "synthetic validation fixture; not a real safe trajectory"
    else:
        chosen = read_trajectory(chosen_path)
        assert not any(step.get("is_attack", False) for step in chosen)
        chosen_source = str(chosen_path)

    with tempfile.TemporaryDirectory(prefix="dpo-reference-") as directory:
        run = Path(directory) / label / "1" / "run"
        run.mkdir(parents=True)
        for name, trajectory in (("0.json", rejected), ("1.json", chosen)):
            (run / name).write_text(json.dumps(trajectory), encoding="utf-8")
        dataset = TrajectoryDPODataset(directory, ValidationProcessor())
        assert len(dataset) == len(phases)
        hf_dataset = dataset.to_hf_dataset()
        assert set(hf_dataset.column_names) == {"prompt", "images", "chosen", "rejected"}
        phase_reports = []
        for index, phase in enumerate(phases):
            row = dataset[index]
            expected_rejected = "\n".join(filter(None, (phase_response(step, phase) for step in rejected)))
            expected_chosen = "\n".join(filter(None, (phase_response(step, phase) for step in chosen)))
            assert row["rejected"] == expected_rejected
            assert row["chosen"] == expected_chosen and expected_chosen != expected_rejected
            assert row["images"] and all(image.mode == "RGB" for image in row["images"])
            hf_row = hf_dataset.select([index]).map(lambda value: {"chosen": value["chosen"]})[0]
            assert hf_row["rejected"] == expected_rejected
            assert hf_row["images"][0].size == row["images"][0].size
            phase_reports.append({
                "phase": phase,
                "response_steps": sum(bool(phase_response(step, phase)) for step in rejected),
                "chosen_characters": len(row["chosen"]),
                "rejected_characters": len(row["rejected"]),
                "image_sizes": [list(image.size) for image in row["images"]],
            })
        return {
            "format": label,
            "rejected_source": str(rejected_path),
            "chosen_source": chosen_source,
            "rejected_steps": len(rejected),
            "pairs": len(dataset),
            "phases": phase_reports,
            "hf_columns": hf_dataset.column_names,
            "result": "passed",
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--guiowl", type=Path, required=True)
    parser.add_argument("--maiui", type=Path, required=True)
    parser.add_argument("--guiowl-chosen", type=Path)
    parser.add_argument("--maiui-chosen", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = {
        "scope": "CPU data construction and Hugging Face serialization; no model training",
        "checks": [
            validate_pair("guiowl", args.guiowl, args.guiowl_chosen),
            validate_pair("maiui", args.maiui, args.maiui_chosen),
        ],
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.report:
        args.report.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
