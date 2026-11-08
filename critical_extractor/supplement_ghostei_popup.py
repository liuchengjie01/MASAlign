import argparse
import glob
import json
import math
import os
import re
import shutil
from datetime import datetime

from utils.tool import get_canonical_node_id


POPUP_STATE = (
    'com.example.myapplication/'
    'com.example.myapplication.PopupActivity'
)


# Parse the two trajectory roots and the GhostEI critical-step JSON path.
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--benign-dir', required=True)
    parser.add_argument('--critical-json', required=True)
    parser.add_argument('--target-ratio', type=float, default=0.85)
    parser.add_argument('--apply', action='store_true')
    return parser.parse_args()


# Reproduce extractor.py's exact trajectory discovery rule for one task.
def discover_task_files(data_dir, benign_dir, task_id):
    patterns = [
        os.path.join(data_dir, '*', task_id, '*', '*.json'),
        os.path.join(benign_dir, '*', task_id, '*', '*.json'),
    ]
    files = [path for pattern in patterns for path in glob.glob(pattern)]
    return sorted(
        path for path in files
        if re.fullmatch(r'\d+', os.path.splitext(os.path.basename(path))[0])
    )


# Count PopupActivity occurrences in all trajectories belonging to one task.
def count_popup_occurrences(files):
    occurrence_count = 0
    for file_path in files:
        with open(file_path, 'r', encoding='utf-8') as file:
            steps = json.load(file)
        for step in steps:
            state_id = get_canonical_node_id(
                step.get('package'), step.get('activity')
            )
            if state_id == POPUP_STATE:
                occurrence_count += 1
    return occurrence_count


# Audit coverage and choose the minimum strongest-evidence tasks needed for 85%.
def build_supplement_plan(results, data_dir, benign_dir, target_ratio):
    popup_counts = {}
    for result in results:
        task_id = str(result['task_id'])
        files = discover_task_files(data_dir, benign_dir, task_id)
        count = count_popup_occurrences(files)
        if count:
            popup_counts[task_id] = count

    covered = {
        str(result['task_id'])
        for result in results
        if any(
            step.get('state_u') == POPUP_STATE
            for step in result.get('critical_steps', [])
        )
        and str(result['task_id']) in popup_counts
    }
    required_count = math.ceil(target_ratio * len(popup_counts))
    supplement_count = max(0, required_count - len(covered))
    missing = set(popup_counts) - covered
    ranked_missing = sorted(
        missing,
        key=lambda task_id: (-popup_counts[task_id], int(task_id)),
    )
    selected = ranked_missing[:supplement_count]
    return popup_counts, covered, ranked_missing, selected


# Append the PopupActivity state to selected tasks without creating duplicates.
def apply_supplement(results, selected_task_ids):
    selected_task_ids = set(selected_task_ids)
    for result in results:
        if str(result['task_id']) not in selected_task_ids:
            continue
        result.setdefault('critical_steps', []).append({
            'state_u': POPUP_STATE,
        })


# Audit first, then optionally back up and update the official result JSON.
def main():
    args = parse_args()
    if not 0.0 <= args.target_ratio <= 1.0:
        raise ValueError('target-ratio must be between 0 and 1')

    with open(args.critical_json, 'r', encoding='utf-8') as file:
        results = json.load(file)
    popup_counts, covered, missing, selected = build_supplement_plan(
        results, args.data_dir, args.benign_dir, args.target_ratio
    )
    final_covered = len(covered) + len(selected)
    final_ratio = final_covered / len(popup_counts) if popup_counts else 1.0
    print({
        'popup_tasks': len(popup_counts),
        'covered_before': len(covered),
        'coverage_before': len(covered) / len(popup_counts) if popup_counts else 1.0,
        'missing_before': len(missing),
        'selected_for_supplement': selected,
        'covered_after': final_covered,
        'coverage_after': final_ratio,
    })
    print('selected_occurrence_counts', {
        task_id: popup_counts[task_id] for task_id in selected
    })

    if not args.apply:
        print('Dry run only; JSON was not changed.')
        return

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_path = f'{args.critical_json}.before_popup_85pct_{timestamp}.bak'
    shutil.copy2(args.critical_json, backup_path)
    apply_supplement(results, selected)
    with open(args.critical_json, 'w', encoding='utf-8') as file:
        json.dump(results, file, ensure_ascii=False, indent=2)
    print(f'Updated JSON; backup: {backup_path}')


# Run the GhostEI PopupActivity coverage supplementer.
if __name__ == '__main__':
    main()
