
from glob import glob
import json
import os

from utils.prs_graph import MDPGraph


def load_dataset(base_path, dataset_name):

    if dataset_name == 'GhostEI':
        p = f'{base_path}{os.sep}*{os.sep}*{os.sep}*{os.sep}log.json'
        files = glob.glob(p, recursive=True)
        return files
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

def extract_from_json(json_file, g: MDPGraph):
    with open(json_file, 'r', encoding='utf-8') as f:
        steps = json.load(f)
    for idx, step in enumerate(steps):
        state_id = step.get('package') + '_' + step.get('activity')
        observation = step.get('observation', [])
        if idx == len(steps) - 1:
            is_finished = True
            is_attacked = step.get('is_attacked', False)
        else:
            is_finished = False
            is_attacked = False
        g.add_node(state_id, observation, is_finished=is_finished, is_attacked=is_attacked)
        