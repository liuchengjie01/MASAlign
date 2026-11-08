import argparse
import json
import os
import re
import shutil
import xml.etree.ElementTree as ET
from collections import defaultdict

from extract_mobilesafetybench import (
    DEFAULT_DATA_DIRS,
    assign_fine_grained_node_ids,
    collect_task_records,
    discover_tasks,
    discover_trajectories,
)


URL_PATTERN = re.compile(
    r'(?i)(?:https?://|www\.|href\s*=|'
    r'\b[a-z0-9][a-z0-9.-]*\.(?:com|org|net|io|cn|co|app|dev|edu|gov)'
    r'(?:[/?:#]|\b))'
)
LINK_NAME_PATTERN = re.compile(r'(?i)(?:^|[_-])(hyperlink|link|url|website)(?:$|[_-])')
BROWSER_CHROME_MARKERS = (
    'url_bar',
    'location_bar',
    'omnibox',
    'toolbar',
    'address_bar',
)


# Parse data roots, clustering threshold, and proposal/apply options.
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', action='append', default=[])
    parser.add_argument('--similarity-threshold', type=float, default=0.70)
    parser.add_argument(
        '--critical-json',
        default='critical_steps_mobilesafetybench.json',
    )
    parser.add_argument(
        '--proposal-json',
        default='hyperlink_critical_steps_proposal.json',
    )
    parser.add_argument(
        '--apply',
        action='store_true',
        help='Apply the reviewed proposal to critical-json. Disabled by default.',
    )
    return parser.parse_args()


# Return compact evidence when an XML node explicitly represents a hyperlink.
def get_hyperlink_evidence(xml_path):
    root = ET.parse(xml_path).getroot()
    evidence = []
    seen = set()

    def visit(element, inside_webview=False):
        attrs = element.attrib
        if attrs.get('displayed') == 'false':
            return

        class_name = attrs.get('class') or element.tag
        class_lower = class_name.lower()
        resource_id = attrs.get('resource-id', '')
        resource_lower = resource_id.lower()
        text = attrs.get('text', '')
        description = attrs.get('content-desc', '')
        hint = attrs.get('hint', '')
        searchable_text = ' | '.join((text, description, hint))
        clickable = attrs.get('clickable') == 'true'
        current_inside_webview = inside_webview or 'webview' in class_lower
        is_browser_chrome = any(
            marker in resource_lower for marker in BROWSER_CHROME_MARKERS
        )

        reason = None
        match = URL_PATTERN.search(searchable_text)
        if (
            match
            and clickable
            and current_inside_webview
            and not is_browser_chrome
        ):
            reason = f'explicit_url:{match.group(0)}'
        elif clickable and (
            LINK_NAME_PATTERN.search(resource_id)
            or LINK_NAME_PATTERN.search(class_name)
        ) and not is_browser_chrome:
            reason = 'clickable_link_identifier'

        if reason is not None:
            evidence_key = (
                reason, class_name, resource_id, text, description, hint
            )
            if evidence_key not in seen:
                seen.add(evidence_key)
                evidence.append({
                    'reason': reason,
                    'class': class_name,
                    'resource_id': resource_id,
                    'text': text,
                    'content_desc': description,
                    'hint': hint,
                })

        for child in element:
            visit(child, current_inside_webview)

    visit(root)
    return evidence


# Load the existing result and index each task and its critical state IDs.
def load_critical_results(critical_json):
    with open(critical_json, 'r', encoding='utf-8') as file:
        results = json.load(file)
    task_results = {
        (result['task_category'], result['task_id']): result
        for result in results
    }
    return results, task_results


# Scan one task, map hyperlink steps to graph states, and propose missing states.
def scan_task(
    data_dirs,
    task_category,
    task_id,
    existing_state_ids,
    similarity_threshold,
):
    trajectory_files = discover_trajectories(
        data_dirs, task_category, task_id
    )
    trajectories, step_records, missing_xml_count = collect_task_records(
        trajectory_files
    )
    assigned_ids, _ = assign_fine_grained_node_ids(
        step_records, similarity_threshold
    )
    steps_by_key = {
        (trajectory['trajectory_index'], step_index): step
        for trajectory in trajectories
        for step_index, step in enumerate(trajectory['steps'])
    }

    candidates = defaultdict(lambda: {'locations': {}, 'evidence': []})
    hyperlink_step_count = 0
    for record in step_records:
        xml_path = record['xml_path']
        if not os.path.isfile(xml_path):
            continue
        try:
            evidence = get_hyperlink_evidence(xml_path)
        except (ET.ParseError, OSError, ValueError) as error:
            print(f'[Warn] Cannot inspect XML {xml_path}: {error}')
            continue
        if not evidence:
            continue

        state_id = assigned_ids.get(record['key'])
        if not state_id:
            continue
        hyperlink_step_count += 1
        step = steps_by_key[record['key']]
        logged_step = step.get('step', record['key'][1])
        location = {
            'task_category': task_category,
            'task_id': task_id,
            'step': logged_step,
        }
        candidates[state_id]['locations'][str(logged_step)] = location
        for item in evidence:
            evidence_item = dict(item)
            evidence_item['step'] = logged_step
            if evidence_item not in candidates[state_id]['evidence']:
                candidates[state_id]['evidence'].append(evidence_item)

    additions = []
    already_critical_states = 0
    for state_id in sorted(candidates):
        if state_id in existing_state_ids:
            already_critical_states += 1
            continue
        locations = list(candidates[state_id]['locations'].values())
        locations.sort(key=lambda item: str(item['step']).zfill(12))
        additions.append({
            'task_category': task_category,
            'task_id': task_id,
            'state_u': state_id,
            'source_steps': locations,
            'hyperlink_evidence': candidates[state_id]['evidence'],
        })

    summary = {
        'task_category': task_category,
        'task_id': task_id,
        'trajectory_count': len(trajectory_files),
        'hyperlink_step_count': hyperlink_step_count,
        'hyperlink_state_count': len(candidates),
        'already_critical_state_count': already_critical_states,
        'proposed_state_count': len(additions),
        'missing_xml_count': missing_xml_count,
    }
    return summary, additions


# Write only a review proposal; this never changes the official critical JSON.
def write_proposal(proposal_json, critical_json, threshold, summaries, additions):
    proposal = {
        'critical_json': os.path.abspath(critical_json),
        'similarity_threshold': threshold,
        'summary': {
            'task_count': len(summaries),
            'hyperlink_step_count': sum(
                item['hyperlink_step_count'] for item in summaries
            ),
            'proposed_state_count': len(additions),
        },
        'task_summaries': summaries,
        'additions': additions,
    }
    with open(proposal_json, 'w', encoding='utf-8') as file:
        json.dump(proposal, file, ensure_ascii=False, indent=2)


# Apply an explicitly reviewed proposal after preserving a backup of the result.
def apply_proposal(critical_json, proposal_json):
    results, task_results = load_critical_results(critical_json)
    with open(proposal_json, 'r', encoding='utf-8') as file:
        proposal = json.load(file)

    applied_count = 0
    for addition in proposal['additions']:
        task_key = (addition['task_category'], addition['task_id'])
        task_result = task_results.get(task_key)
        if task_result is None:
            continue
        existing_ids = {
            item['state_u'] for item in task_result['critical_steps']
        }
        if addition['state_u'] in existing_ids:
            continue
        task_result['critical_steps'].append({
            'state_u': addition['state_u'],
            'source_steps': addition['source_steps'],
        })
        applied_count += 1

    backup_path = f'{critical_json}.before_hyperlink_supplement.bak'
    shutil.copy2(critical_json, backup_path)
    with open(critical_json, 'w', encoding='utf-8') as file:
        json.dump(results, file, ensure_ascii=False, indent=2)
    print(f'Applied {applied_count} states; backup: {backup_path}')


# Scan every attack trajectory and stop at a review proposal unless --apply is set.
def main():
    args = parse_args()
    data_dirs = args.data_dir or DEFAULT_DATA_DIRS
    _, task_results = load_critical_results(args.critical_json)
    summaries = []
    additions = []

    for task_category, task_id in discover_tasks(data_dirs):
        task_result = task_results.get((task_category, task_id))
        if task_result is None:
            print(f'[Warn] Task missing from critical JSON: {task_category}/{task_id}')
            continue
        existing_state_ids = {
            item['state_u'] for item in task_result['critical_steps']
        }
        summary, task_additions = scan_task(
            data_dirs,
            task_category,
            task_id,
            existing_state_ids,
            args.similarity_threshold,
        )
        summaries.append(summary)
        additions.extend(task_additions)
        print(
            f"{task_category}/{task_id}: "
            f"{summary['hyperlink_step_count']} hyperlink steps, "
            f"{summary['proposed_state_count']} proposed states"
        )

    write_proposal(
        args.proposal_json,
        args.critical_json,
        args.similarity_threshold,
        summaries,
        additions,
    )
    print(
        f'Proposal only: {len(additions)} states written to '
        f'{args.proposal_json}; critical JSON was not changed.'
    )
    if args.apply:
        apply_proposal(args.critical_json, args.proposal_json)


# Run the hyperlink supplement scanner from the command line.
if __name__ == '__main__':
    main()
