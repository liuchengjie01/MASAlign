import argparse
import glob
import hashlib
import json
import os
import re
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict

from utils.prs_graph import MDPGraph
from utils.tool import get_canonical_node_id, print_graph, save_mermaid


DEFAULT_DATA_DIRS = [
    r'E:\luis\mobilesafetybench\logs',
]

TASK_CATEGORIES = [
    'memo_app_download',
    'memo_completing',
    'social_media_download',
    'social_media_posting_direct',
    'text_message_forwarding',
    'text_message_sending',
]

IGNORED_PACKAGE_MARKERS = (
    'inputmethod',
    'com.android.systemui',
)

URL_PATTERN = re.compile(
    r'(?i)(?:https?://|www\.|href\s*=|'
    r'\b[a-z0-9][a-z0-9.-]*\.(?:com|org|net|io|cn|co|app|dev|edu|gov)'
    r'(?:[/?:#]|\b))'
)
LINK_NAME_PATTERN = re.compile(
    r'(?i)(?:^|[_-])(hyperlink|link|url|website)(?:$|[_-])'
)
BROWSER_CHROME_MARKERS = (
    'url_bar',
    'location_bar',
    'omnibox',
    'toolbar',
    'address_bar',
)


# Parse input locations, similarity threshold, and output locations.
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data-dir',
        action='append',
        default=[],
        help='Trajectory root. May be supplied more than once.',
    )
    parser.add_argument(
        '--mmd-dir',
        default=f'critical_extractor{os.sep}mmds_mobilesafetybench',
        help='Directory used to save Mermaid graphs.',
    )
    parser.add_argument(
        '--output-json',
        default=f'critical_extractor{os.sep}critical_steps_mobilesafetybench.json',
        help='Path used to save the final analysis result.',
    )
    parser.add_argument(
        '--cluster-audit-json',
        default='critical_extractor{os.sep}mobilesafetybench_ui_clusters.json',
        help='Path used to save page-clustering details for review.',
    )
    parser.add_argument(
        '--similarity-threshold',
        type=float,
        default=0.70,
        help='Minimum complete-link XML similarity for one graph node.',
    )
    parser.add_argument(
        '--task',
        action='append',
        default=[],
        help='Only run category/task_id. May be supplied more than once.',
    )
    return parser.parse_args()


# Detect transitions caused by pressing the system HOME button.
def is_home_action(action):
    if action is None:
        return False
    action_text = str(action)
    return 'system_button' in action_text and 'home' in action_text.lower()


# Normalize text while removing values that commonly change between steps.
def normalize_text(value):
    value = re.sub(r'\s+', ' ', str(value or '')).strip().lower()
    if not value:
        return ''
    value = re.sub(r'https?://\S+|www\.\S+', '<url>', value)
    value = re.sub(r'\b[\w.+-]+@[\w.-]+\.[a-z]{2,}\b', '<email>', value)
    value = re.sub(r'\b[0-9a-f]{8,}\b', '<id>', value)
    value = re.sub(r'\d+', '<num>', value)
    return value[:120]


# Reduce a resource ID to its stable, app-independent suffix.
def normalize_resource_id(value):
    value = str(value or '').strip()
    if not value:
        return ''
    return value.rsplit('/', 1)[-1]


# Convert exact screen coordinates into a small, resolution-independent grid.
def coarse_layout_bucket(bounds, screen_width, screen_height):
    numbers = [int(number) for number in re.findall(r'-?\d+', bounds or '')]
    if len(numbers) != 4 or screen_width <= 0 or screen_height <= 0:
        return ''
    left, top, right, bottom = numbers
    center_x = max(0.0, min(1.0, ((left + right) / 2) / screen_width))
    center_y = max(0.0, min(1.0, ((top + bottom) / 2) / screen_height))
    column = min(2, int(center_x * 3))
    row = min(3, int(center_y * 4))
    return f'{column},{row}'


# Classify a UI element by function instead of its current value.
def get_control_role(class_name, attrs):
    class_name = class_name.lower()
    if attrs.get('password') == 'true':
        return 'password_input'
    if 'edittext' in class_name or attrs.get('text-entry-key') == 'true':
        return 'input'
    if 'button' in class_name or attrs.get('clickable') == 'true':
        return 'action'
    if attrs.get('scrollable') == 'true':
        return 'scroll'
    if attrs.get('checkable') == 'true':
        return 'choice'
    if 'image' in class_name:
        return 'image'
    if 'text' in class_name:
        return 'label'
    return 'container'


# Detect potentially inducive hyperlinks in non-Chrome accessibility XML.
def get_inducement_hyperlink_evidence(xml_path, step_package):
    if 'com.android.chrome' in str(step_package or '').lower():
        return []

    root = ET.parse(xml_path).getroot()
    evidence = []
    seen = set()

    # Keep only explicit clickable links in page content, not browser chrome.
    def visit(element, inside_webview=False):
        attrs = element.attrib
        if attrs.get('displayed') == 'false':
            return

        class_name = attrs.get('class') or element.tag
        class_lower = class_name.lower()
        node_package = attrs.get('package', '').lower()
        if 'com.android.chrome' in node_package:
            return

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
        url_match = URL_PATTERN.search(searchable_text)
        if (
            url_match
            and clickable
            and current_inside_webview
            and not is_browser_chrome
        ):
            reason = 'explicit_url'
        elif (
            clickable
            and not is_browser_chrome
            and (
                LINK_NAME_PATTERN.search(resource_id)
                or LINK_NAME_PATTERN.search(class_name)
            )
        ):
            reason = 'clickable_link_identifier'

        if reason:
            evidence_key = (
                reason, class_name, resource_id, text, description, hint
            )
            if evidence_key not in seen:
                seen.add(evidence_key)
                evidence.append({
                    'reason': reason,
                    'text': text,
                    'content_desc': description,
                    'hint': hint,
                })

        for child in element:
            visit(child, current_inside_webview)

    visit(root)
    return evidence


# Add a weighted token while limiting repeated list items from dominating.
def add_feature(features, token, weight, repeat_limit=3):
    if not token:
        return
    features[token] = min(features[token] + weight, weight * repeat_limit)


# Extract a value-insensitive functional fingerprint from one accessibility XML.
def extract_ui_signature(xml_path):
    root = ET.parse(xml_path).getroot()
    screen_width = int(root.attrib.get('width') or 0)
    screen_height = int(root.attrib.get('height') or 0)
    structure = Counter()
    controls = Counter()
    layout = Counter()
    meaningful_nodes = 0

    # Walk the XML tree while retaining the nearest stable parent anchor.
    def visit(element, depth=0, parent_anchor='root'):
        nonlocal meaningful_nodes
        attrs = element.attrib
        node_package = attrs.get('package', '')
        package_lower = node_package.lower()
        if any(marker in package_lower for marker in IGNORED_PACKAGE_MARKERS):
            return
        if attrs.get('displayed') == 'false':
            return

        class_name = attrs.get('class') or element.tag
        class_name = class_name.rsplit('.', 1)[-1].lower()
        resource_id = normalize_resource_id(attrs.get('resource-id'))
        role = get_control_role(class_name, attrs)
        anchor = resource_id or f'{role}:{class_name}'
        depth_bucket = min(depth, 10)

        add_feature(structure, f'class:{class_name}', 1)
        add_feature(structure, f'role:{role}', 2)
        add_feature(
            structure,
            f'node:{parent_anchor}>{anchor}:{class_name}:d{depth_bucket}',
            3,
        )
        if resource_id:
            meaningful_nodes += 1
            add_feature(structure, f'id:{resource_id}', 5)
            add_feature(controls, f'control:{role}:{resource_id}', 5)
        elif role != 'container':
            meaningful_nodes += 1
            add_feature(controls, f'control:{role}:{class_name}', 2)

        # Input values, focus, selection, and keyboard state are intentionally
        # excluded so that typing does not create a new functional state.
        if role not in {'input', 'password_input'}:
            text = normalize_text(attrs.get('text'))
            description = normalize_text(attrs.get('content-desc'))
            if text:
                add_feature(controls, f'text:{role}:{text}', 1)
            if description:
                add_feature(controls, f'desc:{role}:{description}', 2)

        position = coarse_layout_bucket(
            attrs.get('bounds'), screen_width, screen_height
        )
        if position and (resource_id or role != 'container'):
            add_feature(layout, f'layout:{anchor}:{position}', 1)

        next_parent_anchor = resource_id or parent_anchor
        for child in element:
            visit(child, depth + 1, next_parent_anchor)

    visit(root)
    signature = {
        'structure': structure,
        'controls': controls,
        'layout': layout,
        'meaningful_nodes': meaningful_nodes,
    }
    signature['fingerprint'] = signature_fingerprint(signature)
    return signature


# Produce a stable hash for exact-signature deduplication and node naming.
def signature_fingerprint(signature):
    serializable = {
        component: sorted(signature[component].items())
        for component in ('structure', 'controls', 'layout')
    }
    payload = json.dumps(
        serializable, ensure_ascii=False, sort_keys=True, separators=(',', ':')
    )
    return hashlib.sha1(payload.encode('utf-8')).hexdigest()


# Compute weighted multiset Jaccard similarity for one feature component.
def weighted_jaccard(left, right):
    keys = left.keys() | right.keys()
    if not keys:
        return None
    intersection = sum(min(left[key], right[key]) for key in keys)
    union = sum(max(left[key], right[key]) for key in keys)
    return intersection / union if union else 1.0


# Combine structure, functional controls, and coarse layout similarities.
def compute_ui_similarity(left, right):
    component_weights = {
        'structure': 0.60,
        'controls': 0.30,
        'layout': 0.10,
    }
    weighted_score = 0.0
    used_weight = 0.0
    for component, weight in component_weights.items():
        score = weighted_jaccard(left[component], right[component])
        if score is None:
            continue
        weighted_score += score * weight
        used_weight += weight
    return weighted_score / used_weight if used_weight else 0.0


# Return the minimum pairwise similarity between two complete-link clusters.
def complete_link_similarity(left_cluster, right_cluster, similarity_cache):
    return min(
        similarity_cache[tuple(sorted((left, right)))]
        for left in left_cluster
        for right in right_cluster
    )


# Cluster one package/activity group without allowing similarity chaining.
def cluster_signature_group(signatures, threshold):
    signatures_by_hash = {
        signature['fingerprint']: signature for signature in signatures
    }
    fingerprints = sorted(signatures_by_hash)
    if not fingerprints:
        return [], signatures_by_hash, {}

    similarity_cache = {}
    for left_index, left in enumerate(fingerprints):
        similarity_cache[(left, left)] = 1.0
        for right in fingerprints[left_index + 1:]:
            similarity_cache[(left, right)] = compute_ui_similarity(
                signatures_by_hash[left], signatures_by_hash[right]
            )

    clusters = [{fingerprint} for fingerprint in fingerprints]
    while True:
        best_pair = None
        best_score = threshold
        for left_index in range(len(clusters)):
            for right_index in range(left_index + 1, len(clusters)):
                score = complete_link_similarity(
                    clusters[left_index],
                    clusters[right_index],
                    similarity_cache,
                )
                if score >= best_score:
                    candidate_key = (
                        score,
                        sorted(clusters[left_index]),
                        sorted(clusters[right_index]),
                    )
                    if best_pair is None or candidate_key > best_pair[0]:
                        best_pair = (candidate_key, left_index, right_index)
                        best_score = score
        if best_pair is None:
            break
        _, left_index, right_index = best_pair
        clusters[left_index] = clusters[left_index] | clusters[right_index]
        del clusters[right_index]

    return clusters, signatures_by_hash, similarity_cache


# Choose the cluster member with the highest average similarity as its medoid.
def choose_cluster_medoid(cluster, similarity_cache):
    candidates = []
    for fingerprint in sorted(cluster):
        scores = [
            similarity_cache[tuple(sorted((fingerprint, other)))]
            for other in cluster
        ]
        candidates.append((sum(scores) / len(scores), fingerprint))
    return max(candidates)[1]


# Assign deterministic fine-grained node IDs to every collected valid step.
def assign_fine_grained_node_ids(step_records, threshold=0.80):
    records_by_base_id = defaultdict(list)
    for record in step_records:
        records_by_base_id[record['base_id']].append(record)

    assigned_ids = {}
    audit_entries = []
    for base_id in sorted(records_by_base_id):
        records = records_by_base_id[base_id]
        valid_records = [record for record in records if record['signature']]
        missing_records = [record for record in records if not record['signature']]

        clusters, _, similarity_cache = cluster_signature_group(
            [record['signature'] for record in valid_records], threshold
        )
        for cluster in clusters:
            medoid = choose_cluster_medoid(cluster, similarity_cache)
            node_id = f'{base_id}::ui_{medoid[:10]}'
            members = [
                record for record in valid_records
                if record['signature']['fingerprint'] in cluster
            ]
            for record in members:
                assigned_ids[record['key']] = node_id

            minimum_similarity = min(
                (
                    similarity_cache[tuple(sorted((left, right)))]
                    for left in cluster
                    for right in cluster
                ),
                default=1.0,
            )
            audit_entries.append({
                'base_id': base_id,
                'node_id': node_id,
                'representative_fingerprint': medoid,
                'minimum_internal_similarity': minimum_similarity,
                'member_count': len(members),
                'members': [record['xml_path'] for record in members],
            })

        if missing_records:
            node_id = f'{base_id}::ui_xml_missing'
            for record in missing_records:
                assigned_ids[record['key']] = node_id
            audit_entries.append({
                'base_id': base_id,
                'node_id': node_id,
                'representative_fingerprint': None,
                'minimum_internal_similarity': None,
                'member_count': len(missing_records),
                'members': [record['xml_path'] for record in missing_records],
            })

    return assigned_ids, audit_entries


# Resolve step_XX.xml beside its trajectory JSON using the logged step number.
def get_step_xml_path(trajectory_file, step, step_index):
    step_number = step.get('step', step_index)
    try:
        step_number = int(step_number)
    except (TypeError, ValueError):
        step_number = step_index
    return os.path.join(
        os.path.dirname(trajectory_file), f'step_{step_number:02d}.xml'
    )


# Discover all attack-only trajectory files for one category/task pair.
def discover_trajectories(data_dirs, task_category, task_id):
    trajectories = []
    for data_dir in data_dirs:
        pattern = os.path.join(
            data_dir,
            '*',
            task_category,
            task_id,
            '*',
            'log_*_-1.json',
        )
        trajectories.extend(
            file_path
            for file_path in glob.glob(pattern)
            if not file_path.endswith('_detailed.json')
        )
    return sorted(set(trajectories))


# Discover only category/task combinations that physically exist on disk.
def discover_tasks(data_dirs):
    tasks = set()
    for data_dir in data_dirs:
        for task_category in TASK_CATEGORIES:
            pattern = os.path.join(data_dir, '*', task_category, '*')
            for task_path in glob.glob(pattern):
                task_id = os.path.basename(os.path.normpath(task_path))
                if os.path.isdir(task_path) and not task_id.endswith('_no_attack'):
                    tasks.add((task_category, task_id))
    return sorted(tasks)


# Load all trajectories first and extract XML signatures before graph creation.
def collect_task_records(trajectory_files):
    trajectories = []
    step_records = []
    missing_xml_count = 0

    for trajectory_index, trajectory_file in enumerate(trajectory_files):
        with open(trajectory_file, 'r', encoding='utf-8') as file:
            steps = json.load(file)
        if not isinstance(steps, list) or not steps:
            continue

        trajectory = {
            'trajectory_index': trajectory_index,
            'file_path': trajectory_file,
            'steps': steps,
        }
        trajectories.append(trajectory)

        for step_index, step in enumerate(steps):
            base_id = get_canonical_node_id(
                step.get('package'), step.get('activity')
            )
            if base_id == 'UNKNOWN_NODE':
                continue

            xml_path = get_step_xml_path(trajectory_file, step, step_index)
            signature = None
            hyperlink_evidence = []
            if os.path.isfile(xml_path):
                try:
                    signature = extract_ui_signature(xml_path)
                    hyperlink_evidence = get_inducement_hyperlink_evidence(
                        xml_path, step.get('package')
                    )
                except (ET.ParseError, OSError, ValueError) as error:
                    print(f'[Warn] Cannot parse XML {xml_path}: {error}')
            if signature is None:
                missing_xml_count += 1

            step_records.append({
                'key': (trajectory_index, step_index),
                'base_id': base_id,
                'xml_path': xml_path,
                'signature': signature,
                'hyperlink_evidence': hyperlink_evidence,
            })

    return trajectories, step_records, missing_xml_count


# Build an MDP graph after all step-to-node assignments are fixed.
def build_task_graph(trajectories, assigned_ids):
    graph = MDPGraph()
    for trajectory in trajectories:
        trajectory_index = trajectory['trajectory_index']
        steps = trajectory['steps']
        state_ids = []

        for step_index, step in enumerate(steps):
            key = (trajectory_index, step_index)
            state_id = assigned_ids.get(key, 'UNKNOWN_NODE')
            state_ids.append(state_id)
            if state_id == 'UNKNOWN_NODE':
                continue
            graph.add_node(
                state_id,
                '',
                is_finished=(step_index == len(steps) - 1),
                is_attacked=step.get('is_attack', False),
            )

        for step_index, (src_state_id, dest_state_id) in enumerate(
            zip(state_ids, state_ids[1:])
        ):
            if is_home_action(steps[step_index].get('action')):
                continue
            if (
                src_state_id == dest_state_id
                or src_state_id == 'UNKNOWN_NODE'
                or dest_state_id == 'UNKNOWN_NODE'
            ):
                continue
            graph.add_edge(src_state_id, dest_state_id, 1)

    return graph


# Make a trajectory path relative to the configured logs root.
def get_relative_trajectory_path(trajectory_file, data_dirs):
    absolute_file = os.path.abspath(trajectory_file)
    for data_dir in data_dirs:
        relative_path = os.path.relpath(absolute_file, os.path.abspath(data_dir))
        if relative_path != '..' and not relative_path.startswith(f'..{os.sep}'):
            return relative_path.replace('/', '\\')
    raise ValueError(f'Trajectory is outside configured data dirs: {trajectory_file}')


# Index every graph state by its exact relative JSON file and logged step number.
def build_state_source_index(
    trajectories, assigned_ids, data_dirs, included_step_keys=None
):
    source_index = defaultdict(list)
    seen_locations = defaultdict(set)
    for trajectory in trajectories:
        trajectory_index = trajectory['trajectory_index']
        relative_file = get_relative_trajectory_path(
            trajectory['file_path'], data_dirs
        )
        for step_index, step in enumerate(trajectory['steps']):
            step_key = (trajectory_index, step_index)
            if included_step_keys is not None and step_key not in included_step_keys:
                continue
            state_id = assigned_ids.get(step_key, 'UNKNOWN_NODE')
            if state_id == 'UNKNOWN_NODE':
                continue
            logged_step = step.get('step', step_index)
            location_key = (relative_file, str(logged_step))
            if location_key in seen_locations[state_id]:
                continue
            seen_locations[state_id].add(location_key)
            source_index[state_id].append({
                'file': relative_file,
                'step': logged_step,
            })
    for source_steps in source_index.values():
        source_steps.sort(key=lambda item: (item['file'], str(item['step']).zfill(12)))
    return source_index


# Merge graph-derived and hyperlink-derived critical states with exact sources.
def merge_critical_steps(
    graph_critical_steps,
    hyperlink_state_ids,
    source_index,
    hyperlink_source_index,
):
    graph_state_ids = [item['state_u'] for item in graph_critical_steps]
    ordered_state_ids = graph_state_ids + sorted(
        hyperlink_state_ids - set(graph_state_ids)
    )
    critical_steps = []
    for state_id in ordered_state_ids:
        reasons = []
        if state_id in graph_state_ids:
            reasons.append('graph_branch')
        if state_id in hyperlink_state_ids:
            reasons.append('inducement_hyperlink')
        relevant_sources = (
            hyperlink_source_index.get(state_id, [])
            if state_id in hyperlink_state_ids
            else source_index.get(state_id, [])
        )
        critical_steps.append({
            'state_u': state_id,
            'critical_reasons': reasons,
            'source_steps': relevant_sources,
        })
    return critical_steps


# Parse repeated category/task_id selectors into exact task tuples.
def parse_task_selectors(task_selectors):
    selected_tasks = set()
    for selector in task_selectors:
        parts = selector.replace('\\', '/').split('/', 1)
        if len(parts) != 2 or not all(parts):
            raise ValueError(f'Invalid --task value: {selector}')
        selected_tasks.add(tuple(parts))
    return selected_tasks


# Run two-phase XML clustering and graph analysis for MobileSafetyBench.
def main_mobile_safety_bench(target_dir=None):
    args = parse_args()
    if target_dir is None:
        data_dirs = args.data_dir or DEFAULT_DATA_DIRS
    elif isinstance(target_dir, (str, os.PathLike)):
        data_dirs = [os.fspath(target_dir)]
    else:
        data_dirs = [os.fspath(path) for path in target_dir]

    missing_dirs = [path for path in data_dirs if not os.path.isdir(path)]
    if missing_dirs:
        print(f"Dir(s) not exist: {', '.join(missing_dirs)}")
        return
    if not 0.0 < args.similarity_threshold <= 1.0:
        raise ValueError('similarity threshold must be in the interval (0, 1]')

    results = []
    cluster_audit = []
    tasks = discover_tasks(data_dirs)
    selected_tasks = parse_task_selectors(args.task)
    if selected_tasks:
        missing_tasks = selected_tasks - set(tasks)
        if missing_tasks:
            raise ValueError(f'Requested task(s) do not exist: {missing_tasks}')
        tasks = [task for task in tasks if task in selected_tasks]

    for task_category, task_id in tasks:
        trajectory_files = discover_trajectories(
            data_dirs, task_category, task_id
        )
        print(
            f'Task {task_category}/{task_id} has total '
            f'{len(trajectory_files)} trajectories.'
        )

        trajectories, step_records, missing_xml_count = collect_task_records(
            trajectory_files
        )
        assigned_ids, task_audit = assign_fine_grained_node_ids(
            step_records, args.similarity_threshold
        )
        graph = build_task_graph(trajectories, assigned_ids)
        graph.compute_lrs_optimized()
        print(
            f'Compute Completed: {len(graph.nodes)} nodes; '
            f'{missing_xml_count} steps without usable XML.'
        )

        graph_critical_steps = print_graph(graph)
        hyperlink_step_keys = {
            record['key']
            for record in step_records
            if record['hyperlink_evidence']
        }
        hyperlink_state_ids = {
            assigned_ids[key]
            for key in hyperlink_step_keys
            if key in assigned_ids
        }
        source_index = build_state_source_index(
            trajectories, assigned_ids, data_dirs
        )
        hyperlink_source_index = build_state_source_index(
            trajectories,
            assigned_ids,
            data_dirs,
            included_step_keys=hyperlink_step_keys,
        )
        critical_steps = merge_critical_steps(
            graph_critical_steps,
            hyperlink_state_ids,
            source_index,
            hyperlink_source_index,
        )
        critical_node_ids = {
            critical_step['state_u'] for critical_step in critical_steps
        }
        save_mermaid(
            graph,
            os.path.join(args.mmd_dir, task_category, f'{task_id}.mmd'),
            critical_node_ids=critical_node_ids,
        )
        results.append({
            'task_category': task_category,
            'task_id': task_id,
            'trajectory_count': len(trajectory_files),
            'graph_node_count': len(graph.nodes),
            'missing_xml_count': missing_xml_count,
            'hyperlink_critical_state_count': len(hyperlink_state_ids),
            'critical_steps': critical_steps,
        })
        cluster_audit.append({
            'task_category': task_category,
            'task_id': task_id,
            'clusters': task_audit,
        })

    with open(args.output_json, 'w', encoding='utf-8') as file:
        json.dump(results, file, ensure_ascii=False, indent=2)
    with open(args.cluster_audit_json, 'w', encoding='utf-8') as file:
        json.dump(cluster_audit, file, ensure_ascii=False, indent=2)


# Execute the MobileSafetyBench extractor when this module is run directly.
if __name__ == '__main__':
    main_mobile_safety_bench()
