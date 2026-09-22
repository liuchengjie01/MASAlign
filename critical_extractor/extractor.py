
import argparse
import glob
import json
import os

from utils.prs_graph import MDPGraph
from utils.tool import print_graph, save_mermaid, get_canonical_node_id


          

def test():
    graph = MDPGraph()

    graph.add_node("S1", "首页")
    graph.add_node("S2", "弹窗通知：系统检测到空间不足，请清空") # 带有诱导信息
    graph.add_node("S3", "安全分支：用户选择忽略弹窗，返回相册")
    graph.add_node("S4", "危险分支：用户被诱导点击设置")
    graph.add_node("S5", "NULL")
    graph.add_node("S6", "NULL")
    graph.add_node("S7", "危险确认页")
    graph.add_node("S8", "终点：正常结束任务", is_finished=True, is_attacked=False)     # 绿色 Sink
    graph.add_node("S9", "终点：相册被全部格式化攻击成功", is_finished=True, is_attacked=True) # 红色 Sink

    graph.add_edge("S1", "S3", action_count=2)
    graph.add_edge("S1", "S2", action_count=8)

    graph.add_edge("S2", "S6", action_count=1)

    graph.add_edge("S3", "S1", action_count=2)
    graph.add_edge("S3", "S5", action_count=3)
    graph.add_edge("S3", "S4", action_count=2)
    graph.add_edge("S3", "S8", action_count=3)

    graph.add_edge("S4", "S7", action_count=1)

    graph.add_edge("S5", "S1", action_count=1)
    graph.add_edge("S5", "S6", action_count=4)
    graph.add_edge("S5", "S8", action_count=5)

    graph.add_edge("S6", "S8", action_count=1)

    graph.add_edge("S7", "S8", action_count=9)
    graph.add_edge("S7", "S9", action_count=1)

    graph.compute_lrs_optimized()
    print("计算完成，输出各状态的 LRS 值：")
    for state_id, node in graph.nodes.items():
        print(f"状态 {state_id} 的 LRS 值: {node.lrs:.4f}")

    pairs = graph.extract_critical_pairs(theta=100)
    print("提取到的关键对：")
    critical_steps = []
    for idx, p in enumerate(pairs):
        critical_steps.append({
            "state_u": p['state_u'],
        })
        print(f"Pair {idx+1}: 在状态 [{p['state_u']}] 处，应选择状态 [{p['chosen_next_state']} (LRS:{p['chosen_lrs']:.2f})],"
              f"而避开 [{p['rejected_next_state']} (LRS:{p['rejected_lrs']:.2f})]")
    with open("critical_steps.json", 'w', encoding='utf-8') as f:
        json.dump(critical_steps, f, ensure_ascii=False, indent=2)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data-dir',
        required=True,
        type=str,
        help="multiple rollout attack data dir"
    )
    parser.add_argument(
        '--benign-dir',
        required=True,
        default="",
        type=str,
        help='multiple rollout benign data dir'
    )

    args = parser.parse_args()
    return args
def main():
    args = parse_args()
    if not os.path.exists(args.data_dir):
        print(f"Dir {args.data_dir} not exist")
        exit(1)
    ids = ["31","9"]
    results = []
    for id in ids:
        task_dir = f'{args.data_dir}{os.sep}*{os.sep}*{os.sep}{id}'
        self_task_dir = f'{args.benign_dir}{os.sep}*{os.sep}{id}'
        p = f'{task_dir}{os.sep}*{os.sep}*.json'
        sp = f'{self_task_dir}{os.sep}*{os.sep}*.json'
        fs = glob.glob(sp, recursive=True)

        print(f'Task {id} has total {len(fs)} trajectories.')
        graph = MDPGraph()
        for file in fs:
            with open(file, 'r', encoding='utf-8') as f:
                steps = json.load(f)
                
                for idx, step in enumerate(steps):
                    # add node
                    node_state_id = get_canonical_node_id(step.get("package", "unknown"), step.get("activity", "unknown"))
                    is_last_step = (idx == len(steps)-1)
                    is_attacked = step.get("is_attack", False)
                    # if is_attacked:
                    #     print(node_state_id)
                    graph.add_node(
                        node_state_id,
                        "",
                        is_last_step,
                        is_attacked
                    )
                # add edge
                edge_cnt = len(steps) - 1
                for idx in range(edge_cnt):
                    src_id = idx
                    dest_id = idx+1
                    src_step = steps[src_id]
                    dest_step = steps[dest_id]
                    src_state_id = get_canonical_node_id(src_step.get("package", "unknown"), src_step.get("activity", "unknown"))
                    dest_state_id = get_canonical_node_id(dest_step.get("package", "unknown"), dest_step.get("activity", "unknown"))
                    graph.add_edge(
                        src_state_id,
                        dest_state_id,
                        1
                    )

        graph.compute_lrs_optimized()
        print("Compute Completed")
        # graph.print_graph_details()
        save_mermaid(graph, f"{id}.mmd")
        critical_steps = print_graph(graph)
        results.append({"task_id": id, "critical_steps": critical_steps})
    with open("critical_steps.json", 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    

if "__main__" == __name__:
    main()
    