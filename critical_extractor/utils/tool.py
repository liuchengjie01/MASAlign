import base64
import io
import os
import re

import numpy as np
from PIL import Image

from critical_extractor.utils.prs_graph import MDPGraph


def save_mermaid(graph: MDPGraph, output_filepath: str = "mdp_graph.mmd"):
    """
    导出纯净、兼容性极强的 Mermaid Flowchart 文件 (.mmd)
    自动应用 Times New Roman 字体与定制的流程图样式
    """
    lines = []
    
    # 1. 注入主题与全局样式配置 (Flowchart 专用配置)
    lines.append("%%{init: {")
    lines.append("  'theme': 'base',")
    lines.append("  'themeVariables': {")
    lines.append("    'fontFamily': 'Times New Roman, serif',")
    lines.append("    'fontSize': '16px'")
    lines.append("  }")
    lines.append("}}%%")
    
    lines.append("graph TD")

    # ID 清理函数：替换所有非字母数字字符为下划线，防止语法报错
    def clean_id(raw_id: str) -> str:
        return re.sub(r'[^a-zA-Z0-9]', '_', str(raw_id))

    # 2. 生成节点定义（格式化 HTML 换行与引号转义）
    for state_id, node in graph.nodes.items():
        cid = clean_id(state_id)
        # 转义 HTML 字符与单双引号，用 <br/> 替代换行
        safe_id_text = str(state_id).replace('"', '&quot;').replace("'", "&#39;")
        label = f"{safe_id_text}<br/><b>LRS: {node.lrs:.2f}</b>"
        
        if node.is_finished:
            if node.is_attacked:
                # 攻击成功的终点 (使用 {{...}} 双重菱形/六角形)
                lines.append(f'    {cid}{{{{"{label}"}}}}')
            else:
                # 正常结束的终点 (使用 (...) 圆角矩形)
                lines.append(f'    {cid}("{label}")')
        else:
            # 普通中间状态节点 (使用 [...] 直角矩形)
            lines.append(f'    {cid}["{label}"]')

    # 3. 生成边与转移概率标记
    for (src_id, dest_id), count in graph.edge_visits.items():
        src_node = graph.nodes.get(src_id)
        dest_node = graph.nodes.get(dest_id)
        if src_node and dest_node:
            prob = graph.get_policy_prob(src_node, dest_node)
            c_src = clean_id(src_id)
            c_dest = clean_id(dest_id)
            edge_label = f"P={prob:.2f} (N={count})"
            lines.append(f'    {c_src} -->|"{edge_label}"| {c_dest}')

    # 4. 节点样式类定义 (Class Defs)
    lines.append("")
    lines.append("    %% 节点着色与样式定义")
    lines.append("    classDef attackedNode fill:#ffcccc,stroke:#ff0000,stroke-width:2px,color:#990000;")
    lines.append("    classDef safeSink fill:#ccffcc,stroke:#009900,stroke-width:2px,color:#006600;")
    lines.append("    classDef normalNode fill:#f4f4f4,stroke:#333333,stroke-width:1.5px,color:#111111;")

    attacked_ids = [clean_id(n.state_id) for n in graph.nodes.values() if n.is_finished and n.is_attacked]
    safe_ids = [clean_id(n.state_id) for n in graph.nodes.values() if n.is_finished and not n.is_attacked]
    normal_ids = [clean_id(n.state_id) for n in graph.nodes.values() if not n.is_finished]

    if attacked_ids:
        lines.append(f"    class {','.join(attacked_ids)} attackedNode;")
    if safe_ids:
        lines.append(f"    class {','.join(safe_ids)} safeSink;")
    if normal_ids:
        lines.append(f"    class {','.join(normal_ids)} normalNode;")

    # 写入文件（确保 utf-8 编码，并且文件内不包含额外的 Markdown ```mermaid 包裹）
    os.makedirs(os.path.dirname(os.path.abspath(output_filepath)), exist_ok=True)
    with open(output_filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
        
    print(f"[Info] 适配版 Mermaid 文件已完美保存至: {output_filepath}")


def print_graph(graph: MDPGraph):
    for state_id, node in graph.nodes.items():
            print(f"状态 {state_id} 的 LRS 值: {node.lrs:.4f}")
        
    pairs = graph.extract_critical_pairs()
    print("提取到的关键对：")
    critical_steps = []
    for idx, p in enumerate(pairs):
        critical_steps.append({
            "state_u": p['state_u'],
        })
        print(f"Pair {idx+1}: 在状态 [{p['state_u']}] 处，应选择状态 [{p['chosen_next_state']} (LRS:{p['chosen_lrs']:.2f})],"
                f"而避开 [{p['rejected_next_state']} (LRS:{p['rejected_lrs']:.2f})]")

    return critical_steps

def get_canonical_node_id(package: str, activity: str) -> str:
    """将不同来源的 (package, activity) 统一归一化为标准的全局唯一 Node ID"""
    if not package or not activity:
        return "UNKNOWN_NODE"

    package = package.strip()
    activity = activity.strip()

    # 情况 1: activity 以 '.' 开头，属于相对路径，需要补全 package 前缀
    if activity.startswith("."):
        full_activity = f"{package}{activity}"

    # 情况 2: activity 已经包含完整路径（无论是否包含 package 前缀）
    else:
        full_activity = activity

    # 最终的统一 Node 标识符（采用统一的格式：package/full_activity）
    return f"{package}/{full_activity}"


def helloworld() -> str:
    """Return a simple greeting."""
    return "Hello, World!"
