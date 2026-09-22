
import math
from collections import defaultdict, deque


class StateNode:
    def __init__(self, state_id, observation, is_finished=False, is_attacked=False):
        self.state_id = state_id          # 状态哈希 ID (App + 骨架签名)
        self.observation = observation    # 当前页面的文本/UI观测内容
        self.is_finished = is_finished    # 是否为终点 (Sink 节点)
        self.is_attacked = is_attacked    # 如果是终点，是否被注入攻击成功
        self.prs = 0.0                    # 后验风险分数 (PRS)
        self.children = []                # 后继子节点列表 (v)
        self.lrs = 0.0                    # 累积风险分数 (LRS)

class MDPGraph:
    def __init__(self):
        self.nodes = {}                   # state_id -> StateNode
        self.edge_visits = defaultdict(int)       # (u_id, v_id) -> 访问频次 N(u -> v)
        self.action_visits = defaultdict(int)     # u_id -> 该状态总访问频次

    def add_node(self, state_id, observation, is_finished=False, is_attacked=False):
        if state_id not in self.nodes:
            self.nodes[state_id] = StateNode(state_id, observation, is_finished, is_attacked)
        else:
            # 如果节点已存在，更新 observation (可根据需求选择更新或保留)
            if observation and not self.nodes[state_id].observation:
                self.nodes[state_id].observation = observation
            if is_finished:
                self.nodes[state_id].is_finished = True
            if is_attacked:
                self.nodes[state_id].is_attacked = True
        return self.nodes[state_id]

    def add_edge(self, u_id, v_id, action_count=1):            
        self.edge_visits[(u_id, v_id)] += action_count
        self.action_visits[u_id] += action_count

        u_node = self.nodes.get(u_id)
        v_node = self.nodes.get(v_id)
        if u_node and v_node and v_node not in u_node.children:
            u_node.children.append(v_node)

    def get_policy_prob(self, u_node: StateNode, v_node: StateNode) -> float:
        """计算策略/转移概率 P(v | u) = N(u -> v) / Total_N(u)"""
        total = self.action_visits[u_node.state_id]
        if total == 0:
            return 0.0
        count = self.edge_visits[(u_node.state_id, v_node.state_id)]
        return count / total * 1.0

    def evaluate_immediate_threat(self, observation: str) -> float:
        keywords = ["立即清空", "转账", "点击领取", "授权登录"]
        for kw in keywords:
            if kw in observation:
                return 0.85  # 发现高危诱导词
        return 0.0         # 正常页面

    def compute_prs_value_iteration(self, max_iter=100, tol=1e-6):
        """
        使用值迭代法(Value Iteration)在有环图上计算 PRS 
        对应公式: PRS(u) = E(u) + (1 - E(u)) * Max[ P(v|u) * PRS(v) ]
        """
        # 1. 初始化
        for node in self.nodes.values():
            if node.is_finished:
                node.prs = 1.0 if node.is_attacked else 0.0
            else:
                node.prs = self.evaluate_immediate_threat(node.observation)

        # 2. 迭代收敛
        for iteration in range(max_iter):
            max_diff = 0.0
            old_prs = {n_id: node.prs for n_id, node in self.nodes.items()}

            for node in self.nodes.values():
                # if it is a terminal node, skip
                if node.is_finished:
                    continue

                # 计算子树的最大加权风险
                max_child_risk = 0.0
                if node.children:
                    for child in node.children:
                        p_v_u = self.get_policy_prob(node, child)
                        weighted_risk = p_v_u * old_prs[child.state_id]
                        if weighted_risk > max_child_risk:
                            max_child_risk = weighted_risk

                e_u = self.evaluate_immediate_threat(node.observation)
                
                # 核心 PRS 公式应用
                new_prs = e_u + (1.0 - e_u) * max_child_risk

                diff = abs(new_prs - old_prs[node.state_id])
                if diff > max_diff:
                    max_diff = diff

                node.prs = new_prs

            if max_diff < tol:
                print(f"[Info] PRS 风险传播在第 {iteration} 轮成功收敛。")
                break
    def compute_loop_probability_matrix(self, target_node_id: str, max_steps=4, default_p_loop=0.0, tol=1e-6) -> dict:
        """
        【带步数限制的系统性环路求解】
        仅求解在 max_steps 步之内能够折返回 target_node_id 的概率。
        对于路径步数超过 max_steps 的长环路，自动截断并降级使用 default_p_loop。
        
        Args:
            target_node_id: 目标折返节点 (父节点 u)
            max_steps: 允许的最大环路步数 (比如 3 或 4 步)
            default_p_loop: 超出步数限制时的默认折返概率 (默认为 0.0)
        """
        # prob_by_step[step][node_id] 表示从 node_id 出发，恰好在 step 步内到达 target_node_id 的概率
        prob_by_step = {s: {n_id: 0.0 for n_id in self.nodes} for s in range(max_steps + 1)}
        
        # 基础状态：0 步到达 target 节点的概率为 1.0
        for s in range(max_steps + 1):
            prob_by_step[s][target_node_id] = 1.0

        # 按步数动态规划：step 1 -> max_steps
        for s in range(1, max_steps + 1):
            for n_id, node in self.nodes.items():
                if n_id == target_node_id:
                    continue
                if node.is_finished or node.is_attacked:
                    prob_by_step[s][n_id] = 0.0
                    continue

                # P_s(k -> u) = sum_w P(w | k) * P_{s-1}(w -> u)
                step_prob = 0.0
                if node.children:
                    for child in node.children:
                        p_trans = self.get_policy_prob(node, child)
                        step_prob += p_trans * prob_by_step[s - 1][child.state_id]

                prob_by_step[s][n_id] = step_prob

        # 汇总结果：提取在 max_steps 步内可达的最终累计概率
        final_prob_map = {}
        for n_id in self.nodes:
            if n_id == target_node_id:
                final_prob_map[n_id] = 1.0
                continue
                
            p_val = prob_by_step[max_steps][n_id]
            # 如果在 max_steps 步内概率为 0（说明折返路径太长，超过了限制），则使用 default_p_loop
            if p_val == 0.0:
                final_prob_map[n_id] = default_p_loop
            else:
                final_prob_map[n_id] = p_val

        return final_prob_map

    def compute_lrs_optimized(self, tol=1e-6):
        """
        optimized: 基于 Queue 的逆向拓扑传播 + 单调性剪枝，支持有环图
        """
        epsilon = 1e-9

        # 1. 维护一个反向邻接表（即每个节点的前驱节点列表），用于从 finished 节点往回推
        predecessors = {n_id: set() for n_id in self.nodes}
        for (u_id, v_id), _ in self.edge_visits.items():
            predecessors[v_id].add(u_id)

        # 2. 初始化所有节点的 PRS
        queue = deque()
        for node in self.nodes.values():
            if node.is_attacked:
                prs_val = 1.0
                node.lrs = math.log(prs_val + epsilon)  # 对数空间存储，避免数值下溢
                # 终点节点作为源头，首先入队，准备向外辐射反向传播
                queue.append(node.state_id)
            else:
                # 初始赋予当步威胁作为底分
                e_u = self.evaluate_immediate_threat(node.observation)
                node.lrs = math.log(e_u + epsilon)  # 对数空间存储，避免数值下溢

        # 3. 基于 Queue 的反向传播与单调性剪枝循环
        while queue:
            current_id = queue.popleft()
            current_node = self.nodes[current_id]

            # 遍历当前节点的所有“前驱节点”（即能走到 current 的上一步节点 u）
            for p_id in predecessors[current_id]:
                u_node = self.nodes[p_id]
                if u_node.is_attacked:
                    continue  # attack 节点 PRS 固定，不被反向污染

                # 计算前驱节点 u 的最大子树风险
                max_child_log_risk = float('-inf')
                if u_node.children:
                    for child in u_node.children:
                        p_v_u = self.get_policy_prob(u_node, child)
                        if p_v_u > 0:
                            weighted_log_risk = math.log(p_v_u) + child.lrs
                        else:
                            weighted_log_risk = float('-inf')

                        if weighted_log_risk > max_child_log_risk:
                            max_child_log_risk = weighted_log_risk

                e_u = self.evaluate_immediate_threat(u_node.observation)
                
                # 计算 u 的新 PRS 分数
                if max_child_log_risk == float('-inf'):
                    max_propagated_risk = 0.0
                else:
                    max_propagated_risk = math.exp(max_child_log_risk)

                new_prs = e_u + (1.0 - e_u) * max_propagated_risk
                new_lrs = math.log(new_prs + epsilon)

                # 【核心改造：单调性剪枝与收敛控制】
                # 因为 PRS 是风险累加，如果新算出来的风险没有变大（或者变大的幅度小于容忍阈值 tol），
                # 说明该路径的风险已经稳定，无需继续向下游前驱扩散，直接跳过！
                if new_lrs - u_node.lrs > tol:
                    u_node.lrs = new_lrs
                    # 风险发生了有效增长，把此前驱节点入队，让它的上游继续感知风险
                    if p_id not in queue:
                        queue.append(p_id)

        print("[Info] Compute LRS Completed.")


    def extract_critical_pairs(self, max_loop_steps=4, default_p_loop=0.0):
        critical_pairs = []
        epsilon = 1e-9

        for node in self.nodes.values():
            if len(node.children) < 2:
                continue  # 至少需要两个分支才能做对比
            print(f"Compute reverse probability for {node.state_id}")
            loop_prob_map = self.compute_loop_probability_matrix(node.state_id, max_loop_steps, default_p_loop)
            contributions = []
            for child in node.children:
                if child.state_id == node.state_id:
                    continue  # 自环节点不参与关键对提取
                prs_u = math.exp(node.lrs) - epsilon
                prs_v = math.exp(child.lrs) - epsilon
                prs_u = max(0.0, min(1.0, prs_u))
                prs_v = max(0.0, min(1.0, prs_v))
                if child.state_id == node.state_id:
                    net_lrs = node.lrs  # 自环节点，直接使用当前节点的 LRS
                    p_v_to_u = self.get_policy_prob(node, child)
                else:
                    p_v_to_u = loop_prob_map.get(child.state_id, default_p_loop)
                    if p_v_to_u > 0 and p_v_to_u < 1.0:
                        net_prs = (prs_v - p_v_to_u * prs_u) / (1.0 - p_v_to_u)
                        net_prs = max(0.0, min(1.0, net_prs))  # 裁切在 [0, 1]
                        net_lrs = math.log(net_prs + epsilon)
                    else:
                        net_lrs = child.lrs
                advantage = net_lrs - node.lrs


                contributions.append({
                    "child_node": child,
                    "raw_lrs": child.lrs,
                    "net_lrs": net_lrs,
                    "advantage": advantage,
                    "p_loop_step_bounded": p_v_to_u
                })
            if len(contributions) < 2:
                continue

            # 按 LRS 分数对后继节点降序排序
            contributions.sort(key=lambda x: x["advantage"], reverse=True)
            high_risk = contributions[0]   # 危险分支 (Rejected 倾向)
            low_risk = contributions[-1]   # 安全分支 (Chosen 倾向)
            high_log_contrib = high_risk["advantage"]
            low_log_contrib = low_risk["advantage"]
            is_directional_split = (high_log_contrib > 0 and low_log_contrib < 0)

            # 动态落差门槛：使用当前节点下所有动作 Advantage 的标准差作为基准
            advantages = [m["advantage"] for m in contributions]
            mean_adv = sum(advantages) / len(advantages)
            adv_std = math.sqrt(sum((a - mean_adv) ** 2 for a in advantages) / len(advantages))
            adv_gap = high_log_contrib - low_log_contrib
            if high_log_contrib == float('-inf') or low_log_contrib == float('-inf'):
                continue  # 如果最高或最低贡献也是负无穷，说明没有有效的后继分支
            if is_directional_split or (adv_gap >= max(1.0, adv_std)):
                pair = {
                    "state_u": node.state_id,
                    "observation": node.observation,
                    "rejected_next_state": high_risk["child_node"].state_id,
                    "rejected_lrs": high_risk["net_lrs"],
                    "rejected_advantage": high_log_contrib,
                    "chosen_next_state": low_risk["child_node"].state_id,
                    "chosen_lrs": low_risk["net_lrs"],
                    "chosen_advantage": low_log_contrib,
                    "advantage_gap": adv_gap,
                    "is_directional_split": is_directional_split
                }
                critical_pairs.append(pair)
                    
        return critical_pairs

    def print_graph_details(self):
        """格式化输出图中的所有节点名称、节点状态以及每条出边的转移概率和 Action"""
        print("\n" + "="*80)
        print("                      MDP 图拓扑结构与转移概率明细                      ")
        print("="*80)

        if not self.nodes:
            print("[Info] 当前图为空。")
            return

        for state_id, node in self.nodes.items():
            total_visits = self.action_visits[state_id]
            print(f"\n📍 节点 (State Node): [{state_id}]")
            print(f"   ├─ 节点状态: Terminal={node.is_finished} | Attacked={node.is_attacked}")
            print(f"   ├─ 风险指标: PRS={node.prs:.4f} | LRS={node.lrs:.4f}")
            print(f"   ├─ 访问频次: 该节点共被访问/发出动作 {total_visits} 次")
            print(f"   └─ 出发边与转移概率 (Outgoing Edges):")

            if not node.children:
                print("      └── (无出边 / 终点节点)")
            else:
                for idx, child in enumerate(node.children):
                    # 获取计算出的转移概率
                    prob = self.get_policy_prob(node, child)
                    # 获取对应边的访问频次
                    edge_count = self.edge_visits[(state_id, child.state_id)]
                    # 获取触发转移的动作描述
                    action_text = ""
                    
                    is_last_edge = (idx == len(node.children) - 1)
                    prefix = "      └──" if is_last_edge else "      ├──"

                    print(f"{prefix} 走向节点: [{child.state_id}]")
                    print(f"{' '*10}├─ 转移概率 P(v|u): {prob:.4f} ({prob * 100:.2f}%)")
                    print(f"{' '*10}├─ 边访问频次: {edge_count}/{total_visits}")
                    print(f"{' '*10}└─ 触发动作: {action_text}")
                    
        print("\n" + "="*80)



    