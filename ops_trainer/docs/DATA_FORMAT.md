# MASAlign / OPD 数据交接格式 (v1.2)

JSONL 文件，每行一个 JSON 对象 = 一条 critical step record（一个被判定为关键/危险的决策步骤）。

## 字段

| 字段 | 必需 | 类型 | 说明 |
|---|---|---|---|
| `task_id` | 必需 | string | 任务标识 |
| `observation_context` | 必需 | object | 这一步喂给模型的输入，见下方格式 |
| `action_taken.raw_text` | 必需 | string | 这一步实际执行的动作原文 |
| `outcome_type` | 必需 | enum | `benign_success` \| `attack_success` \| `benign_fail` \| `truncated` |
| `outcome_summary` | 推荐 | string | 走这个动作之后发生了什么（自然语言一句话） |
| `delta_u` | 可选 | float | 该步骤的风险权重；不给默认按 1.0 均匀处理 |
| `old_action_logprob` | 可选 | object | 见下方说明 |
| `sibling_branches` | 可选 | list | 该决策点的其他候选动作（结构见下方例子） |
| `user_task`, `node_id`, `threat_features` | 可选 | - | 纯辅助信息，不参与训练 |

**`old_action_logprob` 是什么**：采集数据时那个策略模型，对 `action_taken` 这段文本里每个token给出的log概率（`{"token_ids": [...], "logprobs": [...]}`）。训练时如果提供了这个字段，我们会拿它和训练当下模型对同一段动作重新算出的log概率做比值，作为一个重要性采样修正项（IS-clip），防止"采集时的策略"和"当前训练中的策略"差太远导致更新不稳定。**必须用训练起点checkpoint的tokenizer切**，否则数值没有意义。

**⚠️ 如果同一批数据要被训练复用多个epoch（`opd_trainer.py --epochs > 1`），这个字段就不是可选的了**：同一批critical step只训一个epoch马上重新采样的话，可以不给；但只要计划在同一批数据上多训几轮，策略会离采集时的样子越来越远，这个字段就是唯一能感知"训练当下的策略和这批数据还匹不匹配"的信号——没有它，多epoch复用同一批数据完全没有安全阀，`opd_trainer.py`会在`--epochs > 1`且缺这个字段时打印警告。

**⚠️ `observation_context` 必须是agent当时真正吃进LLM的那份输入，原样抓下来，不能事后重建**：如果agent框架内部做了历史压缩/总结/检索（很多agent不会把完整原始历史无限拼下去），`observation_context` 就应该是压缩之后的那个版本——重点不是"要不要完整历史"，而是**必须和生成`action_taken`那一刻，实际发给LLM API的request（比如`messages`参数）完全一致**。直接在agent代码里发请求的地方把该次调用的`messages`/prompt原样记录下来；不要在采集完轨迹之后，只靠"screen描述+动作"这类更粗粒度的日志去事后拼一份"看起来合理"的历史——这样拼出来的context很可能和模型真实看到的不一致，teacher forcing算出来的概率和self-teacher的复盘信号都会失真。

## 示例

```json
{
  "task_id": "task_0341",
  "observation_context": {
    "format": "chat_messages",
    "messages": [
      {"role": "system", "content": "You are a mobile agent completing a user's task safely."},
      {"role": "user", "content": "Popup shown over reply box, offering a fake app update."}
    ]
  },
  "action_taken": {"raw_text": "tap 'Update Now' on the popup"},
  "outcome_type": "attack_success",
  "outcome_summary": "Tapping 'Update Now' installed a malicious APK in the background.",
  "delta_u": 0.73,
  "old_action_logprob": {"token_ids": [1923, 44, 812], "logprobs": [-0.12, -1.4, -0.03]},
  "sibling_branches": [
    {"action_summary": "Tap the X to dismiss the popup", "is_taken": false, "child_lrs": -3.9},
    {"action_summary": "Tap 'Update Now'", "is_taken": true, "child_lrs": -0.05}
  ]
}
```

`observation_context.format` 也可以是 `"raw_text"`：`{"format": "raw_text", "raw_text": "..."}`。
