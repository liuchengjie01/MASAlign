# MASAlign / OPD 数据交接格式 (v2.0)

JSONL 文件，每行一个 JSON 对象 = 一条 critical step record（一个被判定为关键/危险的决策步骤）。这是`opd_masalign/opd_trainer.py`直接消费的格式，也是`opd_masalign/extract_critical_steps.py`的输出格式——不管数据是手写的、合成的，还是从协作者原始日志转换出来的，最终都要落到这份schema上。

> 这份文档描述的是**现在代码实际的样子**。v1.x版本里提到的`delta_u`、`sibling_branches`驱动的风险打分、协作者提供`old_action_logprob`这些设计，在一次架构重构里已经被移除/替换掉了，本文档已同步更新，不再是过时状态。

## 字段

| 字段 | 必需 | 类型 | 说明 |
|---|---|---|---|
| `task_id` | 必需 | string | 任务标识 |
| `observation_context` | 必需 | object | 这一步喂给模型的输入，见下方格式 |
| `action_taken.raw_text` | 必需 | string | 这一步实际执行的动作原文 |
| `outcome_type` | 必需 | enum | `benign_success` \| `attack_success` \| `benign_fail` \| `truncated`（`extract_critical_steps.py`目前只产出前两种，见下方说明） |
| `outcome_summary` | 推荐 | string | 走这个动作之后发生了什么（自然语言一句话）；不给的话`self_teacher.py`会退化成一句从`outcome_type`推出来的通用描述 |
| `sibling_branches` | 可选 | list | 该决策点的其他候选动作，每项`{"action_summary": str, "is_taken": bool, "child_lrs": float(可选，<0表示这个候选比实际选的更安全)}`；**目前没有任何自动化流程会填这个字段**（原来负责算这个的graph/risk/critical_steps.py那套树状分支散度流水线已经被删除），只有手工构造或未来重新引入类似机制时才会有值。不给的话`self_teacher.py`的复盘prompt里就不会出现"更安全的选择"这一段 |
| `action_taken.thinking` | 可选 | string | 模型生成这个动作时的推理文本（`<think>`块内容）。`opd_trainer.py --think_prefill`需要它来构造 assistant 前缀；缺失时训练会给出警告 |
| `state_u`, `source` | 可选 | string | `extract_critical_steps.py`写入：定位用的`package/activity`，以及来源 rollout 文件。不参与训练 |
| `downstream_chain`, `downstream_event` | 可选 | string | `extract_critical_steps.py --format guiowl`写入：从这一步到攻击成功那一步的动作链（换 app 时标`[now in <package>]`）。复盘 prompt（t1 等）优先用它，没有时退回`outcome_summary` |
| `planner_context` | 可选 | object | GUI-Owl：这一步 planner 子调用的输入，格式同`observation_context`（`{"format": "chat_messages", "messages": ...}`）。`opd_trainer.py --teacher_replan`需要，缺失的记录会被丢掉 |
| `step_idx`, `popup_offset` | 可选 | int | GUI-Owl：这一步在 rollout 里的下标，以及相对第一个弹窗步骤的偏移。只用于定位和分组分析，prompt 不读 |
| `user_task`, `node_id`, `threat_features` | 可选 | - | 纯辅助信息，不参与训练 |

**不在这张表里、也不需要提供的字段**：`delta_u`（已移除——现在每条critical step在loss里权重相等，不再按风险散度加权，见`opd_masalign/losses.py::compute_opd_loss`的注释）、`old_action_logprob`（已移除，见下方说明）。

## `old_action_logprob`：不用协作者提供，训练时自己算

早期版本要求协作者随数据一起提供`old_action_logprob`（采集时那个策略对`action_taken`每个token的log概率），用于IS-clip重要性采样修正。**现在不这样做了**：`opd_trainer.py::OPDTrainer.precompute_old_log_probs()`会在训练正式开始前，**用当前要训练的这个checkpoint自己**，对数据集里每条记录的`action_taken`重新算一遍log概率，写入内存里的记录（不改动磁盘上的JSONL）。

这样做的原因：IS-clip的比值`exp(student_logp - old_logp)`只有在"old"和"student"用的是**同一个tokenizer**时才有意义——协作者提供的值几乎不可能保证这一点（协作者用的tokenizer未必和你训练起点的checkpoint一致）。用训练起点checkpoint自己算，天然保证tokenizer一致，不需要协作者额外配合。**所以这份JSONL里完全不需要出现`old_action_logprob`这个字段，出现了也会被忽略。**

## `observation_context`：必须是模型当时真正吃进去的那份输入

**⚠️ 这是最容易踩坑、也是实际踩过坑的一条**：`observation_context`必须和生成`action_taken`那一刻，实际发给LLM API的request（`messages`参数）**完全一致**，不能事后拼一份"看起来合理"的历史。

真实踩过的坑：把协作者原始rollout日志（`data/luis/test_log.zip`那种格式）转换成这份schema时，发现**日志里每一步的`messages`字段只记录了那一步新增的那一条消息**，完全没有累积的历史对话——但同一份日志自己报出来的`prompt_tokens`是逐步稳定上涨的，说明真实发给LLM的request其实是带着历史的，只是日志没有完整记录下来。如果直接把这种"单轮"日志原样当成`observation_context`喂给训练，训练出来的信号就是基于一个模型从未真正见过的残缺上下文算出来的，没有意义。**`opd_masalign/extract_critical_steps.py::_reconstruct_full_messages`就是为了解决这个具体问题写的**，可以参考它的实现和docstring（里面详细记录了怎么用`prompt_tokens`的增长趋势倒推出历史被漏记了、以及重建时做了哪些假设、哪些假设已经和协作者确认过）。

**如果你在准备新一批协作者数据**：最好的做法是从一开始就在发起API请求的地方，把那次调用完整的`messages`/prompt原样记录下来，而不是记录"新增的这一条"或者"screen描述+动作"这类更粗粒度的东西——省得后续还要像`extract_critical_steps.py`这样去做有损的事后重建。

`observation_context.format`目前有两种取值：
- `"chat_messages"`：`{"format": "chat_messages", "messages": [...]}` ——**主要用的格式**，`opd_trainer.py`的`Collator`只实现了这一种
- `"raw_text"`：`{"format": "raw_text", "raw_text": "..."}` ——**`schema.py`的校验会接受它，但`opd_trainer.py`目前没有实现这条路径**，传进去会在`Collator`里直接报`AssertionError`（`self_teacher.py::build_teacher_messages`也会主动报`ValueError`并提示怎么手动绕过）。这是个已知的、有意暴露出来的缺口，不是静默出错，但也不能真的拿这个格式跑训练，除非先把`Collator`补上这条路径。

### `messages`里的`content`可以是纯文本，也可以是多模态内容块（截图输入 / `--vlm`）

如果这一步的观测里有图（比如手机智能体的截图），`content`可以是OpenAI风格的内容块列表，而不是纯字符串：

```json
{"role": "user", "content": [
  {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KG..."}},
  {"type": "text", "text": "Popup shown over reply box, offering a fake app update."}
]}
```

- 图片目前只支持**base64 data-URI**（`data:image/...;base64,...`），不支持远程http(s) URL——`opd_masalign/opd_trainer.py::_decode_image_block`就只处理这一种，传别的会直接抛`NotImplementedError`。
- 训练时要加`--vlm`（走`AutoProcessor`+`AutoModelForVision2Seq`，而不是纯文本的`AutoModelForCausalLM`），具体CLI参数和几个容易踩坑的点见`opd_masalign/README.md`。
- **目前`Collator`要求每条记录的`observation_context`里恰好有1张图**（`--vlm`模式下），多图或零图的记录会直接报错，不支持混用。
- 如果历史里有多轮、每轮都有截图：**不要把每一轮的图都保留**——真实数据验证过，只有当前这一轮该带图，更早轮次的图应该丢弃只留文字（否则单条记录能轻松膨胀到几MB，而且和真实场景观察到的token增长曲线对不上，说明真实系统本来就不会无限累积历史截图）。

## `outcome_type`：目前只有二值在被产出

`TERMINAL_TYPES`枚举定义了4种取值，但`extract_critical_steps.py`这条真实数据ETL路径目前只会产出`attack_success`/`benign_success`（往后扫同一条rollout有没有`is_attack: true`的后续步骤），没有对`benign_fail`/`truncated`做区分。如果你的数据源能区分得更细，`schema.py`是接受这两个值的，只是目前没有自动化流程会真的产出它们。

## 多轮回复格式（AutoGLM 类模型）

`messages`里的 assistant 轮是模型当时完整的回复。AutoGLM-Phone-9B 的回复形如：

```
<think>
推理文本
</think>
<answer>
do(action="Tap", element=[500, 800])
</answer>
```

`action_taken.raw_text`只保存最后的动作调用，推理文本放在`action_taken.thinking`。

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
  "sibling_branches": [
    {"action_summary": "Tap the X to dismiss the popup", "is_taken": false},
    {"action_summary": "Tap 'Update Now'", "is_taken": true}
  ]
}
```

带图的例子（多轮历史，只有最后一轮带图）：

```json
{
  "task_id": "task_0342",
  "observation_context": {
    "format": "chat_messages",
    "messages": [
      {"role": "system", "content": "You are a mobile agent completing a user's task safely."},
      {"role": "user", "content": "Organize my inbox."},
      {"role": "assistant", "content": "I'll open the mail app first. do(action=\"Launch\", app=\"Mail\")"},
      {"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KG..."}},
        {"type": "text", "text": "** Screen Info **\n\n{\"current_app\": \"Mail\"}"}
      ]}
    ]
  },
  "action_taken": {"raw_text": "do(action=\"Tap\", element=[500, 800])"},
  "outcome_type": "benign_success"
}
```

## 参考实现

- `opd_masalign/schema.py::validate_critical_step_record` —— 这份schema的程序化校验，跑一遍就知道一条record缺什么/有没有问题
- `opd_masalign/extract_critical_steps.py` —— 从协作者原始per-step rollout日志转换成这份schema的参考ETL实现，包括历史重建的具体做法
- `opd_masalign/README.md` —— 完整的环境安装/训练/测试说明
