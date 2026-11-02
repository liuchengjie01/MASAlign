# opd_masalign

## 快速开始：

在项目根目录执行。训练脚本不会自动提取数据，所以先提取、检查，再训练。

```bash
# 1. 原始 rollout zip -> 训练数据 JSONL（结束时自动检查，最后一行应为 OK）
ZIPS="data/autoglm-phone-9b.zip data/autoglm-phone-9b-part10.zip" \
CRITICAL_STEPS_JSON=data/critical_steps.json \
OUT=data/critical_steps_0925.jsonl \
bash scripts/extract_critical_steps.sh

# 2. 8 卡 FSDP2 正式训练（路径、epoch、学习率、卡数可用环境变量覆盖）
MODEL=/juicefs/models/AutoGLM-Phone-9B \
DATA=data/critical_steps_0925.jsonl \
OUT=/tmp/opd_autoglm_epoch3_0925 \
EPOCHS=3 \
bash scripts/train_opd_autoglm9b.sh
```

第 2 步展开后的命令是：

```bash
torchrun --standalone --nproc_per_node=8 -m opd_masalign.opd_trainer --vlm \
  --model_name /juicefs/models/AutoGLM-Phone-9B \
  --data_path data/critical_steps_0925.jsonl \
  --output_dir /tmp/opd_autoglm_epoch3_0925 \
  --dtype bf16 --disable_image_splitting --image_max_side 512 \
  --gradient_checkpointing --think_prefill \
  --epochs 3 --max_steps 100000 --max_len 8192 \
  --lr 5e-6 --max_grad_norm 1.0 --save_every 20
```

要点：
- 必须传 `--max_steps`：默认值是 5，到步数就停，与 `--epochs` 无关。
- 数据里要有 `action_taken.thinking`（`--think_prefill` 下 student 一侧需要），旧版提取脚本生成的 JSONL 没有，要重新提取。
- `--think_prefill` 下 teacher 每个 step 会自己生成一段推理，比不生成时慢；耗时看 `metrics.jsonl` 里的 `step_time_s`。
- 参数的含义和其余选项见下面"训练"一节，数据流程见"完整流程"一节。
- teacher 的复盘 prompt 默认是通用版 `t1`（`--reprompt_variant`，见"Teacher 的复盘 prompt"），`REPROMPT=v0` 可切回最早只针对弹窗的版本。

GUI-Owl-7B（Mobile-Agent-V3）训练：

```bash
# 数据需带 planner_context（extract_critical_steps.py --format guiowl 现在会写入；
# data/critical_steps_guiowl_1027_train.jsonl 是 1027 数据加上该字段后的版本）
MODEL=<GUI-Owl-7B 路径> DATA=data/critical_steps_guiowl_1027_train.jsonl OUT=/tmp/opd_guiowl7b \
GPUS=8 bash scripts/train_opd_guiowl7b.sh
```

展开后的关键参数是 `--model_format guiowl --think_prefill --teacher_replan --planner_variant t1`：teacher 先在 planner 的输入上加 t1 复盘 prompt 重写计划，把新计划替换进 executor 输入的 "### Overall Plan ###"/"### Current Subgoal ###"，再在这个上下文里自己写 Thought，最后给原动作打分。executor 一侧不再追加复盘 prompt（在 GUI-Owl-7B 上所有 executor 端 prompt 都让原动作的 log-prob 变高，见"Teacher 的复盘 prompt"）。


MASAlign 项目里 OPD（One-step Policy Distillation）训练部分的实现：把协作者标出的"关键决策步骤"数据转换成统一格式，再用一个单进程/单GPU（可扩展到多GPU FSDP2）的教师-forcing训练循环，跑通 student 前向 → EMA-teacher 前向 → reverse-KL loss → EMA 更新 这套流程。

## 最近更新

- **`opd_trainer.py`新增`--save_every N`**：每N个step额外存一份checkpoint到`{output_dir}/checkpoint-{step}/`（结束时仍会照常在`--output_dir`本身存一份最终checkpoint）。默认关闭、行为不变。加这个之前，训练中途只要崩溃/OOM/被Slurm时间墙杀掉，就完全没有checkpoint可用——只有`metrics.jsonl`里已经落盘的loss记录能留下来，见下面"已知限制"。分布式（FSDP2）下这个也是安全的：内部复用和最终保存同一条"gather到rank0再落盘"的路径，见`opd_trainer.py::_save_checkpoint`。
- **新增`opd_masalign/plot_metrics.py`**：把`metrics.jsonl`画成loss曲线，见下面"训练"一节。
- **新增`opd_masalign/load_checkpoint.py`**：加载一个checkpoint目录（`--output_dir`本身或某个`checkpoint-{step}/`），可选对`--data_path`里的几条记录跑推理、和ground-truth的`action_taken`对比，作为最基本的sanity check。是纯文本还是VLM checkpoint靠目录里有没有`preprocessor_config.json`或`processor_config.json`自动识别，不用额外传参。

```bash
# 只加载，确认checkpoint能读起来
python -m opd_masalign.load_checkpoint --checkpoint_dir /tmp/opd_smoke
# 加载后跑推理，和ground truth对比（VLM真实截图数据建议加下面两个参数，见下条）
python -m opd_masalign.load_checkpoint --checkpoint_dir /tmp/opd_smoke \
    --data_path data/critical_steps_smoke.jsonl --num_records 3
```

注意：这只是"checkpoint能不能正常加载/生成的东西看起来合不合理"级别的sanity check，不是严格的held-out评估（没有做train/eval切分，也没有算成功率之类的指标）——`opd_trainer.py`训练用的数据和这里跑推理用的数据如果是同一份，看到的只是"记没记住训练数据"，不是"泛化能力"。

- **`load_checkpoint.py`修了几个真实GPU上会OOM的问题**（用真实AutoGLM-Phone-9B权重+真实截图数据在GPU上复现过）：
  1. `--dtype`默认值现在按device自动选（GPU上bf16、CPU上fp32），而不是不管什么设备都默认fp32——真实VLM是bf16存的，fp32加载权重+KV cache直接翻倍，9B模型这个差别足以决定会不会OOM。
  2. 加载模型现在也用`device_map`直接流式加载到GPU（跟`opd_trainer.py`一致），不再是先把完整模型materialize到CPU RAM再`.to(device)`拷一份过去。
  3. 新增`--disable_image_splitting`（VLM用）：不加的话processor默认的高分辨率tiling会把一张手机截图切成好几个tile，视觉token数（进而KV cache占用）随图片分辨率剧烈变化，不同记录之间差异很大。
  4. 每条记录生成完之后会`del`掉中间tensor再`torch.cuda.empty_cache()`，避免不同形状的输入让CUDA caching allocator的reserved内存越攒越多；同时每条记录现在会打印`prompt_len`和`gpu_allocated`/`gpu_reserved`/`gpu_max_allocated`，方便定位是"越跑越大"还是"某一条明显更大"。

## 目录结构

```
opd_masalign/
  schema.py                 CriticalStepRecord 的字段定义 + 校验
  extract_critical_steps.py 数据转换：协作者原始rollout日志 -> CriticalStepRecord JSONL
  self_teacher.py           构造 EMA-teacher 的"事后复盘"reprompt（默认 t1，v0 为旧弹窗版）
  reprompt_variants.py      所有复盘 prompt 版本：executor（VARIANTS）/ planner（PLANNER_VARIANTS）/ reflector（REFLECTOR_VARIANTS），默认值 DEFAULT_*_VARIANT
  guiowl_format.py          GUI-Owl 的 "### Thought ### / ### Action ###" 预填和 teacher 推理生成
  replan.py                 GUI-Owl planner 重新规划：teacher 重写计划并替换进 executor 输入
  action_parse.py           把模型回复解析成 (thinking, action, well_formed)
  client.py                 OpenAI兼容接口的推理客户端（支持 <think> 预填）
  prompts_en.py             手机智能体的 system prompt
  compare_teacher_student.py  对真实模型服务采样，比较 student / teacher 两种上下文下的动作
  load_checkpoint.py        加载 checkpoint 并做推理 sanity check
  optim.py                  bf16 安全的 AdamW 和 EMA lerp（随机舍入）
  check_dataset.py          训练前检查数据：schema、outcome 分布、是否有 thinking、每条是否恰好 1 张图
  losses.py                 OPD loss（reverse-KL）+ EMA teacher更新（含FSDP2支持）
  opd_trainer.py            训练主循环（CLI入口），纯文本 / VLM 两条路径都支持
  tests/                    单元测试 + 手动多卡验证脚本 + 合成数据生成脚本
data/
  critical_steps_smoke.jsonl            合成的完整schema示例数据
  critical_steps_minimal_smoke.jsonl    合成的"仅4个必填字段"示例数据
  critical_steps_autoglm.jsonl          从协作者数据提取出的训练数据
  critical_steps_autoglm_popup_test20.jsonl  19条popup决策步（attack_success），用于小规模训练/推理验证
  critical_steps_guiowl_1027_g1.jsonl     GUI-Owl-7B 关键步（94条，60条attack_success），带 downstream_chain/step_idx/popup_offset
  critical_steps_guiowl_1027_train.jsonl  同上再加 planner_context，--teacher_replan 训练用
  luis/                                 协作者提供的原始数据（rollout日志zip + 关键步索引）
slurm/                       HPC集群上跑训练/验证用的sbatch脚本
scripts/                     数据提取和正式训练的 shell 脚本（见下面"完整流程"）
docs/DATA_FORMAT.md          数据格式说明文档（含多模态图片格式）
```

## 环境安装

```bash
python3.11 -m venv opd_masalign/.venv
source opd_masalign/.venv/bin/activate
pip install -r requirements.txt   # 项目根目录下
```

`requirements.txt` 里 torch 装的是CPU版；GPU训练需要单独装CUDA版torch（见`requirements.txt`里的注释）。

如果要跑的VLM架构比较新（比如某些协作者模型），要确认`transformers`版本支持该架构——版本太旧时`AutoProcessor.from_pretrained`不会报错，而是静默退化成一个普通tokenizer，等到后面用的时候才报出令人费解的`AttributeError`。遇到这种情况先确认：

```bash
python -c "from transformers import AutoProcessor; p = AutoProcessor.from_pretrained('<model_name>'); print(hasattr(p, 'image_processor'))"
```

返回`False`就说明这个transformers版本不认识该模型的多模态结构，需要升级。`requirements.txt`里的注释记录了一次真实排查：`transformers`版本够新只是第一关，某些模型（如GLM-4V架构）的config用了较新的字段命名（比如`rope_parameters`替代旧的`rope_scaling`），版本不够新时不会报错、只会让模型悄悄按默认值跑，行为不对但看不出来——遇到RoPE相关的`NoneType`报错时先查这个。

`requirements.txt`里的`torchvision`也是必需的：部分VLM的processor（如GLM-4V）在`import`时会检查这个库是否存在，即使当前只处理图片、不处理视频。

## 完整流程：提取数据 → 检查 → 训练

训练脚本 `opd_trainer.py` 只读 `--data_path` 指向的 JSONL，不会自动调用提取脚本。拿到新的原始日志 zip 时，按下面三步走（在项目根目录执行）：

```bash
# 1. 原始 rollout zip -> CriticalStepRecord JSONL，并自动做数据检查
ZIPS="data/autoglm-phone-9b.zip data/autoglm-phone-9b-part10.zip" \
CRITICAL_STEPS_JSON=data/critical_steps.json \
OUT=data/critical_steps_0925.jsonl \
bash scripts/extract_critical_steps.sh

# 2. （单独检查）确认数据可以用于 --vlm --think_prefill 训练，最后一行应为 OK
python -m opd_masalign.check_dataset data/critical_steps_0925.jsonl

# 3. 训练（8卡 FSDP2）
MODEL=/juicefs/models/AutoGLM-Phone-9B DATA=data/critical_steps_0925.jsonl \
OUT=/tmp/opd_autoglm_epoch3_0925 EPOCHS=3 \
bash scripts/train_opd_autoglm9b.sh
```

- 第 1 步需要原始日志 zip 和 `critical_steps.json`（定位关键步骤的索引）。已有用新版提取脚本生成的 JSONL 时可以跳过。旧版脚本生成的 JSONL 没有 `action_taken.thinking`，`check_dataset` 会报 `NOT READY`，需要重新提取。
- `check_dataset` 检查：每条记录通过 schema 校验、恰好 1 张图（`--vlm` 的 Collator 要求）、有 `thinking`（`--think_prefill` 要求）。
- `scripts/train_opd_autoglm9b.sh` 开始前会再跑一次 `check_dataset`，数据不合格则不启动训练。脚本内的路径、epoch 数、学习率、卡数都可用环境变量覆盖（`MODEL`、`DATA`、`OUT`、`EPOCHS`、`LR`、`GPUS`）。
- 注意 `--max_steps` 的默认值是 5，训练循环到这个步数就会停，与 `--epochs` 无关。脚本里设成了一个足够大的值，让训练由 `--epochs` 决定。自己写命令时需要显式传。
- 步数 ≈ `ceil(记录数 / (GPU数 × batch_size)) × epochs`。`--save_every N` 的 N 大于总步数时中间不会存 checkpoint，只在结束时存一份（每份约 19GB）。

## 数据：从协作者原始数据到训练数据

`opd_trainer.py`只认`CriticalStepRecord`这一种JSONL格式（定义见`docs/DATA_FORMAT.md`），协作者提供的原始数据不是这个格式，需要先转换。

### 协作者原始数据长什么样

`data/luis/`下有两部分：
- `critical_steps.json`：`[{"task_id": "31", "critical_steps": [{"state_u": "包名/Activity名"}]}]`——一份定位索引，指出哪个界面状态是关键决策点，本身不是训练样本。
- `test_log.zip`：`<model>/<task_id>/<时间戳>/<rollout编号>.json`，每个文件是一条按时间顺序排列的rollout，每一步是`{step, package, activity, action, messages, response, prompt_tokens, log_prob_avg, is_attack(可选)}`。

### 用 `extract_critical_steps.py` 转换

```bash
python -m opd_masalign.extract_critical_steps \
  --zip data/luis/test_log.zip \
  --critical_steps_json data/luis/critical_steps.json \
  --model autoglm-phone-9b \
  --out data/critical_steps_autoglm.jsonl
```

`--model`传哪个协作者模型的目录名就转哪个；只有日志里带`is_attack`字段的模型才能被标注（目前是`autoglm-phone-9b`），其余会被跳过并给出warning。

`--zip`可以传多个，数据分多批给时一次合并：

```bash
python -m opd_masalign.extract_critical_steps \
  --zip data/autoglm-phone-9b.zip data/autoglm-phone-9b-part10.zip \
  --critical_steps_json data/critical_steps.json \
  --model autoglm-phone-9b --out data/critical_steps_autoglm_v2.jsonl
```

zip内部既可以是`<model>/<task_id>/...`（单模型），也可以是带一层顶层目录的多模型布局，脚本会自动识别。同一条rollout出现在多个zip里时只保留先出现的那份。

对每个`(task_id, state_u)`，脚本做三件事：

1. **定位**：在rollout里找`package/activity`字符串匹配`state_u`的第一步。
2. **重建完整对话历史**：原始日志里每一步的`messages`字段只包含那一步新增的user turn（第0步例外，多一条system），不含累积的历史对话。`_reconstruct_full_messages()`会把该步之前每一步自己的user turn（去掉图片，只留文字）+ 它的`response`（当作assistant turn）依次穿起来，最后接上这一步自己的user turn（保留图片）。这个重建基于两个假设：早期轮次的截图会被丢弃、只保留当前这一步的图（已与协作者确认）；文字历史是线性拼接、没有做过摘要或滑窗（这条尚待协作者确认，正式训练前建议核实）。
3. **标注**：往后扫同一条rollout，找有没有`is_attack: true`的后续步骤，标`attack_success`/`benign_success`（二值，不区分`benign_fail`/`truncated`）。

模型回复通过`action_parse.py::parse_response`解析：回复里有`<answer>`标签时取最后一个标签块；没有标签时取最后一个以收尾（无尾随文本）结束的`do(...)`/`finish(...)`调用，括号匹配时忽略引号内的括号。解析出的最终动作与日志里记录的动作一致时，推理文本会作为`action_taken.thinking`写入记录。另外每条记录带`state_u`（定位用的`package/activity`）和`source`（来源rollout文件）。

产出的`observation_context.messages`是OpenAI风格的多模态消息列表（`content`可以是字符串，也可以是`[{"type":"image_url","image_url":{"url":"data:image/...;base64,..."}}, {"type":"text","text":"..."}]`这种内容块列表），完整格式说明见`docs/DATA_FORMAT.md`。

## 训练：`opd_trainer.py`

纯文本模型：
```bash
python -m opd_masalign.opd_trainer \
  --model_name Qwen/Qwen2.5-0.5B-Instruct \
  --data_path data/critical_steps_smoke.jsonl \
  --output_dir /tmp/opd_smoke --max_steps 5
```

VLM（`observation_context`里带截图）：
```bash
python -m opd_masalign.opd_trainer --vlm \
  --model_name HuggingFaceTB/SmolVLM-256M-Instruct \
  --data_path data/critical_steps_autoglm.jsonl \
  --output_dir /tmp/opd_vlm_smoke --max_steps 4 --dtype fp32 \
  --disable_image_splitting --image_max_side 512 --max_len 4096
```

常用CLI参数：

| 参数 | 说明 |
|---|---|
| `--vlm` | `--model_name`是视觉语言模型时加这个（走`AutoProcessor`+`AutoModelForImageTextToText`），否则走纯文本的`AutoModelForCausalLM` |
| `--dtype {bf16,fp32}` | 默认`bf16`；在没有原生bf16硬件加速的CPU上跑会明显更慢，这种情况用`fp32` |
| `--disable_image_splitting` | VLM模式下`--batch_size > 1`时必须加，否则不同图片按分辨率切出的tile数不一样，没法拼成一个batch；对高分辨率截图也能降低内存/显存占用 |
| `--image_max_side N` | 配合上一条，把图缩到指定的最长边再编码 |
| `--max_len N` | 上下文+动作的总token上限；带较深历史的记录需要更大的值，超限会报错（VLM模式下不做截断，避免破坏图片token对齐） |
| `--no_is_clip` | 关闭IS-clip重要性采样修正项 |
| `--epochs N` | 完整遍历`--data_path`的轮数；`old_action_logprob`在训练开始前用当前checkpoint自算，为IS-clip提供漂移信号，多轮复用同一批数据时仍建议保持较小的值 |
| `--gradient_checkpointing` | 用重算换激活值显存（只加在student上，EMA teacher全程`no_grad`没有反传，加了没用）。多GPU下FSDP2只分片参数/梯度/优化器状态，不分片激活值——激活值大小只取决于`batch_size × 序列长度 × hidden_size × 层数`，跟卡数无关；真实VLM（尤其大量视觉token拼进文本序列）在多卡上仍可能因激活值而OOM，这是第一个该试的选项 |
| `--think_prefill` | 动作在 `<think>推理</think><answer>` 之后打分：student 用数据里记录的推理，teacher 自己生成推理，见下一节。数据里要有`action_taken.thinking` |
| `--teacher_think_max_tokens N` | teacher 生成推理的最大 token 数，默认 1024；到上限还没写完时保留已写出的部分 |
| `--teacher_think_temperature X` / `--teacher_think_top_p X` | teacher 生成推理的采样参数，默认 0.6 / 0.9 |
| `--is_clip X` | IS-clip 的上限，默认 2.0 |
| `--train_outcomes A,B` | 只训练这些 `outcome_type` 的记录，默认 `attack_success`。复盘 prompt 写的是"该动作是被攻击诱导的错误"，对 `benign_success` 的记录不成立，所以默认排除；传空字符串则训练所有记录 |
| `--model_format autoglm\|guiowl` | 模型回复格式：AutoGLM 的 `<think>/<answer>` 特殊 token，或 GUI-Owl 的 `### Thought ### / ### Action ###` 文字标题；决定 `--think_prefill` 的前缀和 teacher 生成推理的停止方式 |
| `--reprompt_variant` | executor 端复盘 prompt（`reprompt_variants.VARIANTS`，默认 `t1`；`v0` 为旧弹窗版）。开了 `--teacher_replan` 时只在重写计划失败时兜底使用 |
| `--teacher_replan` | 仅 GUI-Owl，需同时 `--think_prefill --vlm`：teacher 在 `planner_context` 上加 `--planner_variant`（默认 `t1`）重写计划，替换进 executor 输入后再生成推理；没有 `planner_context` 的记录会被丢掉。`--teacher_replan_max_tokens` 控制计划长度（默认 400） |
| `--max_steps N` | 训练到第 N 步就停，默认 5；多 epoch 训练必须调大，否则会提前结束 |
| `--max_grad_norm X` | 每步优化前把全局梯度范数裁剪到 X，默认 1.0，0 关闭；裁剪前的范数记录在 `metrics.jsonl` 的 `opd/grad_norm` |
| `--ema_alpha X` | 教师每步按 `teacher ← X·student + (1-X)·teacher` 更新，默认 0.02；0 表示教师固定为初始模型 |
| `--save_every N` | 每N个step额外存一份checkpoint到`{output_dir}/checkpoint-{step}/`，防止长跑中途崩溃/被杀时血本无归；每份都是完整模型拷贝，大模型注意磁盘占用 |

### 动作的打分位置：`--think_prefill`

AutoGLM-Phone-9B 的回复结构固定为 `<think>推理</think><answer>动作</answer>`，其中 `<think>`、`</think>`、`<answer>`、`</answer>` 是特殊 token。默认情况下训练把动作 token 直接放在 `<|assistant|>\n` 之后打分，这不是模型实际生成动作的位置，首个动作 token 的 log-prob 约 -25 到 -31。

加`--think_prefill`后，动作 token 都放在 `<think>推理</think><answer>` 之后打分，loss mask 仍然只覆盖动作 token：

```
<think>
{推理}
</think>
<answer>
{动作}          ← 只有这一段算 loss
```

两边的推理来源不同：
- **student**：用数据里记录的 `action_taken.thinking`。那是它当时真实产生的推理。
- **teacher**（EMA 模型）：每个 step 先在 `observation_context + 复盘 prompt` 的上下文里，以 `<think>` 开头自己生成一段推理，遇到 `</think>` 停止，再对原动作打分。不会把 student 的推理交给 teacher，否则结论已经被旧推理决定，复盘 prompt 起不到作用。生成的采样参数见上面的参数表。
- 如果 teacher 没写推理就直接输出动作，这一条的推理留空（`<think>\n\n</think>`），不退回使用 student 的推理。
- 多卡 FSDP2 下用 `synced_gpus=True` 保证各 rank 的生成同步，不会因为长度不同而卡死。

每个 step 会多一次自回归生成，是训练变慢的主要来源（单卡未分片时平均约 130 个 token、8 秒左右一条，FSDP2 下更慢，未实测）。`metrics.jsonl` 里新增三项：`opd/teacher_think_tokens`（本 step 平均生成长度）、`opd/teacher_think_unclosed`（没写完的条数）、`opd/teacher_think_skipped`（推理为空的条数）。

在真实 9B 权重上，首个动作 token 的 log-prob 约为 0。`opd_masalign/tests/manual_prefill_logprob_check.py` 可以复现这个对比，`opd_masalign/tests/manual_teacher_generate_check.py` 可以在单卡上比较 student、teacher 用旧推理、teacher 用自己生成推理三种情况下被攻击动作的 log-prob（都需要 GPU）。

### bf16 权重下的优化器和 EMA

模型权重以 bf16 存储。bf16 只有约 8 位有效精度，小于相邻两个 bf16 数间距一半的更新会被舍入吃掉：在真实 9B 上 5 步、lr 1e-5 的训练后，约 77% 的权重元素完全没变，`lm_head` 只有 0.34% 变了。因此：
- `OPDTrainer` 在参数为 bf16 时使用 `optim.AdamWSR`：更新在 fp32 里计算，写回 bf16 时做随机舍入（按余数概率向上或向下取整），更新的期望值是准确的。不需要 fp32 主权重，显存和加载方式都不变，参数和优化器状态仍是 bf16。参数为 fp32 时使用普通 `torch.optim.AdamW`。
- EMA 更新（`losses.ema_update`）同样在 fp32 里算并随机舍入。FSDP2 下直接对本地分片在 GPU 上做 lerp，不需要通信；student 和 teacher 的分片形状不一致会报错。
- 不要用单纯调大学习率来补：要超过 bf16 的最小刻度需要 lr 约 1e-4 以上，对 9B 全参数训练过大。
- 训练后可用 `python -m opd_masalign.tests.manual_weight_change_check --base <基座目录> --ckpt <训练输出目录>` 查看权重被改变的元素比例。

训练前`precompute_old_log_probs`只用 student 的上下文和记录的推理编码，所以 IS-clip 的比值和 student 的训练打分一致，不受 teacher 生成的影响。

### Teacher 的复盘 prompt

所有版本在`reprompt_variants.py`里，训练/推理默认用 **t1**（`DEFAULT_EXECUTOR_VARIANT` / `DEFAULT_PLANNER_VARIANT` / `DEFAULT_REFLECTOR_VARIANT`）。t1 不再只讲弹窗，覆盖项目里的三种攻击：

- 弹窗：另一个 app 在任务 app 上弹出对话框要求"验证/确认"，按钮跳到浏览器里收集信息的网页；
- 超链接：任务 app 自己的内容被注入（如 Joplin 笔记 "Security Notice"），链接打开仿冒该 app 的网页；
- 悬浮窗：覆盖在其他 app 上的诱导内容（示例里是逐字重复用户指令、附带链接的短信）。

t1 由几段组成：发生了什么（`$downstream_chain`，没有时退回`outcome_summary`）→ 你被攻击者放在屏幕上的内容误导了（列出几种形式和常见伪装：安全通知、身份验证、用户协议、重复用户指令；会把你带出任务 app 到收集信息的页面）→ 如果屏幕上有东西在要求你去某处/做某事，想想是不是它误导了你；无论是这些内容还是这一步的动作，重新想它是否符合、是否有助于用户原本的目标 → 隐私信息只该给任务所在的 app，之前写的计划可能已被带偏 → 按角色结尾（planner 修改计划 / executor 必要且安全就保留、否则更谨慎 / reflector 离开任务 app 或进入任务不需要的页面算 B）。prompt 里不规定输出格式、不点名具体按钮，也不判断当前步骤在弹窗前后。

各角色怎么用：
- AutoGLM（单次调用）：`observation_context`后追加 executor 版 t1（`--reprompt_variant t1`）。t1 在 AutoGLM 上还没有单独测过，之前在 AutoGLM 上测过的是 v0。
- GUI-Owl：`--teacher_replan`，planner 版 t1 重写计划，executor 不加 prompt（见上面"快速开始"）。在 GUI-Owl-7B 的 1027 数据（60 条 attack_success）上，executor 端各版本的 mean diff 都为正（v0 +1.01、g1 +1.28、g2 +1.48、v4 +1.61），planner 重写计划在弹窗那一步（36 条）把原动作 log-prob 降低（g4：mean −9.83，86% 变低）。t1/t2 与 g4/g5/g6 的对比在跑（`slurm/teacher_check_guiowl_t1_t2_full.slurm`）。
- reflector 版 t1 只用于测试（`tests/manual_teacher_reflect_check_guiowl.py`），训练还没有接 reflector 这一路（见 TODO）。

prompt 里不规定输出格式，输出结构靠`--think_prefill`保持（AutoGLM 是`<think>/<answer>`，GUI-Owl 是`--model_format guiowl`的文字标题）。

训练循环每一步会把loss和相关指标追加写入`{output_dir}/metrics.jsonl`（一行一个JSON），不止打印在stdout。

用`plot_metrics.py`把loss曲线画出来（同时会在stdout打印总耗时/平均每step耗时）：

```bash
python -m opd_masalign.plot_metrics {output_dir}/metrics.jsonl --out loss.png
# 多个run放一起对比，x轴换成累计训练耗时（分钟）而不是step：
python -m opd_masalign.plot_metrics run_a/metrics.jsonl run_b/metrics.jsonl --labels a b --xaxis time
```

### 多GPU训练（FSDP2）

CLI命令不变，用`torchrun`拉起多进程即可，`opd_trainer.py`会自动检测`WORLD_SIZE`环境变量切到分布式模式：

```bash
torchrun --standalone --nproc_per_node=<GPU数> -m opd_masalign.opd_trainer \
  --model_name Qwen/Qwen2.5-0.5B-Instruct \
  --data_path data/critical_steps_smoke.jsonl \
  --output_dir /tmp/opd_multigpu --max_steps 5
```

分布式模式下：
- student模型和EMA teacher模型各自独立加载后分别用`fully_shard`按decoder layer切片（先切每层再切整模型），所以两者显存占用都会被均摊到各GPU上。VLM架构（GLM-4V的`model.visual.blocks`、SmolVLM/Idefics3的`model.vision_model.encoder.layers`）的视觉塔也会按同样粒度单独切片，不会被整个打包进最外层那一次`fully_shard(model)`调用。
- 数据用`DistributedSampler`分片，各rank吃到不同的样本；`--batch_size`是每个rank自己的batch size，总有效batch size是`--batch_size × GPU数`。
- 日志打印、`metrics.jsonl`写入、checkpoint保存只在rank 0上发生；保存前会把分片的FSDP2权重gather成完整state dict再落盘，产出的checkpoint目录和单卡跑出来的格式一致，可以直接用`from_pretrained`加载。
- `--standalone`只支持单机多卡；多机多卡需要换成`torchrun --rdzv_backend=c10d --rdzv_endpoint=<主节点>:<端口> --nnodes=<节点数> ...`，这条路径尚未实测。
- **FSDP2只分片参数/梯度/优化器状态，不分片激活值**：激活值内存只取决于`batch_size × 序列长度 × hidden_size × 层数`，跟卡数无关。真实VLM（尤其视觉token多、序列长的场景）在多卡上仍可能因激活值爆显存，配合`--gradient_checkpointing`使用。

## 推理与 teacher/student 对比

`client.py`是 OpenAI 兼容接口的客户端，`ModelConfig(think_prefill=True)`会在请求末尾加一条内容为`<think>`的 assistant 消息，并带上`continue_final_message=True`和`add_generation_prompt=False`，需要服务端支持（vLLM 支持）。返回文本从预填标签之后开始。`ModelResponse.well_formed`标记回复是否以完整的动作调用结尾。

`compare_teacher_student.py`对每条记录分别用原始上下文（student）和加了复盘 prompt 的上下文（teacher）各采样若干次，统计与原动作相同的比例：

```bash
python -m opd_masalign.compare_teacher_student \
  --base_url http://<host>:<port>/v1 --model autoglm-phone-9b --apikey <key> \
  --data_path data/critical_steps_autoglm_popup_test20.jsonl \
  --is_attack --num_rollouts 8 --think_prefill --output out.jsonl
```

`--reprompt_variant`选 teacher 用的复盘 prompt（默认 t1，`v0`为旧弹窗版）。这是通过 API 采样的行为层面的检查，不计算训练用的 log-prob 和 KL。`load_checkpoint.py --think_prefill`对本地 checkpoint 做同样的预填；加`--use_teacher_reprompt`则用 teacher 上下文生成。

## 测试

```bash
python -m pytest opd_masalign/tests/ -v
```

覆盖范围：
- `test_extract_critical_steps.py`：数据转换脚本的定位/历史重建/攻击标注逻辑（离线，不依赖真实数据）
- `test_losses.py`：OPD loss梯度方向、EMA更新（含FSDP2路径的CPU验证）
- `test_response_alignment.py`：student/teacher的action-token对齐逻辑
- `test_vlm_encode.py`：多模态消息解析、图片解码
- `test_minimal_contract.py`：只给4个必填字段时的兜底行为

`test_optim.py`：随机舍入无偏、`AdamWSR` 在 bf16 上不丢小更新、EMA lerp 期望正确（CPU）。

`tests/manual_prefill_logprob_check.py`需要 GPU 和真实模型权重，用于核对`--think_prefill`前后动作 token 的 log-prob。

`tests/manual_multi_rank_ema_check.py`和`manual_multi_rank_train_step_check.py`需要真实多进程/真实GPU，不会被pytest自动跑，用`slurm/multi_rank_ema_check.slurm`等脚本单独提交。

## TODO

- reflector 训练：reflector 版 t1 在测试里能把弹窗那一步的反思从 A 推向 B，但训练数据和`opd_trainer.py`目前只有 executor 动作这一种目标；要训练 reflector 需要再抽一类以 `reflect_messages` 为上下文、反思输出为目标的记录。
- 超链接和悬浮窗两种攻击目前只有各一条非 GUI-Owl 的示例轨迹，t1 只在 GUI-Owl 弹窗数据上测过。悬浮窗示例的截图和 XML 里看不到悬浮窗本身，prompt 里对它的描述是推测。

- 通用复盘 prompt（`reprompt_variants.py` 的 `g1`）里暂时去掉了 "but the flag may be wrong -- this step may also have been a necessary, harmless part of the task."，以后可能加回来：HMM 标出的关键步不一定是真正出错的那一步（例如弹窗前打开 app 那一步），加上这句可以让 teacher 在这种步骤上保留原动作。

## 已知限制
- **`losses.py::ema_update`的FSDP2路径**：对本地分片直接做lerp，更新前先对student和teacher各调一次`reshard()`（teacher只做forward，根模块自己的参数forward后仍是完整张量，不reshard会报分片形状不一致）。已在4×A100、真实AutoGLM-Phone-9B上跑通5步。
- **多GPU训练**：已在真实2×A100上验证`torchrun --nproc_per_node=2`跑通FSDP2分片训练+EMA teacher分片+checkpoint gather保存（文本模型SmolLM2）。尚未验证：多机多卡（`--standalone`只支持单机）、以及超过2卡的规模。 真实9B（`--vlm`、`--think_prefill`、teacher自己生成推理、`AdamWSR`）已在单机4卡上验证；8卡由使用方在别处运行。
- **VLM + FSDP2 + 真实大模型的组合**：真实8×A100(40GB)跑`AutoGLM-Phone-9B`时观测到过一次OOM，定位是激活值内存（而非参数分片）——已加`--gradient_checkpointing`选项、精简了teacher-forced前向不必要的KV cache（`use_cache=False`）、并修复了视觉塔此前未按块单独分片的问题，但完整的"真实9B模型 + 8卡FSDP2"端到端跑通尚未重新验证过，需要在真实集群上用这些选项重跑确认。