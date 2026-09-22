# opd_masalign

MASAlign 项目里 OPD（One-step Policy Distillation，暂定名）训练部分的参考实现：把协作者标出的"关键决策步骤"数据，转换成统一格式，再用一个单进程/单GPU的教师-forcing训练循环，跑通 student 前向 → EMA-teacher 前向 → reverse-KL loss → EMA 更新 这套流程。

**这份 README 只描述"现在代码库里实际是什么样"，不描述论文/设计文档里理想中应该是什么样。** 如果和 `docs/DATA_FORMAT.md` 或 `tutorials/` 里的说法冲突，以这份 README 和源码为准——那两处还没跟上最近一次架构重构，本文档最后的"已知问题 / TODO"里列了具体要修的地方。

## 目录结构

```
opd_masalign/
  schema.py                 CriticalStepRecord 的字段定义 + 校验
  extract_critical_steps.py 数据转换：协作者原始rollout日志 -> CriticalStepRecord JSONL
  self_teacher.py           构造 EMA-teacher 的"事后复盘"reprompt
  losses.py                 OPD loss（reverse-KL）+ EMA teacher更新（含FSDP2支持）
  opd_trainer.py            训练主循环（CLI入口），纯文本 / VLM 两条路径都支持
  tests/                    单元测试 + 手动多卡验证脚本 + 合成数据生成脚本
data/
  critical_steps_smoke.jsonl            合成的完整schema示例数据
  critical_steps_minimal_smoke.jsonl    合成的"仅4个必填字段"示例数据
  critical_steps_autoglm.jsonl          从真实协作者数据里提取出来的真实训练数据
  luis/                                 协作者提供的原始数据（rollout日志zip + 关键步索引）
slurm/                       在这台HPC集群上跑各种验证用的sbatch脚本
docs/DATA_FORMAT.md          数据格式的原始设计文档（部分内容已过时，见下）
tutorials/                   6个中文教学notebook（部分已过时，见下）
```

## 环境安装

```bash
python3.11 -m venv opd_masalign/.venv
source opd_masalign/.venv/bin/activate
pip install -r requirements.txt   # 项目根目录下
```

`requirements.txt` 里 torch 装的是CPU版；GPU训练/验证要单独装CUDA版torch（看`requirements.txt`里的注释），本仓库里另外维护了一份`opd_masalign/.venv_gpu`专门给需要真实GPU的验证脚本用。

**在这台集群上跑东西，永远不要直接在登录节点跑**（哪怕看起来只是跑个pytest）——用`srun`/`sbatch`提交到`standard`或其他计算分区。`slurm/`目录下的脚本都是这么写的，照抄格式即可。

## 数据：从协作者原始数据到训练数据

### 第一步：协作者给的是什么

`data/luis/`下有两个东西：
- `critical_steps.json`：`[{"task_id": "31", "critical_steps": [{"state_u": "包名/Activity名"}]}]`——只是个**定位索引**，指出哪个界面状态是关键决策点，本身不是训练样本。
- `test_log.zip`：`<model>/<task_id>/<时间戳>/<rollout编号>.json`，每个文件是一条按时间顺序排列的rollout，每一步是`{step, package, activity, action, messages, response, prompt_tokens, log_prob_avg, is_attack(可选)}`。

这两个东西都**不是**`opd_trainer.py`能直接吃的格式。

### 第二步：`extract_critical_steps.py` 做转换

```bash
python -m opd_masalign.extract_critical_steps \
  --zip data/luis/test_log.zip \
  --critical_steps_json data/luis/critical_steps.json \
  --model autoglm-phone-9b \
  --out data/critical_steps_autoglm.jsonl
```

`--model`不是写死的，传哪个协作者模型的目录名就转哪个——但目前只有`autoglm-phone-9b`（开源模型）的日志带`is_attack`字段，能支撑下面第2步的真实标注；`claude-fable-5`/`gemini-3.1-pro-preview`/`gpt-6-astra`这些闭源模型的日志里没有这个字段，会被跳过并给出warning。

对每个`(task_id, state_u)`，脚本做两件事：

1. **定位**：在rollout里找`package/activity`字符串匹配`state_u`的第一步。
2. **⚠️ 重建完整对话历史**（这是最近才修的，之前是个真实bug）：`test_log.zip`里每一步的`messages`字段**只有那一步新增的那一条user turn**（第0步例外，多一条system），完全没有累积的历史对话——但同一份日志自己报出来的`prompt_tokens`是逐步稳定上涨的（比如某条rollout从4058涨到5066），说明真实发给LLM的request其实是带着历史的，只是没被完整记录下来。`_reconstruct_full_messages()`会把该rollout里**这一步之前的每一步自己的user turn（去掉图片，只留文字）+ 它的`response`（当作assistant turn）**依次穿起来，最后接上这一步自己的user turn（保留图片）。这里做了两个假设：(a) 缺的历史就是本条rollout自己的前几步，文字部分线性拼接、没做摘要/压缩/滑窗——**这条还没确认**；(b) 早期轮次的图会被丢弃，只有当前这一步保留截图——**这条已经和协作者确认过，是对的**。(a)在正式训练前仍需要协作者确认。
3. **标注**：往后扫同一条rollout，找有没有`is_attack: true`的后续步骤，标`attack_success`/`benign_success`（二值，没有`benign_fail`/`truncated`）。

产出的`observation_context.messages`是OpenAI风格的多模态消息列表（`content`可以是字符串，也可以是`[{"type":"image_url","image_url":{"url":"data:image/...;base64,..."}}, {"type":"text","text":"..."}]`这种块列表）——`docs/DATA_FORMAT.md`目前还没写这部分，见文末TODO。

## 训练：`opd_trainer.py`

纯文本：
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
  --disable_image_splitting --image_max_side 512
```

几个容易踩坑的CLI参数，都是真实跑出问题之后加的：

| 参数 | 什么时候要用 |
|---|---|
| `--dtype fp32` | 在没有原生bf16硬件加速的CPU上跑，默认的`bf16`会慢10-50倍 |
| `--disable_image_splitting` | VLM模式下`--batch_size > 1`时必须加，否则不同图片按分辨率切出的tile数不一样，没法拼成一个batch；对手机截图这种高分辨率图也能避免把内存/显存撑爆 |
| `--image_max_side N` | 配合上一条，把图缩到合理大小再编码 |
| `--no_is_clip` | 协作者数据的tokenizer没法验证和`--model_name`一致时，关掉IS-clip（`old_action_logprob`已经改成训练开始前用当前checkpoint自算，不再依赖协作者提供，正常情况不需要这个开关） |

训练循环每一步都会把loss/蒸馏信号相关的指标追加写到`{output_dir}/metrics.jsonl`（一行一个JSON），不止打印在stdout——想看某次训练的信号变化直接读这个文件。

## 测试

```bash
python -m pytest opd_masalign/tests/ -v
```

现在是33个测试全过，覆盖：
- `test_extract_critical_steps.py`：数据转换脚本的定位/历史重建/攻击标注逻辑（离线，不需要真实数据）
- `test_losses.py`：OPD loss梯度方向、EMA更新（含FSDP2路径的CPU验证）
- `test_response_alignment.py`：student/teacher的action-token对齐逻辑（按batch内最长的padding，不是裁到最短的——之前是个bug）
- `test_vlm_encode.py`：多模态消息解析/图片解码
- `test_minimal_contract.py`：只给4个必填字段时的兜底行为

`tests/manual_multi_rank_ema_check.py`和`manual_multi_rank_train_step_check.py`不会被pytest自动跑（需要真实多进程/真实GPU），用`slurm/multi_rank_ema_check.slurm`等脚本手动提交。

## 已知问题 / TODO（按优先级）

1. **`data/luis/`里协作者真实原始日志的"完整messages"重建，还有一条假设没确认**——见上面"重建完整对话历史"那段的假设(a)：文字历史是不是真的线性拼接、没有摘要/压缩/滑窗。假设(b)（旧截图会被丢弃）已经和协作者确认过了。
2. **`tutorials/`里6个notebook，有5个已经和当前代码对不上**：`01`提到`delta_u`，`02`/`03`直接import已经被删掉的`opd_masalign.graph`/`risk`/`critical_steps`模块，`05`/`06`用到已经删掉的`delta_weights`/`group_id`等字段。只有`04`没问题。需要按现在的架构重写。
3. **`docs/DATA_FORMAT.md`本身也过时了**：还在讲`delta_u`/`sibling_branches`/协作者提供`old_action_logprob`这些已经被移除的设计，也完全没提多模态`image_url`内容块。需要重写成和这份README、和`schema.py`一致。
4. **`losses.py::ema_update`的FSDP2路径**：验证发现"不gather、直接对分片DTensor做naive lerp"效果和现在的gather版本完全一样（见该函数docstring），但因为"必须两边分片方式逐字节一致"这个前提没有强校验，暂时没换成更便宜的naive版本，算是个待定的性能优化项。
5. **闭源协作者模型（claude-fable-5/gemini-3.1-pro-preview/gpt-6-astra）的数据目前没法参与真实outcome标注**（日志里没有`is_attack`字段），也没有真实逐token logprob（`log_prob_avg`固定是0.0占位符）——目前`extract_critical_steps.py`只支持`autoglm-phone-9b`。
