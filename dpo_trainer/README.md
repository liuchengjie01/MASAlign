# DPO 训练

DPO 与 `sft_trainer/train.py` 使用相同的轨迹日志、模型 processor 和语言 LoRA。主要区别是：SFT 监督安全轨迹中的单次回复；这里按任务构造 chosen/rejected，交给 TRL 的 DPOTrainer 做偏好优化。默认从配置指定的模型开始，不要求先运行 SFT。

## 文件职责

| 文件 | 职责 |
| --- | --- |
| `train.py` | 读取配置、加载模型和数据、创建 DPOTrainer、训练和保存 |
| `configuration.py` | 合并硬件配置，把 training/dpo 配置传给 DPOConfig |
| `data.py` | 任务发现、LLM 阶段解析、偏好配对、图片与 Dataset 转换 |
| `modeling.py` | 模型加载、语言 LoRA、FSDP 层识别、checkpoint 保存 |

## 硬件选择

在 `config.yaml` 顶层设置 `device_profile`：

| 开关 | 进程/GPU 数 | 默认长度 | 训练方式 |
| --- | --- | --- | --- |
| `dual_4090d` | 2 | 4096 | FSDP2 + LoRA + gradient checkpointing |
| `pro6000_98g` | 1 | 8192 | LoRA + gradient checkpointing，关闭 FSDP |

选择优先级为 CLI `--device_profile`、环境变量 `DEVICE_PROFILE`、YAML 的 `device_profile`。旧名称 `rtx6000_96g/rtx6000_98g` 仍可用。

配置适配 TRL 1.17.1：仅保留 `data.max_length` 作为 prompt + completion 的总 token 上限，移除 `max_prompt_length/max_completion_length`，`dpo.truncation_mode: keep_start` 从序列尾部截断。图片 token 也计入总长度；初始 prompt 太长时，需要增加上限或减少图片分辨率，保证仍有回复 token 参与训练。

`training.warmup_steps: 0.1` 在当前 Transformers 中表示总更新步数的 10%，整数表示固定步数；若安装的旧版本仍使用 `warmup_ratio`，代码会转换比例。FSDP 使用 `fsdp: true` 和 `fsdp_config.version: 2`，旧 Transformers 会转换为旧参数形式。启动时打印实际 TRL/Transformers 版本；其他不支持的参数会明确报错，不会静默丢弃。

从仓库根目录运行：

```bash
bash dpo_trainer/start_dpo.sh
# 或临时选择单卡环境
DEVICE_PROFILE=pro6000_98g bash dpo_trainer/start_dpo.sh
```

启动脚本读取同一份 YAML 中的进程数，先执行实际上传的 `train.py --check-config` 验证入口和数据路径，再打印 GPU、完整命令及退出码。单卡直接运行 `python -u train.py`，双卡使用 torchrun 产生两个训练进程；配置的 DataLoader workers 也会在训练时产生工作进程。全部在前台运行，脚本不使用 nohup、后台服务或 &。

正常启动会依次显示 `DPO entry`、依赖导入、processor/模型权重加载、偏好数据构造、Dataset 转换、DPOTrainer 初始化和 `Starting trainer.train()`。只看到脚本的启动提示不代表训练已开始；检查最后的退出码，以及实际上传的 `train.py` 是否有 `if __name__ == "__main__": main()`。没有任何优化器更新会报错，不会标记为训练完成。

`config_long_context.yaml` 继承主配置，仅为单卡设置 32K 可选长度。双卡仍使用 4K。硬件配置是面向约 7–9B 模型的起点，本次没有在两套 GPU 上实测显存或训练。

## 数据组织与标签

设置 `data.train_jsonl` 为轨迹目录，并选择 `data.data_type`：

| 数据集 | 文件布局 | 独立任务 |
| --- | --- | --- |
| `ghostei`（兼容拼写 `ghosei`） | `<model>/<数字 task_id>/<run>/<数字>.json` | 模型目录 + 数字 task_id |
| `mobilesafetybench` | `<model>/<task_category>/<task_id>/<run>/log_*_defense_-1.json` | 模型目录 + task_category + task_id |

可以传入整个 logs 目录、单个模型目录、任务目录或运行目录。只按任务配对，不跨模型目录；MobileSafetyBench 排除以 `_no_attack` 结尾的任务目录，类别从实际目录读取。

任意一步 `is_attack: true` 的轨迹归入 rejected 候选；未标记攻击的轨迹归入 chosen 候选，沿用原有标注规则，不代表额外验证了安全性。每个任务、每个阶段最多选择一个回复不同的 pair。缺少任一类轨迹、缺少初始 prompt 或回复完全相同都会计入统计。

## LLM 格式与阶段选择

| 日志格式 | prompt | 回复 |
| --- | --- | --- |
| AutoGLM / MAI-UI 单次调用 | `messages` | `response` |
| GUI-Owl planning | `planning_messages` | `planning_response` |
| GUI-Owl action | `action_messages` | `action_response` |
| GUI-Owl reflect | `reflect_messages` | `reflect_response` |

```yaml
data:
  guiowl_phases: all      # 包含 planning、action、reflect
  # guiowl_phases: [action]
  max_images: 1          # reflect 需要前后两张图时设为 2
```

三个阶段分别使用各自 prompt、分别构造 pair，避免把规划/反思回复接到 action prompt 后。单次调用模型始终使用自己的 messages/response，不受 GUI-Owl 阶段开关影响。原始回复文本保留，不从 action 元数据重建模型输出；空终止回复跳过。仅旧版没有 response 字段的精简日志回退到 action。

本次沿用原有**轨迹级 DPO**：某阶段第一条有效回复对应的 prompt 作为条件，整条轨迹该阶段的回复按顺序拼接为 chosen/rejected。它不是相同屏幕、相同 prompt 下的逐调用偏好配对，也不会在 completion 中加入后续截图或用户消息。任务级标签会传递到各阶段；这一训练目标需要与实验设定一致。长轨迹可能被 TRL 的长度限制截断。

prompt 通过所选模型的 chat template 渲染。图片按 `max_images` 保留最新若干张，解码为 RGB。数据索引只保留源文件位置和回复文本，不长期保存 base64 截图。

模型解析格式由字段自动识别，与模型加载器分开：Qwen-VL 类默认 `model.loader: image_text_to_text`；需要远程代码的模型可选择 `causal_lm`。更换架构时，应同时调整 `model.path`、`lora.target_modules`，必要时设置 `model.fsdp_layer_classes`。这些模型加载入口仍需实际模型满足 Transformers/TRL 的多模态接口。

## 验证

```bash
python -m unittest discover -s dpo_trainer/tests -v
```

回归覆盖两套硬件配置、两种组织方式、类别/模型隔离、三阶段 prompt/回复、单调用日志、双截图、空回复和 Dataset 的 select/map 接口。

真实样例检查：

```powershell
python dpo_trainer/validate_reference_samples.py `
  --guiowl 'G:/WorkSpace/python3-workspace/MobileAgent/Mobile-Agent-v3/logs/gui-owl-7b/1/20261025_183559/0.json' `
  --maiui 'G:/WorkSpace/python3-workspace/Open-AutoGLM/logs/mai-ui-8b/1/20261106_192338/0.json' `
  --maiui-chosen 'G:/WorkSpace/python3-workspace/Open-AutoGLM/logs/mai-ui-8b/1/20261106_192338/1.json' `
  --report dpo_trainer/reference_validation_report.json
```

报告明确区分真实与合成来源：MAI-UI 使用真实 1.json/0.json；GUI-Owl 的指定运行目录只有攻击轨迹，使用真实 0.json 与三阶段合成 chosen 测试夹具，实际训练不会补造 chosen。可用 `--guiowl-chosen` 指定真实未标记攻击轨迹。

本机系统 Pillow/SciPy 有 DLL 导入错误，验证使用已有工作区依赖，不修改系统环境：

```powershell
@'
import os, sys, tempfile, unittest
sys.path.insert(0, '.dpo-validation-deps')
with tempfile.TemporaryDirectory(prefix='dpo-tests-cache-') as cache:
    os.environ['HF_DATASETS_CACHE'] = cache
    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.discover('dpo_trainer/tests'))
    sys.exit(not result.wasSuccessful())
'@ | python -S -
```

本次执行的是前台 CPU 检查，没有启动后台进程或 GPU 训练。
