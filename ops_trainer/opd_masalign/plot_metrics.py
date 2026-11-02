"""把 opd_trainer.py 写出的 metrics.jsonl 画成曲线图。

用法：
    python -m opd_masalign.plot_metrics /tmp/opd_smoke/metrics.jsonl
    python -m opd_masalign.plot_metrics run_a/metrics.jsonl run_b/metrics.jsonl --labels a b
    python -m opd_masalign.plot_metrics /tmp/opd_smoke/metrics.jsonl --keys opd/loss opd/mean_is_ratio
    python -m opd_masalign.plot_metrics /tmp/opd_smoke/metrics.jsonl --smooth 20 --out loss.png
    python -m opd_masalign.plot_metrics /tmp/opd_smoke/metrics.jsonl --xaxis time  # x轴改成累计训练耗时(分钟)

每次运行都会在stdout打印一份耗时汇总（总耗时/平均每step耗时/steps-per-sec），
并且只要metrics.jsonl里有`step_time_s`字段，就会自动多画一个"每step耗时"子图，
方便看有没有变慢（比如显存碎片、数据IO卡顿之类的问题）。
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # 无GUI环境（如集群/slurm）下也能直接存图，不依赖$DISPLAY
import matplotlib.pyplot as plt


def load_metrics(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def moving_average(values: list[float], window: int) -> list[float]:
    if window <= 1:
        return values
    out = []
    for i in range(len(values)):
        lo = max(0, i - window + 1)
        chunk = values[lo : i + 1]
        out.append(sum(chunk) / len(chunk))
    return out


def format_duration(seconds: float) -> str:
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    if h:
        return f"{int(h)}h{int(m):02d}m{s:04.1f}s"
    if m:
        return f"{int(m)}m{s:04.1f}s"
    return f"{s:.1f}s"


def print_timing_summary(label: str, rows: list[dict]) -> None:
    times = [row["step_time_s"] for row in rows if "step_time_s" in row]
    if not times:
        print(f"[{label}] metrics.jsonl里没有step_time_s字段，跳过耗时统计")
        return
    total = sum(times)
    avg = total / len(times)
    print(
        f"[{label}] {len(times)} steps, 总耗时 {format_duration(total)}, "
        f"平均每step {avg:.2f}s ({1 / avg:.2f} step/s)"
    )


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("metrics_files", nargs="+", type=Path, help="一个或多个 metrics.jsonl 路径")
    p.add_argument("--labels", nargs="*", default=None, help="每个文件对应的图例名，默认用父目录名")
    p.add_argument(
        "--keys", nargs="*", default=None,
        help="要画的指标key（如 opd/loss）。不指定则自动画出所有文件里出现过的、名字含'loss'的key",
    )
    p.add_argument("--smooth", type=int, default=1, help="滑动平均窗口大小（step数），默认1即不平滑")
    p.add_argument("--logy", action="store_true", help="loss轴用对数坐标（loss有负值时不适用，会自动跳过）")
    p.add_argument(
        "--xaxis", choices=["step", "time"], default="step",
        help="x轴用step序号，还是用累计训练耗时（分钟，需要metrics.jsonl里有step_time_s）",
    )
    p.add_argument("--out", type=Path, default=Path("loss_curve.png"), help="输出图片路径")
    args = p.parse_args()

    # matplotlib把以'_'开头的label当成"不进图例"的内部标记，目录名凑巧以'_'开头
    # （比如这份demo数据用的_demo_output）时得去掉，不然图例会静默地漏掉这条线
    labels = args.labels or [(f.parent.name or str(f)).lstrip("_") or str(f) for f in args.metrics_files]
    if len(labels) != len(args.metrics_files):
        raise SystemExit(f"--labels 数量({len(labels)})必须和 metrics_files 数量({len(args.metrics_files)})一致")

    runs = [load_metrics(f) for f in args.metrics_files]
    for f, rows in zip(args.metrics_files, runs):
        if not rows:
            raise SystemExit(f"{f} 是空文件，没有可画的数据")

    for label, rows in zip(labels, runs):
        print_timing_summary(label, rows)

    # 每个run自己的x值序列：step模式直接用metrics里的step字段；
    # time模式用累计的step_time_s（换算成分钟），缺这个字段就报错退回step模式提示。
    run_xs = []
    for label, rows in zip(labels, runs):
        if args.xaxis == "time":
            if not all("step_time_s" in row for row in rows):
                raise SystemExit(f"[{label}] --xaxis time 需要每行都有step_time_s字段，但这个run里缺失")
            cum = 0.0
            xs = []
            for row in rows:
                cum += row["step_time_s"]
                xs.append(cum / 60.0)
        else:
            xs = [row.get("step", i) for i, row in enumerate(rows)]
        run_xs.append(xs)

    keys = args.keys
    if keys is None:
        all_keys = set()
        for rows in runs:
            for row in rows:
                all_keys.update(row.keys())
        keys = sorted(k for k in all_keys if "loss" in k.lower())
        if not keys:
            raise SystemExit("没有自动找到含'loss'的字段，用 --keys 显式指定要画哪些key")
        # 有耗时数据的话自动多加一个"每step耗时"子图，方便看训练有没有变慢
        if "step_time_s" not in keys and any("step_time_s" in row for rows in runs for row in rows):
            keys = keys + ["step_time_s"]

    n = len(keys)
    fig, axes = plt.subplots(n, 1, figsize=(8, 3.2 * n), squeeze=False, sharex=True)
    axes = axes[:, 0]

    for ax, key in zip(axes, keys):
        any_plotted = False
        for label, rows, xs_all in zip(labels, runs, run_xs):
            xs, ys = [], []
            for x, row in zip(xs_all, rows):
                if key in row:
                    xs.append(x)
                    ys.append(row[key])
            if not ys:
                continue
            any_plotted = True
            raw_line, = ax.plot(xs, ys, alpha=0.25)
            smoothed = moving_average(ys, args.smooth)
            smooth_line, = ax.plot(xs, smoothed, label=label)
            raw_line.set_color(smooth_line.get_color())  # 原始曲线用同色但透明，平滑线在上面盖住
        ax.set_ylabel(key)
        ax.grid(alpha=0.3)
        if any_plotted:
            ax.legend(fontsize=8)
        if args.logy and key != "step_time_s" and all(
            v > 0 for rows in runs for row in rows for k, v in row.items() if k == key
        ):
            ax.set_yscale("log")

    # matplotlib默认字体(DejaVu Sans)不带中文字形，图里的文字一律用英文，避免存出来的图缺字/方框
    axes[-1].set_xlabel("wall-clock time (min)" if args.xaxis == "time" else "step")
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"saved to {args.out}")


if __name__ == "__main__":
    main()
