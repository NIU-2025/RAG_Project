"""
评估结果对比脚本：将当前 eval_report.json 与 baseline.json 对比，输出差分报告。

用法:
    cd d:/AI_code/RAGProject
    python tests/evaluation/compare_eval.py                  # 对比 eval_report.json vs baseline.json
    python tests/evaluation/compare_eval.py --help           # 查看帮助
"""

import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent
CURRENT_PATH = BASE_DIR / "eval_report.json"
BASELINE_PATH = BASE_DIR / "baseline.json"

THRESHOLDS = {
    "context_precision": 0.70,
    "context_recall": 0.80,
    "faithfulness": 0.75,
    "answer_relevancy": 0.70,
    "retrieval/recall@k": 0.70,
    "retrieval/precision@k": 0.60,
}


def load_json(path: Path) -> dict:
    if not path.exists():
        print(f"[ERROR] 文件不存在: {path}")
        sys.exit(1)
    return json.loads(path.read_text(encoding="utf-8"))


def compare(current: dict, baseline: dict) -> dict:
    cur_s = current.get("summary", {})
    base_s = baseline.get("summary", {})

    all_metrics = set(cur_s.keys()) | set(base_s.keys())
    diffs = {}

    for metric in sorted(all_metrics):
        cur_val = cur_s.get(metric)
        base_val = base_s.get(metric)
        cur_is_nan = cur_val != cur_val if cur_val is not None else True
        base_is_nan = base_val != base_val if base_val is not None else True

        if cur_is_nan or base_is_nan:
            diffs[metric] = {
                "baseline": base_val,
                "current": cur_val,
                "delta": None,
                "status": "nan",
            }
        else:
            delta = round(cur_val - base_val, 4)
            threshold = THRESHOLDS.get(metric)
            below_threshold = threshold is not None and cur_val < threshold

            if below_threshold:
                status = "alert"
            elif delta >= 0.005:
                status = "improved"
            elif delta <= -0.005:
                status = "degraded"
            else:
                status = "stable"

            diffs[metric] = {
                "baseline": base_val,
                "current": cur_val,
                "delta": delta,
                "status": status,
                "threshold": threshold,
            }

    return diffs


def print_comparison(diffs: dict, baseline_info: dict, current_info: dict):
    sep = "=" * 64
    print(f"\n{sep}")
    print(f"  RAG 评估对比报告")
    print(f"  基线: {baseline_info.get('timestamp', '?')}  |  当前: {current_info.get('timestamp', '?')}")
    print(f"{sep}\n")

    print(f"  {'指标':<24s} {'基线':>8s} {'当前':>8s} {'Δ':>8s}  {'状态'}")
    print(f"  {'─' * 60}")

    alerts = []
    for metric, d in sorted(diffs.items()):
        status = d["status"]
        icon = {"improved": "[+]", "stable": "[=]", "degraded": "[-]", "nan": "[WARN]", "alert": "[ALERT]"}.get(status, "?")
        base_str = f"{d['baseline']:.4f}" if d["baseline"] == d["baseline"] else "NaN"
        cur_str = f"{d['current']:.4f}" if d["current"] == d["current"] else "NaN"
        delta_str = f"{d['delta']:+.4f}" if d["delta"] is not None else "  N/A"
        note = ""
        if d.get("threshold") and d["current"] == d["current"] and d["current"] < d["threshold"]:
            note = f" (阈值: {d['threshold']:.2f})"
            alerts.append(f"{metric}: {cur_str} < 阈值 {d['threshold']:.2f}")

        print(f"  {metric:<24s} {base_str:>8s} {cur_str:>8s} {delta_str:>8s}  {icon} {status}{note}")

    if alerts:
        print(f"\n  {'─' * 60}")
        print(f"  [ALERT] 报警项 ({len(alerts)}):")
        for a in alerts:
            print(f"     - {a}")
        print(f"\n  [WARN]  建议: 排查当前参数配置，回退或调整后重新评估")

    cfg_diff = []
    base_cfg = baseline_info.get("config", {})
    cur_cfg = current_info.get("config", {})
    for key in set(base_cfg.keys()) | set(cur_cfg.keys()):
        bv = base_cfg.get(key)
        cv = cur_cfg.get(key)
        if bv != cv:
            cfg_diff.append(f"     {key}: {bv} → {cv}")

    if cfg_diff:
        print(f"\n  [Config] 配置变更:")
        for line in cfg_diff:
            print(line)

    print(f"\n{sep}\n")


def main():
    if len(sys.argv) > 1 and sys.argv[1] in ("--help", "-h"):
        print(__doc__)
        return

    current = load_json(CURRENT_PATH)
    baseline = load_json(BASELINE_PATH)

    diffs = compare(current, baseline)
    print_comparison(diffs, baseline, current)


if __name__ == "__main__":
    main()
