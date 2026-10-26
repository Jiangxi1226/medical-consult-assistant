# -*- coding: utf-8 -*-
"""成本与延迟量化：从链路追踪文件汇总**实测**指标。

数据来源是 `data/traces/*.json`（utils/tracer.py 落的盘），不做事后估算。
token 数只在开启 `MEDICAL_STREAM_USAGE=1` 跑过之后才有——因为那些数字来自
服务端返回的 usage 字段，而不是用字数反推。没有就如实报"未采集"。

费用：**不内置任何价格**。DeepSeek 等厂商单价会调整，硬编码一个数字写进
简历就是给自己挖坑。要算钱就显式传单价（单位：元 / 百万 token）：

    python eval/cost_probe.py --price-in 2 --price-out 8

用法：
    python eval/cost_probe.py                       # 只看延迟与 token
    python eval/cost_probe.py --price-in 2 --price-out 8
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRACE_DIR = os.getenv("MEDICAL_TRACE_DIR", "") or os.path.join(_ROOT, "data", "traces")


def _walk(node: dict):
    yield node
    for c in node.get("children", []) or []:
        yield from _walk(c)


def _pct(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return round(values[0], 1)
    ordered = sorted(values)
    k = (len(ordered) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(ordered) - 1)
    return round(ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo), 1)


def collect() -> dict:
    files = sorted(glob.glob(os.path.join(TRACE_DIR, "*.json")))
    totals, ttfts, dept_ms = [], [], []
    tok_in = tok_out = 0
    runs_with_usage = 0
    per_dept = {}

    for path in files:
        try:
            data = json.load(open(path, encoding="utf-8"))
        except Exception:
            continue
        root = data.get("root", {})
        if root.get("name") != "consult":
            continue
        totals.append(root.get("duration_ms", 0))
        run_has_usage = False
        for node in _walk(root):
            name = node.get("name", "")
            attrs = node.get("attributes", {}) or {}
            if name.startswith("dept:"):
                dept_ms.append(node.get("duration_ms", 0))
                key = name.split(":", 1)[1]
                slot = per_dept.setdefault(key, {"runs": 0, "ms": [], "ttft": []})
                slot["runs"] += 1
                slot["ms"].append(node.get("duration_ms", 0))
                if "ttft_ms" in attrs:
                    slot["ttft"].append(attrs["ttft_ms"])
                    ttfts.append(attrs["ttft_ms"])
                if "prompt_tokens" in attrs or "completion_tokens" in attrs:
                    tok_in += attrs.get("prompt_tokens", 0)
                    tok_out += attrs.get("completion_tokens", 0)
                    run_has_usage = True
        if run_has_usage:
            runs_with_usage += 1

    return {
        "runs": len(totals),
        "runs_with_usage": runs_with_usage,
        "total_ms": {"p50": _pct(totals, 0.5), "p95": _pct(totals, 0.95),
                     "mean": round(statistics.fmean(totals), 1) if totals else 0.0,
                     "max": round(max(totals), 1) if totals else 0.0},
        "dept_ms": {"p50": _pct(dept_ms, 0.5), "p95": _pct(dept_ms, 0.95)},
        "ttft_ms": {"p50": _pct(ttfts, 0.5), "p95": _pct(ttfts, 0.95)},
        "tokens": {"prompt": tok_in, "completion": tok_out,
                   "total": tok_in + tok_out,
                   "avg_per_run": round((tok_in + tok_out) / runs_with_usage, 1)
                   if runs_with_usage else None},
        "per_dept": {k: {"runs": v["runs"],
                         "avg_ms": round(statistics.fmean(v["ms"]), 1) if v["ms"] else 0,
                         "avg_ttft_ms": round(statistics.fmean(v["ttft"]), 1) if v["ttft"] else None}
                     for k, v in per_dept.items()},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--price-in", type=float, default=None, help="输入单价：元/百万 token")
    ap.add_argument("--price-out", type=float, default=None, help="输出单价：元/百万 token")
    args = ap.parse_args()

    if not os.path.isdir(TRACE_DIR):
        print(f"未找到追踪目录：{TRACE_DIR}\n先跑几次会诊（可开 MEDICAL_STREAM_USAGE=1）再回来。")
        return

    m = collect()
    print(f"会诊次数: {m['runs']}（其中 {m['runs_with_usage']} 次采集到真实 token 用量）")
    print(f"总耗时  P50={m['total_ms']['p50']}ms  P95={m['total_ms']['p95']}ms  "
          f"均值={m['total_ms']['mean']}ms  最大={m['total_ms']['max']}ms")
    print(f"各科耗时 P50={m['dept_ms']['p50']}ms  P95={m['dept_ms']['p95']}ms")
    print(f"首字延迟 TTFT  P50={m['ttft_ms']['p50']}ms  P95={m['ttft_ms']['p95']}ms")
    print(f"Token 累计: 输入={m['tokens']['prompt']}  输出={m['tokens']['completion']}  "
          f"合计={m['tokens']['total']}")
    if m["tokens"]["avg_per_run"] is not None:
        print(f"单次会诊平均 token: {m['tokens']['avg_per_run']}")
    else:
        print("单次 token: 未采集（需 MEDICAL_STREAM_USAGE=1 重跑）")

    for k, v in sorted(m["per_dept"].items()):
        ttft = f"{v['avg_ttft_ms']}ms" if v["avg_ttft_ms"] is not None else "n/a"
        print(f"  - {k}: 跑 {v['runs']} 次, 平均 {v['avg_ms']}ms, TTFT {ttft}")

    if args.price_in is not None and args.price_out is not None and m["tokens"]["total"]:
        cost = (m["tokens"]["prompt"] / 1e6 * args.price_in
                + m["tokens"]["completion"] / 1e6 * args.price_out)
        per_run = cost / max(m["runs_with_usage"], 1)
        print(f"\n按输入 {args.price_in} 元/百万、输出 {args.price_out} 元/百万 估算：")
        print(f"  这批会诊合计 {cost:.4f} 元，单次约 {per_run:.4f} 元")
        print("  （单价由调用方传入，脚本不内置任何厂商价格）")


if __name__ == "__main__":
    main()
