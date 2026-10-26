# -*- coding: utf-8 -*-
"""流式链路探测：跑若干例会诊，产出链路追踪，供 cost_probe.py 汇总延迟与成本。

为什么单独一个脚本：run_eval.py 走的是**图链路**（结构化、可评估），
而链路追踪与真实 token 采集埋在**流式链路**（患者端打字机）里。
两条链路是并存的，指标和性能要分别测。

用法：
    MEDICAL_STREAM_USAGE=1 python eval/stream_probe.py            # 默认跑 5 例
    MEDICAL_STREAM_USAGE=1 python eval/stream_probe.py --n 10     # 跑 10 例

采集到的 token 来自服务端 usage 字段（不是字数反推）；端点不支持时
usage 为空，cost_probe 会如实报"未采集"。
"""
from __future__ import annotations

import argparse
import json
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from graph.stream_flow import stream_consult  # noqa: E402


def _cases(n: int) -> list[dict]:
    path = os.path.join(_ROOT, "eval", "cases.json")
    data = json.load(open(path, encoding="utf-8"))
    # 只用核心集、且主诉信息已足够直接进入会诊的样例，避免耗时花在追问上
    core = [c for c in data if c.get("split", "core") == "core"]
    return core[:n]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5, help="跑前 N 个核心用例")
    args = ap.parse_args()

    cases = _cases(args.n)
    print(f"流式探测：{len(cases)} 例（token 采集 {'开' if os.getenv('MEDICAL_STREAM_USAGE') == '1' else '关'}）")

    for c in cases:
        tokens = 0
        final = None
        for evt, data in stream_consult(c["query"]):
            if evt == "token":
                tokens += 1
            elif evt == "final":
                final = data
        trace = (final or {}).get("_audit", {}).get("trace", {})
        total = trace.get("total_ms", 0)
        summary = (final or {}).get("summary", "")[:36]
        print(f"  {c['id']}  {total:>7.0f}ms  tokens={tokens:<4}  {summary or '（无结案）'}")

    print(f"\n追踪已写入 data/traces/，接着跑：python eval/cost_probe.py")


if __name__ == "__main__":
    main()
