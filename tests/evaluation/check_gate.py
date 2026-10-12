"""
Evaluation gate for CI. Compares reports/metrics.json against thresholds.json and exits
non-zero if any metric regressed. A MISSING metric also fails the gate (fail closed), so
renaming or dropping a metric cannot silently disable the check.

    python tests/evaluation/check_gate.py --metrics reports/metrics.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent


def lookup(d: dict, dotted: str):
    cur = d
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def evaluate_gate(metrics: dict, thresholds: dict) -> list[dict]:
    rows = []
    for path, rule in thresholds.items():
        if path.startswith("_"):
            continue
        value = lookup(metrics, path)
        if not isinstance(value, (int, float)):
            rows.append({"metric": path, "value": None, "rule": rule, "passed": False, "reason": "missing"})
            continue
        ok = value >= rule["min"] if "min" in rule else True
        if "max" in rule:
            ok = ok and value <= rule["max"]
        rows.append({"metric": path, "value": value, "rule": rule, "passed": ok,
                     "reason": "" if ok else "regressed"})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", default="reports/metrics.json")
    ap.add_argument("--thresholds", default=str(HERE / "thresholds.json"))
    args = ap.parse_args()

    rows = evaluate_gate(json.loads(Path(args.metrics).read_text()), json.loads(Path(args.thresholds).read_text()))
    width = max(len(r["metric"]) for r in rows)
    for r in rows:
        rule = ", ".join(f"{k} {v}" for k, v in r["rule"].items())
        val = "MISSING" if r["value"] is None else f"{r['value']:.4f}"
        print(f"{'PASS' if r['passed'] else 'FAIL'}  {r['metric']:<{width}}  {val:>8}  ({rule})")
    failed = [r for r in rows if not r["passed"]]
    print(f"\nEvaluation gate: {'FAILED' if failed else 'passed'} ({len(rows) - len(failed)}/{len(rows)} checks)")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
