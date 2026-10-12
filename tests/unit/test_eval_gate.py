import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation"))
from check_gate import evaluate_gate  # noqa: E402

THRESH = {"a.recall": {"min": 0.9}, "b.cost": {"max": 10}}


def test_passes_when_metrics_meet_thresholds():
    assert all(r["passed"] for r in evaluate_gate({"a": {"recall": 0.95}, "b": {"cost": 5}}, THRESH))


def test_fails_on_regression():
    rows = evaluate_gate({"a": {"recall": 0.80}, "b": {"cost": 5}}, THRESH)
    assert [r["passed"] for r in rows] == [False, True]


def test_missing_metric_fails_closed():
    rows = evaluate_gate({"b": {"cost": 5}}, THRESH)
    assert rows[0]["passed"] is False and rows[0]["reason"] == "missing"


def test_cli_exit_code_reflects_result(tmp_path):
    gate = Path(__file__).resolve().parents[1] / "evaluation" / "check_gate.py"
    t = tmp_path / "t.json"
    t.write_text(json.dumps(THRESH))
    good, bad = tmp_path / "good.json", tmp_path / "bad.json"
    good.write_text(json.dumps({"a": {"recall": 0.95}, "b": {"cost": 5}}))
    bad.write_text(json.dumps({"a": {"recall": 0.10}, "b": {"cost": 5}}))

    def run(m):
        return subprocess.run(
            [sys.executable, str(gate), "--metrics", str(m), "--thresholds", str(t)], capture_output=True, text=True
        ).returncode

    assert run(good) == 0 and run(bad) == 1
