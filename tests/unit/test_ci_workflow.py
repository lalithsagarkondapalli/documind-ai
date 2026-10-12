"""Guards the claim 'deployment is gated on the evaluation job' so it cannot silently regress."""

from pathlib import Path

import yaml

WORKFLOW = yaml.safe_load((Path(__file__).resolve().parents[2] / ".github/workflows/ci.yml").read_text())
JOBS = WORKFLOW["jobs"]


def test_deploy_requires_the_evaluation_gate_and_all_checks():
    needs = set(JOBS["deploy"]["needs"])
    assert {"eval-gate", "lint-test", "integration-postgres", "infra-validate", "docker"} <= needs


def test_deploy_only_runs_on_main_pushes_and_when_explicitly_enabled():
    cond = JOBS["deploy"]["if"]
    assert "refs/heads/main" in cond and "DEPLOY_ENABLED" in cond and "push" in cond


def test_eval_gate_job_runs_the_threshold_check():
    steps = " ".join(s.get("run", "") for s in JOBS["eval-gate"]["steps"])
    assert "run_eval.py" in steps and "check_gate.py" in steps


def test_deploy_uses_oidc_not_static_aws_keys():
    text = (Path(__file__).resolve().parents[2] / ".github/workflows/ci.yml").read_text()
    assert "id-token: write" in text
    assert "AWS_ACCESS_KEY_ID" not in text and "AWS_SECRET_ACCESS_KEY" not in text
