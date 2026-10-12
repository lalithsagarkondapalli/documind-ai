.PHONY: install test lint data eval gate reports run validate-infra

install:
	pip install -r requirements-dev.txt -r requirements-eval.txt

test:
	pytest

lint:
	ruff check app tests/unit tests/integration tests/conftest.py
	ruff format --check app tests/unit tests/integration tests/conftest.py

data:
	python tests/evaluation/synthetic_dataset_generator.py --n-docs 60 --n-questions 180 --out-dir data

eval: data
	python tests/evaluation/run_eval.py --data-dir data --reports-dir reports

gate:
	python tests/evaluation/check_gate.py --metrics reports/metrics.json

reports:
	python tests/evaluation/benchmark_dashboard.py --metrics-file reports/metrics.json --out reports/dashboard.html
	python tests/evaluation/pdf_report.py --metrics-file reports/metrics.json --out reports/benchmark_report.pdf

run:
	uvicorn app.main:app --reload

validate-infra:
	python scripts/validate_manifests.py infra/k8s
