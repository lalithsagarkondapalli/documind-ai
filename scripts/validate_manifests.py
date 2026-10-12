"""Static validation of the Kubernetes manifests (no cluster needed).

Checks structure AND policy, e.g. selectors really match, HPA targets a real Deployment,
no ':latest' images, every container has resource limits + a hardened securityContext.
This is NOT a substitute for `kubectl apply --dry-run=server` against a real cluster.

    python scripts/validate_manifests.py infra/k8s
"""
from __future__ import annotations

import sys
from pathlib import Path

import yaml

SKIP = {"secret.example.yaml", "kustomization.yaml"}


def load(dirpath: Path) -> list[tuple[str, dict]]:
    docs = []
    for f in sorted(dirpath.glob("*.yaml")):
        if f.name in SKIP:
            continue
        for d in yaml.safe_load_all(f.read_text()):
            if d:
                docs.append((f.name, d))
    return docs


def pod_specs(doc: dict):
    kind = doc["kind"]
    if kind in ("Deployment", "StatefulSet"):
        return doc["spec"]["template"]
    if kind == "Job":
        return doc["spec"]["template"]
    return None


def main(path: str) -> int:
    errors: list[str] = []
    docs = load(Path(path))
    by_kind: dict[str, list[dict]] = {}
    for fname, d in docs:
        for key in ("apiVersion", "kind"):
            if key not in d:
                errors.append(f"{fname}: missing {key}")
        if "name" not in d.get("metadata", {}):
            errors.append(f"{fname}: missing metadata.name")
        by_kind.setdefault(d.get("kind", "?"), []).append(d)

    deployments = {d["metadata"]["name"]: d for d in by_kind.get("Deployment", [])}

    for fname, d in docs:
        tpl = pod_specs(d)
        if not tpl:
            continue
        ident = f"{d['kind']}/{d['metadata']['name']}"
        labels = tpl["metadata"]["labels"]
        sel = d["spec"].get("selector", {}).get("matchLabels")
        if sel and not all(labels.get(k) == v for k, v in sel.items()):
            errors.append(f"{ident}: selector.matchLabels is not a subset of the pod template labels")
        pod = tpl["spec"]
        if not pod.get("securityContext", {}).get("runAsNonRoot"):
            errors.append(f"{ident}: pod securityContext.runAsNonRoot must be true")
        for c in pod["containers"]:
            cid = f"{ident}/{c['name']}"
            img = c["image"]
            if img.endswith(":latest") or ":" not in img.split("/")[-1]:
                errors.append(f"{cid}: image '{img}' must pin a tag (not :latest / untagged)")
            res = c.get("resources", {})
            if not res.get("limits") or not res.get("requests"):
                errors.append(f"{cid}: resources.requests and resources.limits are required")
            sc = c.get("securityContext", {})
            if sc.get("allowPrivilegeEscalation") is not False:
                errors.append(f"{cid}: allowPrivilegeEscalation must be false")
            if "ALL" not in sc.get("capabilities", {}).get("drop", []):
                errors.append(f"{cid}: must drop ALL capabilities")
            if d["kind"] in ("Deployment", "StatefulSet"):
                for probe in ("readinessProbe", "livenessProbe"):
                    if probe not in c:
                        errors.append(f"{cid}: missing {probe}")

    for hpa in by_kind.get("HorizontalPodAutoscaler", []):
        t = hpa["spec"]["scaleTargetRef"]
        if t["name"] not in deployments:
            errors.append(f"HPA/{hpa['metadata']['name']}: targets unknown Deployment '{t['name']}'")
        elif hpa["spec"]["minReplicas"] > hpa["spec"]["maxReplicas"]:
            errors.append(f"HPA/{hpa['metadata']['name']}: minReplicas > maxReplicas")

    for svc in by_kind.get("Service", []):
        sel = svc["spec"].get("selector", {})
        targets = [d for _, d in docs if d["kind"] in ("Deployment", "StatefulSet")]
        if sel and not any(all(t["spec"]["template"]["metadata"]["labels"].get(k) == v for k, v in sel.items())
                           for t in targets):
            errors.append(f"Service/{svc['metadata']['name']}: selector matches no workload")

    for pdb in by_kind.get("PodDisruptionBudget", []):
        sel = pdb["spec"]["selector"]["matchLabels"]
        if not any(all(d["spec"]["template"]["metadata"]["labels"].get(k) == v for k, v in sel.items())
                   for d in deployments.values()):
            errors.append(f"PDB/{pdb['metadata']['name']}: selector matches no Deployment")

    for ing in by_kind.get("Ingress", []):
        svc_names = {s["metadata"]["name"] for s in by_kind.get("Service", [])}
        for rule in ing["spec"]["rules"]:
            for p in rule["http"]["paths"]:
                if p["backend"]["service"]["name"] not in svc_names:
                    errors.append(f"Ingress/{ing['metadata']['name']}: backend service not defined")

    for e in errors:
        print("ERROR", e)
    print(f"{len(docs)} manifests checked, {len(errors)} problem(s)")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "infra/k8s"))
