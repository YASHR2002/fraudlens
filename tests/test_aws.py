"""Static checks of the AWS blueprint files (nothing here calls AWS)."""

from __future__ import annotations

import json
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).parents[1]
AWS = ROOT / "aws"


def mib(limit: str) -> int:
    value, unit = int(limit[:-1]), limit[-1].lower()
    return value * 1024 if unit == "g" else value


def test_compose_fits_a_t3_micro() -> None:
    spec = yaml.safe_load((ROOT / "docker-compose.aws.yml").read_text(encoding="utf-8"))
    services = spec["services"]
    assert set(services) == {"api", "prometheus", "grafana"}
    # 1 GiB of RAM, minus the OS and Docker: the containers' limits must leave room for them.
    assert sum(mib(s["mem_limit"]) for s in services.values()) <= 1024
    assert services["api"]["environment"]["MODEL_SOURCE"] == "s3"
    assert services["grafana"]["environment"]["GF_AUTH_ANONYMOUS_ENABLED"] == "false"
    assert all(s.get("restart") == "unless-stopped" for s in services.values())
    for config in ("docker/prometheus/prometheus.yml", "docker/grafana/provisioning"):
        assert (ROOT / config).exists()


def test_user_data_script() -> None:
    raw = (AWS / "user-data.sh").read_bytes()
    assert b"\r\n" not in raw, "user data must have LF line endings"
    text = raw.decode("utf-8")
    assert text.startswith("#!/bin/bash\n") and "set -euo pipefail" in text
    assert "/swapfile" in text and "count=2048" in text  # the 2 GiB swap file
    assert "docker-compose.aws.yml" in text and "X-aws-ec2-metadata-token" in text  # IMDSv2
    # The bucket naming convention is shared by the script, the IAM policy and the guide.
    assert "S3_BUCKET=fraudlens-${ACCOUNT_ID}-${REGION}" in text


def test_iam_policies_are_least_privilege() -> None:
    trust = json.loads((AWS / "ec2-trust-policy.json").read_text(encoding="utf-8"))
    assert trust["Statement"][0]["Principal"] == {"Service": "ec2.amazonaws.com"}
    policy = json.loads((AWS / "ec2-permissions-policy.json").read_text(encoding="utf-8"))
    actions = {a for st in policy["Statement"] for a in ([st["Action"]] if isinstance(
        st["Action"], str) else st["Action"])}  # fmt: skip
    assert not any(a.endswith("*") for a in actions)  # no wildcard actions
    assert all(a.split(":")[1].startswith(("Get", "List", "Batch")) for a in actions)
    wildcard = [st["Sid"] for st in policy["Statement"] if st["Resource"] == "*"]
    assert wildcard == ["EcrLogin"]  # ecr:GetAuthorizationToken cannot be scoped
    text = json.dumps(policy)
    assert set(re.findall(r"[A-Z_]{6,}", text)) <= {"ACCOUNT_ID", "REGION"}  # placeholders
    assert "fraudlens-ACCOUNT_ID-REGION" in text


def test_instance_disk_and_budget() -> None:
    (disk,) = json.loads((AWS / "block-devices.json").read_text(encoding="utf-8"))
    ebs = disk["Ebs"]
    assert ebs["VolumeSize"] == 20 and ebs["VolumeType"] == "gp3" and ebs["DeleteOnTermination"]
    budget = json.loads((AWS / "budget.json").read_text(encoding="utf-8"))
    assert budget["BudgetType"] == "COST" and float(budget["BudgetLimit"]["Amount"]) <= 10
    notes = json.loads((AWS / "budget-notifications.json").read_text(encoding="utf-8"))
    assert {n["Notification"]["NotificationType"] for n in notes} == {"ACTUAL", "FORECASTED"}
