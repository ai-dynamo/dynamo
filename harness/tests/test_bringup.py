# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for :mod:`dynamo_test.bringup`, against a real subprocess.

The test that matters is the **negative control**: attaching to a system that is
running the wrong thing must fail at bind time. Without it, VERIFY is a mode that
always succeeds, which is worse than not having it.
"""

import socket
import sys
import time
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

from dynamo_test.bringup import bring_up, refuse_if_not_owned  # noqa: E402
from dynamo_test.manifest import Plan  # noqa: E402
from dynamo_test.providers import LocalProvider, LocalRole  # noqa: E402
from dynamo_test.roles import Role, at  # noqa: E402
from dynamo_test.site import Mismatch, Ownership, Refused, Site, Topology  # noqa: E402
from dynamo_test.stack import Stack  # noqa: E402

STUB = Path(__file__).parent / "stub_frontend.py"

MODEL = "Qwen/Qwen3-0.6B"
OTHER_MODEL = "meta-llama/Llama-3.1-8B"


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def site(ownership: Ownership) -> Site:
    return Site(
        name=f"t-{ownership.value}",
        topology=Topology.ALL_IN_ONE,
        ownership=ownership,
        shares_process_tree=True,
        shares_network=True,
        substrate_owned=ownership is Ownership.IMPOSE,
    )


@pytest.fixture
def env(tmp_path):
    """A one-role stack and a provider that can actually start it."""
    port = free_port()
    manifest = {
        "apiVersion": "nvidia.com/v1beta1",
        "kind": "DynamoGraphDeployment",
        "metadata": {"name": "bringup-test"},
        "spec": {
            "components": [
                {
                    "name": "Frontend",
                    "container": {
                        "name": "main",
                        "image": "stub",
                        "command": ["python3"],
                        "args": ["-m", "dynamo.frontend", "--model", MODEL],
                    },
                }
            ]
        },
    }
    path = tmp_path / "deploy.yaml"
    path.write_text(yaml.safe_dump(manifest))
    stack = Stack(Plan.from_file(str(path)))

    def provider_for(model: str) -> LocalProvider:
        return LocalProvider(
            {
                Role.FRONTEND: LocalRole(
                    role=Role.FRONTEND,
                    argv=[
                        sys.executable,
                        str(STUB),
                        "--port",
                        str(port),
                        "--model",
                        model,
                    ],
                    port=port,
                )
            },
            log_dir=tmp_path / "logs",
        )

    providers: list[LocalProvider] = []

    def make(model=MODEL):
        p = provider_for(model)
        providers.append(p)
        return p

    yield stack, make
    for p in providers:
        p.shutdown()


def wait_serving(provider, timeout=30):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if provider.request(at("frontend"), "/v1/models", timeout=2).is_known:
            return True
        time.sleep(0.1)
    return False


# ------------------------------------------------------------------ IMPOSE


def test_impose_starts_and_owns(env):
    stack, make = env
    provider = make()

    result = bring_up(stack, site(Ownership.IMPOSE), provider)

    assert result.owns is True
    assert result.started == ("frontend",)
    assert wait_serving(provider)
    assert "(owned)" in result.describe()


# ------------------------------------------------------------------- ADOPT


def test_adopt_starts_what_is_missing_but_owns_nothing(env):
    """Transcribed from the MinIO precedent: probe first, never destroy."""
    stack, make = env
    provider = make()

    result = bring_up(stack, site(Ownership.ADOPT), provider)

    assert result.started == ("frontend",)
    assert result.owns is False, "ADOPT promises never to destroy"
    assert "teardown must not destroy it" in result.describe()


def test_adopt_joins_something_already_running(env):
    stack, make = env
    provider = make()
    provider.start(at("frontend"))
    assert wait_serving(provider)

    result = bring_up(stack, site(Ownership.ADOPT), provider)

    assert result.adopted == ("frontend",)
    assert result.started == ()
    assert result.owns is False


# ------------------------------------------------------------------ VERIFY


def test_verify_binds_to_a_matching_system(env):
    stack, make = env
    provider = make(MODEL)
    provider.start(at("frontend"))
    assert wait_serving(provider)

    result = bring_up(stack, site(Ownership.VERIFY), provider, timeout=20)

    assert result.checked == ("frontend",)
    assert result.owns is False
    assert result.divergences == ()


def test_verify_fails_at_bind_when_the_model_is_wrong(env):
    """**The negative control.** This is the test that makes VERIFY worth having.

    The system is up and healthy — it is simply serving something else. Without
    a comparison at bind time the first symptom is a 404 at query time, which
    reads like a routing bug and sends the reader to the wrong place entirely.
    """
    stack, make = env  # the stack declares Qwen/Qwen3-0.6B
    provider = make(OTHER_MODEL)  # ...the running system serves Llama
    provider.start(at("frontend"))
    assert wait_serving(provider), "the wrong system must still be healthy"

    with pytest.raises(Mismatch) as exc:
        bring_up(stack, site(Ownership.VERIFY), provider, timeout=20)

    message = str(exc.value)
    assert "frontend.model" in message
    assert MODEL in message and OTHER_MODEL in message
    assert "1 divergence" in message


def test_verify_reports_nothing_running_as_a_divergence(env):
    """Distinct from "running the wrong thing", and it must not be silent."""
    stack, make = env
    provider = make()  # never started

    with pytest.raises(Mismatch) as exc:
        bring_up(stack, site(Ownership.VERIFY), provider, timeout=1)
    assert "replicas" in str(exc.value)


def test_an_unreachable_frontend_is_unverified_not_a_disagreement(env):
    """A connection failure is not evidence that the model is wrong.

    Scoring it as a divergence would blame the deployment for a network problem.
    """
    from dynamo_test.bringup import observe

    stack, make = env
    provider = make()
    got = observe(provider, at("frontend"), timeout=0)
    assert got.is_unknown
    assert "unreachable" in got.detail


# --------------------------------------------------------------- refusal


def test_teardown_is_refused_when_the_run_did_not_create_it(env):
    """The quiet no-op is the hazard; this raises instead."""
    for ownership in (Ownership.VERIFY, Ownership.ADOPT):
        with pytest.raises(Refused) as exc:
            refuse_if_not_owned(site(ownership), "stop")
        assert "does not own" in str(exc.value)

    # IMPOSE created it, so tearing it down is exactly right.
    refuse_if_not_owned(site(Ownership.IMPOSE), "stop")


def test_the_result_serialises_for_the_run_record(env):
    import json

    stack, make = env
    result = bring_up(stack, site(Ownership.IMPOSE), make())
    record = result.to_record()
    assert record["ownership"] == "impose"
    assert record["owns"] is True
    assert json.loads(json.dumps(record))
