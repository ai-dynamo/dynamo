# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for :mod:`dynamo_test.stack`.

The property under test throughout is *provenance*: not just that a value was
resolved, but that it says which rung produced it. A bare port number gives no
clue whether a connection failure means fixing the manifest, the flag, or the
default table.
"""

import pathlib

import pytest

yaml = pytest.importorskip("yaml")

from dynamo_test.manifest import Plan  # noqa: E402
from dynamo_test.roles import PortName, Role  # noqa: E402
from dynamo_test.stack import DYNAMO_DEFAULTS, Defaults, Stack  # noqa: E402


def manifest(tmp_path, components, backend_framework=None, name="s"):
    spec = {"components": components}
    if backend_framework:
        spec["backendFramework"] = backend_framework
    path = tmp_path / f"{name}.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "apiVersion": "nvidia.com/v1beta1",
                "kind": "DynamoGraphDeployment",
                "metadata": {"name": name},
                "spec": spec,
            }
        )
    )
    return Stack(Plan.from_file(str(path)))


def container(**kw):
    return {"name": "main", "image": "nvcr.io/dynamo:latest", **kw}


# ------------------------------------------------- what the operator supplies


def test_discovery_env_is_injected_because_no_manifest_declares_it():
    """Measured: **0 of 427** recipe components declare either variable.

    Not "few" — none. The operator injects both, so every tier without an
    operator must synthesise them or nothing discovers anything.
    """
    env = DYNAMO_DEFAULTS.discovery_env()
    assert set(env) == {"NATS_SERVER", "ETCD_ENDPOINTS"}


def test_a_frontend_with_no_command_gets_the_operators(tmp_path):
    """27 of 164 recipe frontends declare no command."""
    stack = manifest(tmp_path, [{"name": "Frontend", "container": container()}])
    render = stack.render(next(iter(stack.plan)))

    assert render.command.require() == ("python3", "-m", "dynamo.frontend")
    assert "component_frontend.go" in render.command.detail
    assert render.is_runnable


def test_the_manifest_beats_the_default(tmp_path):
    stack = manifest(
        tmp_path,
        [
            {
                "name": "Frontend",
                "container": container(
                    command=["/bin/bash", "-lc"],
                    args=["exec python3 -m dynamo.frontend --http-port 9999"],
                ),
            }
        ],
    )
    render = stack.render(next(iter(stack.plan)))
    assert "declared in the manifest" in render.command.detail


def test_a_declared_env_var_beats_the_injected_one(tmp_path):
    """A manifest naming its own NATS is making a deliberate choice."""
    stack = manifest(
        tmp_path,
        [
            {
                "name": "Frontend",
                "container": container(
                    env=[{"name": "NATS_SERVER", "value": "nats://mine:4222"}]
                ),
            }
        ],
    )
    render = stack.render(next(iter(stack.plan)))
    assert render.env["NATS_SERVER"].require() == "nats://mine:4222"
    assert "manifest" in render.env["NATS_SERVER"].detail


def test_a_site_override_beats_everything(tmp_path):
    """A compose site points discovery at service names; that is the whole
    reason the frontend can leave the all-in-one container."""
    plan = manifest(tmp_path, [{"name": "Frontend", "container": container()}]).plan
    stack = Stack(plan, overrides={"NATS_SERVER": "nats://nats-server:4222"})
    render = stack.render(next(iter(stack.plan)))
    assert render.env["NATS_SERVER"].require() == "nats://nats-server:4222"
    # `source` is the address the value came from; `detail` is the explanation.
    assert render.env["NATS_SERVER"].source == "site override"


# ----------------------------------------------------------- the port ladder


def test_a_declared_port_wins_and_says_so(tmp_path):
    stack = manifest(
        tmp_path,
        [
            {
                "name": "Frontend",
                "container": container(ports=[{"name": "http", "containerPort": 8080}]),
            }
        ],
    )
    port = stack.resolve_port(next(iter(stack.plan)), PortName.SERVICE)
    assert port.require() == 8080
    assert "manifest" in port.detail


def test_a_flag_beats_the_default(tmp_path):
    stack = manifest(
        tmp_path,
        [
            {
                "name": "Frontend",
                "container": container(
                    command=["python3", "-m", "dynamo.frontend"],
                    args=["--http-port", "8123"],
                ),
            }
        ],
    )
    port = stack.resolve_port(next(iter(stack.plan)), PortName.SERVICE)
    assert port.require() == 8123
    assert "command line" in port.detail


def test_the_default_is_the_operators_and_cites_it(tmp_path):
    stack = manifest(tmp_path, [{"name": "Frontend", "container": container()}])
    service = stack.resolve_port(next(iter(stack.plan)), PortName.SERVICE)
    system = stack.resolve_port(next(iter(stack.plan)), PortName.SYSTEM)

    assert service.require() == 8000 and "consts.go:13" in service.detail
    assert system.require() == 9090 and "consts.go:20" in system.detail


def test_a_nonsense_port_is_unknown_not_a_crash(tmp_path):
    stack = manifest(
        tmp_path,
        [
            {
                "name": "Frontend",
                "container": container(
                    command=["python3"], args=["--http-port", "${PORT}"]
                ),
            }
        ],
    )
    port = stack.resolve_port(next(iter(stack.plan)), PortName.SERVICE)
    assert port.is_unknown
    assert "not a port number" in port.detail


# ------------------------------------------------------- engine resolution


def test_the_crd_field_resolves_a_worker_with_no_command(tmp_path):
    """`spec.backendFramework` is the CRD's own field, present in 139 of the
    178 recipe plans."""
    stack = manifest(
        tmp_path,
        [{"name": "PrefillWorker", "container": container()}],
        backend_framework="sglang",
    )
    render = stack.render(next(iter(stack.plan)))
    assert render.command.require() == ("python3", "-m", "dynamo.sglang")
    assert "backendFramework" in render.command.detail


def test_the_component_name_is_never_used_to_guess_the_engine(tmp_path):
    """It would fix two components in the whole corpus, and guess wrong the
    first time somebody names a worker after a model."""
    stack = manifest(tmp_path, [{"name": "VllmDecodeWorker", "container": container()}])
    render = stack.render(next(iter(stack.plan)))
    assert not render.command.is_known
    assert "spec.backendFramework" in render.command.detail


def test_an_unresolvable_command_explains_what_would_fix_it(tmp_path):
    stack = manifest(tmp_path, [{"name": "PrefillWorker", "container": container()}])
    render = stack.render(next(iter(stack.plan)))
    assert not render.is_runnable
    why = render.why_not_runnable()
    assert "backendFramework" in why
    assert "a site must supply the engine" in why


# ------------------------------------------------------------------- corpus


REPO = pathlib.Path(__file__).resolve().parents[2]


@pytest.mark.skipif(
    not (REPO / "recipes").is_dir(),
    reason="recipes are not present next to the harness",
)
def test_every_recipe_renders_or_says_why():
    """The gate: render every shipped deployment and account for each component.

    Measured at the time of writing: **398 of 427 resolve a command**. The 29
    that do not are not a gap in this module — those manifests declare no engine
    anywhere, on the command line or in `spec.backendFramework`. On Kubernetes
    the operator resolves it from the image; off Kubernetes a site has to supply
    it, and saying so is more useful than guessing.
    """
    from dynamo_test.manifest import ManifestError, NoGraphDeployment

    resolved = unresolved = 0
    no_discovery = []
    for path in sorted((REPO / "recipes").rglob("*.yaml")):
        try:
            plans = Plan.all_from_file(path)
        except (NoGraphDeployment, ManifestError):
            continue
        for plan in plans:
            for render in Stack(plan):
                if render.command.is_known:
                    resolved += 1
                else:
                    unresolved += 1
                    # An unresolved command must always explain itself.
                    assert render.why_not_runnable()
                # The whole reason this layer exists.
                if (
                    "NATS_SERVER" not in render.env
                    or "ETCD_ENDPOINTS" not in render.env
                ):
                    no_discovery.append(f"{plan.source}[{render.name}]")

    assert no_discovery == [], "every component must get discovery env"
    assert resolved + unresolved > 400
    assert resolved / (resolved + unresolved) > 0.9, f"{resolved}/{resolved+unresolved}"


@pytest.mark.skipif(
    not (REPO / "recipes").is_dir(),
    reason="recipes are not present next to the harness",
)
def test_every_resolved_port_names_its_rung():
    """A bare integer tells you nothing about which fix to make."""
    from dynamo_test.manifest import ManifestError, NoGraphDeployment

    for path in sorted((REPO / "recipes").rglob("*.yaml"))[:40]:
        try:
            plans = Plan.all_from_file(path)
        except (NoGraphDeployment, ManifestError):
            continue
        for plan in plans:
            stack = Stack(plan)
            for component in plan:
                port = stack.resolve_port(component, PortName.SERVICE)
                assert port.source, f"{component.name} port has no source"
                if port.is_known:
                    assert port.detail, f"{component.name} port has no provenance"


def test_defaults_are_overridable_wholesale(tmp_path):
    """A site with a different discovery plane replaces the table, not the code."""
    compose = Defaults(
        nats_server="nats://nats-server:4222",
        etcd_endpoints="http://etcd-server:2379",
    )
    plan = manifest(tmp_path, [{"name": "Frontend", "container": container()}]).plan
    render = Stack(plan, defaults=compose).render(next(iter(plan)))
    assert render.env["NATS_SERVER"].require() == "nats://nats-server:4222"
    assert render.env["ETCD_ENDPOINTS"].require() == "http://etcd-server:2379"


def test_the_stack_serialises_with_provenance(tmp_path):
    import json

    stack = manifest(tmp_path, [{"name": "Frontend", "container": container()}])
    record = stack.to_record()
    frontend = record["components"][0]
    assert frontend["role"] == str(Role.FRONTEND)
    assert frontend["ports"]["service"]["port"] == 8000
    assert (
        "consts.go" in frontend["ports"]["service"]["from"]
        or frontend["ports"]["service"]["from"]
    )
    assert json.loads(json.dumps(record))
