# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for schema-aware DynamoGraphDeployment helpers."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import aiohttp
import httpx
import kr8s
import pytest
import requests
import yaml

from tests.deploy.dgd_utils import DeploymentSpec, ManagedDeployment

pytestmark = [pytest.mark.unit, pytest.mark.pre_merge, pytest.mark.gpu_0]


def managed_deployment(tmp_path) -> ManagedDeployment:
    spec = SimpleNamespace(name="test-deployment", services=[], api_version="v1beta1")
    return ManagedDeployment(
        log_dir=str(tmp_path),
        deployment_spec=spec,  # type: ignore[arg-type]
        namespace="default",
    )


def test_logging_config_reads_existing_v1beta1_env(tmp_path) -> None:
    """Recognize JSONL logging already declared in a v1beta1 manifest."""
    manifest = {
        "apiVersion": "nvidia.com/v1beta1",
        "kind": "DynamoGraphDeployment",
        "metadata": {"name": "logging-test"},
        "spec": {
            "components": [],
            "env": [{"name": "DYN_LOGGING_JSONL", "value": "1"}],
        },
    }
    manifest_path = tmp_path / "deploy.yaml"
    manifest_path.write_text(yaml.safe_dump(manifest))

    deployment_spec = DeploymentSpec(str(manifest_path))

    assert deployment_spec.get_logging_config()["jsonl_enabled"] is True


@pytest.mark.parametrize(
    "connection_error",
    [
        httpx.RemoteProtocolError("tunnel disconnected"),
        kr8s.APITimeoutError("vCluster request timed out"),
    ],
)
def test_get_pods_retries_transient_vcluster_disconnect(
    monkeypatch, tmp_path, connection_error
) -> None:
    deployment = managed_deployment(tmp_path)
    pod = MagicMock()
    get_pods = MagicMock(side_effect=[connection_error, [pod]])
    sleep = MagicMock()
    monkeypatch.setattr("tests.deploy.dgd_utils.kr8s.get", get_pods)
    monkeypatch.setattr("tests.deploy.vcluster_utils.time.sleep", sleep)

    result = deployment.get_pods(["Frontend"])

    assert result == {"Frontend": [pod]}
    assert get_pods.call_count == 2
    sleep.assert_called_once_with(5)


async def test_delete_deployment_retries_vcluster_connection_failure(
    monkeypatch, tmp_path
) -> None:
    deployment = managed_deployment(tmp_path)
    connection_key = MagicMock(host="127.0.0.1", port=8443, ssl=True)
    deployment._custom_api = MagicMock()
    deployment._custom_api.delete_namespaced_custom_object = AsyncMock(
        side_effect=[
            aiohttp.ClientConnectorError(
                connection_key,
                ConnectionRefusedError(111, "vCluster tunnel unavailable"),
            ),
            None,
        ]
    )
    sleep = AsyncMock()
    monkeypatch.setattr("tests.deploy.vcluster_utils.asyncio.sleep", sleep)

    await deployment._delete_deployment()

    assert deployment._custom_api.delete_namespaced_custom_object.await_count == 2
    sleep.assert_awaited_once_with(5)


@pytest.mark.parametrize(
    "cleanup_error",
    [
        httpx.ConnectError("tunnel unavailable"),
        kr8s.APITimeoutError("vCluster request timed out"),
    ],
)
async def test_context_exit_preserves_original_error_when_cleanup_fails(
    tmp_path, cleanup_error
) -> None:
    deployment = managed_deployment(tmp_path)
    deployment._cleanup = AsyncMock(side_effect=cleanup_error)

    result = await deployment.__aexit__(ValueError, ValueError("test failed"), None)

    assert result is False


async def test_context_enter_preserves_setup_error_when_cleanup_times_out(
    tmp_path,
) -> None:
    deployment = managed_deployment(tmp_path)
    deployment._init_kubernetes = AsyncMock(side_effect=ValueError("setup failed"))
    deployment._cleanup = AsyncMock(
        side_effect=kr8s.APITimeoutError("vCluster request timed out")
    )

    with pytest.raises(ValueError, match="setup failed"):
        await deployment.__aenter__()


async def test_context_exit_reraises_unexpected_cleanup_error(tmp_path) -> None:
    deployment = managed_deployment(tmp_path)
    deployment._cleanup = AsyncMock(side_effect=RuntimeError("cleanup defect"))

    with pytest.raises(RuntimeError, match="cleanup defect"):
        await deployment.__aexit__(ValueError, ValueError("test failed"), None)


@pytest.mark.parametrize(
    "transport_error",
    [
        requests.ConnectionError("forward dropped"),
        requests.Timeout("forward stalled"),
    ],
)
def test_request_rebuilds_port_forward_after_transport_failure(
    monkeypatch, tmp_path, transport_error
) -> None:
    deployment = managed_deployment(tmp_path)
    original_port_forward = MagicMock(local_port=31001)
    replacement_port_forward = MagicMock(local_port=31002)
    deployment.port_forward = MagicMock(return_value=replacement_port_forward)
    response = MagicMock(spec=requests.Response)
    request_sender = MagicMock(side_effect=[transport_error, response])
    sleep = MagicMock()
    monkeypatch.setattr("tests.deploy.dgd_utils.time.sleep", sleep)

    result = deployment.send_request_with_port_forward_retry(
        pod=MagicMock(),
        remote_port=8000,
        endpoint="/v1/chat/completions",
        payload={"model": "test"},
        timeout=120,
        port_forward=original_port_forward,
        request_sender=request_sender,
    )

    assert result is response
    assert (
        request_sender.call_args_list[0].args[0].startswith("http://localhost:31001/")
    )
    assert (
        request_sender.call_args_list[1].args[0].startswith("http://localhost:31002/")
    )
    original_port_forward.stop.assert_called_once_with()
    deployment.port_forward.assert_called_once()
    sleep.assert_called_once_with(5)


def test_request_reraises_unexpected_port_forward_stop_error(tmp_path) -> None:
    deployment = managed_deployment(tmp_path)
    original_port_forward = MagicMock(local_port=31001)
    original_port_forward.stop.side_effect = ValueError("stop defect")
    request_sender = MagicMock(side_effect=requests.ConnectionError("forward dropped"))

    with pytest.raises(ValueError, match="stop defect"):
        deployment.send_request_with_port_forward_retry(
            pod=MagicMock(),
            remote_port=8000,
            endpoint="/v1/chat/completions",
            payload={"model": "test"},
            timeout=120,
            port_forward=original_port_forward,
            request_sender=request_sender,
        )


async def test_in_flight_restart_preserves_bounded_previous_log(tmp_path) -> None:
    """Keep a bounded previous-instance log before Kubernetes rotates again."""
    deployment = ManagedDeployment(
        log_dir=str(tmp_path),
        deployment_spec=SimpleNamespace(name="test-dgd"),
        namespace="default",
    )
    terminated = SimpleNamespace(reason="Error", exit_code=1)
    container_status = SimpleNamespace(
        name="main",
        restart_count=1,
        last_state=SimpleNamespace(terminated=terminated),
    )
    pod = SimpleNamespace(
        metadata=SimpleNamespace(name="worker-0"),
        status=SimpleNamespace(container_statuses=[container_status]),
    )
    deployment._core_api = SimpleNamespace(
        list_namespaced_pod=AsyncMock(return_value=SimpleNamespace(items=[pod])),
        read_namespaced_pod_log=AsyncMock(
            return_value="first line\nsecond line\nthird line\n"
        ),
    )

    warnings = await deployment._dump_in_flight_restart_logs(prev_log_tail_lines=2)

    assert len(warnings) == 1
    assert "first line" not in warnings[0]
    assert "second line" in warnings[0]
    assert "third line" in warnings[0]
    preserved = tmp_path / "restarts" / "worker-0.main.restart-1.previous.log"
    assert preserved.read_text() == "first line\nsecond line\nthird line\n"
    deployment._core_api.read_namespaced_pod_log.assert_awaited_once_with(
        name="worker-0",
        namespace="default",
        container="main",
        previous=True,
        tail_lines=50000,
    )


def test_multi_document_manifest_selects_the_graph_deployment(tmp_path) -> None:
    """Recipe manifests bundle the DGD with ConfigMaps and friends.

    Most files under ``recipes/`` are multi-document; loading them with
    ``yaml.safe_load`` raises ComposerError, which previously made the majority
    of the recipe corpus unusable with DeploymentSpec.
    """
    config_map = {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {"name": "engine-config"},
        "data": {"engine.yaml": "tensor_parallel_size: 1\n"},
    }
    graph_deployment = {
        "apiVersion": "nvidia.com/v1beta1",
        "kind": "DynamoGraphDeployment",
        "metadata": {"name": "multi-doc-test"},
        "spec": {"components": []},
    }
    manifest_path = tmp_path / "deploy.yaml"
    manifest_path.write_text(yaml.safe_dump_all([config_map, graph_deployment]))

    deployment_spec = DeploymentSpec(str(manifest_path))

    assert deployment_spec.name == "multi-doc-test"
    assert deployment_spec.schema == "v1beta1"


def test_manifest_without_a_graph_deployment_is_rejected(tmp_path) -> None:
    """A manifest carrying no DGD must fail loudly, not silently pick a Service."""
    manifest_path = tmp_path / "deploy.yaml"
    manifest_path.write_text(
        yaml.safe_dump_all(
            [
                {"apiVersion": "v1", "kind": "Service", "metadata": {"name": "svc"}},
                {"apiVersion": "v1", "kind": "ConfigMap", "metadata": {"name": "cm"}},
            ]
        )
    )

    with pytest.raises(ValueError, match="no DynamoGraphDeployment"):
        DeploymentSpec(str(manifest_path))


def _multi_document_manifest(tmp_path):
    """A recipe shaped like the 97 that bundle prerequisites.

    The DGD references the ConfigMap by name, so applying the DGD alone leaves
    the worker in CreateContainerConfigError.
    """
    documents = [
        {
            "apiVersion": "v1",
            "kind": "ConfigMap",
            "metadata": {"name": "engine-config"},
            "data": {"prefill.yaml": "model_path: Qwen/Qwen3-0.6B"},
        },
        {
            "apiVersion": "resource.nvidia.com/v1beta1",
            "kind": "ComputeDomain",
            "metadata": {"name": "test-compute-domain"},
            "spec": {"numNodes": 2},
        },
        {
            "apiVersion": "nvidia.com/v1beta1",
            "kind": "DynamoGraphDeployment",
            "metadata": {"name": "recipe-under-test"},
            "spec": {
                "components": [
                    {
                        "name": "Worker",
                        "podTemplate": {
                            "spec": {
                                "containers": [
                                    {
                                        "name": "main",
                                        "command": ["/bin/bash", "-lc"],
                                        "args": ["exec python3 -m dynamo.vllm"],
                                        "volumeMounts": [
                                            {
                                                "name": "engine-config",
                                                "mountPath": "/etc/engine",
                                            }
                                        ],
                                    }
                                ],
                                "volumes": [
                                    {
                                        # Named after the ConfigMap it mounts, as
                                        # 22 volumes across ``recipes/`` are. A
                                        # fixture that names it something else
                                        # cannot exhibit the collision that
                                        # test_uniquify_leaves_pod_local_volume_names_alone
                                        # exists to pin.
                                        "name": "engine-config",
                                        "configMap": {"name": "engine-config"},
                                    }
                                ],
                            }
                        },
                    }
                ]
            },
        },
    ]
    path = tmp_path / "deploy.yaml"
    path.write_text(yaml.safe_dump_all(documents))
    return path


def test_companion_documents_are_retained_not_discarded(tmp_path) -> None:
    """97 of 178 recipes bundle a resource their own DGD names.

    Selecting the DGD and dropping the rest makes the manifest loadable but
    incomplete: the operator creates the deployment and its workers then wait
    for a ConfigMap nothing ever created.
    """
    spec = DeploymentSpec(str(_multi_document_manifest(tmp_path)))

    assert spec.name == "recipe-under-test"
    assert [d["kind"] for d in spec.companions] == ["ConfigMap", "ComputeDomain"]
    # The DGD itself is never among them.
    assert all(d["kind"] != "DynamoGraphDeployment" for d in spec.companions)


def test_companions_keep_their_file_order(tmp_path) -> None:
    """A ComputeDomain must exist before the workers that claim it."""
    spec = DeploymentSpec(str(_multi_document_manifest(tmp_path)))
    names = [d["metadata"]["name"] for d in spec.companions]
    assert names == ["engine-config", "test-compute-domain"]


def test_a_single_document_manifest_has_no_companions(tmp_path) -> None:
    manifest = {
        "apiVersion": "nvidia.com/v1beta1",
        "kind": "DynamoGraphDeployment",
        "metadata": {"name": "solo"},
        "spec": {"components": []},
    }
    path = tmp_path / "solo.yaml"
    path.write_text(yaml.safe_dump(manifest))

    assert DeploymentSpec(str(path)).companions == []


async def test_companions_are_applied_before_the_deployment(tmp_path) -> None:
    """Order is the whole point: the DGD references them by name."""
    spec = DeploymentSpec(str(_multi_document_manifest(tmp_path)))
    deployment = ManagedDeployment(
        log_dir=str(tmp_path),
        deployment_spec=spec,
        namespace="default",
    )

    applied: list = []
    deployment._kubectl = lambda verb, docs, *extra: applied.append(
        (verb, [d["kind"] for d in docs])
    )
    deployment._custom_api = SimpleNamespace(
        create_namespaced_custom_object=AsyncMock(return_value=None)
    )

    await deployment._create_deployment()

    assert applied == [("apply", ["ConfigMap", "ComputeDomain"])]
    deployment._custom_api.create_namespaced_custom_object.assert_awaited_once()


async def test_a_failed_companion_apply_stops_the_deployment(tmp_path) -> None:
    """Creating the DGD anyway would leave a worker stuck on a missing resource,
    which reads as a deployment timeout rather than a manifest problem."""
    spec = DeploymentSpec(str(_multi_document_manifest(tmp_path)))
    deployment = ManagedDeployment(
        log_dir=str(tmp_path),
        deployment_spec=spec,
        namespace="default",
    )
    deployment._kubectl = lambda *a, **k: SimpleNamespace(
        returncode=1, stderr="forbidden", stdout=""
    )
    deployment._custom_api = SimpleNamespace(
        create_namespaced_custom_object=AsyncMock(return_value=None)
    )

    with pytest.raises(RuntimeError, match="companion resources"):
        await deployment._create_deployment()
    deployment._custom_api.create_namespaced_custom_object.assert_not_awaited()


async def test_companions_are_removed_after_the_deployment(tmp_path) -> None:
    """Deleted after the DGD, so nothing is still mounting them."""
    spec = DeploymentSpec(str(_multi_document_manifest(tmp_path)))
    deployment = ManagedDeployment(
        log_dir=str(tmp_path),
        deployment_spec=spec,
        namespace="default",
    )
    calls: list = []
    deployment._kubectl = lambda verb, docs, *extra: calls.append((verb, extra))
    deployment._deployment_name = "recipe-under-test"
    deployment._custom_api = SimpleNamespace(
        delete_namespaced_custom_object=AsyncMock(return_value=None)
    )

    await deployment._delete_deployment()

    assert calls == [("delete", ("--ignore-not-found=true", "--wait=false"))]


async def test_teardown_survives_a_failed_companion_delete(tmp_path) -> None:
    """Teardown must not mask the failure the test was reporting."""
    spec = DeploymentSpec(str(_multi_document_manifest(tmp_path)))
    deployment = ManagedDeployment(
        log_dir=str(tmp_path),
        deployment_spec=spec,
        namespace="default",
    )

    def boom(*a, **k):
        raise OSError("kubectl not found")

    deployment._kubectl = boom
    deployment._deployment_name = "recipe-under-test"
    deployment._custom_api = SimpleNamespace(
        delete_namespaced_custom_object=AsyncMock(return_value=None)
    )

    await deployment._delete_deployment()  # must not raise


def test_a_placeholder_namespace_is_retargeted(tmp_path) -> None:
    """Four companions in the corpus declare `namespace: <your-namespace>`.

    That is neither a valid namespace name nor the one under test, and kubectl
    rejects the whole apply with a namespace mismatch rather than just that
    document.
    """
    documents = [
        {
            "apiVersion": "resource.nvidia.com/v1beta1",
            "kind": "ComputeDomain",
            "metadata": {"name": "cd", "namespace": "<your-namespace>"},
        },
        {
            "apiVersion": "nvidia.com/v1beta1",
            "kind": "DynamoGraphDeployment",
            "metadata": {"name": "d"},
            "spec": {"components": []},
        },
    ]
    path = tmp_path / "deploy.yaml"
    path.write_text(yaml.safe_dump_all(documents))

    deployment = ManagedDeployment(
        log_dir=str(tmp_path),
        deployment_spec=DeploymentSpec(str(path)),
        namespace="under-test",
    )
    companion = deployment.deployment_spec.companions[0]

    assert deployment._retarget(companion)["metadata"]["namespace"] == "under-test"
    # The loaded spec is not mutated; -n and the document stay consistent.
    assert companion["metadata"]["namespace"] == "<your-namespace>"


def test_a_companion_without_a_namespace_is_left_alone(tmp_path) -> None:
    """Cluster-scoped kinds must not be given a namespace they cannot have."""
    deployment = ManagedDeployment(
        log_dir=str(tmp_path),
        deployment_spec=DeploymentSpec(str(_multi_document_manifest(tmp_path))),
        namespace="under-test",
    )
    configmap = deployment.deployment_spec.companions[0]

    assert "namespace" not in deployment._retarget(configmap)["metadata"]


# ---------------------------------------------------------------------------
# Per-run isolation of companion resources
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_uniquify_renames_every_companion_not_just_the_deployment(tmp_path) -> None:
    """Suffixing only the DGD leaves the companions colliding.

    `_create_companions` applies them by name and `_delete_companions` deletes
    them by name, from `_delete_deployment` -- which `__aenter__` calls *before*
    creating. Two runs sharing a companion name therefore overwrite each other's
    content on the way up and delete each other's prerequisites on the way down.
    """
    spec = DeploymentSpec(str(_multi_document_manifest(tmp_path)))

    renames = spec.uniquify("-tx-abc123")

    assert spec.name == "recipe-under-test-tx-abc123"
    assert [d["metadata"]["name"] for d in spec.companions] == [
        "engine-config-tx-abc123",
        "test-compute-domain-tx-abc123",
    ]
    assert renames["engine-config"] == "engine-config-tx-abc123"


def test_uniquify_rewrites_the_references_that_point_at_companions(
    tmp_path,
) -> None:
    """A renamed ConfigMap whose reference still names the old one is worse than
    no rename at all: the worker sits in CreateContainerConfigError instead."""
    spec = DeploymentSpec(str(_multi_document_manifest(tmp_path)))

    spec.uniquify("-tx-abc123")

    volumes = spec.spec()["spec"]["components"][0]["podTemplate"]["spec"]["volumes"]
    assert volumes[0]["configMap"]["name"] == "engine-config-tx-abc123"


def test_uniquify_leaves_pod_local_volume_names_alone(tmp_path) -> None:
    """`volumes[].name` names the volume, not the ConfigMap.

    It only has to match `volumeMounts[].name` inside the same pod. 22 volumes
    across ``recipes/`` are nevertheless named after the ConfigMap they mount,
    so a rename keyed on the *name* rather than on the reference key would
    rewrite it -- and the hazard is the **partial** case, where one side of the
    pairing moves and the other does not. (A substitution consistent enough to
    rewrite both sides happens to be harmless here; measured, it kills no test
    in this module. The partial one is caught by
    ``test_uniquify_leaves_every_volume_mount_resolvable_across_the_corpus``.)
    """
    spec = DeploymentSpec(str(_multi_document_manifest(tmp_path)))

    spec.uniquify("-tx-abc123")

    pod = spec.spec()["spec"]["components"][0]["podTemplate"]["spec"]
    # The ConfigMap moved; the volume that mounts it did not.
    assert pod["volumes"][0]["configMap"]["name"] == "engine-config-tx-abc123"
    assert pod["volumes"][0]["name"] == "engine-config"
    assert pod["containers"][0]["volumeMounts"][0]["name"] == "engine-config"


def test_uniquify_renames_the_compute_domain_channel_template(tmp_path) -> None:
    """A pod never names the ComputeDomain; it names the channel template.

    All 28 ComputeDomains in `recipes/` are wired this way, so renaming only
    `metadata.name` would leave every run creating the same cluster-scoped
    ResourceClaimTemplate -- the collision the rename exists to prevent.
    """
    documents = [
        {
            "apiVersion": "resource.nvidia.com/v1beta1",
            "kind": "ComputeDomain",
            "metadata": {"name": "cd"},
            "spec": {
                "numNodes": 0,
                "channel": {"resourceClaimTemplate": {"name": "cd-channel"}},
            },
        },
        {
            "apiVersion": "nvidia.com/v1beta1",
            "kind": "DynamoGraphDeployment",
            "metadata": {"name": "dgd"},
            "spec": {
                "components": [
                    {
                        "name": "Worker",
                        "podTemplate": {
                            "spec": {
                                "containers": [{"name": "main"}],
                                "resourceClaims": [
                                    {
                                        "name": "compute-domain-channel",
                                        "resourceClaimTemplateName": "cd-channel",
                                    }
                                ],
                            }
                        },
                    }
                ]
            },
        },
    ]
    path = tmp_path / "deploy.yaml"
    path.write_text(yaml.safe_dump_all(documents))
    spec = DeploymentSpec(str(path))

    spec.uniquify("-tx-abc123")

    channel = spec.companions[0]["spec"]["channel"]["resourceClaimTemplate"]
    assert channel["name"] == "cd-channel-tx-abc123"
    claims = spec.spec()["spec"]["components"][0]["podTemplate"]["spec"][
        "resourceClaims"
    ]
    assert claims[0]["resourceClaimTemplateName"] == "cd-channel-tx-abc123"
    # The claim's pod-local name is not a cluster resource and stays put.
    assert claims[0]["name"] == "compute-domain-channel"


def _recipe_manifests():
    recipes = REPO_ROOT / "recipes"
    if not recipes.is_dir():
        pytest.skip("recipes/ not present in this checkout")
    for path in sorted(recipes.rglob("*.yaml")):
        try:
            documents = [
                d for d in yaml.safe_load_all(path.read_text()) if isinstance(d, dict)
            ]
        except yaml.YAMLError:
            continue
        if len([d for d in documents if d.get("kind") == "DynamoGraphDeployment"]) == 1:
            yield path


def _owned_names(spec):
    """Every cluster-scoped name the manifest's companions bring into existence.

    Includes the ResourceClaimTemplate a ComputeDomain's controller creates for
    its channel, which is named by the manifest and so collides like any other.
    """
    names = set()
    for document in spec.companions:
        names.add((document.get("metadata") or {}).get("name"))
        if document.get("kind") == "ComputeDomain":
            names.add(
                (
                    ((document.get("spec") or {}).get("channel") or {}).get(
                        "resourceClaimTemplate"
                    )
                    or {}
                ).get("name")
            )
    names.discard(None)
    return names


def _iter_reference_values(node, path="") -> list:
    """Every companion reference in a DGD, as (path, value)."""
    found = []
    if isinstance(node, dict):
        for key, value in node.items():
            if key in ("configMap", "configMapRef", "configMapKeyRef") and isinstance(
                value, dict
            ):
                if isinstance(value.get("name"), str):
                    found.append((f"{path}.{key}.name", value["name"]))
            elif key == "resourceClaimTemplateName" and isinstance(value, str):
                found.append((f"{path}.{key}", value))
            else:
                found.extend(_iter_reference_values(value, f"{path}.{key}"))
    elif isinstance(node, list):
        for index, item in enumerate(node):
            found.extend(_iter_reference_values(item, f"{path}[{index}]"))
    return found


def _iter_pod_specs(node) -> list:
    """Every mapping that carries both `volumes` and `containers`."""
    found = []
    if isinstance(node, dict):
        if "volumes" in node and isinstance(node.get("volumes"), list):
            found.append(node)
        for value in node.values():
            found.extend(_iter_pod_specs(value))
    elif isinstance(node, list):
        for item in node:
            found.extend(_iter_pod_specs(item))
    return found


def test_uniquify_suffixes_every_owned_name_across_the_recipe_corpus() -> None:
    """Replay the rename over every real recipe, not a synthetic stand-in.

    A hand-written manifest only proves the shapes its author thought of. The
    corpus is what this is pointed at: 91 ConfigMaps, 28 ComputeDomains and 2
    ResourceClaimTemplates across ~101 manifests. Every cluster-scoped name any
    of them creates has to carry the per-run suffix, or two runs collide on it.
    """
    checked = 0
    for path in _recipe_manifests():
        spec = DeploymentSpec(str(path))
        if not spec.companions:
            continue
        checked += 1
        before = _owned_names(spec)

        renames = spec.uniquify("-tx-abc123")

        for name in before:
            assert name in renames, f"{path}: {name!r} was not renamed"
            assert renames[name] == f"{name}-tx-abc123"
        assert _owned_names(spec) == {f"{n}-tx-abc123" for n in before}

    assert checked >= 90, f"expected ~101 manifests with companions, got {checked}"


def test_uniquify_keeps_every_companion_reference_resolving_across_the_corpus() -> None:
    """A rename that leaves a reference behind is worse than no rename.

    The worker sits in CreateContainerConfigError minutes later instead of
    failing at apply time. Two properties, over the whole corpus: a reference
    that named a bundled companion now names its renamed form, and a reference
    to anything else is untouched. The second matters -- some recipes reference
    resources they do not bundle (`kimi-k2.5/trtllm/agg-eagle-kv-router`
    references a `your-compute-domain-channel` placeholder the operator
    supplies), and rewriting those would invent a dangling name.
    """
    for path in _recipe_manifests():
        spec = DeploymentSpec(str(path))
        if not spec.companions:
            continue
        owned = _owned_names(spec)
        before = dict(_iter_reference_values(spec.spec()))

        spec.uniquify("-tx-abc123")

        after = dict(_iter_reference_values(spec.spec()))
        assert after.keys() == before.keys(), f"{path}: reference sites changed shape"
        for site, old_value in before.items():
            expected = f"{old_value}-tx-abc123" if old_value in owned else old_value
            assert after[site] == expected, (
                f"{path}: {site} was {old_value!r}, expected {expected!r}, "
                f"got {after[site]!r}"
            )


def test_uniquify_leaves_every_volume_mount_resolvable_across_the_corpus() -> None:
    """`volumes[].name` is pod-local and must keep matching `volumeMounts[].name`.

    22 volumes across ``recipes/`` are named after the ConfigMap they mount, so
    a rename keyed on the name rather than on the reference key can rewrite one
    side of that pairing and leave the other. Checked over the corpus because
    the two sides live in different subtrees and a partial rename still parses.

    Measured: this is the only test in this module that fails when
    ``_rewrite_companion_refs`` is mutated to follow ``volumes[].name``.
    """
    for path in _recipe_manifests():
        spec = DeploymentSpec(str(path))
        if not spec.companions:
            continue

        spec.uniquify("-tx-abc123")

        for pod in _iter_pod_specs(spec.spec()):
            declared = {
                v.get("name") for v in pod.get("volumes") or [] if isinstance(v, dict)
            }
            containers = list(pod.get("containers") or [])
            main = pod.get("mainContainer")
            if isinstance(main, dict):
                containers.append(main)
            for container in containers:
                if not isinstance(container, dict):
                    continue
                for mount in container.get("volumeMounts") or []:
                    if not isinstance(mount, dict):
                        continue
                    assert mount.get("name") in declared, (
                        f"{path}: volumeMount {mount.get('name')!r} no longer "
                        f"matches any volume in {sorted(declared)}"
                    )


def test_uniquify_refuses_a_companion_kind_it_cannot_repoint(tmp_path) -> None:
    """Renaming a kind the reference walker does not know is silent breakage.

    ``_rewrite_companion_refs`` repoints ConfigMap and resource-claim-template
    references. A bundled PersistentVolumeClaim would get a fresh name while
    ``persistentVolumeClaim.claimName`` kept the old one, and the pod would fail
    to schedule for a reason that looks nothing like "the harness renamed it".
    Leaving it unrenamed is no better -- that is the cross-run collision
    ``uniquify`` exists to prevent. So refuse, and say what would fix it.

    No recipe bundles such a kind today (measured: 87 ConfigMaps, 28
    ComputeDomains, 2 ResourceClaimTemplates and nothing else), so this rejects
    nothing that ships; it makes the next kind an explicit decision.
    """
    manifest_path = tmp_path / "deploy.yaml"
    manifest_path.write_text(
        yaml.safe_dump_all(
            [
                {
                    "apiVersion": "v1",
                    "kind": "PersistentVolumeClaim",
                    "metadata": {"name": "model-cache"},
                    "spec": {"accessModes": ["ReadWriteMany"]},
                },
                {
                    "apiVersion": "nvidia.com/v1beta1",
                    "kind": "DynamoGraphDeployment",
                    "metadata": {"name": "recipe-under-test"},
                    "spec": {"components": []},
                },
            ]
        )
    )
    spec = DeploymentSpec(str(manifest_path))

    with pytest.raises(ValueError, match="cannot isolate per run"):
        spec.uniquify("-tx-abc123")

    # And the refusal must not have half-renamed anything on its way out.
    assert spec.name == "recipe-under-test"
    assert spec.companions[0]["metadata"]["name"] == "model-cache"


def test_uniquify_accepts_every_companion_kind_the_recipes_actually_bundle(
    tmp_path,
) -> None:
    """The guard is only honest if it does not reject the shipped corpus."""
    spec = DeploymentSpec(str(_multi_document_manifest(tmp_path)))

    spec.uniquify("-tx-abc123")  # ConfigMap + ComputeDomain: must not raise

    assert spec.name.endswith("-tx-abc123")
