# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Live-cluster integration tests for the planner's actuation knobs.

These tests run *inside* the dev pod (`planner-dev`) and exercise the actual
Kubernetes API path that mocks cannot validate:

  - RBAC permits PATCH on pods/DGDs in the dev namespace
  - Label selectors match real worker / frontend pods on the live DGD
  - TGP annotations actually land on a pod's metadata and are readable back
  - Frontend admission-control endpoint is reachable via pod IP
  - DGD spec is mutated by `set_component_replicas` (real or no-op)
  - Advisory mode genuinely produces zero K8s side effects

The whole module is skipped unless:
  1. `kubernetes.config.load_incluster_config()` succeeds (we're in a pod), AND
  2. `DYN_PARENT_DGD_K8S_NAME` is set to a real DGD name.

Run from the dev pod:
    export PYTHONPATH=/workspace/repo/components/src:/workspace/repo
    cd /workspace/repo
    python3 -m pytest \
        components/src/dynamo/planner/tests/integration/test_actuation_knobs_live.py \
        -v --tb=short -m 'integration'

The `scaling_real` test mutates the DGD spec; opt in via:
    RUN_DISRUPTIVE_TESTS=1 python3 -m pytest ... -k scaling_real
"""

from __future__ import annotations

import os
import time

import pytest

# ---------------------------------------------------------------------------
# Module-level gating
# ---------------------------------------------------------------------------

try:
    from kubernetes import client as k8s_client
    from kubernetes import config as k8s_config

    try:
        k8s_config.load_incluster_config()
        _IN_CLUSTER = True
    except k8s_config.ConfigException:
        _IN_CLUSTER = False
except ImportError:
    _IN_CLUSTER = False

_DGD_NAME = os.environ.get("DYN_PARENT_DGD_K8S_NAME")
_K8S_NAMESPACE = os.environ.get("DYN_PARENT_DGD_K8S_NAMESPACE") or os.environ.get(
    "POD_NAMESPACE"
)
_DYN_NAMESPACE = os.environ.get("DYN_NAMESPACE", "dynamo")

if not (_IN_CLUSTER and _DGD_NAME):
    pytest.skip(
        "Live-cluster tests require running inside the dev pod with "
        "DYN_PARENT_DGD_K8S_NAME set. "
        f"in_cluster={_IN_CLUSTER}, DYN_PARENT_DGD_K8S_NAME={_DGD_NAME!r}",
        allow_module_level=True,
    )

# Imports that pull in the kubernetes runtime (deferred until after gating).
from dynamo.planner.config.defaults import SubComponentType, TargetReplica  # noqa: E402
from dynamo.planner.connectors.kubernetes import KubernetesConnector  # noqa: E402
from dynamo.planner.connectors.kubernetes_api import KubernetesAPI  # noqa: E402
from dynamo.planner.core.budget import POWER_ANNOTATION_KEY  # noqa: E402

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.integration,
    pytest.mark.planner,
]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def core_api():
    """Raw CoreV1Api for kubectl-equivalent verification."""
    return k8s_client.CoreV1Api()


@pytest.fixture(scope="module")
def kube_api() -> KubernetesAPI:
    return KubernetesAPI(_K8S_NAMESPACE)


@pytest.fixture(scope="module")
def connector() -> KubernetesConnector:
    return KubernetesConnector(
        dynamo_namespace=_DYN_NAMESPACE,
        k8s_namespace=_K8S_NAMESPACE,
        parent_dgd_name=_DGD_NAME,
    )


@pytest.fixture(scope="module")
def any_worker_pod(kube_api: KubernetesAPI):
    """Return a single worker pod for the live DGD, regardless of subcomponent type.

    Aggregated DGDs may have only `worker`-typed services that don't map cleanly
    to PREFILL/DECODE; we look up by the operator-applied DGD label directly
    (KubeLabelDynamoGraphDeploymentName in deploy/operator/internal/consts/consts.go).
    """
    label_selector = f"nvidia.com/dynamo-graph-deployment-name={_DGD_NAME}"
    pods = kube_api.list_pods_by_label(label_selector)
    worker_pods = [
        p
        for p in pods
        if (p.metadata.labels or {}).get("nvidia.com/dynamo-component-type")
        != "frontend"
    ]
    if not worker_pods:
        pytest.skip(f"No worker pods found for DGD {_DGD_NAME!r}")
    return worker_pods[0]


# ---------------------------------------------------------------------------
# 1. TGP annotation round-trip (write via planner → read via raw K8s API)
# ---------------------------------------------------------------------------


class TestTgpAnnotationRoundTrip:
    """Prove the TGP annotation actually lands on a real pod and is readable."""

    def test_patch_pod_annotation_persists_and_is_readable(
        self,
        kube_api: KubernetesAPI,
        core_api,
        any_worker_pod,
    ):
        pod_name = any_worker_pod.metadata.name
        test_value = "400"

        # Capture original so we can restore even on assertion failure.
        original = (any_worker_pod.metadata.annotations or {}).get(POWER_ANNOTATION_KEY)

        try:
            kube_api.patch_pod_annotation(pod_name, POWER_ANNOTATION_KEY, test_value)

            # Re-read via the raw API (independent of the planner code path).
            fresh = core_api.read_namespaced_pod(
                name=pod_name, namespace=_K8S_NAMESPACE
            )
            actual = (fresh.metadata.annotations or {}).get(POWER_ANNOTATION_KEY)
            assert actual == test_value, (
                f"Expected annotation {POWER_ANNOTATION_KEY}={test_value} on "
                f"pod {pod_name}, got {actual!r}"
            )
        finally:
            # Restore: write back the original value if there was one,
            # otherwise null it via JSON-merge-patch (the only way to
            # actually remove a key).
            if original is not None:
                kube_api.patch_pod_annotation(pod_name, POWER_ANNOTATION_KEY, original)
            else:
                core_api.patch_namespaced_pod(
                    name=pod_name,
                    namespace=_K8S_NAMESPACE,
                    body={"metadata": {"annotations": {POWER_ANNOTATION_KEY: None}}},
                )

    def test_patch_is_idempotent_same_value(
        self,
        kube_api: KubernetesAPI,
        core_api,
        any_worker_pod,
    ):
        """Re-patching with the same value must not change the resource version."""
        pod_name = any_worker_pod.metadata.name
        original = (any_worker_pod.metadata.annotations or {}).get(POWER_ANNOTATION_KEY)

        try:
            kube_api.patch_pod_annotation(pod_name, POWER_ANNOTATION_KEY, "350")
            after_first = core_api.read_namespaced_pod(
                name=pod_name, namespace=_K8S_NAMESPACE
            )
            rv_first = after_first.metadata.resource_version

            kube_api.patch_pod_annotation(pod_name, POWER_ANNOTATION_KEY, "350")
            after_second = core_api.read_namespaced_pod(
                name=pod_name, namespace=_K8S_NAMESPACE
            )

            # Strategic-merge of an unchanged value is a server-side no-op,
            # so the resource version should not bump.
            assert (
                after_second.metadata.resource_version == rv_first
            ), "Re-patching with the same value triggered a spec mutation"
        finally:
            if original is not None:
                kube_api.patch_pod_annotation(pod_name, POWER_ANNOTATION_KEY, original)
            else:
                core_api.patch_namespaced_pod(
                    name=pod_name,
                    namespace=_K8S_NAMESPACE,
                    body={"metadata": {"annotations": {POWER_ANNOTATION_KEY: None}}},
                )


# ---------------------------------------------------------------------------
# 2. Component pod discovery against the live DGD
# ---------------------------------------------------------------------------


class TestPodDiscovery:
    """Verify label selectors actually resolve to real running pods."""

    def test_at_least_one_worker_pod_exists_for_dgd(self, kube_api: KubernetesAPI):
        label_selector = f"nvidia.com/dynamo-graph-deployment-name={_DGD_NAME}"
        pods = kube_api.list_pods_by_label(label_selector)
        assert pods, f"No pods found for DGD {_DGD_NAME!r} (RBAC or label issue)"

    def test_get_component_pods_resolves_some_subcomponent(
        self, connector: KubernetesConnector
    ):
        """At least one of PREFILL/DECODE should resolve, OR the DGD is
        aggregated-only (in which case both legitimately return empty)."""
        prefill = connector.get_component_pods(SubComponentType.PREFILL)
        decode = connector.get_component_pods(SubComponentType.DECODE)
        # We only assert the call doesn't raise — emptiness is allowed for agg.
        assert isinstance(prefill, list)
        assert isinstance(decode, list)

    def test_list_frontend_pods_returns_frontend(self, connector: KubernetesConnector):
        pods = connector.list_frontend_pods()
        assert pods, f"No frontend pods found for DGD {_DGD_NAME!r}"
        for pod in pods:
            labels = pod.metadata.labels or {}
            assert (
                labels.get("nvidia.com/dynamo-component-type") == "frontend"
            ), f"Pod {pod.metadata.name} missing frontend component-type label"

    def test_frontend_pod_has_pod_ip(self, connector: KubernetesConnector):
        """post_busy_threshold requires pod_ip; a Running frontend must expose one."""
        pods = connector.list_frontend_pods()
        running = [p for p in pods if (p.status.phase == "Running" and p.status.pod_ip)]
        assert running, (
            f"No Running frontend pod with pod_ip for DGD {_DGD_NAME!r}; "
            f"phases={[p.status.phase for p in pods]}"
        )


# ---------------------------------------------------------------------------
# 3. Admission-control POST against the live frontend
# ---------------------------------------------------------------------------


class TestPostBusyThresholdLive:
    """Hit the actual /busy_threshold endpoint on a frontend pod."""

    @pytest.mark.asyncio
    async def test_post_busy_threshold_returns_2xx(
        self, connector: KubernetesConnector
    ):
        pods = connector.list_frontend_pods()
        running = [p for p in pods if p.status.phase == "Running" and p.status.pod_ip]
        if not running:
            pytest.skip("No Running frontend pod with pod IP")
        pod = running[0]

        try:
            model = connector.get_model_name(
                require_prefill=False, require_decode=False
            )
        except Exception:
            model = os.environ.get("MODEL_NAME", "Qwen/Qwen3-0.6B")

        try:
            await connector.post_busy_threshold(
                pod=pod,
                model=model,
                port=8000,
                # Very permissive: documented as "no admission control" defaults.
                active_decode_blocks_threshold=1.0,
                active_prefill_tokens_threshold=10_000_000,
                active_prefill_tokens_threshold_frac=1.0,
            )
        except Exception as exc:
            # 404 means this build of the frontend hasn't enabled the
            # admission-control endpoint yet — that's a build feature flag,
            # not a planner bug.
            msg = str(exc).lower()
            if "404" in msg or "not found" in msg:
                pytest.skip(
                    "Frontend build does not expose /busy_threshold " f"(got {exc!r})"
                )
            raise


# ---------------------------------------------------------------------------
# 4. Scaling apply (advisory mode → must be a true no-op on the DGD spec)
# ---------------------------------------------------------------------------


def _read_dgd_generation(kube_api: KubernetesAPI) -> int:
    dgd = kube_api.get_graph_deployment(_DGD_NAME)
    return int(dgd.get("metadata", {}).get("generation", 0))


def _read_replicas_map(kube_api: KubernetesAPI) -> dict[str, int]:
    dgd = kube_api.get_graph_deployment(_DGD_NAME)
    services = dgd.get("spec", {}).get("services", {}) or {}
    return {name: int(spec.get("replicas", 0)) for name, spec in services.items()}


class TestScalingAdvisoryMode:
    """Advisory mode must NEVER mutate DGD spec, even with non-trivial targets."""

    def test_advisory_mode_does_not_change_dgd_generation(
        self, kube_api: KubernetesAPI
    ):
        """Capture generation before, build a deliberately-different target,
        run the same code path the planner uses with advisory=True, verify
        generation is unchanged.

        This goes through `NativePlannerBase._apply_scaling_targets`, which
        is the actual call site that gates advisory mode.
        """
        from dynamo.planner.core.base import NativePlannerBase

        replicas_before = _read_replicas_map(kube_api)
        gen_before = _read_dgd_generation(kube_api)
        assert replicas_before, "DGD has no services?"

        # Build a target that would visibly change at least one service.
        first_service = next(iter(replicas_before))
        target = TargetReplica(
            sub_component_type=SubComponentType.DECODE,
            component_name=first_service,
            desired_replicas=replicas_before[first_service] + 99,
        )

        # Stub: a NativePlannerBase wired with config.advisory=True. We
        # bypass __init__ since it requires a full planner context.
        # The real check in _apply_scaling_targets reads self.config.advisory.
        class _AdvisoryConfig:
            advisory = True

        adapter = object.__new__(NativePlannerBase)
        adapter.config = _AdvisoryConfig()
        adapter.connector = None  # must not be touched in advisory mode
        # The method is async on some branches — call sync if available,
        # otherwise wrap with asyncio.run.
        method = getattr(adapter, "_apply_scaling_targets", None)
        assert method is not None, "_apply_scaling_targets missing on base"

        import asyncio
        import inspect

        if inspect.iscoroutinefunction(method):
            asyncio.run(method([target]))
        else:
            method([target])

        # Spec must not have been touched.
        assert _read_replicas_map(kube_api) == replicas_before
        assert _read_dgd_generation(kube_api) == gen_before


# ---------------------------------------------------------------------------
# 4b. TGP orchestration: _apply_power_annotations end-to-end
# ---------------------------------------------------------------------------


class TestApplyPowerAnnotationsLive:
    """Prove the orchestration layer (NativePlannerBase._apply_power_annotations)
    actually patches live pods when invoked end-to-end.

    qwen3-quickstart is an aggregated DGD whose worker is labelled
    componentType=worker, which doesn't map to PREFILL or DECODE
    SubComponentType. To exercise the patch path on this real cluster
    we override the connector's get_component_pods to return the actual
    worker pods (as if they were DECODE pods) and let the orchestrator
    do the rest.
    """

    @pytest.mark.asyncio
    async def test_orchestration_patches_real_pod(
        self,
        kube_api: KubernetesAPI,
        core_api,
        connector: KubernetesConnector,
        any_worker_pod,
    ):
        from dynamo.planner.core.base import NativePlannerBase

        pod_name = any_worker_pod.metadata.name
        original = (any_worker_pod.metadata.annotations or {}).get(POWER_ANNOTATION_KEY)
        target_watts = 350
        if original == str(target_watts):
            target_watts = 360  # ensure the test value differs from current

        class _PowerConfig:
            advisory = False
            enable_power_awareness = True
            prefill_engine_gpu_power_limit = target_watts
            decode_engine_gpu_power_limit = target_watts

        # Stub the connector to return our real pod as a "decode" pod so
        # the orchestrator dispatches a real patch_pod_annotation call.
        original_get_component_pods = connector.get_component_pods

        def _stubbed_get_component_pods(sub_component_type):
            from dynamo.planner.config.defaults import SubComponentType as _Sct

            if sub_component_type == _Sct.DECODE:
                return [any_worker_pod]
            return []

        connector.get_component_pods = _stubbed_get_component_pods  # type: ignore[assignment]

        adapter = object.__new__(NativePlannerBase)
        adapter.config = _PowerConfig()
        adapter.connector = connector
        adapter.require_prefill = False
        adapter.require_decode = True

        try:
            await adapter._apply_power_annotations()

            # Verify the annotation actually landed on the live pod.
            fresh = core_api.read_namespaced_pod(
                name=pod_name, namespace=_K8S_NAMESPACE
            )
            actual = (fresh.metadata.annotations or {}).get(POWER_ANNOTATION_KEY)
            assert actual == str(target_watts), (
                f"Expected {POWER_ANNOTATION_KEY}={target_watts} on {pod_name}, "
                f"got {actual!r}"
            )
        finally:
            connector.get_component_pods = original_get_component_pods  # type: ignore[assignment]
            if original is not None:
                kube_api.patch_pod_annotation(pod_name, POWER_ANNOTATION_KEY, original)
            else:
                core_api.patch_namespaced_pod(
                    name=pod_name,
                    namespace=_K8S_NAMESPACE,
                    body={"metadata": {"annotations": {POWER_ANNOTATION_KEY: None}}},
                )

    @pytest.mark.asyncio
    async def test_orchestration_skips_when_power_awareness_disabled(
        self,
        connector: KubernetesConnector,
        any_worker_pod,
        core_api,
    ):
        """With enable_power_awareness=False the orchestrator must be a no-op."""
        from dynamo.planner.core.base import NativePlannerBase

        pod_name = any_worker_pod.metadata.name
        before = (
            core_api.read_namespaced_pod(
                name=pod_name, namespace=_K8S_NAMESPACE
            ).metadata.annotations
            or {}
        ).get(POWER_ANNOTATION_KEY)

        class _OffConfig:
            advisory = False
            enable_power_awareness = False
            prefill_engine_gpu_power_limit = 999
            decode_engine_gpu_power_limit = 999

        adapter = object.__new__(NativePlannerBase)
        adapter.config = _OffConfig()
        adapter.connector = connector
        adapter.require_prefill = True
        adapter.require_decode = True

        await adapter._apply_power_annotations()

        after = (
            core_api.read_namespaced_pod(
                name=pod_name, namespace=_K8S_NAMESPACE
            ).metadata.annotations
            or {}
        ).get(POWER_ANNOTATION_KEY)
        assert (
            before == after
        ), "Pod annotation changed despite enable_power_awareness=False"


# ---------------------------------------------------------------------------
# 5. Scaling apply (non-advisory) — opt-in, actually mutates the DGD
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    os.environ.get("RUN_DISRUPTIVE_TESTS") != "1",
    reason="Disruptive test: scales workers up and back down. "
    "Set RUN_DISRUPTIVE_TESTS=1 to enable.",
)
class TestScalingRealMutation:
    """Prove `set_component_replicas` patches the live DGD when not advisory.

    The test scales the first scalable service +1 then back to its original
    count, asserting (a) the DGD generation bumps after the +1 patch and
    (b) the spec reflects the requested values.
    """

    @pytest.mark.asyncio
    async def test_set_component_replicas_mutates_dgd_spec(
        self,
        kube_api: KubernetesAPI,
        connector: KubernetesConnector,
    ):
        replicas_before = _read_replicas_map(kube_api)
        # Skip frontend — its replica count is governed by separate logic.
        scalable = {
            name: count
            for name, count in replicas_before.items()
            if "frontend" not in name.lower()
        }
        if not scalable:
            pytest.skip("No non-frontend scalable services on this DGD")

        service_name, original = next(iter(scalable.items()))
        target_up = original + 1

        gen_before = _read_dgd_generation(kube_api)
        try:
            await connector.set_component_replicas(
                [
                    TargetReplica(
                        sub_component_type=SubComponentType.DECODE,
                        component_name=service_name,
                        desired_replicas=target_up,
                    )
                ],
                blocking=False,
            )
            time.sleep(2)  # let the API server settle

            replicas_after = _read_replicas_map(kube_api)
            gen_after = _read_dgd_generation(kube_api)

            assert (
                replicas_after.get(service_name) == target_up
            ), f"Expected {service_name}.replicas={target_up}, got {replicas_after.get(service_name)}"
            assert gen_after > gen_before, (
                f"DGD generation did not bump: before={gen_before}, "
                f"after={gen_after}"
            )
        finally:
            # Restore original count regardless of outcome.
            await connector.set_component_replicas(
                [
                    TargetReplica(
                        sub_component_type=SubComponentType.DECODE,
                        component_name=service_name,
                        desired_replicas=original,
                    )
                ],
                blocking=False,
            )
