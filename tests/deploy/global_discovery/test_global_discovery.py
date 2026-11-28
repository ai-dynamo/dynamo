# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A pool in one cluster is listed by the Global Discovery Service in another.

The most basic multi-cluster Global Discovery case, on two clusters (two
vClusters in CI)::

    hub cluster                          member cluster
    Global Discovery Service  <--------  DynamoPoolRelay (operator-managed)
      --pool-discovery registrations       POST /pools on every heartbeat
                                         DGD: Frontend + mocker worker

The relay reaches the service through the hub Service's ClusterIP. vCluster
syncs every pod and Service to the host cluster, so an address from one
vCluster is routable from a pod of the other: the routable network the
Kubernetes discovery modes assume. The test reaches the service through a
port-forward.

The test checks that:

1. ``GET /pools`` on the hub lists the member's pool under its key (the
   relay's cluster ID and the DGD's namespace and name), with the model the
   relay read from the DGD, the relay's region and the frontend URL it
   registered, and calls the list complete.
2. Heartbeats keep the pool listed for three leases.
3. Once the relay is gone, the pool leaves ``GET /pools`` within one lease.

README.md lists what the test does not cover yet.
"""

import logging
import time
from pathlib import Path
from typing import Any, Optional

import pytest

from tests.deploy.global_discovery.cluster import Cluster, PortForwardHttp, wait_for
from tests.utils.output_paths import resolve_test_output_path

logger = logging.getLogger(__name__)

MODEL = "Qwen/Qwen3-0.6B"

# Hub: the Global Discovery Service.
DISCOVERY = "gd-e2e-discovery"
DISCOVERY_PORT = 8080
HEARTBEAT_S = 2
LEASE_S = 10

# Member: the pool. The DGD and its DynamoPoolRelay share a name.
POOL = "gd-e2e-pool"
POOL_COMPONENTS = ("Frontend", "worker")
MEMBER_CLUSTER_ID = "gd-e2e-member"
MEMBER_REGION = "gd-e2e-region"
RELAY_SELECTOR = "app.kubernetes.io/name=pool-relay"
RELAY_TERMINATION_GRACE_S = 10

DISCOVERY_READY_TIMEOUT_S = 300
POOL_READY_TIMEOUT_S = 900
RELAY_READY_TIMEOUT_S = 300
REGISTRATION_TIMEOUT_S = 300
RELAY_GONE_TIMEOUT_S = 120
# The relay stops renewing on SIGTERM, before its pod is gone, so the pool
# leaves within one lease of that; the rest covers the polling.
POOL_EXPIRY_TIMEOUT_S = LEASE_S + 20

# The DGD's transport. The relay must use the same; see
# deploy/operator/samples/dynamo-pool-relay.yaml.
TRANSPORT_ENV = [
    {"name": "DYN_REQUEST_PLANE", "value": "tcp"},
    {"name": "DYN_EVENT_PLANE", "value": "zmq"},
]
# A secretKeyRef, not envFrom: the PoolRelay controller refuses a DGD whose
# containers take envFrom without setting the discovery backend and mode.
HF_TOKEN_ENV = {
    "name": "HF_TOKEN",
    "valueFrom": {
        "secretKeyRef": {
            "name": "hf-token-secret",
            "key": "HF_TOKEN",
            "optional": True,
        }
    },
}


def discovery_manifests(image: str) -> list[dict[str, Any]]:
    """The Global Discovery Service in registrations mode with memory storage."""
    labels = {"app.kubernetes.io/name": DISCOVERY}
    container = {
        "name": "global-discovery-service",
        "image": image,
        "args": [
            f"--listen=0.0.0.0:{DISCOVERY_PORT}",
            "--pool-discovery=registrations",
            "--registration-storage=memory",
            f"--heartbeat-interval-secs={HEARTBEAT_S}",
            f"--lease-duration-secs={LEASE_S}",
        ],
        "env": [
            {"name": "DYN_LOG", "value": "info,dynamo_global_discovery_service=debug"}
        ],
        "ports": [{"name": "http", "containerPort": DISCOVERY_PORT}],
        "readinessProbe": {
            "httpGet": {"path": "/healthz", "port": "http"},
            "periodSeconds": 2,
        },
        "resources": {
            "requests": {"cpu": "50m", "memory": "64Mi"},
            "limits": {"memory": "256Mi"},
        },
        "securityContext": {
            "allowPrivilegeEscalation": False,
            "capabilities": {"drop": ["ALL"]},
        },
    }
    deployment = {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {"name": DISCOVERY, "labels": labels},
        "spec": {
            "replicas": 1,
            "selector": {"matchLabels": labels},
            "template": {
                "metadata": {"labels": labels},
                "spec": {
                    "securityContext": {
                        "runAsNonRoot": True,
                        "seccompProfile": {"type": "RuntimeDefault"},
                    },
                    "containers": [container],
                },
            },
        },
    }
    service = {
        "apiVersion": "v1",
        "kind": "Service",
        "metadata": {"name": DISCOVERY, "labels": labels},
        "spec": {
            "selector": labels,
            "ports": [{"name": "http", "port": DISCOVERY_PORT, "targetPort": "http"}],
        },
    }
    return [deployment, service]


def pool_dgd(image: str) -> dict[str, Any]:
    """The pool: a frontend and one mocker worker serving ``MODEL``."""

    def component(name: str, kind: str, **main: Any) -> dict[str, Any]:
        container = {
            "name": "main",
            "image": image,
            "env": [*TRANSPORT_ENV, HF_TOKEN_ENV],
            "resources": {
                "requests": {"cpu": "500m", "memory": "1Gi"},
                "limits": {"memory": "4Gi"},
            },
            **main,
        }
        return {
            "name": name,
            "type": kind,
            "replicas": 1,
            "podTemplate": {"spec": {"containers": [container]}},
        }

    frontend, worker = POOL_COMPONENTS
    return {
        "apiVersion": "nvidia.com/v1beta1",
        "kind": "DynamoGraphDeployment",
        "metadata": {"name": POOL},
        "spec": {
            "components": [
                component(frontend, "frontend"),
                component(
                    worker,
                    "worker",
                    command=["python3", "-m", "dynamo.mocker"],
                    args=["--model-path", MODEL, "--model-name", MODEL],
                ),
            ]
        },
    }


def pool_relay(
    image: str, namespace: str, discovery_url: str, frontend_url: str
) -> dict[str, Any]:
    """The DGD's PoolRelay, registering the DGD as a routing pool.

    The operator does not set the Global Discovery flags yet, so they go in
    as environment variables (components/src/dynamo/pool_relay/cli.py).
    """
    env = [
        *TRANSPORT_ENV,
        {"name": "DYN_RELAY_GLOBAL_DISCOVERY_URL", "value": discovery_url},
        {
            "name": "DYN_RELAY_ROUTING_POOL",
            "value": f"{MEMBER_CLUSTER_ID}/{namespace}/{POOL}",
        },
        {"name": "DYN_RELAY_ROUTING_POOL_REGION", "value": MEMBER_REGION},
        {"name": "DYN_RELAY_ROUTING_POOL_FRONTENDS", "value": frontend_url},
    ]
    container = {
        "name": "main",
        "image": image,
        "env": env,
        "resources": {
            "requests": {"cpu": "250m", "memory": "512Mi"},
            "limits": {"memory": "2Gi"},
        },
    }
    return {
        "apiVersion": "nvidia.com/v1alpha1",
        "kind": "DynamoPoolRelay",
        "metadata": {"name": POOL},
        "spec": {
            "clusterId": MEMBER_CLUSTER_ID,
            "replicas": 1,
            "podTemplate": {
                "spec": {
                    "terminationGracePeriodSeconds": RELAY_TERMINATION_GRACE_S,
                    "containers": [container],
                }
            },
        },
    }


def condition(obj: Optional[dict[str, Any]], kind: str) -> Optional[dict[str, Any]]:
    conditions = ((obj or {}).get("status") or {}).get("conditions") or []
    return next((c for c in conditions if c.get("type") == kind), None)


def wait_for_discovery(hub: Cluster) -> None:
    def available() -> tuple[Optional[bool], str]:
        status = (hub.get("deployment", DISCOVERY) or {}).get("status") or {}
        return (True if status.get("availableReplicas", 0) >= 1 else None), str(status)

    wait_for("the Global Discovery Service", available, DISCOVERY_READY_TIMEOUT_S)


def wait_for_pool_dgd(member: Cluster) -> None:
    def is_ready() -> tuple[Optional[bool], str]:
        dgd = member.get("dynamographdeployment", POOL)
        state = ((dgd or {}).get("status") or {}).get("state")
        ready = condition(dgd, "Ready")
        done = state == "successful" and (ready or {}).get("status") == "True"
        return (True if done else None), f"state={state} Ready={ready}"

    wait_for(f"DGD {POOL}", is_ready, POOL_READY_TIMEOUT_S, interval_s=5)


def wait_for_relay(member: Cluster) -> None:
    def is_ready() -> tuple[Optional[bool], str]:
        ready = condition(member.get("dynamopoolrelay", POOL), "Ready")
        return (True if (ready or {}).get("status") == "True" else None), str(ready)

    wait_for(f"DynamoPoolRelay {POOL}", is_ready, RELAY_READY_TIMEOUT_S, interval_s=5)


def wait_for_relay_pods_gone(member: Cluster) -> None:
    def gone() -> tuple[Optional[bool], str]:
        pods = member.names("pod", RELAY_SELECTOR)
        return (None if pods else True), str(pods)

    wait_for("the PoolRelay pods to go", gone, RELAY_GONE_TIMEOUT_S)


def listed_pool(
    listing: dict[str, Any], key: dict[str, str]
) -> Optional[dict[str, Any]]:
    return next((p for p in listing.get("pools", []) if p.get("key") == key), None)


def dump_diagnostics(hub: Cluster, member: Cluster, directory: Path) -> None:
    """Save the state of both clusters, and log what explains most failures."""
    hub.save(directory, "resources.txt", "get", "deployment,pod,service", "-o", "wide")
    hub.save(directory, "events.txt", "get", "events", "--sort-by=.lastTimestamp")
    discovery_log = hub.save(
        directory, "discovery.log", "logs", f"deployment/{DISCOVERY}", "--tail=500"
    )
    member.save(
        directory,
        "resources.txt",
        "get",
        "dynamographdeployment,dynamopoolrelay,deployment,pod,service",
        "-o",
        "wide",
    )
    member.save(
        directory,
        "pool.yaml",
        "get",
        "dynamographdeployment,dynamopoolrelay",
        POOL,
        "-o",
        "yaml",
    )
    member.save(directory, "events.txt", "get", "events", "--sort-by=.lastTimestamp")
    for component in POOL_COMPONENTS:
        member.save(
            directory,
            f"{component.lower()}.log",
            "logs",
            "-l",
            f"nvidia.com/dynamo-graph-deployment-name={POOL},"
            f"nvidia.com/dynamo-component={component}",
            "--tail=300",
            "--all-containers",
            "--prefix",
        )
    # The relay logs "pool registered with Global Discovery", or why a call
    # failed. It does not log a registration it withholds, for example while
    # the DGD's serving topology has no model yet.
    relay_log = member.save(
        directory,
        "relay.log",
        "logs",
        "-l",
        RELAY_SELECTOR,
        "--tail=500",
        "--all-containers",
        "--prefix",
    )
    logger.error("PoolRelay log tail:\n%s", relay_log[-3000:])
    logger.error("Global Discovery Service log tail:\n%s", discovery_log[-3000:])
    logger.error("Diagnostics saved under %s", directory)


@pytest.mark.k8s
@pytest.mark.deploy
@pytest.mark.post_merge
@pytest.mark.e2e
@pytest.mark.timeout(2100)
def test_member_pool_is_listed_by_hub_discovery(
    hub: Cluster,
    member: Cluster,
    global_discovery_image: str,
    pool_image: str,
    request: pytest.FixtureRequest,
) -> None:
    output_dir = Path(resolve_test_output_path(request.node.name))
    output_dir.mkdir(parents=True, exist_ok=True)
    key = {
        "cluster_id": MEMBER_CLUSTER_ID,
        "namespace": member.namespace,
        "dgd_name": POOL,
    }

    try:
        logger.info("Hub: deploying the Global Discovery Service")
        hub.apply(*discovery_manifests(global_discovery_image))
        wait_for_discovery(hub)
        discovery_url = f"http://{hub.service_address(DISCOVERY, 'http')}"

        logger.info("Member: deploying DGD %s", POOL)
        member.apply(pool_dgd(pool_image))
        wait_for_pool_dgd(member)
        frontend_service = f"{POOL}-{POOL_COMPONENTS[0].lower()}"
        frontend_url = f"http://{member.service_address(frontend_service, 'http')}"

        logger.info(
            "Member: deploying the PoolRelay; Global Discovery at %s, frontend at %s",
            discovery_url,
            frontend_url,
        )
        member.apply(
            pool_relay(pool_image, member.namespace, discovery_url, frontend_url)
        )
        wait_for_relay(member)

        with PortForwardHttp(
            hub, DISCOVERY, DISCOVERY_PORT, output_dir / "port-forward.log"
        ) as discovery:

            def registered() -> tuple[Optional[dict[str, Any]], str]:
                listing = discovery.get_json("/pools")
                return (listing if listed_pool(listing, key) else None), str(listing)

            def expired() -> tuple[Optional[bool], str]:
                listing = discovery.get_json("/pools")
                return (None if listed_pool(listing, key) else True), str(listing)

            logger.info("Hub: waiting for GET /pools to list %s", key)
            listing = wait_for(
                f"GET /pools to list {key}", registered, REGISTRATION_TIMEOUT_S
            )
            pool = listed_pool(listing, key)
            assert pool is not None
            logger.info("Hub lists the member's pool: %s", pool)
            assert listing["complete"] is True, listing
            assert pool["model"] == MODEL, pool
            assert pool["location"]["region"] == MEMBER_REGION, pool
            frontends = [url.rstrip("/") for url in pool["frontends"]]
            assert frontends == [frontend_url], pool

            logger.info("Hub: the pool must stay listed for three leases")
            stay_until = time.monotonic() + 3 * LEASE_S
            while time.monotonic() < stay_until:
                listing = discovery.get_json("/pools")
                assert listed_pool(listing, key), f"the pool left early: {listing}"
                time.sleep(1)

            logger.info("Member: deleting the PoolRelay")
            member.delete("dynamopoolrelay", POOL)
            wait_for_relay_pods_gone(member)
            wait_for(
                f"GET /pools to drop {key}",
                expired,
                POOL_EXPIRY_TIMEOUT_S,
                interval_s=1,
            )
    except BaseException:
        dump_diagnostics(hub, member, output_dir)
        raise
    finally:
        for cluster, kind, name in (
            (member, "dynamopoolrelay", POOL),
            (member, "dynamographdeployment", POOL),
            (hub, "service", DISCOVERY),
            (hub, "deployment", DISCOVERY),
        ):
            try:
                cluster.delete(kind, name)
            except Exception as error:
                logger.warning(
                    "%s: deleting %s/%s failed: %s", cluster.role, kind, name, error
                )
