# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Expose the local engine's Prometheus metrics through the Dynamo worker endpoint.

The in-process backends (``dynamo.sglang``, ``dynamo.vllm``) share a registry with
the engine, so they pass engine metrics through with a ``MultiProcessCollector``.
The OpenAI backend instead launches the engine as a separate HTTP server, so its
metrics are only reachable over HTTP — and, by default, only on loopback. Without
this module the worker's ``/metrics`` carries ``dynamo_*`` runtime metrics but none
of the engine's own ``sglang:*`` / ``vllm:*`` series.

Federate them on scrape: fetch the engine's exposition text, parse it, and hand the
families to the shared registration helper so they pick up the same
``dynamo_namespace`` / ``dynamo_component`` / ``dynamo_endpoint`` / ``worker_id`` /
``model`` labels the in-process backends inject.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Iterator, Optional
from urllib.parse import urlsplit, urlunsplit
from urllib.request import ProxyHandler, build_opener

if TYPE_CHECKING:
    from dynamo.runtime import Endpoint

LOGGER = logging.getLogger(__name__)

# Scrapes are seconds apart and the engine is on loopback, so a short timeout is
# enough. Exceeding it must not stall the worker's own metrics.
FETCH_TIMEOUT_SECONDS = 2.0

# Prefixes emitted by the engines the OpenAI backend can launch. Anything else the
# engine's endpoint serves (python_*, process_*, http_*) is dropped by the filter.
ENGINE_METRIC_PREFIXES = ["sglang:", "vllm:", "trtllm_"]

# The engine is a local process, so never route this through a proxy: the ambient
# HTTP(S)_PROXY of the pod would add latency and send metrics off-box.
_OPENER = build_opener(ProxyHandler({}))


def engine_metrics_url(base_url: str) -> str:
    """Derive the engine's ``/metrics`` URL from one of its HTTP URLs.

    Raises:
        ValueError: if ``base_url`` has no scheme or host. Without both,
            ``urlopen`` would fall through to a local file path.
    """
    parts = urlsplit(base_url)
    if parts.scheme not in ("http", "https") or not parts.netloc:
        raise ValueError(f"engine URL must be absolute http(s), got {base_url!r}")
    return urlunsplit((parts.scheme, parts.netloc, "/metrics", "", ""))


class EngineMetricsCollector:
    """Prometheus collector that federates the engine's ``/metrics`` on scrape.

    Returns nothing when the engine is unreachable or serves no metrics — that is
    the normal state for an engine started without ``--enable-metrics``, and the
    worker's own metrics must still be served in that case.
    """

    def __init__(self, metrics_url: str, timeout: float = FETCH_TIMEOUT_SECONDS):
        self._metrics_url = metrics_url
        self._timeout = timeout

    def collect(self) -> Iterator:
        from prometheus_client.parser import text_string_to_metric_families

        try:
            with _OPENER.open(self._metrics_url, timeout=self._timeout) as response:
                payload = response.read().decode("utf-8")
        except Exception as exc:  # noqa: BLE001 - never break the worker's scrape
            LOGGER.debug(
                "Could not fetch engine metrics from %s: %s", self._metrics_url, exc
            )
            return

        try:
            yield from text_string_to_metric_families(payload)
        except Exception as exc:  # noqa: BLE001 - malformed engine output
            LOGGER.warning(
                "Could not parse engine metrics from %s: %s", self._metrics_url, exc
            )


def register_engine_metrics(
    endpoint: "Endpoint",
    engine_base_url: str,
    *,
    namespace_name: Optional[str] = None,
    component_name: Optional[str] = None,
    endpoint_name: Optional[str] = None,
    model_name: Optional[str] = None,
) -> None:
    """Publish the engine's metrics alongside the worker's on the system port.

    ``engine_base_url`` must address the engine itself. In router mode the worker's
    upstream is the router, which does not emit the engine's series — pass the
    engine origin instead.

    Safe to call unconditionally: registration itself performs no I/O, and the
    engine is only contacted when ``/metrics`` is scraped.
    """
    try:
        from prometheus_client import CollectorRegistry

        from dynamo.common.utils.prometheus import register_engine_metrics_callback

        metrics_url = engine_metrics_url(engine_base_url)
        registry = CollectorRegistry()
        registry.register(EngineMetricsCollector(metrics_url))

        register_engine_metrics_callback(
            endpoint=endpoint,
            registry=registry,
            metric_prefix_filters=ENGINE_METRIC_PREFIXES,
            namespace_name=namespace_name,
            component_name=component_name,
            endpoint_name=endpoint_name,
            model_name=model_name,
        )
        LOGGER.info("Publishing engine metrics from %s", metrics_url)
    except Exception as exc:  # noqa: BLE001 - metrics must never block serving
        LOGGER.warning("Could not register engine metrics passthrough: %s", exc)
