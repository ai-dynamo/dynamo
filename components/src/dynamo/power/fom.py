# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Figure-of-merit (FoM) providers for EPD's controller.

EPD pulls one float per decision cycle through ``get_perf()``. Its signal
contract: higher is better, a per-interval rate (not a cumulative total),
computed over the whole pool, and exactly ``0.0`` when there is no reading.
Every provider here returns a float ``>= 0`` and ``0.0`` for "no signal":
before the first sample, when any ``/metrics`` endpoint is unreachable (a
partial sum would look like a collapse), when a counter went backwards (a
restart) or when no request was served in the interval.

Sources:

* Prometheus ``/metrics`` endpoints (the frontend's HTTP port and the pool
  workers' system-status ports), scraped with the standard library; a
  ``prom:`` spec's ``scope`` picks the endpoints the launcher tagged
  ``frontend=`` or ``worker=`` (``all`` scrapes every endpoint);
* EPD's own energy snapshots (``{LOG_DIR}/epd/nvml/node_energy/``) for the
  tokens-per-joule preset;
* any user class exposing ``get_perf() -> float`` (``python:<module>:<Class>``).

Presets map onto the metric names Dynamo really exposes:

===================== ==========================================================
preset                 source
===================== ==========================================================
``output_tokens``      rate of ``dynamo_frontend_output_tokens_total`` [tokens/s]
``tokens_per_gpu``     the same rate / the pool's GPU count [tokens/s/GPU]
``per_user_speed_p25`` 1 / p75 of ``dynamo_frontend_inter_token_latency_seconds``
                       [tokens/s per stream]; the slow quartile's speed
``tokens_per_joule``   output tokens / GPU energy from EPD's snapshots [tokens/J]
``ctx_tokens``         rate of new context tokens on the pool's workers:
                       ``trtllm_prompt_tokens_total`` minus
                       ``trtllm_prompt_cached_tokens_total`` (vLLM / SGLang:
                       ``vllm:`` / ``sglang:prompt_tokens_total``) [tokens/s]
``gen_tokens``         rate of ``trtllm_generation_tokens_total`` (vLLM / SGLang:
                       ``vllm:`` / ``sglang:generation_tokens_total``) [tokens/s]
``inv_ttft_p50``       1 / p50 of ``dynamo_frontend_time_to_first_token_seconds`` [1/s]
===================== ==========================================================

The engine presets read the workers' own ``/metrics`` (``DYN_SYSTEM_PORT``).
TensorRT-LLM workers publish the ``trtllm_*`` counters only when started with
``--publish-metrics`` (``--publish-events-and-metrics`` before it), which also
turns on the engine's ``enable_iter_perf_stats`` and ``return_perf_metrics``;
both counters are credited when a request finishes.
"""

from __future__ import annotations

import csv
import http.client
import importlib
import json
import logging
import math
import os
import re
import time
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

from dynamo.power.config import FomSpec, PowerControlConfigError

logger = logging.getLogger(__name__)

Scraper = Callable[[str], "str | None"]
Clock = Callable[[], float]

FRONTEND_OUTPUT_TOKENS_TOTAL = "dynamo_frontend_output_tokens_total"
FRONTEND_TTFT_SECONDS = "dynamo_frontend_time_to_first_token_seconds"
FRONTEND_ITL_SECONDS = "dynamo_frontend_inter_token_latency_seconds"
ENGINE_PROMPT_TOKENS_TOTAL = (
    "trtllm_prompt_tokens_total",
    "vllm:prompt_tokens_total",
    "sglang:prompt_tokens_total",
)
# Prompt tokens served from the KV cache; only TensorRT-LLM publishes the counter.
ENGINE_PROMPT_CACHED_TOKENS_TOTAL = ("trtllm_prompt_cached_tokens_total",)
ENGINE_GENERATION_TOKENS_TOTAL = (
    "trtllm_generation_tokens_total",
    "vllm:generation_tokens_total",
    "sglang:generation_tokens_total",
)
ENERGY_SNAPSHOT_SUBDIR = os.path.join("epd", "nvml", "node_energy")
FOM_TRACE_FILENAME = "fom.csv"
DEFAULT_SCRAPE_TIMEOUT_S = 5.0

_SAMPLE_RE = re.compile(
    r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{(.*)\})?\s+(\S+)(?:\s+-?\d+(?:\.\d+)?)?\s*$"
)
_LABEL_RE = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)="((?:[^"\\]|\\.)*)"')


# ---------------------------------------------------------------- exposition


@dataclass(frozen=True)
class Sample:
    """One series sample from a Prometheus text exposition."""

    name: str
    labels: Mapping[str, str]
    value: float


def _unescape_label(value: str) -> str:
    return value.replace('\\"', '"').replace("\\n", "\n").replace("\\\\", "\\")


def parse_exposition(text: str) -> list[Sample]:
    """Parse Prometheus text format into samples; comment and blank lines are skipped."""
    samples = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        match = _SAMPLE_RE.match(stripped)
        if match is None:
            logger.debug("skipping unparseable metrics line: %r", stripped)
            continue
        name, label_text, value_text = match.group(1), match.group(2), match.group(3)
        try:
            value = float(value_text)
        except ValueError:
            logger.warning(
                "skipping metrics line with a non-numeric value: %r", stripped
            )
            continue
        labels = {
            key: _unescape_label(raw)
            for key, raw in _LABEL_RE.findall(label_text or "")
        }
        samples.append(Sample(name=name, labels=labels, value=value))
    return samples


def http_scrape(url: str, timeout_s: float = DEFAULT_SCRAPE_TIMEOUT_S) -> str | None:
    """Fetch one ``/metrics`` page; ``None`` (after a warning) when unreachable."""
    request = urllib.request.Request(url, headers={"Accept": "text/plain"})
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            return response.read().decode("utf-8", errors="replace")
    except (OSError, http.client.HTTPException) as exc:
        # URLError, HTTPError and socket timeouts are all OSError subclasses.
        logger.warning("metrics scrape of %s failed: %s", url, exc)
        return None


@dataclass(frozen=True)
class SeriesSelector:
    """Selects the samples of one metric (any of ``names``) matching ``labels``."""

    names: tuple[str, ...]
    labels: tuple[tuple[str, str], ...] = ()

    def matches(self, sample: Sample) -> bool:
        if sample.name not in self.names:
            return False
        return all(sample.labels.get(key) == value for key, value in self.labels)

    def describe(self) -> str:
        names = "|".join(self.names)
        if not self.labels:
            return names
        return names + "{" + ",".join(f'{k}="{v}"' for k, v in self.labels) + "}"


class MetricsEndpoints:
    """A fixed set of ``/metrics`` URLs scraped together every cycle."""

    def __init__(self, urls: Sequence[str], scraper: Scraper = http_scrape) -> None:
        if not urls:
            raise PowerControlConfigError("at least one metrics URL is required")
        self.urls = tuple(urls)
        self._scraper = scraper

    def scrape_all(self) -> list[list[Sample]] | None:
        """Samples per URL, or ``None`` when any endpoint is unreachable."""
        pages = []
        for url in self.urls:
            body = self._scraper(url)
            if body is None:
                return None
            pages.append(parse_exposition(body))
        return pages

    def unreachable(self) -> list[str]:
        """URLs that do not answer right now (used by the start-up probe)."""
        return [url for url in self.urls if self._scraper(url) is None]


class ScalarReader:
    """Aggregates one metric across every endpoint into a single number."""

    def __init__(
        self, endpoints: MetricsEndpoints, selector: SeriesSelector, aggregation: str
    ) -> None:
        if aggregation not in ("sum", "avg"):
            raise PowerControlConfigError(f"unknown aggregation {aggregation!r}")
        self.endpoints = endpoints
        self.selector = selector
        self.aggregation = aggregation

    def read(self) -> float | None:
        """Aggregated value; ``None`` when unreachable or the metric is absent."""
        pages = self.endpoints.scrape_all()
        if pages is None:
            return None
        values = [
            sample.value
            for page in pages
            for sample in page
            if self.selector.matches(sample) and math.isfinite(sample.value)
        ]
        if not values:
            return None
        total = sum(values)
        return total if self.aggregation == "sum" else total / len(values)

    def describe(self) -> str:
        return f"{self.aggregation}({self.selector.describe()})"


class DifferenceReader:
    """``sum(minuend) - sum(subtrahend)`` from one scrape of every endpoint.

    The subtrahend may be absent (vLLM and SGLang publish no cached-prompt
    counter): the difference is then the minuend alone. ``None`` when an
    endpoint is unreachable or the minuend is absent everywhere.
    """

    def __init__(
        self,
        endpoints: MetricsEndpoints,
        minuend: SeriesSelector,
        subtrahend: SeriesSelector,
    ) -> None:
        self.endpoints = endpoints
        self.minuend = minuend
        self.subtrahend = subtrahend

    def read(self) -> float | None:
        pages = self.endpoints.scrape_all()
        if pages is None:
            return None
        samples = [
            sample for page in pages for sample in page if math.isfinite(sample.value)
        ]
        minuend = [sample.value for sample in samples if self.minuend.matches(sample)]
        if not minuend:
            return None
        subtrahend = [
            sample.value for sample in samples if self.subtrahend.matches(sample)
        ]
        return sum(minuend) - sum(subtrahend)

    def describe(self) -> str:
        return f"sum({self.minuend.describe()}) - sum({self.subtrahend.describe()})"


class HistogramReader:
    """Cumulative bucket counts of one histogram, summed across endpoints and series."""

    def __init__(
        self,
        endpoints: MetricsEndpoints,
        base_name: str,
        labels: tuple[tuple[str, str], ...] = (),
    ) -> None:
        self.endpoints = endpoints
        self.base_name = base_name
        self.selector = SeriesSelector((f"{base_name}_bucket",), labels)

    def read(self) -> dict[float, float] | None:
        """``{upper_bound: cumulative_count}``; ``None`` when unreachable or absent."""
        pages = self.endpoints.scrape_all()
        if pages is None:
            return None
        buckets: dict[float, float] = {}
        for page in pages:
            for sample in page:
                if not self.selector.matches(sample) or "le" not in sample.labels:
                    continue
                try:
                    upper = float(sample.labels["le"])
                except ValueError:
                    logger.warning(
                        "histogram %s has a bad le=%r",
                        self.base_name,
                        sample.labels["le"],
                    )
                    continue
                buckets[upper] = buckets.get(upper, 0.0) + sample.value
        return buckets or None

    def describe(self) -> str:
        return self.base_name


class EnergySnapshotReader:
    """Total GPU energy of the pool's nodes from EPD's ``node_energy_<node>.json`` files.

    Each node's node_agent rotates its snapshot to ``.prev`` before writing a
    new one, so a missing current file falls back to ``.prev`` (energy is
    monotonic, a slightly older reading only shifts the interval by a few
    milliseconds). A node with neither file yields ``None``.
    """

    def __init__(self, snapshot_dir: str, nodes: Sequence[str]) -> None:
        self.snapshot_dir = snapshot_dir
        self.nodes = tuple(nodes)

    def read_total_mj(self) -> float | None:
        total = 0.0
        for node in self.nodes:
            energy = self._read_node_mj(node)
            if energy is None:
                return None
            total += energy
        return total

    def _read_node_mj(self, node: str) -> float | None:
        current = os.path.join(self.snapshot_dir, f"node_energy_{node}.json")
        for path in (current, f"{current}.prev"):
            try:
                with open(path, encoding="utf-8") as handle:
                    payload = json.load(handle)
            except FileNotFoundError:
                continue
            except json.JSONDecodeError as exc:
                logger.warning("energy snapshot %s is not valid JSON: %s", path, exc)
                continue
            energy_mj = (
                payload.get("energy_mJ") if isinstance(payload, Mapping) else None
            )
            if not isinstance(energy_mj, Mapping):
                logger.warning("energy snapshot %s has no energy_mJ map", path)
                continue
            return float(sum(float(value) for value in energy_mj.values()))
        return None

    def describe(self) -> str:
        return f"energy snapshots of {len(self.nodes)} nodes in {self.snapshot_dir}"


# ----------------------------------------------------------------- providers


def histogram_quantile(quantile: float, buckets: Mapping[float, float]) -> float | None:
    """Prometheus-style ``histogram_quantile`` over per-interval bucket deltas.

    ``buckets`` maps the upper bound (``+Inf`` included) to the cumulative
    count. Returns ``None`` when no observation falls in the histogram.
    """
    if not buckets or not 0.0 <= quantile <= 1.0:
        return None
    bounds = sorted(buckets)
    if not math.isinf(bounds[-1]):
        return None
    total = buckets[bounds[-1]]
    if total <= 0:
        return None
    rank = quantile * total
    previous_bound = 0.0
    previous_count = 0.0
    for index, bound in enumerate(bounds):
        count = buckets[bound]
        if count >= rank:
            if math.isinf(bound):
                return previous_bound if index > 0 else None
            if count <= previous_count:
                return bound
            fraction = (rank - previous_count) / (count - previous_count)
            return previous_bound + (bound - previous_bound) * fraction
        previous_bound = bound
        previous_count = count
    return None


@dataclass
class _Anchor:
    at: float
    value: float


class CounterRateProvider:
    """``get_perf()`` = (counter delta) / (seconds since the previous sample) / divisor."""

    def __init__(
        self,
        reader: ScalarReader | DifferenceReader,
        *,
        divisor: float = 1.0,
        unit: str = "tokens/s",
        clock: Clock = time.monotonic,
    ) -> None:
        if divisor <= 0:
            raise PowerControlConfigError(f"divisor must be > 0, got {divisor}")
        self.reader = reader
        self.divisor = divisor
        self.unit = unit
        self._clock = clock
        self._anchor: _Anchor | None = None

    def get_perf(self) -> float:
        value = self.reader.read()
        now = self._clock()
        if value is None:
            return 0.0
        if self._anchor is None:
            self._anchor = _Anchor(at=now, value=value)
            return 0.0
        elapsed_s = now - self._anchor.at
        delta = value - self._anchor.value
        self._anchor = _Anchor(at=now, value=value)
        if delta < 0:
            logger.warning(
                "counter %s went backwards (%.0f -> %.0f); re-anchoring",
                self.reader.describe(),
                value - delta,
                value,
            )
            return 0.0
        if elapsed_s <= 0 or delta == 0:
            return 0.0
        return delta / elapsed_s / self.divisor

    def probe(self) -> list[str]:
        return [
            f"unreachable metrics URL {url}"
            for url in self.reader.endpoints.unreachable()
        ]

    def describe(self) -> str:
        divisor = f" / {self.divisor:g}" if self.divisor != 1.0 else ""
        return f"rate({self.reader.describe()}){divisor} [{self.unit}]"


class GaugeProvider:
    """``get_perf()`` = the aggregated gauge / divisor.

    EPD's higher-is-better contract is checked on the aggregate, the
    pool-wide number EPD sees: a negative sum or mean is reported as no
    signal (``0.0``), while a single negative sample that leaves the
    aggregate at or above zero only lowers it.
    """

    def __init__(
        self, reader: ScalarReader, *, divisor: float = 1.0, unit: str = ""
    ) -> None:
        if divisor <= 0:
            raise PowerControlConfigError(f"divisor must be > 0, got {divisor}")
        self.reader = reader
        self.divisor = divisor
        self.unit = unit

    def get_perf(self) -> float:
        value = self.reader.read()
        if value is None:
            return 0.0
        if value < 0:
            logger.warning(
                "gauge %s is negative (%.3f); EPD needs higher-is-better, treating "
                "as no signal",
                self.reader.describe(),
                value,
            )
            return 0.0
        return value / self.divisor

    def probe(self) -> list[str]:
        return [
            f"unreachable metrics URL {url}"
            for url in self.reader.endpoints.unreachable()
        ]

    def describe(self) -> str:
        divisor = f" / {self.divisor:g}" if self.divisor != 1.0 else ""
        unit = f" [{self.unit}]" if self.unit else ""
        return f"{self.reader.describe()}{divisor}{unit}"


class HistogramQuantileProvider:
    """A per-interval quantile of a Prometheus histogram, optionally inverted.

    Bucket counts are differenced between cycles, so the quantile describes
    only the requests observed in the last interval. With ``invert`` the value
    is ``1 / quantile`` so that a latency becomes higher-is-better.
    """

    def __init__(
        self,
        reader: HistogramReader,
        *,
        quantile: float,
        invert: bool,
        unit: str,
    ) -> None:
        if not 0.0 < quantile < 1.0:
            raise PowerControlConfigError(f"quantile must be in (0, 1), got {quantile}")
        self.reader = reader
        self.quantile = quantile
        self.invert = invert
        self.unit = unit
        self._previous: dict[float, float] | None = None

    def get_perf(self) -> float:
        buckets = self.reader.read()
        if buckets is None:
            return 0.0
        previous = self._previous
        self._previous = dict(buckets)
        if previous is None or set(previous) != set(buckets):
            return 0.0
        deltas = {bound: buckets[bound] - previous[bound] for bound in buckets}
        if any(delta < 0 for delta in deltas.values()):
            logger.warning(
                "histogram %s went backwards; re-anchoring", self.reader.describe()
            )
            return 0.0
        value = histogram_quantile(self.quantile, deltas)
        if value is None or value <= 0:
            return 0.0
        return 1.0 / value if self.invert else value

    def probe(self) -> list[str]:
        return [
            f"unreachable metrics URL {url}"
            for url in self.reader.endpoints.unreachable()
        ]

    def describe(self) -> str:
        core = f"p{round(self.quantile * 100)}({self.reader.describe()})"
        return f"1 / {core} [{self.unit}]" if self.invert else f"{core} [{self.unit}]"


class TokensPerJouleProvider:
    """``get_perf()`` = output tokens in the interval / GPU energy in the interval [tokens/J]."""

    def __init__(self, tokens: ScalarReader, energy: EnergySnapshotReader) -> None:
        self.tokens = tokens
        self.energy = energy
        self._anchor: tuple[float, float] | None = None

    def get_perf(self) -> float:
        token_total = self.tokens.read()
        energy_mj = self.energy.read_total_mj()
        if token_total is None or energy_mj is None:
            return 0.0
        if self._anchor is None:
            self._anchor = (token_total, energy_mj)
            return 0.0
        last_tokens, last_energy_mj = self._anchor
        self._anchor = (token_total, energy_mj)
        delta_tokens = token_total - last_tokens
        delta_energy_j = (energy_mj - last_energy_mj) / 1000.0
        if delta_tokens < 0 or delta_energy_j < 0:
            logger.warning(
                "tokens or energy went backwards; re-anchoring tokens_per_joule"
            )
            return 0.0
        if delta_tokens == 0 or delta_energy_j <= 0:
            return 0.0
        return delta_tokens / delta_energy_j

    def probe(self) -> list[str]:
        return [
            f"unreachable metrics URL {url}"
            for url in self.tokens.endpoints.unreachable()
        ]

    def describe(self) -> str:
        return f"delta({self.tokens.describe()}) / delta({self.energy.describe()}) [tokens/J]"


class TracingProvider:
    """Wraps a provider and appends every reading to a CSV (``ts,value``)."""

    def __init__(self, inner, csv_path: str) -> None:
        self.inner = inner
        self.csv_path = csv_path

    def get_perf(self) -> float:
        value = float(self.inner.get_perf())
        new_file = (
            not os.path.exists(self.csv_path) or os.path.getsize(self.csv_path) == 0
        )
        os.makedirs(os.path.dirname(self.csv_path) or ".", exist_ok=True)
        with open(self.csv_path, "a", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            if new_file:
                writer.writerow(["unix_time_s", "fom"])
            writer.writerow([f"{time.time():.3f}", f"{value:.6g}"])
        return value

    def probe(self) -> list[str]:
        return probe_provider(self.inner)

    def describe(self) -> str:
        return describe_provider(self.inner) + f" (traced to {self.csv_path})"


def load_python_provider(target: str):
    """Instantiate ``module:Class`` and check it exposes ``get_perf``."""
    module_name, separator, class_name = target.partition(":")
    if not separator:
        raise PowerControlConfigError(
            f"python provider {target!r}: expected 'module:Class'"
        )
    module = importlib.import_module(module_name)
    try:
        provider_class = getattr(module, class_name)
    except AttributeError as exc:
        raise PowerControlConfigError(
            f"python provider {target!r}: module {module_name} has no {class_name}"
        ) from exc
    provider = provider_class()
    if not callable(getattr(provider, "get_perf", None)):
        raise PowerControlConfigError(
            f"python provider {target!r} has no callable get_perf() method"
        )
    return provider


def probe_provider(provider) -> list[str]:
    """Problems that must stop the launch (unreachable endpoints, a bad custom class)."""
    probe = getattr(provider, "probe", None)
    if callable(probe):
        return list(probe())
    value = provider.get_perf()
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return [f"get_perf() returned {value!r}, not a number"]
    if not math.isfinite(float(value)) or float(value) < 0:
        return [f"get_perf() returned {value!r}; EPD needs a finite value >= 0"]
    return []


def describe_provider(provider) -> str:
    describe = getattr(provider, "describe", None)
    if callable(describe):
        return str(describe())
    return type(provider).__qualname__


@dataclass(frozen=True)
class ProviderContext:
    """What a pool's controller knows when it builds its FoM provider."""

    urls: tuple[str, ...]
    pool_gpu_count: int
    pool_nodes: tuple[str, ...]
    log_dir: str
    scraper: Scraper = http_scrape
    clock: Clock = time.monotonic
    trace: bool = True
    extra: Mapping[str, str] = field(default_factory=dict)
    # The subsets of ``urls`` the launcher tagged ``frontend=`` / ``worker=``.
    frontend_urls: tuple[str, ...] = ()
    worker_urls: tuple[str, ...] = ()

    @property
    def snapshot_dir(self) -> str:
        return os.path.join(os.path.abspath(self.log_dir), ENERGY_SNAPSHOT_SUBDIR)

    def urls_for_scope(self, scope: str) -> tuple[str, ...]:
        """The endpoints a ``prom:`` FoM of ``scope`` scrapes (``FOM_SCOPES``)."""
        if scope == "frontend":
            return self.frontend_urls
        if scope == "pool_workers":
            return self.worker_urls
        return self.urls


def build_provider(spec: FomSpec, context: ProviderContext):
    """Build the provider for ``spec``; wraps it in a CSV trace unless disabled."""
    provider = _build_provider(spec, context)
    if context.trace:
        return TracingProvider(
            provider, os.path.join(os.path.abspath(context.log_dir), FOM_TRACE_FILENAME)
        )
    return provider


def _build_provider(spec: FomSpec, context: ProviderContext):
    if spec.source == "python":
        return load_python_provider(spec.name)
    if spec.source == "prom":
        scoped_urls = context.urls_for_scope(spec.scope)
        if not scoped_urls:
            tag = "frontend" if spec.scope == "frontend" else "worker"
            raise PowerControlConfigError(
                f"fom scope {spec.scope!r} selects no metrics URL; tag the matching "
                f"--metrics-url values as {tag}=<url>"
            )
        reader = ScalarReader(
            MetricsEndpoints(scoped_urls, context.scraper),
            SeriesSelector((spec.name,), spec.labels),
            str(spec.aggregation),
        )
        if spec.metric_kind == "counter":
            return CounterRateProvider(reader, unit="1/s", clock=context.clock)
        return GaugeProvider(reader)
    if spec.source != "preset":
        raise PowerControlConfigError(f"unknown fom source {spec.source!r}")
    endpoints = MetricsEndpoints(context.urls, context.scraper)

    def output_tokens() -> ScalarReader:
        return ScalarReader(
            endpoints, SeriesSelector((FRONTEND_OUTPUT_TOKENS_TOTAL,)), "sum"
        )

    if spec.name == "output_tokens":
        return CounterRateProvider(
            output_tokens(), unit="tokens/s", clock=context.clock
        )
    if spec.name == "tokens_per_gpu":
        if context.pool_gpu_count <= 0:
            raise PowerControlConfigError("tokens_per_gpu needs a pool GPU count > 0")
        return CounterRateProvider(
            output_tokens(),
            divisor=float(context.pool_gpu_count),
            unit="tokens/s/GPU",
            clock=context.clock,
        )
    if spec.name in ("ctx_tokens", "gen_tokens"):
        # The workers' own /metrics; an untagged URL list falls back to every endpoint.
        worker_endpoints = MetricsEndpoints(
            context.worker_urls or context.urls, context.scraper
        )
        if spec.name == "ctx_tokens":
            difference = DifferenceReader(
                worker_endpoints,
                SeriesSelector(ENGINE_PROMPT_TOKENS_TOTAL),
                SeriesSelector(ENGINE_PROMPT_CACHED_TOKENS_TOTAL),
            )
            return CounterRateProvider(difference, unit="tokens/s", clock=context.clock)
        reader = ScalarReader(
            worker_endpoints, SeriesSelector(ENGINE_GENERATION_TOKENS_TOTAL), "sum"
        )
        return CounterRateProvider(reader, unit="tokens/s", clock=context.clock)
    if spec.name == "per_user_speed_p25":
        return HistogramQuantileProvider(
            HistogramReader(endpoints, FRONTEND_ITL_SECONDS),
            quantile=0.75,
            invert=True,
            unit="tokens/s per stream",
        )
    if spec.name == "inv_ttft_p50":
        return HistogramQuantileProvider(
            HistogramReader(endpoints, FRONTEND_TTFT_SECONDS),
            quantile=0.5,
            invert=True,
            unit="1/s",
        )
    if spec.name == "tokens_per_joule":
        return TokensPerJouleProvider(
            output_tokens(),
            EnergySnapshotReader(context.snapshot_dir, context.pool_nodes),
        )
    raise PowerControlConfigError(f"unknown fom preset {spec.name!r}")


__all__ = [
    "CounterRateProvider",
    "DifferenceReader",
    "EnergySnapshotReader",
    "GaugeProvider",
    "HistogramQuantileProvider",
    "HistogramReader",
    "MetricsEndpoints",
    "ProviderContext",
    "Sample",
    "ScalarReader",
    "SeriesSelector",
    "TokensPerJouleProvider",
    "TracingProvider",
    "build_provider",
    "describe_provider",
    "histogram_quantile",
    "http_scrape",
    "load_python_provider",
    "parse_exposition",
    "probe_provider",
]
