# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the FoM providers, with injected scrape results and clocks."""

import csv
import json
import logging

import pytest

from dynamo.power import fom
from dynamo.power.config import FomSpec, PowerControlConfigError, parse_fom
from dynamo.power.fom import (
    CounterRateProvider,
    EnergySnapshotReader,
    GaugeProvider,
    HistogramQuantileProvider,
    HistogramReader,
    MetricsEndpoints,
    ProviderContext,
    ScalarReader,
    SeriesSelector,
    TokensPerJouleProvider,
    TracingProvider,
    build_provider,
    describe_provider,
    histogram_quantile,
    parse_exposition,
    probe_provider,
)

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]

FRONTEND = "http://frontend:8000/metrics"
WORKER_A = "http://worker-a:9090/metrics"
WORKER_B = "http://worker-b:9090/metrics"


class FakeScraper:
    """Serves canned ``/metrics`` bodies; ``None`` means unreachable."""

    def __init__(self, pages):
        self.pages = dict(pages)
        self.calls = []

    def __call__(self, url):
        self.calls.append(url)
        return self.pages[url]


class FakeClock:
    def __init__(self, start=1000.0):
        self.now = start

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


def frontend_page(tokens, model="m"):
    return (
        "# HELP dynamo_frontend_output_tokens_total Total output tokens\n"
        "# TYPE dynamo_frontend_output_tokens_total counter\n"
        f'dynamo_frontend_output_tokens_total{{model="{model}"}} {tokens}\n'
        'dynamo_frontend_inflight_requests{model="m"} 12\n'
    )


def worker_page(prompt, generation, prefix="vllm", cached=None):
    """A worker ``/metrics`` page; ``prefix="trtllm"`` uses TensorRT-LLM's counter names."""
    if prefix == "trtllm":
        lines = [
            f'trtllm_prompt_tokens_total{{model_name="m"}} {prompt}\n',
            f'trtllm_generation_tokens_total{{model_name="m"}} {generation}\n',
        ]
        if cached is not None:
            lines.append(
                f'trtllm_prompt_cached_tokens_total{{model_name="m"}} {cached}\n'
            )
        return "".join(lines) + 'dynamo_component_inflight_requests{dp_rank="0"} 3\n'
    return (
        f'{prefix}:prompt_tokens_total{{model_name="m"}} {prompt}\n'
        f'{prefix}:generation_tokens_total{{model_name="m"}} {generation}\n'
        'dynamo_component_inflight_requests{dp_rank="0"} 3\n'
    )


def histogram_page(name, counts):
    """``counts`` maps ``le`` (as written in the exposition) to cumulative counts."""
    lines = [f"# TYPE {name} histogram"]
    for upper, count in counts.items():
        lines.append(f'{name}_bucket{{le="{upper}",model="m"}} {count}')
    lines.append(f'{name}_count{{model="m"}} {counts["+Inf"]}')
    lines.append(f'{name}_sum{{model="m"}} 1.0')
    return "\n".join(lines) + "\n"


def output_tokens_reader(scraper, urls=(FRONTEND,)):
    endpoints = MetricsEndpoints(list(urls), scraper)
    return ScalarReader(
        endpoints, SeriesSelector((fom.FRONTEND_OUTPUT_TOKENS_TOTAL,)), "sum"
    )


def context(
    scraper,
    clock,
    urls=(FRONTEND,),
    pool_gpu_count=8,
    log_dir="/unused",
    trace=False,
    frontend_urls=(),
    worker_urls=(),
):
    return ProviderContext(
        urls=tuple(urls),
        pool_gpu_count=pool_gpu_count,
        pool_nodes=("n1", "n2"),
        log_dir=log_dir,
        scraper=scraper,
        clock=clock,
        trace=trace,
        frontend_urls=tuple(frontend_urls),
        worker_urls=tuple(worker_urls),
    )


# --------------------------------------------------------------- exposition


def test_parse_exposition_handles_labels_inf_timestamps_and_comments():
    text = (
        "# HELP x help\n"
        "# TYPE x counter\n"
        'x{a="b",c="quote\\"d"} 1.5 1700000000\n'
        'h_bucket{le="+Inf"} 42\n'
        "plain 7\n"
        "broken line without value\n"
        "bad_value abc\n"
    )
    samples = parse_exposition(text)
    assert [(s.name, s.value) for s in samples] == [
        ("x", 1.5),
        ("h_bucket", 42.0),
        ("plain", 7.0),
    ]
    assert samples[0].labels == {"a": "b", "c": 'quote"d'}
    assert samples[1].labels == {"le": "+Inf"}


def test_series_selector_matches_any_candidate_name_and_labels():
    selector = SeriesSelector(("vllm:x", "sglang:x"), (("model", "m"),))
    assert selector.matches(fom.Sample("sglang:x", {"model": "m"}, 1.0))
    assert not selector.matches(fom.Sample("sglang:x", {"model": "other"}, 1.0))
    assert not selector.matches(fom.Sample("trtllm_x", {"model": "m"}, 1.0))


# ------------------------------------------------------------- counter rate


def test_counter_rate_is_zero_on_first_sample_then_a_rate():
    scraper = FakeScraper({FRONTEND: frontend_page(1000)})
    clock = FakeClock()
    provider = CounterRateProvider(output_tokens_reader(scraper), clock=clock)
    assert provider.get_perf() == 0.0
    scraper.pages[FRONTEND] = frontend_page(4000)
    clock.advance(30)
    assert provider.get_perf() == pytest.approx(100.0)
    assert "tokens/s" in provider.describe()


def test_counter_rate_sums_across_urls_and_ignores_pages_without_the_metric():
    scraper = FakeScraper(
        {
            FRONTEND: frontend_page(1000),
            WORKER_A: worker_page(10, 20),
            "http://frontend-2:8000/metrics": frontend_page(500),
        }
    )
    clock = FakeClock()
    reader = output_tokens_reader(
        scraper, urls=(FRONTEND, WORKER_A, "http://frontend-2:8000/metrics")
    )
    provider = CounterRateProvider(reader, clock=clock)
    provider.get_perf()
    scraper.pages[FRONTEND] = frontend_page(1300)
    scraper.pages["http://frontend-2:8000/metrics"] = frontend_page(800)
    clock.advance(10)
    assert provider.get_perf() == pytest.approx(60.0)


def test_counter_reset_is_no_signal_then_recovers():
    scraper = FakeScraper({FRONTEND: frontend_page(5000)})
    clock = FakeClock()
    provider = CounterRateProvider(output_tokens_reader(scraper), clock=clock)
    provider.get_perf()
    scraper.pages[FRONTEND] = frontend_page(100)  # frontend restarted
    clock.advance(30)
    assert provider.get_perf() == 0.0
    scraper.pages[FRONTEND] = frontend_page(400)
    clock.advance(30)
    assert provider.get_perf() == pytest.approx(10.0)


def test_unreachable_url_is_no_signal_and_keeps_the_anchor():
    scraper = FakeScraper({FRONTEND: frontend_page(1000), WORKER_A: worker_page(1, 2)})
    clock = FakeClock()
    provider = CounterRateProvider(
        output_tokens_reader(scraper, urls=(FRONTEND, WORKER_A)), clock=clock
    )
    provider.get_perf()
    scraper.pages[WORKER_A] = None
    scraper.pages[FRONTEND] = frontend_page(2000)
    clock.advance(30)
    assert provider.get_perf() == 0.0
    scraper.pages[WORKER_A] = worker_page(1, 2)
    scraper.pages[FRONTEND] = frontend_page(4000)
    clock.advance(30)
    assert provider.get_perf() == pytest.approx(3000.0 / 60.0)


def test_metric_absent_everywhere_is_no_signal():
    scraper = FakeScraper({WORKER_A: worker_page(1, 2)})
    provider = CounterRateProvider(
        output_tokens_reader(scraper, urls=(WORKER_A,)), clock=FakeClock()
    )
    assert provider.get_perf() == 0.0
    assert provider.get_perf() == 0.0


def test_counter_rate_label_filter_selects_one_model():
    page = (
        frontend_page(1000, model="a")
        + 'dynamo_frontend_output_tokens_total{model="b"} 50\n'
    )
    scraper = FakeScraper({FRONTEND: page})
    clock = FakeClock()
    reader = ScalarReader(
        MetricsEndpoints([FRONTEND], scraper),
        SeriesSelector((fom.FRONTEND_OUTPUT_TOKENS_TOTAL,), (("model", "a"),)),
        "sum",
    )
    provider = CounterRateProvider(reader, clock=clock)
    provider.get_perf()
    scraper.pages[FRONTEND] = (
        frontend_page(1600, model="a")
        + 'dynamo_frontend_output_tokens_total{model="b"} 9999\n'
    )
    clock.advance(60)
    assert provider.get_perf() == pytest.approx(10.0)


def test_zero_advance_is_no_signal():
    scraper = FakeScraper({FRONTEND: frontend_page(1000)})
    clock = FakeClock()
    provider = CounterRateProvider(output_tokens_reader(scraper), clock=clock)
    provider.get_perf()
    clock.advance(30)
    assert provider.get_perf() == 0.0


# -------------------------------------------------------------------- gauge


def test_gauge_avg_across_endpoints_and_negative_aggregate_is_no_signal(caplog):
    scraper = FakeScraper({WORKER_A: worker_page(1, 2), WORKER_B: worker_page(3, 4)})
    reader = ScalarReader(
        MetricsEndpoints([WORKER_A, WORKER_B], scraper),
        SeriesSelector(("dynamo_component_inflight_requests",)),
        "avg",
    )
    assert GaugeProvider(reader).get_perf() == pytest.approx(3.0)
    scraper.pages[WORKER_B] = 'dynamo_component_inflight_requests{dp_rank="0"} -9\n'
    with caplog.at_level(logging.WARNING):
        assert GaugeProvider(reader).get_perf() == 0.0  # mean (3 - 9) / 2 = -3
    assert "is negative" in caplog.text
    # The aggregate, not each sample, is what EPD sees: one negative sample that
    # leaves the mean at or above zero only lowers it.
    scraper.pages[WORKER_B] = 'dynamo_component_inflight_requests{dp_rank="0"} -1\n'
    assert GaugeProvider(reader).get_perf() == pytest.approx(1.0)


def test_gauge_divisor():
    scraper = FakeScraper({WORKER_A: worker_page(1, 2)})
    reader = ScalarReader(
        MetricsEndpoints([WORKER_A], scraper),
        SeriesSelector(("dynamo_component_inflight_requests",)),
        "sum",
    )
    assert GaugeProvider(reader, divisor=3.0).get_perf() == pytest.approx(1.0)


# --------------------------------------------------------------- histograms


def test_histogram_quantile_interpolates_like_prometheus():
    buckets = {0.1: 10.0, 0.5: 30.0, 1.0: 40.0, float("inf"): 40.0}
    assert histogram_quantile(0.5, buckets) == pytest.approx(0.3)
    assert histogram_quantile(0.75, buckets) == pytest.approx(0.5)
    assert histogram_quantile(0.99, buckets) == pytest.approx(0.98)
    assert histogram_quantile(0.5, {float("inf"): 0.0}) is None
    assert histogram_quantile(0.5, {0.5: 1.0}) is None  # no +Inf bucket


def test_inv_ttft_p50_provider_differences_buckets_between_cycles():
    name = fom.FRONTEND_TTFT_SECONDS
    first = {"0.1": 100, "0.5": 300, "1.0": 400, "+Inf": 400}
    scraper = FakeScraper({FRONTEND: histogram_page(name, first)})
    provider = HistogramQuantileProvider(
        HistogramReader(MetricsEndpoints([FRONTEND], scraper), name),
        quantile=0.5,
        invert=True,
        unit="1/s",
    )
    assert provider.get_perf() == 0.0
    second = {"0.1": 110, "0.5": 330, "1.0": 440, "+Inf": 440}  # deltas 10/30/40/40
    scraper.pages[FRONTEND] = histogram_page(name, second)
    assert provider.get_perf() == pytest.approx(1.0 / 0.3)
    scraper.pages[FRONTEND] = histogram_page(name, second)  # no requests this cycle
    assert provider.get_perf() == 0.0
    scraper.pages[FRONTEND] = histogram_page(name, first)  # counters went backwards
    assert provider.get_perf() == 0.0


def test_per_user_speed_p25_is_inverse_p75_of_itl():
    name = fom.FRONTEND_ITL_SECONDS
    scraper = FakeScraper(
        {FRONTEND: histogram_page(name, {"0.01": 0, "0.02": 0, "+Inf": 0})}
    )
    provider = build_provider(
        parse_fom("per_user_speed_p25"), context(scraper, FakeClock())
    )
    provider.get_perf()
    scraper.pages[FRONTEND] = histogram_page(
        name, {"0.01": 50, "0.02": 100, "+Inf": 100}
    )
    # p75 of the interval: rank 75 lies in (0.01, 0.02] -> 0.015 s -> 66.7 tokens/s
    assert provider.get_perf() == pytest.approx(1.0 / 0.015)
    assert "inter_token_latency" in describe_provider(provider)


# --------------------------------------------------------- tokens per joule


def write_snapshot(directory, node, energy_mj_per_gpu, suffix=""):
    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "sequence": 3,
        "consumed": 0,
        "timestamp": 1.0,
        "node": node,
        "num_gpus": len(energy_mj_per_gpu),
        "energy_mJ": {str(i): value for i, value in enumerate(energy_mj_per_gpu)},
        "limits_mW": {},
    }
    (directory / f"node_energy_{node}.json{suffix}").write_text(json.dumps(payload))


def test_tokens_per_joule_from_snapshots(tmp_path):
    snapshot_dir = tmp_path / "epd" / "nvml" / "node_energy"
    write_snapshot(snapshot_dir, "n1", [1_000_000, 1_000_000])
    write_snapshot(snapshot_dir, "n2", [2_000_000, 2_000_000])
    scraper = FakeScraper({FRONTEND: frontend_page(1000)})
    provider = TokensPerJouleProvider(
        output_tokens_reader(scraper),
        EnergySnapshotReader(str(snapshot_dir), ("n1", "n2")),
    )
    assert provider.get_perf() == 0.0
    write_snapshot(snapshot_dir, "n1", [1_500_000, 1_500_000])
    write_snapshot(snapshot_dir, "n2", [2_250_000, 2_250_000])  # +1,500,000 mJ total
    scraper.pages[FRONTEND] = frontend_page(4000)  # +3000 tokens
    assert provider.get_perf() == pytest.approx(2.0)


def test_energy_reader_falls_back_to_prev_and_reports_missing_nodes(tmp_path):
    snapshot_dir = tmp_path / "node_energy"
    write_snapshot(snapshot_dir, "n1", [10, 20], suffix=".prev")
    reader = EnergySnapshotReader(str(snapshot_dir), ("n1", "n2"))
    assert reader.read_total_mj() is None
    write_snapshot(snapshot_dir, "n2", [5])
    assert reader.read_total_mj() == pytest.approx(35.0)
    (snapshot_dir / "node_energy_n2.json").write_text("{broken")
    assert reader.read_total_mj() is None


# ------------------------------------------------------- presets & builders


@pytest.mark.parametrize(
    ("preset", "expected"),
    [
        ("output_tokens", "dynamo_frontend_output_tokens_total"),
        ("tokens_per_gpu", "dynamo_frontend_output_tokens_total)) / 30"),
        (
            "ctx_tokens",
            "trtllm_prompt_tokens_total|vllm:prompt_tokens_total|sglang:prompt_tokens_total"
            ") - sum(trtllm_prompt_cached_tokens_total)",
        ),
        (
            "gen_tokens",
            "trtllm_generation_tokens_total|vllm:generation_tokens_total"
            "|sglang:generation_tokens_total",
        ),
        ("per_user_speed_p25", "p75(dynamo_frontend_inter_token_latency_seconds)"),
        ("inv_ttft_p50", "p50(dynamo_frontend_time_to_first_token_seconds)"),
        ("tokens_per_joule", "energy snapshots of 2 nodes"),
    ],
)
def test_presets_map_onto_real_metric_names(preset, expected):
    scraper = FakeScraper({FRONTEND: frontend_page(1)})
    provider = build_provider(
        parse_fom(preset), context(scraper, FakeClock(), pool_gpu_count=30)
    )
    assert expected in describe_provider(provider)


def test_tokens_per_gpu_normalises_by_pool_gpu_count():
    scraper = FakeScraper({FRONTEND: frontend_page(0)})
    clock = FakeClock()
    provider = build_provider(
        parse_fom("tokens_per_gpu"), context(scraper, clock, pool_gpu_count=30)
    )
    provider.get_perf()
    scraper.pages[FRONTEND] = frontend_page(90_000)
    clock.advance(30)
    assert provider.get_perf() == pytest.approx(100.0)
    with pytest.raises(PowerControlConfigError):
        build_provider(
            parse_fom("tokens_per_gpu"), context(scraper, clock, pool_gpu_count=0)
        )


def test_engine_token_presets_sum_over_workers():
    scraper = FakeScraper(
        {
            FRONTEND: frontend_page(1),
            WORKER_A: worker_page(100, 10),
            WORKER_B: worker_page(200, 20, prefix="sglang"),
        }
    )
    clock = FakeClock()
    ctx = context(scraper, clock, urls=(FRONTEND, WORKER_A, WORKER_B))
    ctx_tokens = build_provider(parse_fom("ctx_tokens"), ctx)
    gen_tokens = build_provider(parse_fom("gen_tokens"), ctx)
    ctx_tokens.get_perf()
    gen_tokens.get_perf()
    scraper.pages[WORKER_A] = worker_page(400, 40)
    scraper.pages[WORKER_B] = worker_page(500, 50, prefix="sglang")
    clock.advance(10)
    assert ctx_tokens.get_perf() == pytest.approx(60.0)
    assert gen_tokens.get_perf() == pytest.approx(6.0)


def test_ctx_tokens_subtracts_cached_prompt_tokens_on_trtllm_workers():
    scraper = FakeScraper(
        {
            FRONTEND: frontend_page(1),
            WORKER_A: worker_page(1000, 10, prefix="trtllm", cached=400),
            WORKER_B: worker_page(100, 20),
        }
    )
    clock = FakeClock()
    ctx = context(
        scraper,
        clock,
        urls=(FRONTEND, WORKER_A, WORKER_B),
        frontend_urls=(FRONTEND,),
        worker_urls=(WORKER_A, WORKER_B),
    )
    ctx_tokens = build_provider(parse_fom("ctx_tokens"), ctx)
    gen_tokens = build_provider(parse_fom("gen_tokens"), ctx)
    ctx_tokens.get_perf()
    gen_tokens.get_perf()
    # TRT-LLM worker: +1200 prompt tokens of which +600 came from the KV cache;
    # vLLM worker: +300 prompt tokens and no cached-prompt counter at all.
    scraper.pages[WORKER_A] = worker_page(2200, 70, prefix="trtllm", cached=1000)
    scraper.pages[WORKER_B] = worker_page(400, 50)
    clock.advance(10)
    assert ctx_tokens.get_perf() == pytest.approx(90.0)
    assert gen_tokens.get_perf() == pytest.approx(9.0)
    assert "trtllm_prompt_cached_tokens_total" in describe_provider(ctx_tokens)
    # Tagged worker URLs are the only ones scraped: the frontend page is not fetched.
    assert FRONTEND not in scraper.calls[-4:]


def test_prom_specs_build_counter_or_gauge_providers():
    scraper = FakeScraper({WORKER_A: worker_page(1, 2)})
    counter = build_provider(
        parse_fom("prom:vllm:generation_tokens_total:counter:sum"),
        context(scraper, FakeClock(), urls=(WORKER_A,)),
    )
    gauge = build_provider(
        parse_fom("prom:dynamo_component_inflight_requests:dp_rank=0:gauge:avg"),
        context(scraper, FakeClock(), urls=(WORKER_A,)),
    )
    assert isinstance(counter, CounterRateProvider)
    assert isinstance(gauge, GaugeProvider)
    assert gauge.get_perf() == pytest.approx(3.0)


def test_prom_scope_scrapes_only_the_tagged_endpoints():
    scraper = FakeScraper(
        {
            FRONTEND: frontend_page(1),
            WORKER_A: worker_page(1, 2),
            WORKER_B: worker_page(3, 4),
        }
    )
    ctx = context(
        scraper,
        FakeClock(),
        urls=(FRONTEND, WORKER_A, WORKER_B),
        frontend_urls=(FRONTEND,),
        worker_urls=(WORKER_A, WORKER_B),
    )
    workers = build_provider(
        parse_fom(
            "prom:dynamo_component_inflight_requests:scope=pool_workers:gauge:sum"
        ),
        ctx,
    )
    assert workers.get_perf() == pytest.approx(6.0)
    assert sorted(scraper.calls) == [WORKER_A, WORKER_B]
    scraper.calls.clear()
    frontend = build_provider(
        parse_fom("prom:dynamo_frontend_inflight_requests:scope=frontend:gauge:sum"),
        ctx,
    )
    assert frontend.get_perf() == pytest.approx(12.0)
    assert scraper.calls == [FRONTEND]
    scraper.calls.clear()
    everywhere = build_provider(
        parse_fom("prom:dynamo_component_inflight_requests:gauge:sum"), ctx
    )
    assert everywhere.get_perf() == pytest.approx(6.0)
    assert sorted(scraper.calls) == sorted([FRONTEND, WORKER_A, WORKER_B])


def test_prom_scope_without_a_tagged_endpoint_is_refused():
    ctx = context(FakeScraper({FRONTEND: frontend_page(1)}), FakeClock())
    with pytest.raises(PowerControlConfigError, match="worker=<url>"):
        build_provider(parse_fom("prom:x_total:scope=pool_workers:counter:sum"), ctx)
    with pytest.raises(PowerControlConfigError, match="frontend=<url>"):
        build_provider(parse_fom("prom:x_total:scope=frontend:counter:sum"), ctx)


def test_python_provider_loader_and_probe():
    ctx = context(FakeScraper({}), FakeClock(), urls=(FRONTEND,))
    provider = build_provider(
        FomSpec("python", "tests.power.fake_provider:FakeProvider"), ctx
    )
    assert provider.get_perf() == 42.0
    assert probe_provider(provider) == []
    assert describe_provider(provider) == "FakeProvider"
    with pytest.raises(PowerControlConfigError, match="no callable get_perf"):
        build_provider(
            FomSpec("python", "tests.power.fake_provider:NoPerfProvider"), ctx
        )
    with pytest.raises(PowerControlConfigError, match="has no Missing"):
        build_provider(FomSpec("python", "tests.power.fake_provider:Missing"), ctx)
    with pytest.raises(ImportError):
        build_provider(FomSpec("python", "tests.power.no_such_module:X"), ctx)
    negative = build_provider(
        FomSpec("python", "tests.power.fake_provider:NegativeProvider"), ctx
    )
    assert (
        probe_provider(negative) and "finite value >= 0" in probe_provider(negative)[0]
    )


def test_probe_reports_unreachable_urls():
    scraper = FakeScraper({FRONTEND: frontend_page(1), WORKER_A: None})
    provider = build_provider(
        parse_fom("output_tokens"),
        context(scraper, FakeClock(), urls=(FRONTEND, WORKER_A)),
    )
    assert probe_provider(provider) == [f"unreachable metrics URL {WORKER_A}"]


def test_tracing_provider_appends_every_reading(tmp_path):
    scraper = FakeScraper({FRONTEND: frontend_page(0)})
    clock = FakeClock()
    provider = build_provider(
        parse_fom("output_tokens"),
        context(scraper, clock, log_dir=str(tmp_path), trace=True),
    )
    assert isinstance(provider, TracingProvider)
    provider.get_perf()
    scraper.pages[FRONTEND] = frontend_page(300)
    clock.advance(30)
    provider.get_perf()
    with open(tmp_path / "fom.csv", newline="") as handle:
        rows = list(csv.reader(handle))
    assert rows[0] == ["unix_time_s", "fom"]
    assert [float(row[1]) for row in rows[1:]] == [0.0, pytest.approx(10.0)]
