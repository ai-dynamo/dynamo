# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pytest
from learned_routing import HARNESS_VERSION, report
from learned_routing.cache import ResultCache, bindings_build_id, cache_key
from learned_routing.paths import Layout
from learned_routing.policy import default_reference


# -- report -------------------------------------------------------------------------------
def test_bootstrap_is_seeded_and_brackets_the_point():
    cells = {f"c{i}": [1.0 + 0.1 * i, 1.05 + 0.1 * i] for i in range(8)}
    point, lo, hi = report.bootstrap_ci(cells, n_boot=500, seed=3)
    assert lo < point < hi
    assert report.bootstrap_ci(cells, n_boot=500, seed=3) == (point, lo, hi)
    # clustering every cell into one segment leaves only replicate noise
    one = report.bootstrap_ci(cells, {c: "seg" for c in cells}, n_boot=500, seed=3)
    assert one[0] == pytest.approx(point) and (one[2] - one[1]) < (hi - lo)


def test_iqm_and_geomean():
    assert report.iqm([1, 2, 3, 4, 100, 5, 6, 7]) == pytest.approx(4.5)
    assert report.geomean([1.0, 4.0]) == pytest.approx(2.0)
    assert report.geomean([1.0, 0.0]) is None


def records_for(policies, cells, ks, value):
    out = []
    for name, sha in policies:
        for cell in cells:
            for k in ks:
                out.append(
                    {
                        "cache_key": f"{sha}-{cell}-{k}",
                        "policy_sha": sha,
                        "policy_name": name,
                        "cell_id": cell,
                        "repeat": k,
                        "split": "val",
                        "family": "mooncake",
                        "num_workers": 4 if cell.endswith("a") else 8,
                        "goodput_rps_window": value(name, cell, k),
                        "rescore": {
                            "1.0": {"goodput_rps_window": value(name, cell, k)}
                        },
                        "guards": {"worker_share_max": 0.3},
                        "error": None,
                    }
                )
    return out


def test_report_pairs_by_cell_and_replicate_and_writes_outputs(tmp_path):
    ref = default_reference().sha
    policies = [("default@defaults", ref), ("cand", "f" * 64)]

    def value(name, cell, k):
        base = 1.0 + 0.5 * k  # replicate effect shared by both policies (CRN)
        return base * (1.2 if name == "cand" else 1.0)

    records = records_for(policies, ["xa", "ya", "zb"], [0, 1, 2], value)
    rep = report.Report(records, n_boot=300)
    rows = {r["policy"]: r for r in rep.summarize()}
    assert rows["cand"]["norm_mean"] == pytest.approx(1.2)
    # paired ratios remove the replicate effect, so the CI collapses onto 1.2
    assert rows["cand"]["norm_lo"] == pytest.approx(1.2) and rows["cand"][
        "norm_hi"
    ] == pytest.approx(1.2)
    assert rows["default@defaults"]["norm_mean"] == pytest.approx(1.0)
    by_n = rep.grouped("num_workers")
    assert {r["num_workers"] for r in by_n} == {"4", "8"}
    payload = report.write_report(rep, tmp_path / "out")
    assert (tmp_path / "out" / "report.md").exists() and (
        tmp_path / "out" / "summary.json"
    ).exists()
    assert any(f.endswith("policy_summary.png") for f in payload["figures"])
    assert payload["winners"][0]["winner"] == "cand"


def test_report_drops_pairs_without_a_positive_reference():
    ref = default_reference().sha
    records = records_for(
        [("default@defaults", ref), ("cand", "e" * 64)],
        ["xa"],
        [0, 1],
        lambda name, cell, k: 0.0 if (name == "default@defaults" and k == 0) else 1.0,
    )
    rows = {r["policy"]: r for r in report.Report(records, n_boot=50).summarize()}
    assert rows["cand"]["dropped_pairs"] == 1 and rows["cand"][
        "norm_mean"
    ] == pytest.approx(1.0)


def content_records():
    """build audit goodput-normalization F1: one cell_id under two loads (cell SHAs)."""
    ref = default_reference().sha

    def rec(sha, cell_sha, build, value, key):
        return {
            "cache_key": key,
            "policy_sha": sha,
            "policy_name": "default@defaults" if sha == ref else "cand",
            "cell_id": "c",
            "repeat": 0,
            "cell_sha": cell_sha,
            "build_id": build,
            "replicate_protocol": "crn-order-v1",
            "harness_version": HARNESS_VERSION,
            "goodput_rps_window": value,
            "error": None,
        }

    return ref, [
        rec(ref, "LOAD_A", "B1", 1.0, "1"),
        rec("cand", "LOAD_A", "B1", 1.1, "2"),
        rec(ref, "LOAD_B", "B1", 2.0, "3"),
    ]


def test_paired_ratios_match_cell_content_not_just_cell_id():
    ref, records = content_records()
    ratios, dropped = report.paired_ratios(records, ref, "goodput_rps_window")
    # The candidate pairs with its own load's reference (1.1, not 0.55), and each reference
    # pairs with itself.
    assert ratios[("cand", "c")] == [pytest.approx(1.1)]
    assert ratios[(ref, "c")] == [1.0, 1.0]
    # A candidate whose content has no reference is dropped, never paired across contents.
    lone = {**records[1], "cell_sha": "LOAD_C", "cache_key": "4"}
    ratios, dropped = report.paired_ratios(
        [records[0], lone], ref, "goodput_rps_window"
    )
    assert ("cand", "c") not in ratios and dropped["cand"] == 1


def test_report_rejects_or_splits_a_cell_id_with_several_contents(tmp_path):
    ref, records = content_records()
    with pytest.raises(report.MixedContentError, match="1 cell_id"):
        report.Report(records, n_boot=20)
    split = report.Report(records, n_boot=20, contents="split")
    rows = {r["policy"]: r for r in split.summarize()}
    assert rows["cand"]["norm_mean"] == pytest.approx(1.1)
    assert rows["default@defaults"]["cells"] == 2
    assert rows["default@defaults"]["norm_mean"] == pytest.approx(1.0)
    # Results spanning a rebuild: selecting one build removes the conflict.
    rebuilt = [
        {**r, "build_id": "B2", "cache_key": r["cache_key"] + "b"} for r in records[:2]
    ]
    with pytest.raises(report.MixedContentError):
        report.Report(records[:2] + rebuilt, n_boot=20)
    only_b2 = report.Report(records[:2] + rebuilt, n_boot=20, build_id="B2")
    assert {r["build_id"] for r in only_b2.records} == {"B2"}
    # The CLI refuses mixed input with exit code 2 instead of reporting mixed ratios.
    from learned_routing import report_cli

    path = tmp_path / "results.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    assert (
        report_cli.main(
            ["--results", str(path), "--out-dir", str(tmp_path / "o"), "--no-figures"]
        )
        == 2
    )
    assert (
        report_cli.main(
            [
                "--results",
                str(path),
                "--out-dir",
                str(tmp_path / "o"),
                "--no-figures",
                "--mixed-contents",
                "split",
            ]
        )
        == 0
    )


# -- E0 -----------------------------------------------------------------------------------
ENGINE = str(Layout.resolve().engine_json)


@pytest.fixture(scope="module")
def engine():
    pytest.importorskip("dynamo._internal.ais")
    if not Path(ENGINE).exists():
        pytest.skip("campaign engine.json missing (set LR_ROOT to the campaign root)")
    return json.loads(open(ENGINE).read())


def test_e0_is_chunked_prefill_plus_batch1_decode_and_persists(engine, tmp_path):
    from learned_routing.e0 import E0Table

    table = E0Table(engine, tmp_path)
    chunk = engine["mock_engine_args"]["max_num_batched_tokens"]
    s = table.session
    prefill = s.predict_prefill(1, chunk, 0) + s.predict_prefill(1, 1000, chunk)
    # the step producing token j + 2 runs at ISL + j + 2, replay's context (build audit r2 F1)
    decode = sum(
        s._estimate(
            {"num_decode_requests": 1, "sum_decode_kv_tokens": chunk + 1000 + j + 2}
        )
        for j in range(4)
    )
    assert table(chunk + 1000, 5) == pytest.approx(prefill + decode, rel=1e-12)
    # never a single unchunked call (1.5% low at 120K, setup audit r1 F6)
    assert table(chunk + 1000, 1) != pytest.approx(
        s.predict_prefill(1, chunk + 1000, 0), rel=1e-9
    )
    table.persist()
    again = E0Table(engine, tmp_path)
    assert again.values == table.values and again._session is None


def test_e0_merges_values_from_concurrent_tables(engine, tmp_path):
    from learned_routing.e0 import E0Table

    a, b = E0Table(engine, tmp_path), E0Table(engine, tmp_path)
    a(1000, 2)
    b(2000, 3)
    a.persist()
    b.persist()
    merged = json.loads(b.path.read_text())
    assert set(merged) == {"1000,2", "2000,3"}


# -- ingest -------------------------------------------------------------------------------
def test_ingest_recomputes_keys_and_rejects_foreign_records(tmp_path):
    pytest.importorskip("dynamo._core")
    from learned_routing.bundle import ingest

    layout = Layout(tmp_path / "local")
    build = bindings_build_id(layout.cache_dir)["build_id"]
    remote = ResultCache(tmp_path / "bundle" / "runs" / "cache")

    def record(**fields):
        base = dict(
            policy_sha="p",
            cell_id="c",
            cell_sha="s",
            repeat=0,
            replicate_protocol="crn-order-v1",
            harness_version=HARNESS_VERSION,
            build_id=build,
            error=None,
        )
        base.update(fields)
        key = cache_key(
            policy_sha=base["policy_sha"],
            cell_id=base["cell_id"],
            cell_sha=base["cell_sha"],
            repeat=base["repeat"],
            protocol=base["replicate_protocol"],
            harness_version=base["harness_version"],
            build_id=base["build_id"],
        )
        return key, {**base, "cache_key": key}

    good_key, good = record()
    remote.put(good_key, good)
    foreign_key, foreign = record(build_id="0" * 64, repeat=1)
    remote.put(foreign_key, foreign)
    tampered_key, tampered = record(repeat=2)
    tampered["repeat"] = 3  # fields no longer hash to the key
    remote.put(tampered_key, tampered)
    summary = ingest(tmp_path / "bundle", layout)
    assert summary["ingested"] == 1
    reasons = sorted(r["reason"].split(" ")[0] for r in summary["rejected"])
    assert reasons == ["build_id", "cache_key"]
    assert ResultCache(layout.cache_dir).get(good_key)["ingested_from"]
    assert ingest(tmp_path / "bundle", layout)["already_cached"] == 1


# -- bundle isolation (build audit B1) ------------------------------------------------------
PACKAGE_ROOT = str(Path(__file__).resolve().parents[1])


def test_workers_inherit_the_parents_isolation_flags():
    import subprocess
    import sys

    probe = "from learned_routing.pool import isolation_flags; print(isolation_flags())"
    env = {"PATH": "/usr/bin:/bin", "PYTHONPATH": PACKAGE_ROOT}
    isolated = subprocess.run(
        [sys.executable, "-S", "-c", probe],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )
    assert "'-S'" in isolated.stdout
    plain = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )
    assert "'-S'" not in plain.stdout


def test_site_check_sees_only_the_site(tmp_path):
    """An empty site must fail, so the check cannot be satisfied by the venv or dist-packages."""
    from learned_routing.bundle import BundleError, check_site

    (tmp_path / "site").mkdir()
    (tmp_path / "engine.json").write_text(json.dumps({"mock_engine_args": {}}))
    with pytest.raises(BundleError, match="No module named 'learned_routing'"):
        check_site(tmp_path / "site", tmp_path / "engine.json")
