# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the nightly selector generator: selector rows and the ledger.

Run: pytest -c docs/fern/scripts/pytest.ini docs/fern/scripts/test_gen_nightly_selector.py

The repository's root pytest run ignores `docs/`, so the `gen-llms-tables`
pre-commit hook is the runner for this file. Everything here is stubbed: no
network, no git history, no clock.

The data mirrors the real failure modes. `NIGHTS` is newest first and longer
than the ledger window; a "missing companion" is a night where the `ai-dynamo`
wheel published but the `ai-dynamo-runtime` wheel it pins did not, which has
happened (20260616).
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

import gen_nightly_selector as gen  # noqa: E402

pytestmark = [pytest.mark.pre_merge, pytest.mark.gpu_0, pytest.mark.unit]

BASE = "1.6.0"
NIGHTS = ["20260929", "20260928", "20260927", "20260926", "20260925"]
# One night past the three-row ledger window, to prove the window closes.
OUTSIDE_WINDOW = "20260923"
ALL_NIGHTS = [*NIGHTS, OUTSIDE_WINDOW]
# Every backend pin name in these tests resolves to its night's version through
# gen.backend_version(): leading "v" optional, "-cu<N>-runtime" stripped. One
# version per night keeps the selector rows on the multi-version path.
BACKEND_VERSIONS = {night: f"1.6.{index}" for index, night in enumerate(ALL_NIGHTS)}
MONTHS = {
    "20260929": "Sep 29, 2026",
    "20260928": "Sep 28, 2026",
    "20260927": "Sep 27, 2026",
    "20260926": "Sep 26, 2026",
    "20260925": "Sep 25, 2026",
}


def wheel(night: str) -> str:
    """The wheel version a dated nightly tag publishes."""
    return f"{BASE}.dev{night}"


def sha(night: str) -> str:
    return f"abc{night}"


def ledger_index(
    nights: list[str],
    kvbm: list[str] | None = None,
    runtime: list[str] | None = None,
) -> dict[str, set[str]]:
    """A published-version map: ``nights`` published both required wheels.

    ``runtime`` replaces the ``ai-dynamo-runtime`` set, for the night where the
    companion wheel is missing. ``kvbm`` defaults to publishing every night.
    """
    published: dict[str, set[str]] = {
        package: set() for package in gen.NIGHTLY_PACKAGES
    }
    published["ai-dynamo"] = {wheel(night) for night in nights}
    published["ai-dynamo-runtime"] = (
        published["ai-dynamo"]
        if runtime is None
        else {wheel(night) for night in runtime}
    )
    published["kvbm"] = {wheel(night) for night in (nights if kvbm is None else kvbm)}
    return published


def context_doc(sha: str) -> dict:
    """A ``container/context.yaml`` stand-in for the night a tag names."""
    tag = f"{BACKEND_VERSIONS[sha.removeprefix('abc')]}-cu13-runtime"
    return {fw.key: {fw.device: {"runtime_image_tag": tag}} for fw in gen.FRAMEWORKS}


@pytest.fixture
def ngc(monkeypatch):
    """Fake the NGC tag walk and the git lookups ``build()`` reads.

    ``install(backend_nights)`` gives each backend its own dated tags, so a
    backend whose registry call failed can be modelled with an empty list.
    """

    def install(backend_nights: dict[str, list[str]] | None = None) -> None:
        overrides = backend_nights or {}
        tags = {
            fw.image: [
                (night, sha(night)) for night in overrides.get(fw.backend, ALL_NIGHTS)
            ]
            for fw in gen.FRAMEWORKS
        }
        monkeypatch.setattr(gen, "dated_tags", lambda image: tags[image])
        monkeypatch.setattr(gen, "context_at", context_doc)
        monkeypatch.setattr(gen, "base_version_at", lambda _sha: BASE)
        monkeypatch.setattr(gen, "pins_moved_since", lambda _sha: False)

    return install


@pytest.fixture
def index(monkeypatch):
    """Fake the package indexes ``main()`` reads."""

    def install(published: dict[str, set[str]] | None):
        monkeypatch.setattr(gen, "published_nightly_packages", lambda: published)

    return install


class TestInstallableWheels:
    """A wheel command is only emitted where pip can actually resolve it."""

    def test_installable_set_is_the_required_package_intersection(self):
        published = ledger_index(NIGHTS, runtime=NIGHTS[1:])

        assert gen.installable_wheels(published) == {wheel(n) for n in NIGHTS[1:]}

    def test_night_published_by_only_one_required_package_is_not_installable(self):
        # ai-dynamo 1.6.0.devN pins ai-dynamo-runtime==1.6.0.devN, so a night
        # without the companion wheel installs nothing.
        published = ledger_index([NIGHTS[0]], runtime=[])

        assert not gen.ledger_version_published(wheel(NIGHTS[0]), published)
        assert gen.installable_wheels(published) == set()

    def test_both_required_packages_make_a_night_installable(self):
        published = ledger_index([NIGHTS[0]], runtime=[NIGHTS[0]])

        assert gen.ledger_version_published(wheel(NIGHTS[0]), published)

    def test_optional_package_is_never_required(self):
        # kvbm is deprecated with removal targeted for v1.6.0: its wheel missing
        # (or its index gone) must not cost the ledger a row.
        published = ledger_index([NIGHTS[0]], kvbm=[])

        assert gen.ledger_version_published(wheel(NIGHTS[0]), published)


class TestLedgerRows:
    """Rows advertise exactly the packages that published that night."""

    def test_packages_column_lists_every_package_that_published(self):
        published = ledger_index([NIGHTS[0]], kvbm=[NIGHTS[0]])

        assert gen.ledger_packages_published(wheel(NIGHTS[0]), published) == [
            "ai-dynamo",
            "ai-dynamo-runtime",
            "kvbm",
        ]

    def test_packages_column_omits_a_package_that_did_not_publish(self):
        published = ledger_index([NIGHTS[0]], kvbm=[])

        assert gen.ledger_packages_published(wheel(NIGHTS[0]), published) == [
            "ai-dynamo",
            "ai-dynamo-runtime",
        ]


class TestLedgerAssembly:
    def test_ledger_is_newest_first_and_stops_at_the_window(self):
        published = ledger_index(ALL_NIGHTS)
        ledger = gen.build_ledger(gen.installable_wheels(published), published)

        assert [row.version for row in ledger] == [wheel(n) for n in NIGHTS[:3]]
        assert [row.date for row in ledger] == [MONTHS[n] for n in NIGHTS[:3]]

    def test_ledger_skips_a_night_whose_companion_wheel_is_missing(self):
        # Guard probe: dropping the completeness rule would let the incomplete
        # night stand in for a row that installs nothing.
        missing = NIGHTS[0]
        published = ledger_index(NIGHTS, runtime=NIGHTS[1:])
        ledger = gen.build_ledger(gen.installable_wheels(published), published)

        assert wheel(missing) not in [row.version for row in ledger]
        assert [row.version for row in ledger] == [wheel(n) for n in NIGHTS[1:4]]

    def test_ledger_newest_row_is_the_newest_installable_nightly(self):
        # Both views resolve from one wheel set, so the ledger can never name an
        # older nightly than a latest row; it can name a newer one.
        published = ledger_index(NIGHTS)
        wheels = gen.installable_wheels(published)
        ledger = gen.build_ledger(wheels, published)

        assert ledger[0].version == gen.newest_published(wheels) == wheel(NIGHTS[0])

    def test_ledger_window_is_the_declared_size(self):
        published = ledger_index(ALL_NIGHTS)
        ledger = gen.build_ledger(gen.installable_wheels(published), published)

        assert gen.NIGHTLY_LEDGER_BUILDS == 3
        assert len(ledger) == gen.NIGHTLY_LEDGER_BUILDS

    def test_ledger_is_empty_when_no_night_is_installable(self):
        published = ledger_index([], runtime=[])

        assert gen.build_ledger(gen.installable_wheels(published), published) == []


class TestSelectorRows:
    """A backend's registry failure costs that backend's rows, nothing else."""

    def test_one_backend_without_dated_tags_is_skipped(self, ngc):
        # A transport failure on one backend's tag list returns [], which used
        # to empty the ledger and fail the publish. It must cost only that
        # backend's selector rows.
        ngc({"vllm": []})
        published = ledger_index(ALL_NIGHTS)

        rows = gen.build(gen.installable_wheels(published))

        assert not [row for row in rows if row.backend == "vllm"]
        assert [row.backend for row in rows if row.latest] == ["sglang", "trtllm"]
        assert len(rows) == 2 * gen.NIGHTLY_VERSIONS_BACK

    def test_latest_row_pins_the_newest_installable_wheel(self, ngc):
        ngc()
        published = ledger_index(ALL_NIGHTS)

        rows = gen.build(gen.installable_wheels(published))

        assert [row.dynamo for row in rows if row.latest] == [wheel(NIGHTS[0])] * 3


class TestMain:
    """The publish-time entry point, with both indexes faked."""

    def test_main_feeds_one_installable_set_to_both_views(
        self, tmp_path, ngc, monkeypatch
    ):
        # The newest night has no ai-dynamo-runtime wheel, and the kvbm index is
        # down. main() must still write the module, and neither the ledger nor
        # the selector rows can name the incomplete night. Stubbing
        # published_wheels() rather than published_nightly_packages() is what
        # exercises the wiring in main().
        ngc()
        published = ledger_index(ALL_NIGHTS, runtime=ALL_NIGHTS[1:])
        published["kvbm"] = None
        monkeypatch.setattr(gen, "published_wheels", lambda name: published[name])
        out = tmp_path / "nightly.generated.ts"

        assert gen.main(["--out", str(out)]) == 0

        module = out.read_text()
        assert wheel(NIGHTS[0]) not in module
        assert f'{{ version: "{wheel(NIGHTS[1])}"' in module
        assert '"kvbm"' not in module

    def test_writes_the_module_when_the_ledger_is_complete(self, tmp_path, ngc, index):
        ngc()
        index(ledger_index(ALL_NIGHTS))
        out = tmp_path / "nightly.generated.ts"

        assert gen.main(["--out", str(out)]) == 0

        module = out.read_text()
        assert module.count("{ version: ") == gen.NIGHTLY_LEDGER_BUILDS
        assert wheel(NIGHTS[0]) in module
        assert wheel(OUTSIDE_WINDOW) not in module

    def test_refuses_to_replace_the_ledger_with_too_few_rows(
        self, tmp_path, ngc, index, capsys
    ):
        # The guard that keeps an empty or truncated ledger off the site.
        ngc()
        index(ledger_index(NIGHTS[: gen.NIGHTLY_LEDGER_BUILDS - 1]))
        out = tmp_path / "nightly.generated.ts"

        assert gen.main(["--out", str(out)]) == 1

        assert not out.exists()
        assert "fewer than 3 complete nightly ledger rows" in capsys.readouterr().err

    def test_refuses_to_write_when_a_required_index_is_unreachable(
        self, tmp_path, ngc, index
    ):
        ngc()
        index(None)
        out = tmp_path / "nightly.generated.ts"

        assert gen.main(["--out", str(out)]) == 1

        assert not out.exists()

    def test_one_backend_registry_failure_still_writes_the_ledger(
        self, tmp_path, ngc, index
    ):
        # Before this, the ledger needed a dated tag from every backend, so one
        # registry call failing failed the whole docs publish.
        ngc({"vllm": []})
        index(ledger_index(ALL_NIGHTS))
        out = tmp_path / "nightly.generated.ts"

        assert gen.main(["--out", str(out)]) == 0

        module = out.read_text()
        assert module.count("{ version: ") == gen.NIGHTLY_LEDGER_BUILDS
        assert wheel(NIGHTS[0]) in module

    def test_offline_writes_an_empty_module_for_local_previews(
        self, tmp_path, ngc, index
    ):
        index(ledger_index(ALL_NIGHTS))
        out = tmp_path / "nightly.generated.ts"

        assert gen.main(["--offline", "--out", str(out)]) == 0

        module = out.read_text()
        assert "NIGHTLY_BACKEND_BUILDS: NightlyBackendBuild[] = [\n];" in module
        assert "NIGHTLY_BUILDS: NightlyBuild[] = [\n];" in module
