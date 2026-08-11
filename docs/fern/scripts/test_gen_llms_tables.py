# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the generated tables in gen_llms_tables.py.

Run: pytest -c docs/fern/scripts/pytest.ini docs/fern/scripts/test_gen_llms_tables.py

CI runs this through the `gen-llms-tables` pre-commit hook. The repo-root
pytest config ignores docs/, so the standalone pytest.ini beside this file
supplies the marker registrations.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

import gen_llms_tables as gen  # noqa: E402

# Pure-Python table rendering: no GPU, no engine, no network. Registered in
# pyproject.toml even though the root pytest run ignores docs/ -- the gate that
# runs this file is the `gen-llms-tables` pre-commit hook.
pytestmark = [pytest.mark.pre_merge, pytest.mark.gpu_0, pytest.mark.unit]


def data(cuda_rows: list[dict], versions: list[str]) -> dict:
    """A minimal parsed-module stand-in carrying only what the table reads.

    ``versions`` is in RELEASES order (newest first). A ``.postN`` entry is
    recorded as a patch, matching how releases.data.ts classifies post trains.
    """
    return {
        "CUDA_HISTORY": cuda_rows,
        "RELEASES": [
            {"version": f"v{v}", "kind": "patch" if ".post" in v else "stable"}
            for v in versions
        ],
    }


class TestParseDriverFloor:
    def test_reads_the_leading_number(self):
        assert gen.parse_driver_floor("580.xx+") == 580

    def test_orders_numerically_not_lexically(self):
        """A three-digit string compare puts '1000.xx+' before '580.xx+'."""
        floors = sorted(
            [gen.parse_driver_floor(d) for d in ("580.xx+", "1000.xx+", "575.xx+")]
        )
        assert floors == [575, 580, 1000]

    def test_rejects_a_floor_it_cannot_read(self):
        with pytest.raises(gen.TSParseError, match="minDriver"):
            gen.parse_driver_floor("any")


class TestDriverFloorTable:
    def test_one_row_per_distinct_floor_ascending(self):
        table = gen.driver_floor_table(
            data(
                [
                    {"version": "1.3.0", "backend": "vLLM", "minDriver": "580.xx+"},
                    {"version": "1.2.0", "backend": "vLLM", "minDriver": "575.xx+"},
                    {"version": "1.1.0", "backend": "vLLM", "minDriver": "570.xx+"},
                ],
                ["1.3.0", "1.2.0", "1.1.0"],
            )
        )
        floors = [ln.split("|")[1].strip() for ln in table.splitlines()[2:]]
        assert floors == ["570.xx+", "575.xx+", "580.xx+"]

    def test_a_new_floor_flows_through_without_a_code_change(self):
        table = gen.driver_floor_table(
            data(
                [
                    {"version": "1.4.0", "backend": "vLLM", "minDriver": "600.xx+"},
                    {"version": "1.3.0", "backend": "vLLM", "minDriver": "580.xx+"},
                ],
                ["1.4.0", "1.3.0"],
            )
        )
        assert "600.xx+" in table
        assert len(table.splitlines()) == 4  # header, rule, two floors

    def test_a_higher_driver_can_run_the_lower_floor_releases(self):
        """580 satisfies a 575 floor, so the newest 575 release is runnable."""
        table = gen.driver_floor_table(
            data(
                [
                    {"version": "1.3.0", "backend": "SGLang", "minDriver": "580.xx+"},
                    {"version": "1.2.0", "backend": "vLLM", "minDriver": "575.xx+"},
                ],
                ["1.3.0", "1.2.0"],
            )
        )
        rows = {ln.split("|")[1].strip(): ln for ln in table.splitlines()[2:]}
        assert "1.2.0" in rows["580.xx+"]
        assert "1.3.0" in rows["580.xx+"]

    def test_says_none_rather_than_a_dash_when_nothing_runs(self):
        """TensorRT-LLM has no build for the oldest floor; say so."""
        table = gen.driver_floor_table(
            data(
                [
                    {
                        "version": "1.3.0",
                        "backend": "TensorRT-LLM",
                        "minDriver": "580.xx+",
                    },
                    {"version": "1.1.0", "backend": "vLLM", "minDriver": "570.xx+"},
                ],
                ["1.3.0", "1.1.0"],
            )
        )
        oldest = [ln for ln in table.splitlines() if ln.startswith("| 570.xx+")][0]
        assert "None" in oldest
        assert "| - |" not in oldest

    def test_backends_come_from_the_data(self):
        table = gen.driver_floor_table(
            data(
                [{"version": "1.3.0", "backend": "Zebra", "minDriver": "580.xx+"}],
                ["1.3.0"],
            )
        )
        assert "Zebra" in table.splitlines()[0]

    def test_unreleased_versions_do_not_appear(self):
        table = gen.driver_floor_table(
            data(
                [
                    {"version": "1.4.0", "backend": "vLLM", "minDriver": "580.xx+"},
                    {"version": "1.3.0", "backend": "vLLM", "minDriver": "580.xx+"},
                ],
                ["1.3.0"],
            )
        )
        assert "1.4.0" not in table
        assert "1.3.0" in table

    def test_fails_closed_when_no_row_is_released(self):
        with pytest.raises(gen.TSParseError, match="driver"):
            gen.driver_floor_table(
                data(
                    [{"version": "9.9.9", "backend": "vLLM", "minDriver": "580.xx+"}],
                    ["1.3.0"],
                )
            )


class TestPostTrainInheritance:
    """Post trains carry no CUDA_HISTORY row; CUDA_NOTES says they inherit."""

    def test_a_post_train_stands_in_for_its_base(self):
        table = gen.driver_floor_table(
            data(
                [{"version": "0.7.0", "backend": "vLLM", "minDriver": "570.xx+"}],
                ["0.7.0.post1", "0.7.0"],
            )
        )
        assert "0.7.0.post1" in table

    def test_the_newest_post_train_wins(self):
        table = gen.driver_floor_table(
            data(
                [{"version": "0.8.1", "backend": "vLLM", "minDriver": "575.xx+"}],
                ["0.8.1.post3", "0.8.1.post2", "0.8.1.post1", "0.8.1"],
            )
        )
        assert "0.8.1.post3" in table
        assert "0.8.1.post2" not in table

    def test_a_base_without_post_trains_is_left_alone(self):
        table = gen.driver_floor_table(
            data(
                [{"version": "1.3.0", "backend": "vLLM", "minDriver": "580.xx+"}],
                ["1.3.0"],
            )
        )
        assert "| 1.3.0 |" in table


class TestAgainstRealData:
    def test_renders_from_the_checked_in_module(self):
        real = gen.parse_data_module(gen.DATA_TS)
        table = gen.driver_floor_table(real)
        assert table.startswith("| Driver |")
        assert "TensorRT-LLM" in table
