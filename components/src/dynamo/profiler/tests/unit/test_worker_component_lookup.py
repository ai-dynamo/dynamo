# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Current contract of the two "which component serves this role" lookups.

The planner (``get_component_from_type_or_name``) and the profiler
(``get_component_name_by_type``) share one matching core and differ in policy.
Each row of ``LOOKUP_CASES`` runs both on the same ``spec.components``. The rows
where they differ record today's behaviour, so that a change to the shared core
cannot move either caller unnoticed. Aligning the two is a behaviour change and
should edit this table.

``PROFILER_ONLY_CASES`` are rows where the planner agrees with the profiler and
``planner/tests/unit/test_kubernetes_connector.py`` already tests the planner.
"""

from typing import NamedTuple, Optional, Union

import pytest

from dynamo.planner.config.backend_components import WORKER_COMPONENT_NAMES
from dynamo.planner.config.defaults import SubComponentType
from dynamo.planner.errors import DuplicateSubComponentError, SubComponentNotFoundError
from dynamo.planner.monitoring.dgd_services import (
    Service,
    get_component_from_type_or_name,
)
from dynamo.profiler.utils.config import Config, get_component_name_by_type

pytestmark = [
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.planner,
    pytest.mark.parallel,
]

PREFILL = SubComponentType.PREFILL
DECODE = SubComponentType.DECODE
BACKEND = "vllm"
# Canonical component names the profiler falls back to for BACKEND.
DEFAULT_PREFILL = WORKER_COMPONENT_NAMES[BACKEND].prefill_worker_k8s_name
DEFAULT_DECODE = WORKER_COMPONENT_NAMES[BACKEND].decode_worker_k8s_name

DISAGG = (("p", "prefill"), ("d", "decode"))
AGG = (("fe", "frontend"), ("w", "worker"))
TWO_GENERIC_WORKERS = (("w1", "worker"), ("w2", "worker"))
DUPLICATE_PREFILL = (("p1", "prefill"), ("p2", "prefill"), ("d", "decode"))
UNTYPED = (("x", ""), ("y", ""))
NO_WORKER = (("fe", "frontend"),)
FRONTEND_WITH_DEFAULT_DECODE_NAME = ((DEFAULT_DECODE, "frontend"),)


class LookupCase(NamedTuple):
    components: tuple[tuple[str, str], ...]  # (name, type) in spec order
    role: SubComponentType
    profiler: str  # resolved name
    # resolved name, or the error raised; unset in PROFILER_ONLY_CASES
    planner: Union[str, type[Exception], None] = None
    component_name: Optional[str] = None  # passed to the planner lookup only


PROFILER_ONLY_CASES = (
    pytest.param(LookupCase(DISAGG, PREFILL, profiler="p"), id="typed-prefill"),
    pytest.param(LookupCase(DISAGG, DECODE, profiler="d"), id="typed-decode"),
    pytest.param(
        LookupCase(AGG, DECODE, profiler="w"), id="sole-generic-worker-decode"
    ),
)

LOOKUP_CASES = (
    pytest.param(
        LookupCase(
            (("w", "worker"), ("d", "decode")), DECODE, planner="d", profiler="d"
        ),
        id="type-beats-generic-worker",
    ),
    pytest.param(
        LookupCase(
            DUPLICATE_PREFILL,
            PREFILL,
            planner=DuplicateSubComponentError,
            profiler="p1",
        ),
        id="duplicate-type",
    ),
    pytest.param(
        LookupCase(AGG, PREFILL, planner=SubComponentNotFoundError, profiler="w"),
        id="sole-generic-worker-prefill",
    ),
    pytest.param(
        LookupCase(
            TWO_GENERIC_WORKERS,
            DECODE,
            planner=SubComponentNotFoundError,
            profiler=DEFAULT_DECODE,
        ),
        id="two-generic-workers",
    ),
    pytest.param(
        LookupCase(
            UNTYPED,
            DECODE,
            planner="y",
            profiler=DEFAULT_DECODE,
            component_name="y",
        ),
        id="untyped-component-by-name",
    ),
    pytest.param(
        LookupCase(
            FRONTEND_WITH_DEFAULT_DECODE_NAME,
            DECODE,
            planner=SubComponentNotFoundError,
            profiler=DEFAULT_DECODE,
            component_name=DEFAULT_DECODE,
        ),
        id="named-component-has-another-type",
    ),
    pytest.param(
        LookupCase(
            NO_WORKER,
            PREFILL,
            planner=SubComponentNotFoundError,
            profiler=DEFAULT_PREFILL,
        ),
        id="no-worker-prefill",
    ),
    pytest.param(
        LookupCase(
            NO_WORKER,
            DECODE,
            planner=SubComponentNotFoundError,
            profiler=DEFAULT_DECODE,
            component_name="not-in-spec",
        ),
        id="no-worker-decode",
    ),
    pytest.param(
        LookupCase(
            (), DECODE, planner=SubComponentNotFoundError, profiler=DEFAULT_DECODE
        ),
        id="no-components",
    ),
)


def _dgd(components: tuple[tuple[str, str], ...]) -> dict:
    return {
        "metadata": {"name": "graph"},
        "spec": {
            "components": [
                {"name": name, "type": component_type, "replicas": index + 1}
                for index, (name, component_type) in enumerate(components)
            ]
        },
    }


@pytest.mark.parametrize("case", LOOKUP_CASES)
def test_planner_lookup(case):
    deployment = _dgd(case.components)

    if isinstance(case.planner, str):
        service = get_component_from_type_or_name(
            deployment, case.role, case.component_name
        )
        assert service.name == case.planner
    else:
        with pytest.raises(case.planner) as exc_info:
            get_component_from_type_or_name(deployment, case.role, case.component_name)
        assert exc_info.value.sub_component_type == case.role.value


@pytest.mark.parametrize("case", PROFILER_ONLY_CASES + LOOKUP_CASES)
def test_profiler_lookup(case):
    config = Config.model_validate(_dgd(case.components))

    assert get_component_name_by_type(config, BACKEND, case.role) == case.profiler


def test_duplicate_error_names_components_in_spec_order():
    with pytest.raises(DuplicateSubComponentError) as exc_info:
        get_component_from_type_or_name(_dgd(DUPLICATE_PREFILL), PREFILL)

    assert exc_info.value.service_names == ["p1", "p2"]


def test_planner_returns_a_service_and_the_profiler_a_name():
    """The planner result wraps the component spec; the profiler result is the name."""
    deployment = _dgd(DISAGG)

    service = get_component_from_type_or_name(deployment, DECODE)
    name = get_component_name_by_type(
        Config.model_validate(deployment), BACKEND, DECODE
    )

    assert isinstance(service, Service)
    assert service.service == deployment["spec"]["components"][1]
    assert name == "d"
