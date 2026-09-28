# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Behaviour of the ``--elastic-ep-backend mooncake`` startup check.

Imports neither ``sglang`` nor any ``dynamo.sglang`` module that imports it,
so these run in an image where the engine is not installed or not importable
-- which is the only reason the check lives in its own module rather than in
``args.py``. The environment probes are the seams; the decision the check
makes over their results is what is exercised here.
"""

import builtins
import importlib.util
import sys

import pytest

from dynamo.sglang import elastic_ep_preflight
from dynamo.sglang.elastic_ep_preflight import check_elastic_ep_backend

pytestmark = [
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]

_IMPORT_FAILURE = (
    "mooncake.pg: ImportError: Mooncake PG was not built against torch==2.11.0; "
    "mooncake.ep: ModuleNotFoundError: No module named 'mooncake.ep'"
)


def _simulate_environment(
    monkeypatch,
    *,
    import_failure,
    registered_backends,
    mooncake_versions,
    extension_modules,
    torch_version="2.11.0+cu130",
):
    """Pin every probe so the check runs against a known image shape."""
    monkeypatch.setattr(
        elastic_ep_preflight,
        "_import_process_group_extension",
        lambda: import_failure,
    )
    monkeypatch.setattr(
        elastic_ep_preflight,
        "_registered_torch_backends",
        lambda: dict(registered_backends),
    )
    monkeypatch.setattr(
        elastic_ep_preflight,
        "_installed_mooncake_versions",
        lambda: dict(mooncake_versions),
    )
    monkeypatch.setattr(
        elastic_ep_preflight,
        "_available_extension_modules",
        lambda: list(extension_modules),
    )
    monkeypatch.setattr(elastic_ep_preflight, "_torch_version", lambda: torch_version)


def _simulate_broken_mooncake(monkeypatch, **overrides):
    """An image whose mooncake ProcessGroup extension does not load."""
    kwargs = {
        "import_failure": _IMPORT_FAILURE,
        "registered_backends": {"gloo": ("cpu",), "nccl": ("cuda",)},
        "mooncake_versions": {"mooncake-transfer-engine-cuda13": "0.3.11.post1"},
        "extension_modules": ["pg_2_9_1", "pg_2_10_0"],
    }
    kwargs.update(overrides)
    _simulate_environment(monkeypatch, **kwargs)


def _simulate_healthy_mooncake(monkeypatch):
    """An image whose mooncake ProcessGroup extension loads and registers."""
    _simulate_environment(
        monkeypatch,
        import_failure=None,
        registered_backends={
            "gloo": ("cpu",),
            "nccl": ("cuda",),
            "mooncake": ("cuda",),
            "mooncake-cpu": ("cpu",),
        },
        mooncake_versions={"mooncake-transfer-engine-cuda13": "0.3.11.post1"},
        extension_modules=["pg_2_11_0"],
    )


def test_rejects_mooncake_backend_the_image_cannot_serve(monkeypatch):
    """The worker fails at argument parsing, naming what the operator needs."""
    _simulate_broken_mooncake(monkeypatch)

    with pytest.raises(ValueError) as excinfo:
        check_elastic_ep_backend("mooncake", True)

    message = str(excinfo.value)
    assert "--elastic-ep-backend" in message
    # The three facts that separate a bad wheel from a torch-version skew, so
    # the failure is diagnosable without reproducing the crash.
    assert "0.3.11.post1" in message
    assert "2.11.0+cu130" in message
    assert "pg_2_9_1" in message
    # --enable-dp-attention is what drives the collective on the first forward
    # pass, so it has to be called out when it is part of the configuration.
    assert "--enable-dp-attention" in message


def test_reports_missing_mooncake_install_explicitly(monkeypatch):
    """No mooncake at all reads as an absence, not a blank version field."""
    _simulate_broken_mooncake(monkeypatch, mooncake_versions={}, extension_modules=[])

    with pytest.raises(ValueError) as excinfo:
        check_elastic_ep_backend("mooncake")

    assert "none installed" in str(excinfo.value)


@pytest.mark.parametrize("requested_backend", [None, "nccl"])
def test_ignores_backends_other_than_mooncake(monkeypatch, requested_backend):
    """Negative control: the same broken image is fine for everyone else.

    Guards against the check becoming an unconditional startup failure for
    workers that never asked for the mooncake transport.
    """
    _simulate_broken_mooncake(monkeypatch)

    check_elastic_ep_backend(requested_backend, True)


def test_accepts_mooncake_when_the_image_can_serve_it(monkeypatch):
    """Negative control: a working image is not blocked."""
    _simulate_healthy_mooncake(monkeypatch)

    check_elastic_ep_backend("mooncake", True)


def test_rejects_a_wheel_that_registers_only_half_the_process_groups(monkeypatch):
    """A device backend without its CPU peer is not a usable image.

    SGLang builds both an elastic-EP device group and a CPU group for the
    metadata collectives, so accepting on ``mooncake`` alone lets a wheel that
    registered only that one pass here and still fail during engine startup.
    """
    _simulate_environment(
        monkeypatch,
        import_failure=None,
        registered_backends={
            "gloo": ("cpu",),
            "nccl": ("cuda",),
            "mooncake": ("cuda",),
        },
        mooncake_versions={"mooncake-transfer-engine-cuda13": "0.3.11.post1"},
        extension_modules=["pg_2_11_0"],
    )

    with pytest.raises(ValueError) as excinfo:
        check_elastic_ep_backend("mooncake")

    # The operator has to be told which half is absent; the registry line alone
    # shows a mooncake backend present and reads as a working image.
    assert "mooncake-cpu" in str(excinfo.value)


def test_unreadable_torch_backend_registry_does_not_block_startup(monkeypatch):
    """A registry that cannot be read is not evidence that mooncake is absent.

    ``Backend.backend_capability`` is a torch internal. Reading it and failing
    must not be worth more than the extension importing cleanly, or a torch-side
    rename refuses every worker on this transport.
    """
    _simulate_broken_mooncake(monkeypatch, import_failure=None)
    monkeypatch.setattr(
        elastic_ep_preflight, "_registered_torch_backends", lambda: None
    )

    check_elastic_ep_backend("mooncake", True)


def test_import_probe_reports_every_module_name_it_tried(monkeypatch):
    """The real probe, run against an import system with no mooncake in it.

    Blocking at ``sys.meta_path`` rather than by uninstalling makes the case
    behave the same in an image that does ship a working wheel.
    """
    monkeypatch.setattr(
        elastic_ep_preflight,
        "_required_process_group_modules",
        lambda: ("mooncake.pg", "mooncake.ep"),
    )

    class _RefuseMooncake:
        def find_spec(self, name, path=None, target=None):
            if name == "mooncake" or name.startswith("mooncake."):
                raise ImportError(f"blocked for this test: {name}")
            return None

    monkeypatch.setattr(sys, "meta_path", [_RefuseMooncake(), *sys.meta_path])
    for cached in [
        name
        for name in sys.modules
        if name == "mooncake" or name.startswith("mooncake.")
    ]:
        monkeypatch.delitem(sys.modules, cached)

    failure = elastic_ep_preflight._import_process_group_extension()

    assert failure is not None
    # Both names have to appear: which one an image ships is the first thing
    # an operator needs to compare against the engine version.
    assert "mooncake.pg" in failure
    assert "mooncake.ep" in failure


def _install_fake_sglang_sources(monkeypatch, tmp_path, import_line):
    """Point the resolver at a source tree naming one ProcessGroup module."""
    package = tmp_path / "sglang"
    elastic_ep = package / "srt" / "elastic_ep"
    elastic_ep.mkdir(parents=True)
    (elastic_ep / "elastic_ep.py").write_text(import_line, encoding="utf-8")

    spec = importlib.util.spec_from_file_location(
        "sglang", package / "__init__.py", submodule_search_locations=[str(package)]
    )
    monkeypatch.setattr(
        elastic_ep_preflight.importlib.util,
        "find_spec",
        lambda name: spec if name == "sglang" else None,
    )


@pytest.mark.parametrize(
    "import_line, expected",
    [
        ("from mooncake.pg import MooncakeBackendOptions\n", ("mooncake.pg",)),
        ("from mooncake.ep import MooncakeBackendOptions\n", ("mooncake.ep",)),
    ],
)
def test_probes_only_the_module_the_installed_engine_imports(
    monkeypatch, tmp_path, import_line, expected
):
    """The half of the rename the engine does not use must not satisfy the check.

    Accepting either name lets an image whose wheel ships only the other one
    pass here and still fail during engine startup, which is the failure this
    module exists to move earlier.
    """
    _install_fake_sglang_sources(monkeypatch, tmp_path, import_line)

    assert elastic_ep_preflight._required_process_group_modules() == expected


def test_unreadable_engine_sources_probe_both_module_names(monkeypatch, tmp_path):
    """Sources that are present but unreadable are not evidence for either name.

    This is the case the resolver is actually written for: an editable install
    whose engine tree the worker's uid cannot read, which is the same condition
    the repository ``conftest.py`` reports at collection time. Narrowing on a
    guess would refuse workers over a permissions problem, so the resolver
    widens back to both names instead.
    """
    _install_fake_sglang_sources(
        monkeypatch, tmp_path, "from mooncake.pg import MooncakeBackendOptions\n"
    )

    probed = []
    real_open = builtins.open

    def _refuse_engine_sources(file, *args, **kwargs):
        if str(file).endswith(".py") and "sglang" in str(file):
            probed.append(str(file))
            raise PermissionError(13, "Permission denied", str(file))
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", _refuse_engine_sources)

    assert elastic_ep_preflight._required_process_group_modules() == (
        "mooncake.pg",
        "mooncake.ep",
    )
    # The fallback must come from the read failing, not from the resolver
    # skipping the sources it was pointed at.
    assert probed
