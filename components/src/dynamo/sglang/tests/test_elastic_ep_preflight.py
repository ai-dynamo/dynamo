# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Behaviour of the ``--elastic-ep-backend mooncake`` startup check.

Imports neither ``sglang`` nor any ``dynamo.sglang`` module that imports it,
so these run in an image where the engine is not installed or not importable
-- which is the only reason the check lives in its own module rather than in
``args.py``. The environment probes are the seams; the decision the check
makes over their results is what is exercised here.
"""

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


@pytest.mark.parametrize("requested_backend", [None, "", "nccl", "nvshmem"])
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
    """The probe itself, not a stand-in for it.

    The other cases replace the probes to pin the decision logic, which leaves
    the probes themselves untested. This one runs the real
    ``_import_process_group_extension`` against an import system that has no
    mooncake in it, so the import, the exception capture, and the message
    assembly are all the shipped code. Blocking at ``sys.meta_path`` rather
    than by uninstalling makes it behave the same in an image that does have a
    working wheel.
    """

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


def test_absent_mooncake_renders_as_an_explicit_absence():
    """No mooncake installed must read as words, not an empty field."""
    assert elastic_ep_preflight._format_versions({}) == "none installed"
