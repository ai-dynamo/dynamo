# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""EFA packaging checks for the ``-efa`` runtime images.

These run *inside* an EFA-tagged runtime image and assert the properties that
make NIXL able to reach Elastic Fabric Adapter at runtime. They need no GPU, no
EFA device and no cluster, so they run in the existing ``*-efa-test`` CI jobs on
both amd64 and arm64.

They exist because every EFA regression so far has been a packaging regression
that a functional test could not see until it reached a p5/GB200 cluster:

* the TRT-LLM EFA image shipped a NIXL plugin directory with no LIBFABRIC plugin
* NIXL was linked against the generic from-source libfabric instead of the EFA
  SDK's, so HMEM/CUDA dmabuf descriptors did not match at runtime
* the arm64 image put plugins under ``lib/aarch64-linux-gnu`` while
  ``NIXL_PLUGIN_DIR`` still pointed at ``lib64``

Each of those turns into a *silent* fallback to UCX at inference time, which is
why these are hard assertions rather than warnings.

Selected by the ``efa_image`` marker, from the nightly ``*-efa-test`` jobs. EFA
changes are sparse, so nightly is the cadence that matches the risk; nothing
else collects these.
"""

import os
import re
import subprocess

import pytest

pytestmark = [
    pytest.mark.efa_image,
    pytest.mark.nightly,
    pytest.mark.unit,
    pytest.mark.gpu_0,
]

EFA_SDK_ROOT = "/opt/amazon/efa"
FI_INFO = os.path.join(EFA_SDK_ROOT, "bin", "fi_info")
LIBFABRIC_PLUGIN = "libplugin_LIBFABRIC.so"

# The EFA image build (container/templates/aws.Dockerfile) normalizes NIXL into
# a single arch-agnostic layout and points NIXL_PLUGIN_DIR at it. Every value
# below is read from the environment rather than hardcoded, so an image that
# moves the layout but keeps the env consistent still passes.
NIXL_PLUGIN_DIR = os.environ.get("NIXL_PLUGIN_DIR", "")
NIXL_LIB_DIR = os.environ.get("NIXL_LIB_DIR", "")


def _run(cmd: list[str]) -> str:
    """Run a command and return stdout+stderr, failing the test on OSError."""
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    except OSError as exc:
        pytest.fail(f"could not execute {' '.join(cmd)}: {exc}")
    return proc.stdout + proc.stderr


def test_efa_sdk_installed() -> None:
    """The AWS EFA installer's SDK must be present at its canonical prefix."""
    assert os.path.isdir(EFA_SDK_ROOT), (
        f"{EFA_SDK_ROOT} is missing — this image was not built with make_efa=true, "
        "or the EFA installer stage did not run."
    )
    assert os.access(FI_INFO, os.X_OK), f"{FI_INFO} is missing or not executable"

    libs = []
    for lib_dir in (
        os.path.join(EFA_SDK_ROOT, "lib"),
        os.path.join(EFA_SDK_ROOT, "lib64"),
    ):
        if os.path.isdir(lib_dir):
            libs += [f for f in os.listdir(lib_dir) if f.startswith("libfabric.so")]
    assert libs, f"no libfabric.so* under {EFA_SDK_ROOT}/lib*"


def test_efa_version_recorded() -> None:
    """The image must record which EFA installer it was built from.

    Without this there is no way to tell a patched libfabric from a stock one
    after the fact, which is exactly the ambiguity that made the GB200
    VRAM-registration failure expensive to diagnose.
    """
    version = os.environ.get("EFA_VERSION", "")
    assert version, "EFA_VERSION is not set in the image environment"
    assert re.match(
        r"^\d+\.\d+", version
    ), f"EFA_VERSION={version!r} does not look like a version"


def test_efa_provider_compiled_in() -> None:
    """libfabric must actually have the EFA provider built in.

    ``fi_info -l`` lists compiled-in providers and needs no EFA hardware, so a
    CPU-only runner can still catch a libfabric built without EFA support.
    """
    output = _run([FI_INFO, "-l"])
    providers = {
        line.rstrip(":")
        for line in output.splitlines()
        if line and not line[0].isspace()
    }
    assert "efa" in providers, (
        "libfabric has no 'efa' provider. Providers found: "
        f"{sorted(providers) or '<none>'}\n{output}"
    )


def test_nixl_plugin_dir_contains_libfabric_plugin() -> None:
    """NIXL_PLUGIN_DIR must exist and hold the LIBFABRIC backend plugin.

    A plugin directory that resolves but has no LIBFABRIC plugin is the TRT-LLM
    failure mode: NIXL loads, finds only UCX, and silently uses it.
    """
    assert NIXL_PLUGIN_DIR, "NIXL_PLUGIN_DIR is not set in the image environment"
    assert os.path.isdir(
        NIXL_PLUGIN_DIR
    ), f"NIXL_PLUGIN_DIR={NIXL_PLUGIN_DIR} does not exist (wrong arch path?)"

    plugins = sorted(os.listdir(NIXL_PLUGIN_DIR))
    assert LIBFABRIC_PLUGIN in plugins, (
        f"{LIBFABRIC_PLUGIN} missing from NIXL_PLUGIN_DIR={NIXL_PLUGIN_DIR}. "
        f"Present: {plugins}"
    )


def test_libfabric_plugin_links_efa_sdk() -> None:
    """NIXL's LIBFABRIC plugin must link against the EFA SDK's libfabric.

    Linking against a generic from-source libfabric builds a plugin whose
    HMEM/CUDA dmabuf descriptors do not match the EFA provider at runtime.
    """
    plugin = os.path.join(NIXL_PLUGIN_DIR, LIBFABRIC_PLUGIN)
    if not os.path.isfile(plugin):
        pytest.skip("LIBFABRIC plugin absent; covered by the plugin-dir test")

    # Only the libfabric line matters. CUDA libs legitimately resolve to
    # "not found" here because CI runs this on a CPU-only runner.
    resolved = [
        line.strip()
        for line in _run(["ldd", plugin]).splitlines()
        if "libfabric.so" in line
    ]
    assert (
        resolved
    ), f"{plugin} does not link libfabric at all:\n{_run(['ldd', plugin])}"

    for line in resolved:
        _, _, target = line.partition("=>")
        target = target.strip()
        assert target.startswith(EFA_SDK_ROOT), (
            f"{LIBFABRIC_PLUGIN} resolves libfabric to {target or line!r}, expected it "
            f"under {EFA_SDK_ROOT}. NIXL was built against the wrong libfabric — see "
            "the -Dlibfabric_path meson flag in container/templates/wheel_builder.Dockerfile."
        )


def test_nixl_runtime_libs_resolvable() -> None:
    """libnixl.so must sit alongside the plugins, and any LD_PRELOAD must resolve.

    The EFA image sets LD_PRELOAD so the Dynamo-built NIXL wins over any
    framework-bundled copy (see the trtllm branch of aws.Dockerfile); a stale
    path there breaks every worker at startup rather than at transfer time.

    NIXL_LIB_DIR is only exported by some of the runtime stages, so fall back to
    the plugin directory's parent — plugins always live at ``<libdir>/plugins``,
    and a libnixl that does not sit next to its plugins is the bug we care about.
    """
    lib_dir = NIXL_LIB_DIR or os.path.dirname(NIXL_PLUGIN_DIR.rstrip("/"))
    assert lib_dir, "neither NIXL_LIB_DIR nor NIXL_PLUGIN_DIR is set in the image"
    assert os.path.isdir(lib_dir), f"NIXL lib dir {lib_dir} does not exist"
    assert "libnixl.so" in os.listdir(
        lib_dir
    ), f"libnixl.so missing from NIXL lib dir {lib_dir}"

    for entry in os.environ.get("LD_PRELOAD", "").replace(":", " ").split():
        assert os.path.isfile(entry), f"LD_PRELOAD entry {entry} does not exist"
