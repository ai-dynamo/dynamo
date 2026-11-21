# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Builds a Dynamo vLLM runtime image that can serve models whose architecture
# is registered through an out-of-tree (OOT) vLLM plugin instead of vLLM's
# in-tree model registry. Build arguments and the full workflow are
# documented in the adjacent README.md.

# Pin the same Dynamo release you deploy with.
ARG DYNAMO_VLLM_RUNTIME_IMAGE=nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.6.0
FROM ${DYNAMO_VLLM_RUNTIME_IMAGE}

# The runtime image ends on a non-root USER and pip targets the system
# site-packages, so installs need root plus the PEP-668 override on the
# Ubuntu 24.04 base — same layering pattern as container/Dockerfile.test.
# The original USER is restored at the end of this file.
USER root

# Staging directory for local plugin wheels; ./wheels is committed (empty) so
# this COPY works whether or not the builder stages a wheel.
COPY wheels/ /tmp/oot-wheels/

# Required: pip requirement for the plugin — a PyPI name, a git+https URL,
# or an in-image /tmp/oot-wheels/ path for wheels staged in ./wheels.
ARG OOT_PLUGIN_SPEC
# Optional extra requirements the plugin needs (e.g. a transformers pin);
# installed WITH dependency resolution — read "Engine version compatibility"
# in README.md before downgrading anything the base image pins.
ARG OOT_PLUGIN_EXTRA_REQUIREMENTS=""

RUN test -n "${OOT_PLUGIN_SPEC}" \
      || { echo "error: set --build-arg OOT_PLUGIN_SPEC to the plugin pip requirement" >&2; exit 1; }

# Snapshot the plugin-bearing distributions that already exist in the base
# image (e.g. Dynamo's own telemetry plugin) so the compatibility gate below
# validates only what this build adds.
RUN python3 -c "import importlib.metadata as md, json; print(json.dumps(sorted({ep.dist.name for ep in md.entry_points(group='vllm.general_plugins')})))" > /tmp/pre_existing_plugins.json

# The plugin is installed with --no-deps so dependency resolution cannot
# silently replace the vLLM/torch stack the Dynamo image pins; the next RUN
# verifies the plugin's declared requirements against what is installed.
# OOT_PLUGIN_EXTRA_REQUIREMENTS is the explicit escape hatch for pins the
# plugin genuinely needs (e.g. transformers); those DO resolve dependencies.
# Unquoted build args: intentional word splitting so the specs can carry
# multiple requirements.
RUN pip install --no-cache-dir --break-system-packages --no-deps ${OOT_PLUGIN_SPEC} \
 && if [ -n "${OOT_PLUGIN_EXTRA_REQUIREMENTS}" ]; then \
      pip install --no-cache-dir --break-system-packages ${OOT_PLUGIN_EXTRA_REQUIREMENTS}; \
    fi \
 && rm -rf /tmp/oot-wheels

# Build-time gates, in order:
#   1. the install added at least one new vllm.general_plugins entry point
#      (distributions already present in the base image, e.g. Dynamo's own
#      telemetry plugin, are excluded via the snapshot taken before install);
#   2. every requirement the NEW plugin declares (e.g. vllm==X) is present
#      at a satisfying version, since --no-deps skipped resolution — missing
#      or mismatched requirements mean the plugin targets a different stack
#      than this Dynamo release pins (see README.md);
#   3. when OOT_EXPECTED_ARCH is set, loading the plugins the same way vLLM
#      does at engine start actually registers that architecture.
ARG OOT_EXPECTED_ARCH=""
RUN python3 - <<'PY'
import importlib.metadata as md
import json
import os
import re

from packaging.requirements import Requirement


def norm(name):
    # PEP 503: plugins may declare requirements with different
    # dash/underscore/dot spellings than the installed distribution name.
    return re.sub(r"[-_.]+", "-", name).lower()


with open("/tmp/pre_existing_plugins.json") as f:
    pre_existing = set(json.load(f))
print(f"plugin distributions already in the base image: {sorted(pre_existing)}")

plugins = list(md.entry_points(group="vllm.general_plugins"))
new_plugins = [ep for ep in plugins if ep.dist.name not in pre_existing]
print(f"discovered vllm.general_plugins entry points: {[p.name for p in plugins]}")
if not new_plugins:
    raise SystemExit(
        "error: the install added no new vllm.general_plugins entry point; "
        "the plugin wheel does not declare the entry point group"
    )

installed = {
    norm(d.metadata["Name"]): d.version
    for d in md.distributions()
    if d.metadata["Name"]
}
seen = set()
for ep in new_plugins:
    if ep.dist.name in seen:
        continue
    seen.add(ep.dist.name)
    for raw in ep.dist.requires or []:
        req = Requirement(raw)
        if req.marker and not req.marker.evaluate():
            continue
        version = installed.get(norm(req.name))
        if version is None:
            # --no-deps skipped resolution, so nothing else will install
            # this; without failing here the image builds and the worker
            # only crashes at model load.
            raise SystemExit(
                f"error: plugin {ep.dist.name} requires {req.name}{req.specifier}, "
                "which is not installed in this image; add it to "
                "OOT_PLUGIN_EXTRA_REQUIREMENTS or use a plugin build matching "
                "this Dynamo release (see README.md, 'Engine version "
                "compatibility')"
            )
        if not req.specifier.contains(version, prereleases=True):
            raise SystemExit(
                f"error: plugin {ep.dist.name} requires {req.name}{req.specifier} "
                f"but this image has {req.name}=={version}; use a plugin build "
                "matching this Dynamo release's engine pins (see README.md, "
                "'Engine version compatibility')"
            )

from vllm.plugins import load_general_plugins

load_general_plugins()
from vllm import ModelRegistry

archs = ModelRegistry.get_supported_archs()
print(f"model registry reports {len(archs)} architectures after plugin load")
expected = os.environ.get("OOT_EXPECTED_ARCH", "").strip()
if expected:
    if expected not in archs:
        raise SystemExit(
            f"error: expected architecture {expected!r} not registered by any plugin"
        )
    print(f"architecture gate passed: {expected} is registered")
else:
    print("note: OOT_EXPECTED_ARCH not set; architecture gate skipped")

os.remove("/tmp/pre_existing_plugins.json")
PY

# Keep the runtime image's non-root default so the produced image behaves
# like the stock vllm-runtime under the operator (group-0/OpenShift assume it).
USER dynamo
