# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native argument parsing for diffusion (image/video) workers.

Diffusion workers configure SGLang's DiffGenerator, whose options live in
``sglang.multimodal_gen.runtime.server_args.ServerArgs`` — a different
dataclass from the LLM ``ServerArgs``. Building the CLI from that class (the
same way the LLM path builds its CLI from the LLM ``ServerArgs``) makes every
native diffusion engine argument reachable from Dynamo without hand-copying
fields.

``DiffusionWorkerArgs`` is the thin adapter the rest of the Dynamo worker code
sees: it delegates engine fields to the real diffusion ``ServerArgs`` and adds
the few Dynamo-side settings that class does not define. Fields Dynamo's
shared worker code probes on every ``server_args`` (speculative decoding,
disaggregation, load format) are pinned to their inert values here instead of
being hand-maintained in a SimpleNamespace stub.
"""

import argparse
import logging
from typing import Any, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Dynamo-side flags the diffusion ServerArgs may not define. Each entry is
# only added to the parser when the engine parser does not already provide it,
# so a future SGLang version adding the same flag natively wins automatically.
_DYNAMO_SIDE_FLAGS = (
    {
        "flags": ("--served-model-name",),
        "kwargs": {
            "type": str,
            "default": None,
            "help": "Model name reported to the frontend and /v1/models. "
            "Defaults to --model-path.",
        },
    },
    {
        "flags": ("--enable-metrics",),
        "kwargs": {
            "action": "store_true",
            "default": False,
            "help": "Enable Dynamo worker metrics publishing.",
        },
    },
)


class DiffusionWorkerArgs:
    """Adapter combining SGLang's diffusion ServerArgs with Dynamo settings.

    Attribute access falls through to the wrapped diffusion ``ServerArgs``,
    so shared worker code reading e.g. ``model_path``, ``tp_size`` or
    ``log_level`` sees the natively parsed values. ``engine_args`` exposes the
    wrapped object for handing to ``DiffGenerator.from_server_args``.
    """

    # Fields Dynamo's shared worker code reads on any server_args. Diffusion
    # workers do not use these subsystems; pin them to inert values.
    _INERT_DEFAULTS = {
        "speculative_algorithm": None,
        "disaggregation_mode": None,
        "dllm_algorithm": False,
        "load_format": None,
        "kv_events_config": None,
        "enable_forward_pass_metrics": False,
    }

    def __init__(
        self,
        engine_args: Any,
        served_model_name: Optional[str],
        enable_metrics: bool,
    ):
        self.engine_args = engine_args
        self.served_model_name = served_model_name or engine_args.model_path
        self.enable_metrics = enable_metrics
        for name, value in self._INERT_DEFAULTS.items():
            # Prefer a native field if the engine args grow one later.
            setattr(self, name, getattr(engine_args, name, value))

    def __getattr__(self, name: str) -> Any:
        # Only called when normal lookup fails: delegate to the engine args.
        return getattr(self.__dict__["engine_args"], name)


def _existing_option_strings(parser: argparse.ArgumentParser) -> set:
    return {opt for action in parser._actions for opt in action.option_strings}


def _import_diffusion_server_args():
    """Import sglang's diffusion ServerArgs (lazily; layout varies by version)."""
    # Imported lazily: only diffusion/video workers need the multimodal_gen
    # extra, and importing it pulls in heavy dependencies. server_args became
    # a package in newer SGLang releases; older ones expose it as a module.
    try:
        from sglang.multimodal_gen.runtime.server_args.server_args import (
            ServerArgs as DiffusionServerArgs,
        )
    except (ImportError, ModuleNotFoundError):
        from sglang.multimodal_gen.runtime.server_args import (
            ServerArgs as DiffusionServerArgs,  # type: ignore[no-redef]
        )
    return DiffusionServerArgs


def build_diffusion_parser() -> Tuple[argparse.ArgumentParser, List[str]]:
    """Build the diffusion worker CLI: native engine args + Dynamo-side flags.

    Returns:
        (parser, dynamo_side_dests) where dynamo_side_dests names the flags
        registered by Dynamo rather than the engine.
    """
    DiffusionServerArgs = _import_diffusion_server_args()

    try:
        from sglang.multimodal_gen.utils import FlexibleArgumentParser
    except ImportError:
        FlexibleArgumentParser = argparse.ArgumentParser

    parser = FlexibleArgumentParser(
        description="Dynamo SGLang diffusion worker configuration",
        add_help=False,
    )
    DiffusionServerArgs.add_cli_args(parser)

    # Register Dynamo-side flags the engine parser does not define.
    existing = _existing_option_strings(parser)
    dynamo_group = parser.add_argument_group("Dynamo Options")
    dynamo_side_dests = []
    for spec in _DYNAMO_SIDE_FLAGS:
        if any(flag in existing for flag in spec["flags"]):
            continue
        action = dynamo_group.add_argument(*spec["flags"], **spec["kwargs"])
        dynamo_side_dests.append(action.dest)
    return parser, dynamo_side_dests


def parse_diffusion_args(
    unknown_args: List[str],
) -> Tuple[argparse.Namespace, DiffusionWorkerArgs]:
    """Parse worker args against SGLang's native diffusion ServerArgs CLI.

    Args:
        unknown_args: Argument strings left over after Dynamo's own parser.

    Returns:
        (parsed argparse namespace, DiffusionWorkerArgs adapter)
    """
    DiffusionServerArgs = _import_diffusion_server_args()
    parser, dynamo_side_dests = build_diffusion_parser()

    parsed, remaining = parser.parse_known_args(unknown_args)

    # Split Dynamo-side values out of the namespace before handing it to
    # from_cli_args, so they never masquerade as engine arguments.
    dynamo_values = {}
    for dest in dynamo_side_dests:
        dynamo_values[dest] = getattr(parsed, dest)
        delattr(parsed, dest)

    # from_cli_args distinguishes explicitly-set flags from argparse defaults
    # by scanning sys.argv — correct for a worker process, but wrong for any
    # caller that passes an argument list (tests, embedding). Communicate the
    # flags we parsed through the side channel the engine parser supports, so
    # resolution never depends on process argv. Each raw flag is resolved to
    # its parser destination (not its raw spelling): argparse accepts
    # abbreviations and aliases (e.g. --tp for --tp-size), and recording the
    # raw text would make the engine treat the real field as unspecified.
    option_to_dest = {
        opt: action.dest for action in parser._actions for opt in action.option_strings
    }

    def _resolve_dest(raw_flag: str) -> Optional[str]:
        if raw_flag in option_to_dest:
            return option_to_dest[raw_flag]
        # argparse allows unambiguous prefixes; mirror that resolution.
        matches = {d for o, d in option_to_dest.items() if o.startswith(raw_flag)}
        return matches.pop() if len(matches) == 1 else None

    explicit_names = set()
    for arg in unknown_args:
        if not arg.startswith("--"):
            continue
        raw = arg.split("=", 1)[0]
        dest = _resolve_dest(raw)
        if dest is None:
            # Not a registered option (e.g. dynamic --<component>-path flags
            # the engine resolves itself): keep the normalized raw name.
            dest = raw.replace("-", "_").lstrip("_")
        explicit_names.add(dest)
    explicit_names -= set(dynamo_side_dests)

    # The engine defaults num_gpus to 1 and does not derive it from
    # parallelism degrees, so tp/dp without an explicit --num-gpus would
    # under-allocate. Preserve the historical num_gpus = tp * dp behavior.
    if "num_gpus" not in explicit_names:
        tp = getattr(parsed, "tp_size", None) or 1
        dp = getattr(parsed, "dp_size", None) or 1
        if tp * dp > 1:
            parsed.num_gpus = tp * dp
            explicit_names.add("num_gpus")
            logger.info(
                "Derived num_gpus=%d from tp_size=%d * dp_size=%d", tp * dp, tp, dp
            )

    if hasattr(parsed, "_sglang_explicit_arg_names"):
        explicit_names |= set(parsed._sglang_explicit_arg_names)
    parsed._sglang_explicit_arg_names = tuple(sorted(explicit_names))

    engine_args = DiffusionServerArgs.from_cli_args(parsed, remaining)

    adapter = DiffusionWorkerArgs(
        engine_args=engine_args,
        served_model_name=dynamo_values.get(
            "served_model_name", getattr(parsed, "served_model_name", None)
        ),
        enable_metrics=dynamo_values.get(
            "enable_metrics", getattr(parsed, "enable_metrics", False)
        ),
    )

    logger.info(
        "Diffusion worker args parsed natively: model_path=%s, "
        "served_model_name=%s (SGLang diffusion ServerArgs, %d fields reachable)",
        adapter.model_path,
        adapter.served_model_name,
        len(getattr(engine_args, "__dataclass_fields__", ())),
    )
    return parsed, adapter
