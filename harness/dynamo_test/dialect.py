# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Saying what you mean, once, across three engines that spell it differently.

A test that wants a 4096-token context should not have to know that vLLM calls
that ``--max-model-len``, SGLang calls it ``--context-length``, and TensorRT-LLM
calls it ``--max-seq-len``. It says ``context_length=4096`` and the dialect
emits the right flag.

## Read every spelling, write one

The mapping is not one-to-one, and that is measured rather than assumed. Across
the 312 engine workers in ``recipes/`` and ``examples/`` — identified by the
module each actually launches — three settings have more than one live
spelling:

=============== ============= =========================================
setting         engine        spellings in use
=============== ============= =========================================
model           vLLM          ``--model`` x202
                SGLang        ``--model-path`` x45
                TensorRT-LLM  ``--model-path`` x44 **and** ``--model`` x12
tensor_parallel vLLM          ``--tensor-parallel-size`` x164
                **SGLang**    ``--tp`` x30, ``--tensor-parallel-size`` x7,
                              **and** ``--tp-size`` x2
                TensorRT-LLM  ``--tensor-parallel-size`` x15
=============== ============= =========================================

So a reader that knows only the canonical ``--tensor-parallel-size`` finds it in
7 of the 39 SGLang workers that set tensor parallelism, and one that knows only
``--model-path`` misses 12 of TensorRT-LLM's 56. Reading has to accept every
spelling the engine accepts; writing picks one. Those are different lists, and
conflating them is what makes a scan report a flag as absent when it is right
there.

## Settings that are not on the command line at all

**All 56** TensorRT-LLM workers pass ``--extra-engine-args``, and 12 SGLang
workers pass ``--config``. Their settings live in a file this reader cannot
open. Answering "not set" for those would be a false statement, so a setting
missing from a command that defers to a file is reported ``UNKNOWN``, naming the
file — never ``ABSENT``.

## Flags that take no value

``--enforce-eager`` is a switch. Emitting ``--enforce-eager True`` is rejected by
the engine, and it is a defect the existing helpers have. A semantic declared
:data:`SWITCH` emits the bare flag when true and nothing when false.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

from .argv import ArgV
from .facts import Fact
from .roles import Process

__all__ = [
    "SWITCH",
    "Semantic",
    "EngineDialect",
    "Dialect",
    "VLLM",
    "SGLANG",
    "TRTLLM",
    "DIALECTS",
    "for_backend",
    "UnknownSemantic",
]


class _Switch:
    """Marker for a flag that takes no value."""

    def __repr__(self) -> str:
        return "SWITCH"


SWITCH = _Switch()


class UnknownSemantic(KeyError):
    def __init__(self, name: object, backend: str, known: Sequence[str]) -> None:
        super().__init__(
            f"{backend} has no setting called {name!r}; it knows: "
            f"{', '.join(sorted(known))}"
        )


@dataclass(frozen=True)
class Semantic:
    """One tunable, and how this engine spells it.

    ``write`` is the canonical flag. ``read`` lists every spelling the engine
    accepts, canonical first — a manifest may use any of them.
    """

    write: str
    read: tuple[str, ...] = ()
    switch: bool = False

    @property
    def spellings(self) -> tuple[str, ...]:
        return (self.write,) + tuple(r for r in self.read if r != self.write)


@runtime_checkable
class EngineDialect(Protocol):
    """What a backend must tell the harness about itself."""

    backend: str

    def flag(self, semantic: str, value: Any) -> tuple[str, ...]:
        ...

    def read(self, argv: ArgV, semantic: str) -> Fact[str]:
        ...

    def process_pattern(self, process: Process) -> str | None:
        ...


@dataclass(frozen=True)
class Dialect:
    """A concrete engine dialect: a name, a settings table, process patterns.

    ``config_flags`` are flags that move settings out of the command line and
    into a file. When one is present, a setting missing from argv is not
    *absent* — it is somewhere this reader cannot see.
    """

    backend: str
    settings: Mapping[str, Semantic]
    processes: Mapping[Process, str] = None  # type: ignore[assignment]
    config_flags: tuple[str, ...] = ("--config",)

    def __post_init__(self) -> None:
        if self.processes is None:
            object.__setattr__(self, "processes", {})

    def _deferred_to(self, argv: ArgV) -> str | None:
        """The config flag this command defers settings to, if any."""
        for flag in self.config_flags:
            fact = argv.get(flag)
            if fact.is_known:
                return f"{flag} {fact.require()}"
            if argv.has(flag).or_else(False):
                return flag
        return None

    def semantic(self, name: str) -> Semantic:
        try:
            return self.settings[name]
        except KeyError:
            raise UnknownSemantic(name, self.backend, list(self.settings)) from None

    def flag(self, semantic: str, value: Any) -> tuple[str, ...]:
        """The argv fragment that sets ``semantic`` to ``value``.

        A switch emits the bare flag, or nothing at all when false. Emitting
        ``--enforce-eager True`` is rejected by the engine.
        """
        spec = self.semantic(semantic)
        if spec.switch:
            if not isinstance(value, bool):
                raise TypeError(
                    f"{self.backend}.{semantic} is a switch; pass True or False, "
                    f"not {value!r}"
                )
            return (spec.write,) if value else ()
        if isinstance(value, bool):
            raise TypeError(
                f"{self.backend}.{semantic} takes a value, not a bool; a bool here "
                f"would emit '{spec.write} {value}', which the engine rejects"
            )
        return (spec.write, str(value))

    def read(self, argv: ArgV, semantic: str) -> Fact[str]:
        """Read ``semantic`` from a command line, trying every spelling.

        Unparseable input propagates as ``UNKNOWN``. Absence is only reported
        after every spelling this engine accepts has been tried, and the detail
        says which — so "not set" is a claim with its working shown.
        """
        spec = self.semantic(semantic)
        tried: list[str] = []
        for flag in spec.spellings:
            fact = argv.get(flag)
            if fact.is_unknown:
                return fact
            if fact.is_known:
                return fact
            tried.append(flag)
        if spec.switch:
            present = argv.has(spec.write)
            if present.is_unknown:
                return present
        deferred = self._deferred_to(argv)
        if deferred is not None:
            # Not absent: the command hands its settings to a file this reader
            # cannot open. Saying "not set" here would be a false statement, and
            # 12 SGLang workers and all 56 TensorRT-LLM workers are
            # configured exactly this way.
            return Fact.unknown(
                argv.source,
                f"{self.backend}.{semantic} is not on the command line, which "
                f"defers to '{deferred}'; read the config to answer this",
            )
        return Fact.absent(
            argv.source,
            f"{self.backend}.{semantic} is not set; tried {', '.join(tried)}",
        )

    def apply(self, argv: ArgV, **settings: Any) -> ArgV:
        """Set several semantics at once, replacing rather than appending."""
        for name, value in settings.items():
            spec = self.semantic(name)
            if spec.switch:
                argv = argv.set(spec.write, bool(value))
            else:
                argv = argv.set(spec.write, str(value))
        return argv

    def process_pattern(self, process: Process) -> str | None:
        return self.processes.get(process)

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self.settings))


def _s(write: str, *read: str, switch: bool = False) -> Semantic:
    return Semantic(write=write, read=(write,) + read, switch=switch)


# Counts in the comments are worker containers in recipes/ + examples/,
# identified by the module they launch.

VLLM = Dialect(
    backend="vllm",
    settings={
        "model": _s("--model"),  # 202 of 205 vLLM workers
        "served_model_name": _s("--served-model-name"),
        "context_length": _s("--max-model-len"),  # 110
        "tensor_parallel": _s("--tensor-parallel-size"),  # 164
        "pipeline_parallel": _s("--pipeline-parallel-size"),  # 14
        "data_parallel": _s("--data-parallel-size"),  # 24
        "gpu_memory_fraction": _s("--gpu-memory-utilization"),  # 114
        "max_batch_size": _s("--max-num-seqs"),  # 96
        "max_batched_tokens": _s("--max-num-batched-tokens"),  # 87
        "kv_cache_dtype": _s("--kv-cache-dtype"),  # 57
        "eager": _s("--enforce-eager", switch=True),  # 14, takes no value
        "trust_remote_code": _s("--trust-remote-code", switch=True),
        "expert_parallel": _s("--enable-expert-parallel", switch=True),  # 23
        "tool_parser": _s("--dyn-tool-call-parser", "--tool-call-parser"),  # 82 / 4
        "reasoning_parser": _s("--dyn-reasoning-parser", "--reasoning-parser"),
    },
    processes={Process.MAIN: "dynamo.vllm", Process.ENGINE: "VLLM::EngineCore"},
)

SGLANG = Dialect(
    backend="sglang",
    settings={
        "model": _s("--model-path"),  # 45
        "served_model_name": _s("--served-model-name"),
        "context_length": _s("--context-length"),  # 3
        # Three spellings are live: --tp x30, --tensor-parallel-size x7,
        # --tp-size x2. Reading only the canonical one finds 7 of 39.
        "tensor_parallel": _s("--tp", "--tensor-parallel-size", "--tp-size"),
        "data_parallel": _s("--data-parallel-size", "--dp", "--dp-size"),
        "gpu_memory_fraction": _s("--mem-fraction-static"),  # 19
        "max_batch_size": _s("--max-running-requests"),  # 7
        "max_batched_tokens": _s("--chunked-prefill-size"),  # 15
        "kv_cache_dtype": _s("--kv-cache-dtype"),
        "eager": _s("--disable-cuda-graph", switch=True),
        "trust_remote_code": _s("--trust-remote-code", switch=True),
        "tool_parser": _s("--dyn-tool-call-parser", "--tool-call-parser"),
        "reasoning_parser": _s("--dyn-reasoning-parser", "--reasoning-parser"),
    },
    processes={Process.MAIN: "dynamo.sglang", Process.ENGINE: "sglang::scheduler"},
)

TRTLLM = Dialect(
    backend="trtllm",
    settings={
        # Both spellings are live: --model-path x44, --model x12.
        "model": _s("--model-path", "--model"),
        "served_model_name": _s("--served-model-name"),
        "context_length": _s("--max-seq-len"),  # 17
        "tensor_parallel": _s("--tensor-parallel-size", "--tp"),  # 15
        "gpu_memory_fraction": _s("--free-gpu-memory-fraction"),  # 5
        "max_batch_size": _s("--max-batch-size"),  # 29
        "max_batched_tokens": _s("--max-num-tokens"),  # 24
        "kv_cache_dtype": _s("--kv-cache-dtype"),
        "trust_remote_code": _s("--trust-remote-code", switch=True),
        "extra_engine_args": _s("--extra-engine-args"),  # all 56 TRT-LLM workers
        "tool_parser": _s("--dyn-tool-call-parser", "--tool-call-parser"),
        "reasoning_parser": _s("--dyn-reasoning-parser", "--reasoning-parser"),
    },
    processes={Process.MAIN: "dynamo.trtllm", Process.ENGINE: "trtllm_worker"},
    config_flags=("--config", "--extra-engine-args"),
)

DIALECTS: Mapping[str, Dialect] = {d.backend: d for d in (VLLM, SGLANG, TRTLLM)}


def for_backend(backend: str) -> Dialect:
    try:
        return DIALECTS[backend.lower()]
    except KeyError:
        raise KeyError(
            f"no dialect for {backend!r}; known: {', '.join(sorted(DIALECTS))}"
        ) from None


def detect(argv: ArgV) -> Fact[str]:
    """Which engine a command line launches, read from ``python -m dynamo.X``.

    Deliberately not a substring search over the whole command. Matching
    ``"vllm"`` anywhere picks up image names and environment variables, and it
    mis-attributed 12 TensorRT-LLM containers when this was first measured that
    way.

    Reads the *whole* invocation, not just ``args``. v1alpha1 manifests put the
    program in ``command`` (``[python3, -m, dynamo.sglang]``) and only its flags
    in ``args``, so scanning ``args`` alone identifies no engine at all for
    them.
    """
    if not argv.is_parseable:
        return Fact.unknown(
            argv.source, f"command could not be tokenised: {argv.parse_error}"
        )
    tokens = list(argv.invocation())
    for i, token in enumerate(tokens):
        if token != "-m" or i + 1 >= len(tokens):
            continue
        module = tokens[i + 1]
        if not module.startswith("dynamo."):
            continue
        backend = module.split(".", 1)[1]
        if backend in DIALECTS:
            return Fact.known(backend, argv.source, f"launches {module}")
        return Fact.absent(
            argv.source, f"launches {module}, which is not an inference engine"
        )
    return Fact.absent(argv.source, "no 'python -m dynamo.<engine>' invocation found")
