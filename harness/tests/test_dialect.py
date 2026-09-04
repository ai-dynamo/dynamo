# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for :mod:`dynamo_test.dialect`."""

import pytest
from dynamo_test.argv import ArgV
from dynamo_test.dialect import (
    DIALECTS,
    SGLANG,
    TRTLLM,
    VLLM,
    UnknownSemantic,
    detect,
    for_backend,
)
from dynamo_test.roles import Process

# ------------------------------------------------------------------ writing


@pytest.mark.parametrize(
    "dialect, expected",
    [
        (VLLM, ("--max-model-len", "4096")),
        (SGLANG, ("--context-length", "4096")),
        (TRTLLM, ("--max-seq-len", "4096")),
    ],
)
def test_one_semantic_three_spellings(dialect, expected):
    """The point of the whole module: say what you mean once."""
    assert dialect.flag("context_length", 4096) == expected


def test_a_switch_emits_a_bare_flag():
    """``--enforce-eager True`` is rejected by vLLM; it takes no value."""
    assert VLLM.flag("eager", True) == ("--enforce-eager",)
    assert VLLM.flag("eager", False) == ()


def test_a_switch_rejects_a_non_bool():
    with pytest.raises(TypeError, match="pass True or False"):
        VLLM.flag("eager", "yes")


def test_a_valued_flag_rejects_a_bool():
    """Guards the exact defect: a bool would emit `--max-model-len True`."""
    with pytest.raises(TypeError, match="rejects"):
        VLLM.flag("context_length", True)


def test_an_unknown_semantic_lists_what_the_engine_knows():
    with pytest.raises(UnknownSemantic, match="context_length"):
        VLLM.flag("maximum_context", 4096)


# ------------------------------------------------------------------ reading


def test_reading_accepts_every_spelling_the_engine_does():
    """SGLang uses ``--tp`` 10 times and ``--tensor-parallel-size`` 7 times.

    A reader that knows only the canonical spelling misses 10 of 17 workers.
    """
    assert SGLANG.read(ArgV.argv(["--tp", "8"]), "tensor_parallel").require() == "8"
    assert (
        SGLANG.read(
            ArgV.argv(["--tensor-parallel-size", "8"]), "tensor_parallel"
        ).require()
        == "8"
    )


def test_trtllm_reads_both_model_spellings():
    """``--model-path`` ×29 and ``--model`` ×12 are both live."""
    assert TRTLLM.read(ArgV.argv(["--model-path", "A"]), "model").require() == "A"
    assert TRTLLM.read(ArgV.argv(["--model", "A"]), "model").require() == "A"


def test_absence_shows_which_spellings_were_tried():
    """ "Not set" is a claim, and it should carry its working."""
    fact = SGLANG.read(ArgV.argv(["--model-path", "A"]), "tensor_parallel")
    assert fact.is_absent
    assert "--tp" in fact.detail
    assert "--tensor-parallel-size" in fact.detail


def test_an_unreadable_command_is_unknown_not_absent():
    argv = ArgV.shell("exec python3 -m dynamo.vllm --model 'unterminated")
    assert VLLM.read(argv, "model").is_unknown


def test_reading_works_through_a_shell_command():
    argv = ArgV.shell(
        "ulimit -n 65536 && exec python3 -m dynamo.sglang --model-path Qwen/Q --tp 8",
        command=("/bin/bash", "-lc"),
    )
    assert SGLANG.read(argv, "model").require() == "Qwen/Q"
    assert SGLANG.read(argv, "tensor_parallel").require() == "8"


# ------------------------------------------------------------------ applying


def test_apply_sets_several_semantics_at_once():
    argv = ArgV.shell("exec python3 -m dynamo.vllm --model A --max-model-len 2048")
    out = VLLM.apply(argv, context_length=4096, max_batch_size=64)
    assert VLLM.read(out, "context_length").require() == "4096"
    assert VLLM.read(out, "max_batch_size").require() == "64"
    assert out.as_shell_string().count("--max-model-len") == 1


def test_apply_preserves_the_shell_command_around_it():
    argv = ArgV.shell(
        "ulimit -n 65536 && exec python3 -m dynamo.vllm --model A",
        command=("/bin/bash", "-lc"),
    )
    out = VLLM.apply(argv, context_length=4096)
    assert out.as_shell_string().startswith("ulimit -n 65536 && exec")
    assert "'&&'" not in out.as_shell_string()


def test_apply_of_a_switch_emits_no_value():
    out = VLLM.apply(ArgV.argv(["--model", "A"]), eager=True)
    assert out.as_container_args() == ["--model", "A", "--enforce-eager"]


# ----------------------------------------------------------------- detection


@pytest.mark.parametrize("backend", ["vllm", "sglang", "trtllm"])
def test_detect_reads_the_launched_module(backend):
    argv = ArgV.shell(f"exec python3 -m dynamo.{backend} --model A")
    assert detect(argv).require() == backend


def test_detect_ignores_a_backend_name_that_is_not_the_module():
    """A substring search over the whole command mis-attributed 12 containers.

    Image names and environment variables mention engines they do not launch.
    """
    argv = ArgV.shell(
        "export IMAGE=nvcr.io/dynamo-vllm:latest && "
        "exec python3 -m dynamo.trtllm --model-path A"
    )
    assert detect(argv).require() == "trtllm"


def test_detect_reports_a_non_engine_module_as_absent():
    fact = detect(ArgV.shell("exec python3 -m dynamo.frontend"))
    assert fact.is_absent
    assert "not an inference engine" in fact.detail


def test_detect_is_unknown_on_an_unreadable_command():
    assert detect(ArgV.shell("exec python3 -m dynamo.vllm 'unterminated")).is_unknown


# ------------------------------------------------------------------- lookup


def test_for_backend_and_unknown_backend():
    assert for_backend("VLLM") is VLLM
    with pytest.raises(KeyError, match="sglang"):
        for_backend("tensorrt")


def test_every_dialect_names_its_engine_process():
    for dialect in DIALECTS.values():
        assert dialect.process_pattern(Process.MAIN)
        assert dialect.process_pattern(Process.ENGINE)


def test_the_three_dialects_share_their_core_semantics():
    """A deployment-agnostic test can only use semantics all engines have."""
    common = set.intersection(*(set(d.names()) for d in DIALECTS.values()))
    assert {
        "model",
        "context_length",
        "tensor_parallel",
        "max_batch_size",
        "max_batched_tokens",
        "gpu_memory_fraction",
        "kv_cache_dtype",
        "tool_parser",
    } <= common


# ------------------------------------------------- settings held in a file


def test_a_config_file_makes_a_setting_unknown_not_absent():
    """ "Not set" is false when the command hands its settings to a file.

    Four SGLang workers in the corpus pass only ``--config`` plus parser flags;
    the model lives in ``/etc/sglang/*.yaml``. Answering ABSENT there invites
    the caller to conclude no model is configured, which is wrong.
    """
    argv = ArgV.shell("exec python3 -m dynamo.sglang --config /etc/sglang/prefill.yaml")
    fact = SGLANG.read(argv, "model")
    assert fact.is_unknown
    assert "/etc/sglang/prefill.yaml" in fact.detail


def test_trtllm_defers_through_extra_engine_args():
    """``--extra-engine-args`` is TensorRT-LLM's config flag, and its commonest."""
    argv = ArgV.shell("exec python3 -m dynamo.trtllm --extra-engine-args /cfg/e.yaml")
    assert TRTLLM.read(argv, "context_length").is_unknown


def test_a_flag_on_the_command_line_beats_the_config_file():
    """Deferral only applies to settings the command line does not carry."""
    argv = ArgV.shell(
        "exec python3 -m dynamo.sglang --config /etc/s.yaml --model-path Qwen/Q"
    )
    assert SGLANG.read(argv, "model").require() == "Qwen/Q"
    assert SGLANG.read(argv, "tensor_parallel").is_unknown


def test_without_a_config_flag_absence_is_still_absence():
    """Deferral must not turn every missing flag into UNKNOWN."""
    fact = SGLANG.read(ArgV.argv(["--model-path", "Qwen/Q"]), "tensor_parallel")
    assert fact.is_absent
    assert "--tp" in fact.detail
