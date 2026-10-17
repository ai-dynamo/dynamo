# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for :mod:`dynamo_test.argv`.

The interesting cases are all forms that appear in ``recipes/`` — quoted
``$VAR`` values, ``&&`` chains, ``\\``-continued multi-line scripts, and shell
comments containing apostrophes. Each of those broke the previous
split-and-rejoin implementation in a way that produced a green test and a
misconfigured deployment.
"""

import pytest
from dynamo_test.argv import (
    AmbiguousInsertion,
    ArgForm,
    ArgV,
    TokenKind,
    UnparseableCommand,
    is_shell_command_flag,
    tokenize,
)

# A realistic worker command: continuations, comments, quoted vars, an operator,
# and a single-quoted JSON argument.
RECIPE_LIKE = """ulimit -n 1048576 && \\
# vLLM's own scheduler is used here; dynamo's is stale for this image
exec python3 -m dynamo.vllm \\
  --model "${MODEL_ID}" \\
  --tensor-parallel-size 8 \\
  --enforce-eager \\
  --kv-events-config '{"publisher":"zmq","endpoint":"tcp://*:20081"}'"""


# --------------------------------------------------------------------- form


@pytest.mark.parametrize(
    "flag, expected",
    [
        ("-c", True),
        ("-lc", True),
        ("-ec", True),
        ("-euxc", True),
        ("--config", False),
        ("-l", False),
        ("-", False),
        ("c", False),
        (None, False),
    ],
)
def test_shell_command_flag_covers_the_clusters_recipes_use(flag, expected):
    """``-lc`` is the majority spelling in the corpus, not an exception to ``-c``."""
    assert is_shell_command_flag(flag) is expected


@pytest.mark.parametrize("shell_flag", ["-c", "-lc", "-ec"])
def test_container_launched_through_a_shell_is_shell_form(shell_flag):
    container = {
        "command": ["/bin/bash", shell_flag],
        "args": ["python3 -m dynamo.vllm"],
    }
    assert ArgV.from_container(container).form is ArgForm.SHELL


def test_container_with_a_program_command_is_argv_form():
    container = {
        "command": ["python3", "-m", "dynamo.vllm"],
        "args": ["--model", "Qwen/Qwen3-0.6B"],
    }
    argv = ArgV.from_container(container)
    assert argv.form is ArgForm.ARGV
    assert argv.get("--model").require() == "Qwen/Qwen3-0.6B"


def test_single_arg_under_a_non_shell_command_is_one_argument_not_a_script():
    """``--config foo`` is an argument even though ``args`` has one element."""
    container = {"command": ["python3", "-m", "dynamo.vllm"], "args": ["--help me"]}
    assert ArgV.from_container(container).form is ArgForm.ARGV


# ----------------------------------------------------------------- tokenize


def test_tokenize_records_source_spans():
    tokens = tokenize("--model  Qwen/Qwen3-0.6B")
    assert [t.text for t in tokens] == ["--model", "Qwen/Qwen3-0.6B"]
    assert tokens[1].start == 9
    assert (
        "--model  Qwen/Qwen3-0.6B"[tokens[1].start : tokens[1].end] == "Qwen/Qwen3-0.6B"
    )


def test_tokenize_separates_operators_from_words():
    kinds = [(t.text, t.kind) for t in tokenize("a && b | c")]
    assert kinds == [
        ("a", TokenKind.WORD),
        ("&&", TokenKind.OPERATOR),
        ("b", TokenKind.WORD),
        ("|", TokenKind.OPERATOR),
        ("c", TokenKind.WORD),
    ]


def test_tokenize_ignores_whole_line_comments_including_apostrophes():
    """An apostrophe in a comment must not read as an opening quote.

    ``shlex`` has no comment concept, so ``# Dynamo's adapter`` opened a quote
    that never closed and four recipes failed to tokenise at all.
    """
    tokens = tokenize("# Dynamo's metrics adapter is stale\nexec python3 --model X")
    assert [t.text for t in tokens] == ["exec", "python3", "--model", "X"]


def test_tokenize_keeps_a_mid_line_hash_inside_its_token():
    """A ``#`` inside an argument is data, not a comment — as bash agrees."""
    tokens = tokenize("--endpoint https://host/path#frag")
    assert tokens[1].text == "https://host/path#frag"


def test_tokenize_strips_quotes_from_text_but_not_from_the_span():
    tokens = tokenize('--model "${MODEL_ID}"')
    assert tokens[1].text == "${MODEL_ID}"
    assert tokens[1].quoted is True
    assert '--model "${MODEL_ID}"'[tokens[1].start : tokens[1].end] == '"${MODEL_ID}"'


def test_tokenize_treats_a_backslash_newline_as_a_continuation():
    assert [t.text for t in tokenize("--model \\\n  X")] == ["--model", "X"]


@pytest.mark.parametrize(
    "broken", ["--model 'unterminated", '--model "unterminated', "trailing \\"]
)
def test_tokenize_refuses_rather_than_guessing(broken):
    with pytest.raises(UnparseableCommand):
        tokenize(broken)


# --------------------------------------------------------------------- read


def test_reads_a_flag_from_a_multiline_shell_command():
    argv = ArgV.shell(RECIPE_LIKE, command=("/bin/bash", "-lc"))
    assert argv.get("--model").require() == "${MODEL_ID}"
    assert argv.get("--tensor-parallel-size").require() == "8"


def test_reads_the_equals_spelling():
    assert ArgV.argv(["--model=Qwen/Qwen3-0.6B"]).get("--model").require() == (
        "Qwen/Qwen3-0.6B"
    )


def test_a_switch_has_no_value_and_says_so():
    """``ABSENT`` here is a justified claim, not a failure to look."""
    fact = ArgV.shell(RECIPE_LIKE).get("--enforce-eager")
    assert fact.is_absent
    assert "switch" in fact.detail
    assert ArgV.shell(RECIPE_LIKE).has("--enforce-eager").require() is True


def test_a_missing_flag_is_absent_with_the_scan_recorded():
    fact = ArgV.shell(RECIPE_LIKE).get("--speculative-config")
    assert fact.is_absent
    assert "not among" in fact.detail


def test_an_unreadable_command_is_unknown_never_absent():
    """The false-green this type exists to prevent.

    Reporting ``--model`` as absent because the command could not be tokenised
    is a false statement that passes silently.
    """
    argv = ArgV.shell("exec python3 --model 'unterminated")
    assert argv.is_parseable is False
    assert argv.get("--model").is_unknown
    assert argv.has("--model").is_unknown
    assert argv.model().is_unknown


def test_repeated_flags_are_all_visible():
    fact = ArgV.argv(["--x", "1", "--x", "2"]).get_all("--x")
    assert fact.require() == ("1", "2")


def test_model_tries_each_spelling():
    assert ArgV.argv(["--model-path", "A"]).model().require() == "A"
    assert ArgV.argv(["--served-model-name", "B"]).model().require() == "B"
    assert ArgV.argv(["--tp", "8"]).model().is_absent


# -------------------------------------------------------------- write: shell


def test_setting_a_value_touches_only_that_span():
    """Everything else — comments, continuations, operators, quoting — survives.

    Rebuilding from tokens flattens the script to one line, deletes the comment,
    and quotes ``&&`` into a literal argument.
    """
    before = ArgV.shell(RECIPE_LIKE, command=("/bin/bash", "-lc"))
    after = before.set("--model", "meta-llama/Llama-3.1-8B")

    out = after.as_shell_string()
    changed = [
        (a, b) for a, b in zip(RECIPE_LIKE.splitlines(), out.splitlines()) if a != b
    ]
    assert len(changed) == 1
    assert changed[0][1].strip() == "--model meta-llama/Llama-3.1-8B \\"
    assert after.get("--model").require() == "meta-llama/Llama-3.1-8B"
    # Untouched structure.
    assert "ulimit -n 1048576 && \\" in out
    assert "# vLLM's own scheduler is used here" in out
    assert '\'{"publisher":"zmq","endpoint":"tcp://*:20081"}\'' in out
    assert out.count("\n") == RECIPE_LIKE.count("\n")


def test_an_operator_is_never_quoted_into_an_argument():
    """``'&&'`` collapses the command into one call and the worker never starts."""
    out = (
        ArgV.shell("ulimit -l unlimited && exec python3 --model A")
        .set("--model", "B")
        .as_shell_string()
    )
    assert out == "ulimit -l unlimited && exec python3 --model B"


def test_setting_an_absent_flag_lands_beside_the_existing_flags():
    out = ArgV.shell(RECIPE_LIKE).set("--max-model-len", "1024").as_shell_string()
    assert out.rstrip().endswith("--max-model-len 1024")
    assert ArgV.shell(out).get("--max-model-len").require() == "1024"


def test_setting_an_existing_flag_replaces_rather_than_appends():
    """Appending yields ``--tp 8 --tp 4``; the engine takes one and the test the other."""
    argv = ArgV.shell("exec python3 --tensor-parallel-size 8").set(
        "--tensor-parallel-size", "4"
    )
    assert argv.as_shell_string() == "exec python3 --tensor-parallel-size 4"
    assert argv.get_all("--tensor-parallel-size").require() == ("4",)


def test_a_boolean_flag_is_emitted_as_a_switch_not_as_a_value():
    """``--enforce-eager True`` is rejected by engines that declare it store_true."""
    out = ArgV.shell("exec python3 --model A").set("--enforce-eager", True)
    assert out.as_shell_string() == "exec python3 --model A --enforce-eager"


def test_setting_false_removes_the_flag():
    out = ArgV.shell("exec python3 --model A --enforce-eager").set(
        "--enforce-eager", False
    )
    assert out.as_shell_string() == "exec python3 --model A"


def test_unset_removes_the_flag_and_its_value_without_leaving_gaps():
    out = ArgV.shell("exec python3 --model A --tp 8 --trust-remote-code").unset("--tp")
    assert out.as_shell_string() == "exec python3 --model A --trust-remote-code"


def test_a_value_needing_quotes_gets_them():
    out = ArgV.shell("exec python3 --model A").set("--kv-config", '{"publisher":"zmq"}')
    assert out.as_shell_string().endswith('--kv-config \'{"publisher":"zmq"}\'')
    assert ArgV.shell(out.as_shell_string()).get("--kv-config").require() == (
        '{"publisher":"zmq"}'
    )


def test_a_dollar_reference_stays_expandable():
    """Quoting ``$MODEL_PATH`` literally would break expansion at pod start."""
    out = ArgV.shell("exec python3 --model A").set("--model", "$MODEL_PATH")
    assert out.as_shell_string() == "exec python3 --model $MODEL_PATH"


def test_editing_an_unparseable_command_raises_instead_of_corrupting_it():
    with pytest.raises(UnparseableCommand):
        ArgV.shell("exec python3 --model 'unterminated").set("--model", "B")


def test_insertion_refuses_when_there_is_no_defensible_place():
    """Appending after ``&& something-else`` would configure the wrong program."""
    with pytest.raises(AmbiguousInsertion):
        ArgV.shell("setup.sh && teardown.sh").set("--model", "A")


# --------------------------------------------------------------- write: argv


def test_argv_set_replaces_in_place():
    argv = ArgV.argv(["--model", "A", "--tp", "8"]).set("--model", "B")
    assert argv.as_container_args() == ["--model", "B", "--tp", "8"]


def test_argv_set_appends_when_absent():
    argv = ArgV.argv(["--model", "A"]).set("--tp", "8")
    assert argv.as_container_args() == ["--model", "A", "--tp", "8"]


def test_argv_boolean_and_unset():
    assert ArgV.argv(["--model", "A"]).set("--eager", True).as_container_args() == [
        "--model",
        "A",
        "--eager",
    ]
    assert ArgV.argv(["--model", "A", "--tp", "8"]).unset(
        "--tp"
    ).as_container_args() == [
        "--model",
        "A",
    ]


def test_argv_handles_the_equals_spelling_on_write():
    argv = ArgV.argv(["--model=A"]).set("--model", "B")
    assert argv.as_container_args() == ["--model=B"]


# ---------------------------------------------------------------------- emit


def test_shell_form_always_writes_exactly_one_args_element():
    """A second element becomes ``$1`` and the worker never sees it."""
    container = {
        "command": ["/bin/bash", "-lc"],
        "args": ["exec python3 -m dynamo.vllm --model A"],
    }
    argv = ArgV.from_container(container).set("--max-model-len", "1024")
    argv.apply_to(container)
    assert len(container["args"]) == 1
    assert "--max-model-len 1024" in container["args"][0]


def test_edits_are_immutable():
    before = ArgV.shell("exec python3 --model A")
    before.set("--model", "B")
    assert before.as_shell_string() == "exec python3 --model A"
