# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU stand-ins preserve the source-pinned subset used by native strategies."""

import ast
import builtins
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from dynamo.vllm import decision_native_loader as native_loader
from dynamo.vllm.tests import decision_read_test_support as support
from dynamo.vllm.tests.decision_pinned_source import (
    read_pinned_source,
    source_repository,
)
from dynamo.vllm.tests.decision_stub_contract import project_stub_contract

pytest_plugins = ["dynamo.vllm.tests.decision_pinned_fixtures"]

pytestmark = [
    pytest.mark.unit,
    pytest.mark.pre_merge,
    pytest.mark.vllm,
    pytest.mark.core,
    pytest.mark.gpu_0,
]


@pytest.fixture(scope="module")
def api_contract():
    path = Path(__file__).parent / "fixtures/vllm_0_33_0/api-contract.json"
    contract = json.loads(path.read_bytes())
    assert contract["pinned_vllm_commit"] == native_loader.PINNED_VLLM_COMMIT
    blobs = dict(native_loader.PINNED_SOURCE_BLOBS)
    for source, projection in contract["sources"].items():
        assert projection["git_blob"] == blobs[source]
    repository = source_repository()
    if repository is not None:
        assert (
            project_stub_contract(lambda path: read_pinned_source(repository, path))
            == contract
        )
    return contract["sources"]


@pytest.mark.parametrize(
    "source,name,owner,attribute",
    [
        ("vllm/inputs/engine.py", "tokens_input", support, "tokens_input"),
        (
            "vllm/renderers/inputs/preprocess.py",
            "extract_prompt_components",
            support,
            "extract_prompt_components",
        ),
        (
            "vllm/renderers/online_renderer.py",
            "OnlineRenderer.__init__",
            support.OnlineRenderer,
            "__init__",
        ),
        (
            "vllm/renderers/online_renderer.py",
            "OnlineRenderer.preprocess_chat",
            support.OnlineRenderer,
            "preprocess_chat",
        ),
    ],
)
def test_cpu_stub_signatures_match_pinned_owner_contract(
    api_contract, source, name, owner, attribute
):
    assert (
        str(inspect.signature(getattr(owner, attribute)))
        == api_contract[source]["signatures"][name]
    )


def test_sampling_stub_preserves_pinned_defaults_and_label_limit(
    api_contract, native_reads
):
    projection = api_contract["vllm/sampling_params.py"]
    assert (
        vars(support.SamplingParams()) == projection["field_defaults"]["SamplingParams"]
    )
    assert (
        native_reads.strategies.MAX_LOGPROB_TOKEN_IDS
        == projection["constants"]["MAX_LOGPROB_TOKEN_IDS"]
    )
    with pytest.raises(TypeError):
        support.SamplingParams(unknown_field=True)


def test_pinned_strategy_warm_prefix_constant_is_not_an_unchecked_stub(
    pinned_source, native_reads
):
    tree = ast.parse(pinned_source("vllm/entrypoints/openai/decisions/strategies.py"))
    definition = next(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "MIN_WARM_PREFIX_TOKENS"
            for target in node.targets
        )
    )
    assert native_reads.strategies.MIN_WARM_PREFIX_TOKENS == ast.literal_eval(
        definition.value
    )


@pytest.mark.parametrize("prompt,salt", [(None, None), ("hello", "isolated")])
def test_token_input_stub_preserves_pinned_optional_field_omission(prompt, salt):
    actual = support.tokens_input([1, 2], prompt=prompt, cache_salt=salt)
    assert actual["type"] == "token" and actual["prompt_token_ids"] == [1, 2]
    assert ("prompt" in actual) == (prompt is not None)
    assert ("cache_salt" in actual) == (salt is not None)
    components = support.extract_prompt_components(
        SimpleNamespace(is_encoder_decoder=False), actual
    )
    assert components.token_ids == [1, 2]
    assert components.text == prompt and components.embeds is None


def test_hermetic_fixtures_inject_the_verified_stub_owners(native_reads):
    assert native_reads.strategies.tokens_input is support.tokens_input
    assert (
        native_reads.strategies.extract_prompt_components
        is support.extract_prompt_components
    )
    assert native_reads.strategies.SamplingParams is support.SamplingParams
    assert native_reads.online_renderer is support.OnlineRenderer


def test_hermetic_strategy_loading_never_imports_installed_vllm(monkeypatch, request):
    original_import = builtins.__import__

    def reject_vllm(name, *args, **kwargs):
        if name == "vllm" or name.startswith("vllm."):
            raise AssertionError("CPU references must not import the installed backend")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_vllm)
    native = request.getfixturevalue("native_reads")
    assert (
        native.strategies.LiquidStrategy
        and native.strategies.DiffusionGemmaCanvasStrategy
    )
