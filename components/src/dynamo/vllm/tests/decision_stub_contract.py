# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Project CPU contracts or refresh only the existing pinned source fixtures.

Run with --source-repo <local-git> --refresh-bundled to refresh the source subset,
--check-bundled to verify it, or --check to verify the projected API contract.
"""

import argparse
import ast
import inspect
import json
from functools import partial
from pathlib import Path
from tempfile import NamedTemporaryFile

from dynamo.vllm import decision_native_loader as native_loader
from dynamo.vllm.tests.decision_pinned_source import (
    BUNDLED_SOURCE_ROOT,
    read_pinned_source,
)

CONTRACT_PATH = Path(__file__).parent / "fixtures/vllm_0_33_0/api-contract.json"

_SOURCE_APIS = {
    "vllm/inputs/engine.py": ("tokens_input",),
    "vllm/renderers/inputs/preprocess.py": ("extract_prompt_components",),
    "vllm/renderers/online_renderer.py": (
        "OnlineRenderer.__init__",
        "OnlineRenderer.preprocess_chat",
    ),
    "vllm/sampling_params.py": (),
}
_SAMPLING_FIELDS = {"max_tokens", "temperature", "logprob_token_ids", "extra_args"}


def _signature(node):
    arguments = node.args
    positional = arguments.posonlyargs + arguments.args
    first_default = len(positional) - len(arguments.defaults)
    parameters = []
    for index, argument in enumerate(positional):
        kind = (
            inspect.Parameter.POSITIONAL_ONLY
            if index < len(arguments.posonlyargs)
            else inspect.Parameter.POSITIONAL_OR_KEYWORD
        )
        default = (
            inspect.Parameter.empty
            if index < first_default
            else ast.literal_eval(arguments.defaults[index - first_default])
        )
        parameters.append(inspect.Parameter(argument.arg, kind, default=default))
    for argument, default in zip(arguments.kwonlyargs, arguments.kw_defaults):
        parameters.append(
            inspect.Parameter(
                argument.arg,
                inspect.Parameter.KEYWORD_ONLY,
                default=(
                    inspect.Parameter.empty
                    if default is None
                    else ast.literal_eval(default)
                ),
            )
        )
    assert arguments.vararg is None and arguments.kwarg is None
    return str(inspect.Signature(parameters))


def _definition(tree, name):
    owner, _, method = name.partition(".")
    node = next(node for node in tree.body if getattr(node, "name", None) == owner)
    if method:
        node = next(node for node in node.body if getattr(node, "name", None) == method)
    return node


def project_stub_contract(read_source):
    blobs = dict(native_loader.PINNED_SOURCE_BLOBS)
    sources = {}
    for path, names in _SOURCE_APIS.items():
        tree = ast.parse(read_source(path))
        projection = {"git_blob": blobs[path]}
        if names:
            projection["signatures"] = {
                name: _signature(_definition(tree, name)) for name in names
            }
        else:
            sampling = _definition(tree, "SamplingParams")
            defaults = {
                node.target.id: ast.literal_eval(node.value)
                for node in sampling.body
                if isinstance(node, ast.AnnAssign)
                and isinstance(node.target, ast.Name)
                and node.target.id in _SAMPLING_FIELDS
            }
            assert defaults.keys() == _SAMPLING_FIELDS
            projection["field_defaults"] = {"SamplingParams": defaults}
            limit = next(
                node
                for node in tree.body
                if isinstance(node, ast.Assign)
                and any(
                    isinstance(target, ast.Name)
                    and target.id == "MAX_LOGPROB_TOKEN_IDS"
                    for target in node.targets
                )
            )
            projection["constants"] = {
                "MAX_LOGPROB_TOKEN_IDS": ast.literal_eval(limit.value)
            }
        sources[path] = projection
    return {
        "_generated": "Do not edit; regenerate with python3 -m dynamo.vllm.tests.decision_stub_contract --source-repo <repo>",
        "pinned_vllm_commit": native_loader.PINNED_VLLM_COMMIT,
        "scope": "CPU stub signatures and native-read sampling fields only; not renderer, engine, or GPU behavior.",
        "sources": sources,
    }


def refresh_bundled_sources(repository, *, check):
    sources = tuple(
        (
            path,
            read_pinned_source(
                repository, path.relative_to(BUNDLED_SOURCE_ROOT).as_posix()[:-4]
            ).encode(),
        )
        for path in sorted(BUNDLED_SOURCE_ROOT.rglob("*.py.txt"))
    )
    assert sources, "no bundled source fixtures found"
    for path, data in sources:
        if check:
            assert path.read_bytes() == data, f"Bundled source fixture {path} is stale"
            continue
        temporary = None
        try:
            with NamedTemporaryFile(
                dir=path.parent, prefix=f".{path.name}.", delete=False
            ) as output:
                temporary = Path(output.name)
                output.write(data)
            temporary.chmod(path.stat().st_mode & 0o777)
            temporary.replace(path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    return len(sources)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-repo", required=True)
    operations = parser.add_mutually_exclusive_group()
    operations.add_argument(
        "--check", action="store_true", help="Check the projected API contract"
    )
    operations.add_argument(
        "--refresh-bundled",
        action="store_true",
        help="Refresh only existing bundled sources from the pinned local Git commit",
    )
    operations.add_argument(
        "--check-bundled",
        action="store_true",
        help="Check the existing bundled source subset against pinned local Git",
    )
    arguments = parser.parse_args()
    if arguments.refresh_bundled or arguments.check_bundled:
        count = refresh_bundled_sources(
            arguments.source_repo, check=arguments.check_bundled
        )
        print(
            f"{'Verified' if arguments.check_bundled else 'Refreshed'} {count} pinned source fixtures"
        )
        return
    contract = project_stub_contract(partial(read_pinned_source, arguments.source_repo))
    if arguments.check:
        assert (
            json.loads(CONTRACT_PATH.read_bytes()) == contract
        ), "CPU stub contract is stale"
    else:
        print(json.dumps(contract, indent=2))


if __name__ == "__main__":
    main()
