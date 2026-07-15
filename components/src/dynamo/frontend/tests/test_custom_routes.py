# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import sys
from types import ModuleType

import pytest

from dynamo.frontend.frontend_args import FrontendArgGroup, FrontendConfig
from dynamo.frontend.routes import Response, Router, load_custom_routes

pytestmark = [pytest.mark.pre_merge, pytest.mark.gpu_0, pytest.mark.unit]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    FrontendArgGroup().add_arguments(parser)
    return parser


def test_custom_routes_cli_is_repeatable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DYN_CUSTOM_ROUTES", raising=False)
    args, _ = _parser().parse_known_args(
        ["--custom-routes", "first.py", "--custom-routes", "package.routes"]
    )
    assert args.custom_routes == ["first.py", "package.routes"]


def test_custom_routes_env_is_whitespace_separated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DYN_CUSTOM_ROUTES", "first.py package.routes")
    args, _ = _parser().parse_known_args([])
    assert args.custom_routes == ["first.py", "package.routes"]


@pytest.mark.parametrize("mode", ["--interactive", "--kserve-grpc-server"])
def test_custom_routes_reject_non_http_modes(
    monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    monkeypatch.delenv("DYN_CUSTOM_ROUTES", raising=False)
    args, _ = _parser().parse_known_args([mode, "--custom-routes", "routes.py"])
    config = FrontendConfig.from_cli_args(args)

    with pytest.raises(ValueError, match="requires the HTTP frontend"):
        config.validate()


@pytest.mark.asyncio
async def test_loads_sync_and_async_hooks_in_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    order: list[str] = []
    first = ModuleType("test_custom_routes_first")
    second = ModuleType("test_custom_routes_second")

    def register_first(router: Router, runtime: object) -> None:
        order.append("first")

        @router.get("/first")
        async def first_handler(request):
            return Response.text("first")

    async def register_second(router: Router, runtime: object) -> None:
        order.append("second")

        @router.route("/second/{item}", methods=["GET", "POST"])
        async def second_handler(request):
            return Response.json({"item": request.path_params["item"]})

    first.register_routes = register_first  # type: ignore[attr-defined]
    second.register_routes = register_second  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, first.__name__, first)
    monkeypatch.setitem(sys.modules, second.__name__, second)

    routes = await load_custom_routes([first.__name__, second.__name__], object())  # type: ignore[arg-type]

    assert order == ["first", "second"]
    assert [(route.method, route.path) for route in routes] == [
        ("GET", "/first"),
        ("GET", "/second/{item}"),
        ("POST", "/second/{item}"),
    ]


@pytest.mark.asyncio
async def test_loads_python_file_and_identifies_source(tmp_path) -> None:
    plugin = tmp_path / "routes.py"
    plugin.write_text(
        "from dynamo.frontend.routes import Response\n"
        "\n"
        "def register_routes(router, runtime):\n"
        "    @router.delete('/from-file')\n"
        "    async def handler(request):\n"
        "        return Response.text('ok')\n"
    )

    routes = await load_custom_routes([str(plugin)], object())  # type: ignore[arg-type]

    assert len(routes) == 1
    assert routes[0].source == str(plugin)


@pytest.mark.asyncio
async def test_loader_errors_name_the_source(monkeypatch: pytest.MonkeyPatch) -> None:
    module = ModuleType("test_custom_routes_bad")
    monkeypatch.setitem(sys.modules, module.__name__, module)

    with pytest.raises(RuntimeError, match="test_custom_routes_bad.*register_routes"):
        await load_custom_routes([module.__name__], object())  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_route_validation_and_freezing() -> None:
    router = Router()

    def sync_handler(request):
        return Response.text("bad")

    with pytest.raises(TypeError, match="async def"):
        router.get("/sync")(sync_handler)  # type: ignore[arg-type]

    @router.get("/items/{item}")
    async def handler(request):
        return Response.text("ok")

    with pytest.raises(ValueError, match="duplicate custom route"):

        @router.get("/items/{other}")
        async def duplicate(request):
            return Response.text("duplicate")

    with pytest.raises(ValueError, match="catch-all"):
        router.get("/items/{*rest}")

    router.freeze()
    with pytest.raises(RuntimeError, match="frozen"):
        router.post("/late")


def test_response_validation_and_helpers() -> None:
    response = Response.json({"ok": True}, status=201, headers={"x-test": ["a", "b"]})
    assert response.status == 201
    assert response.headers["content-type"] == ["application/json"]
    assert response.headers["x-test"] == ["a", "b"]
    assert response.body == b'{"ok":true}'

    assert Response.text("hello").body == b"hello"
    assert Response(b"\x00\xff").body == b"\x00\xff"
    with pytest.raises(ValueError, match="between 100 and 599"):
        Response.text("bad", status=99)
    with pytest.raises(ValueError, match="invalid HTTP response header"):
        Response.text("bad", headers={"bad header": "value"})
