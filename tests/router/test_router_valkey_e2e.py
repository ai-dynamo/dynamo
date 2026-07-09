# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cross-component direct-worker, frontend, replication, and failover tests."""

from __future__ import annotations

import asyncio
import contextlib
import socket

import aiohttp
import pytest
from dynamo.llm import ValkeyWorkerRegistration

from tests.router.common import valkey_index_key
from tests.router.e2e_harness import allocate_frontend_ports, build_test_payload
from tests.router.helper import (
    generate_random_suffix,
    get_runtime,
    wait_for_frontend_ready,
)
from tests.router.mocker_process import MockerProcess
from tests.router.router_process import (
    FrontendRouterProcess,
    ValkeyModuleProcess,
    ValkeyPrimaryProxy,
    ValkeySentinelProcess,
)
from tests.router.valkey_e2e_scenarios import _test_valkey_three_frontend_routing
from tests.router.valkey_e2e_helpers import (
    _valkey_barrier_and_wait,
    _valkey_integer_array_command,
    _valkey_integer_command,
    _valkey_module_test_paths,
    _valkey_resp_command_on_socket,
    _wait_for_module_convergence,
    _wait_for_replication_roles,
    _wait_for_sentinel_primary,
    router_valkey_config,
)
from tests.utils.constants import ROUTER_MODEL_NAME
from tests.utils.port_utils import allocate_ports, deallocate_ports

BLOCK_SIZE = 16
SPEEDUP_RATIO = 10.0
TEST_PAYLOAD = build_test_payload(ROUTER_MODEL_NAME)

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
    pytest.mark.integration,
    pytest.mark.router,
    pytest.mark.model(ROUTER_MODEL_NAME),
]


@pytest.mark.timeout(300)
def test_valkey_registration_only_dp_ranks(
    request,
    runtime_services_dynamic_ports,
    tmp_path,
    monkeypatch,
):
    """Decode-style ranks register durably without publishing fake KV events."""

    server, module = _valkey_module_test_paths()
    cli = server.with_name("valkey-cli")
    if not cli.is_file():
        pytest.skip("Valkey registration E2E requires valkey-cli beside valkey-server")
    valkey_ports = allocate_ports(2, 15000)
    request.addfinalizer(lambda: deallocate_ports(valkey_ports))
    # Exercise the production stable-primary shape: the client receives one
    # endpoint while still requiring the independently configured replica ACK.
    valkey_url = f"valkey://127.0.0.1:{valkey_ports[0]}"
    namespace = f"valkey-registration-{generate_random_suffix()}"
    index_scope = "decode-only"
    index_key = valkey_index_key(namespace, "decode", index_scope, BLOCK_SIZE).encode()

    config = router_valkey_config([valkey_url], index_scope)
    monkeypatch.setenv("DYN_ROUTER_VALKEY_CONFIG", config)

    with (
        ValkeyModuleProcess(
            request,
            port=valkey_ports[0],
            data_dir=tmp_path / "valkey-registration-primary",
            server=str(server),
            module=str(module),
            require_replica=True,
        ),
        ValkeyModuleProcess(
            request,
            port=valkey_ports[1],
            data_dir=tmp_path / "valkey-registration-replica",
            server=str(server),
            module=str(module),
            replica_of=valkey_ports[0],
        ),
    ):
        asyncio.run(_wait_for_replication_roles(cli, valkey_ports[0], valkey_ports[1]))

        async def run() -> None:
            runtime = get_runtime(store_backend="etcd", request_plane="nats")
            try:
                endpoint = runtime.endpoint(f"{namespace}.decode.generate")
                registration = await ValkeyWorkerRegistration.create_from_env(
                    endpoint=endpoint,
                    kv_block_size=BLOCK_SIZE,
                    dp_ranks=[3, 4],
                )
                assert registration is not None

                expected = (0, 2, 1)
                for port in valkey_ports:
                    stats = await asyncio.to_thread(
                        _valkey_integer_array_command,
                        f"valkey://127.0.0.1:{port}",
                        b"DYNKV.STATS",
                        index_key,
                    )
                    if stats != expected:
                        raise AssertionError(f"port {port}: {stats} != {expected}")
                    lifecycle = await asyncio.to_thread(
                        _valkey_integer_array_command,
                        f"valkey://127.0.0.1:{port}",
                        b"DYNKV.LIFECYCLE_STATS",
                        index_key,
                    )
                    assert lifecycle == (1, 0, 0)

                await registration.shutdown()
                for port in valkey_ports:
                    stats = await asyncio.to_thread(
                        _valkey_integer_array_command,
                        f"valkey://127.0.0.1:{port}",
                        b"DYNKV.STATS",
                        index_key,
                    )
                    if stats != (0, 2, 2):
                        raise AssertionError(f"port {port}: {stats} != (0, 2, 2)")
                    lifecycle = await asyncio.to_thread(
                        _valkey_integer_array_command,
                        f"valkey://127.0.0.1:{port}",
                        b"DYNKV.LIFECYCLE_STATS",
                        index_key,
                    )
                    # Whole-worker unregister retires the epoch directly. The
                    # per-rank tombstone count is reserved for rank-set changes.
                    assert lifecycle == (0, 0, 1)
            finally:
                runtime.shutdown()

        asyncio.run(run())


@pytest.mark.timeout(180)
def test_mocker_valkey_planned_primary_failover(
    request,
    predownload_tokenizers,
    tmp_path,
    monkeypatch,
):
    """Requests and replicated module state survive a coordinated promotion."""

    server, module = _valkey_module_test_paths()
    cli = server.with_name("valkey-cli")
    if not cli.is_file():
        pytest.skip("Valkey failover E2E requires valkey-cli beside valkey-server")

    ports = allocate_ports(8, 15000)
    request.addfinalizer(lambda: deallocate_ports(ports))
    primary_port, replica_port = ports[:2]
    tokenizer_primary_port, tokenizer_replica_port = ports[2:4]
    sentinel_ports = ports[4:7]
    proxy_port = ports[7]
    admission_frontend_port, match_frontend_port, cache_frontend_port = (
        allocate_frontend_ports(request, 3)
    )
    master_name = f"dynkv-{generate_random_suffix()}"
    tokenizer_master_name = f"tokenizer-{generate_random_suffix()}"
    valkey_index_scope = "planned-failover"
    stable_url = f"valkey://127.0.0.1:{proxy_port}"
    discovery_root = tmp_path / "discovery"
    discovery_root.mkdir()
    monkeypatch.setenv("DYN_FILE_KV", str(discovery_root))

    monkeypatch.setenv("DYN_LOG", "info,dynamo_llm::kv_router::indexer::valkey=debug")

    with contextlib.ExitStack() as stack:
        stack.enter_context(
            ValkeyModuleProcess(
                request,
                port=primary_port,
                data_dir=tmp_path / "failover-primary",
                server=str(server),
                module=str(module),
                require_replica=True,
            )
        )
        stack.enter_context(
            ValkeyModuleProcess(
                request,
                port=replica_port,
                data_dir=tmp_path / "failover-replica",
                server=str(server),
                module=str(module),
                replica_of=primary_port,
            )
        )
        stack.enter_context(
            ValkeyModuleProcess(
                request,
                port=tokenizer_primary_port,
                data_dir=tmp_path / "tokenizer-failover-primary",
                server=str(server),
                module=str(module),
                require_replica=True,
            )
        )
        stack.enter_context(
            ValkeyModuleProcess(
                request,
                port=tokenizer_replica_port,
                data_dir=tmp_path / "tokenizer-failover-replica",
                server=str(server),
                module=str(module),
                replica_of=tokenizer_primary_port,
            )
        )
        sentinels = [
            stack.enter_context(
                ValkeySentinelProcess(
                    request,
                    port=sentinel_port,
                    data_dir=tmp_path / f"failover-sentinel-{sentinel_port}",
                    server=str(server),
                    cli=str(cli),
                    master_port=primary_port,
                    master_name=master_name,
                    additional_masters=(
                        (tokenizer_master_name, tokenizer_primary_port),
                    ),
                    down_after_ms=500,
                    failover_timeout_ms=5_000,
                )
            )
            for sentinel_port in sentinel_ports
        ]
        sentinels[0].wait_for_quorum(sentinel_count=3, timeout=10)
        sentinels[0].wait_for_quorum(
            sentinel_count=3,
            master_name=tokenizer_master_name,
            timeout=10,
        )
        sentinel_urls = [
            f"valkey://127.0.0.1:{sentinel_port}" for sentinel_port in sentinel_ports
        ]
        match_config = router_valkey_config(
            [stable_url],
            valkey_index_scope,
            worker_lease_ms=120_000,
            sentinel_urls=sentinel_urls,
            sentinel_master_name=master_name,
            tokenizer_sentinel_master_name=tokenizer_master_name,
            allow_degraded_writes=True,
        )
        admission_config = router_valkey_config(
            [stable_url],
            valkey_index_scope,
            authoritative_admission=True,
            worker_lease_ms=120_000,
            sentinel_urls=sentinel_urls,
            sentinel_master_name=master_name,
            tokenizer_sentinel_master_name=tokenizer_master_name,
            allow_degraded_writes=True,
        )
        monkeypatch.setenv("DYN_ROUTER_VALKEY_CONFIG", match_config)

        proxy = stack.enter_context(
            ValkeyPrimaryProxy(
                listen_port=proxy_port,
                target_port=primary_port,
            )
        )
        pinned_old_connection = socket.create_connection(
            (proxy.listen_host, proxy.listen_port), timeout=5
        )
        stack.callback(pinned_old_connection.close)
        assert _valkey_resp_command_on_socket(pinned_old_connection, b"PING") == "PONG"

        mockers = stack.enter_context(
            MockerProcess(
                request,
                mocker_args={
                    "block_size": BLOCK_SIZE,
                    "kv_bytes_per_token": 128,
                    "speedup_ratio": 1_000.0,
                    "num_gpu_blocks": 4096,
                    "max_num_seqs": 128,
                },
                num_mockers=4,
                store_backend="file",
                request_plane="tcp",
                extra_env={
                    "DYN_EVENT_PLANE": "zmq",
                    "DYN_ROUTER_VALKEY_CONFIG": match_config,
                },
            )
        )
        stack.enter_context(
            FrontendRouterProcess(
                request,
                block_size=BLOCK_SIZE,
                frontend_port=admission_frontend_port,
                namespace=mockers.namespace,
                store_backend="file",
                request_plane="tcp",
                min_initial_workers=mockers.num_workers,
                router_valkey_config=admission_config,
                event_plane="zmq",
            )
        )
        stack.enter_context(
            FrontendRouterProcess(
                request,
                block_size=BLOCK_SIZE,
                frontend_port=match_frontend_port,
                namespace=mockers.namespace,
                store_backend="file",
                request_plane="tcp",
                min_initial_workers=mockers.num_workers,
                router_valkey_config=match_config,
                event_plane="zmq",
            )
        )
        stack.enter_context(
            FrontendRouterProcess(
                request,
                block_size=BLOCK_SIZE,
                frontend_port=cache_frontend_port,
                namespace=mockers.namespace,
                store_backend="file",
                request_plane="tcp",
                min_initial_workers=mockers.num_workers,
                router_valkey_config=match_config,
                event_plane="zmq",
            )
        )
        index_key = valkey_index_key(
            mockers.namespace,
            mockers.component_name,
            valkey_index_scope,
            BLOCK_SIZE,
        ).encode()
        direct_urls = (
            f"valkey://127.0.0.1:{primary_port}",
            f"valkey://127.0.0.1:{replica_port}",
        )
        tokenizer_direct_urls = (
            f"valkey://127.0.0.1:{tokenizer_primary_port}",
            f"valkey://127.0.0.1:{tokenizer_replica_port}",
        )
        empty_match = b"\x01\x00\x00\x00\x00"

        async def run() -> None:
            frontend_urls = {
                admission_frontend_port: (
                    f"http://localhost:{admission_frontend_port}"
                ),
                match_frontend_port: f"http://localhost:{match_frontend_port}",
                cache_frontend_port: f"http://localhost:{cache_frontend_port}",
            }
            await asyncio.gather(
                *(
                    wait_for_frontend_ready(
                        frontend_url=frontend_url,
                        expected_num_workers=mockers.num_workers,
                        timeout=90,
                        engine_workers=mockers,
                        store_backend="file",
                        request_plane="tcp",
                    )
                    for frontend_url in frontend_urls.values()
                )
            )

            async def complete_request(
                session: aiohttp.ClientSession,
                label: str,
                frontend_port: int = admission_frontend_port,
                messages: list[dict[str, str]] | None = None,
            ) -> str:
                payload = {
                    **TEST_PAYLOAD,
                    "messages": messages
                    or [
                        {
                            "role": "user",
                            "content": (
                                TEST_PAYLOAD["messages"][0]["content"] + f" {label}"
                            ),
                        }
                    ],
                    "stream": False,
                    "max_tokens": 4,
                }
                chat_url = f"{frontend_urls[frontend_port]}/v1/chat/completions"
                async with session.post(chat_url, json=payload) as response:
                    body = await response.text()
                    assert response.status == 200, (
                        f"request {label} returned {response.status}: {body}"
                    )
                    return body

            async def metric_value(frontend_port: int, metric_name: str) -> float:
                async with session.get(
                    f"{frontend_urls[frontend_port]}/metrics"
                ) as response:
                    text = await response.text()
                    assert response.status == 200, text
                for line in text.splitlines():
                    if line.startswith(f"{metric_name} "):
                        return float(line.rsplit(" ", maxsplit=1)[1])
                return 0.0

            async def wait_for_tokenizer_replication(previous_keys: int) -> None:
                observed = (-1, -1)
                for _ in range(100):
                    observed = tuple(
                        await asyncio.gather(
                            *(
                                asyncio.to_thread(
                                    _valkey_integer_command,
                                    url,
                                    b"DBSIZE",
                                )
                                for url in tokenizer_direct_urls
                            )
                        )
                    )
                    if observed[0] > previous_keys and observed[0] == observed[1]:
                        return
                    await asyncio.sleep(0.1)
                raise AssertionError(
                    "tokenizer cache did not replicate a new prefix: "
                    f"previous={previous_keys}, observed={observed}"
                )

            async def require_cross_frontend_tokenizer_hit(phase: str) -> None:
                previous_keys = await asyncio.to_thread(
                    _valkey_integer_command,
                    tokenizer_direct_urls[0],
                    b"DBSIZE",
                )
                shared = (
                    f"{TEST_PAYLOAD['messages'][0]['content']} tokenizer-{phase} "
                    + "large-growing-prefix " * 256
                )
                prefix_messages = [
                    {"role": "user", "content": shared},
                    {"role": "assistant", "content": f"ack-{phase}"},
                    {"role": "user", "content": f"continue-{phase}"},
                ]
                await complete_request(
                    session,
                    f"tokenizer-seed-{phase}",
                    admission_frontend_port,
                    prefix_messages,
                )
                await wait_for_tokenizer_replication(previous_keys)

                metric = "dynamo_frontend_tokenizer_cache_l2_hits_total"
                before = await metric_value(cache_frontend_port, metric)
                await complete_request(
                    session,
                    f"tokenizer-grow-{phase}",
                    cache_frontend_port,
                    [
                        *prefix_messages,
                        {
                            "role": "assistant",
                            "content": f"more-{phase}",
                        },
                        {"role": "user", "content": f"next-{phase}"},
                    ],
                )
                after = await metric_value(cache_frontend_port, metric)
                assert after > before, (
                    f"expected a shared tokenizer L2 hit during {phase}; "
                    f"before={before}, after={after}"
                )

            await _wait_for_replication_roles(cli, primary_port, replica_port)
            await _wait_for_replication_roles(
                cli, tokenizer_primary_port, tokenizer_replica_port
            )
            assert (
                await asyncio.to_thread(
                    _valkey_barrier_and_wait, direct_urls[0], index_key
                )
                == 1
            )

            timeout = aiohttp.ClientTimeout(total=60)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                await require_cross_frontend_tokenizer_hit("before-failover")
                await asyncio.gather(
                    *(complete_request(session, f"seed-{index}") for index in range(6))
                )
                await _wait_for_module_convergence(
                    direct_urls, index_key, mockers.num_workers
                )

                failover = await asyncio.to_thread(
                    sentinels[0].run_sentinel_command,
                    "SENTINEL",
                    "FAILOVER",
                    master_name,
                    timeout=3,
                )
                assert failover.returncode == 0, failover.stderr
                assert failover.stdout.strip() == "OK", failover.stdout
                await _wait_for_sentinel_primary(sentinels, replica_port)
                await _wait_for_replication_roles(cli, replica_port, primary_port)

                # Valkey may close pre-promotion client sockets as it changes
                # role. Either signal forces a reconnect; a fresh tunnel while
                # the proxy still targets the old node must then hit the
                # module's explicit replica-role fence.
                with pytest.raises(
                    RuntimeError, match="DYNKV_NOT_PRIMARY|unexpected EOF"
                ):
                    await asyncio.to_thread(
                        _valkey_resp_command_on_socket,
                        pinned_old_connection,
                        b"DYNKV.MATCH_PRIMARY",
                        index_key,
                        empty_match,
                    )
                demoted_connection = socket.create_connection(
                    (proxy.listen_host, proxy.listen_port), timeout=5
                )
                stack.callback(demoted_connection.close)
                with pytest.raises(RuntimeError, match="DYNKV_NOT_PRIMARY"):
                    await asyncio.to_thread(
                        _valkey_resp_command_on_socket,
                        demoted_connection,
                        b"DYNKV.MATCH_PRIMARY",
                        index_key,
                        empty_match,
                    )

                admission_transition = asyncio.create_task(
                    complete_request(session, "during-primary-switch")
                )
                match_transition = asyncio.create_task(
                    complete_request(
                        session,
                        "during-primary-switch-match",
                        match_frontend_port,
                    )
                )
                assert proxy.switch_target(replica_port) == (
                    "127.0.0.1",
                    primary_port,
                )
                # Switching the service endpoint does not retarget established
                # tunnels; MATCH_PRIMARY keeps the demoted backend unusable.
                with pytest.raises(RuntimeError, match="DYNKV_NOT_PRIMARY"):
                    await asyncio.to_thread(
                        _valkey_resp_command_on_socket,
                        demoted_connection,
                        b"DYNKV.MATCH_PRIMARY",
                        index_key,
                        empty_match,
                    )
                with socket.create_connection(
                    (proxy.listen_host, proxy.listen_port), timeout=5
                ) as promoted_connection:
                    promoted_match = await asyncio.to_thread(
                        _valkey_resp_command_on_socket,
                        promoted_connection,
                        b"DYNKV.MATCH_PRIMARY",
                        index_key,
                        empty_match,
                    )
                assert promoted_match == empty_match

                await asyncio.wait_for(
                    asyncio.gather(admission_transition, match_transition),
                    timeout=30,
                )
                await asyncio.gather(
                    *(
                        complete_request(
                            session,
                            f"after-failover-{index}",
                            (
                                admission_frontend_port
                                if index % 2 == 0
                                else match_frontend_port
                            ),
                        )
                        for index in range(6)
                    )
                )

                tokenizer_failover = await asyncio.to_thread(
                    sentinels[0].run_sentinel_command,
                    "SENTINEL",
                    "FAILOVER",
                    tokenizer_master_name,
                    timeout=3,
                )
                assert tokenizer_failover.returncode == 0, tokenizer_failover.stderr
                assert tokenizer_failover.stdout.strip() == "OK", (
                    tokenizer_failover.stdout
                )
                await _wait_for_sentinel_primary(
                    sentinels,
                    tokenizer_replica_port,
                    master_name=tokenizer_master_name,
                )
                await _wait_for_replication_roles(
                    cli,
                    tokenizer_replica_port,
                    tokenizer_primary_port,
                )
                await require_cross_frontend_tokenizer_hit("after-failover")

            assert (
                await asyncio.to_thread(_valkey_barrier_and_wait, stable_url, index_key)
                == 1
            )
            await _wait_for_module_convergence(
                direct_urls, index_key, mockers.num_workers
            )

        asyncio.run(run())


@pytest.mark.timeout(300)
def test_mocker_valkey_three_frontends(
    request,
    predownload_tokenizers,
    tmp_path,
    monkeypatch,
):
    """Three frontends use one replicated Valkey index with four workers."""
    server, module = _valkey_module_test_paths()
    valkey_ports = allocate_ports(2, 15000)
    request.addfinalizer(lambda: deallocate_ports(valkey_ports))
    valkey_url_list = [f"valkey://127.0.0.1:{port}" for port in valkey_ports]
    valkey_urls = ",".join(valkey_url_list)
    valkey_index_scope = "mocker-shared-router"
    config = router_valkey_config(valkey_url_list, valkey_index_scope)
    discovery_root = tmp_path / "discovery"
    discovery_root.mkdir()
    monkeypatch.setenv("DYN_FILE_KV", str(discovery_root))

    with (
        ValkeyModuleProcess(
            request,
            port=valkey_ports[0],
            data_dir=tmp_path / "valkey-0",
            server=str(server),
            module=str(module),
            require_replica=True,
        ),
        ValkeyModuleProcess(
            request,
            port=valkey_ports[1],
            data_dir=tmp_path / "valkey-1",
            server=str(server),
            module=str(module),
            replica_of=valkey_ports[0],
        ),
        MockerProcess(
            request,
            mocker_args={
                "block_size": BLOCK_SIZE,
                "kv_bytes_per_token": 128,
                "speedup_ratio": SPEEDUP_RATIO,
            },
            num_mockers=4,
            store_backend="file",
            request_plane="tcp",
            extra_env={
                "DYN_EVENT_PLANE": "zmq",
                "DYN_ROUTER_VALKEY_CONFIG": config,
            },
        ) as mockers,
    ):
        _test_valkey_three_frontend_routing(
            engine_workers=mockers,
            block_size=BLOCK_SIZE,
            request=request,
            frontend_ports=allocate_frontend_ports(request, 3),
            test_payload=TEST_PAYLOAD,
            valkey_urls=valkey_urls,
            valkey_index_scope=valkey_index_scope,
            router_valkey_config=config,
            store_backend="file",
            request_plane="tcp",
        )


@pytest.mark.timeout(300)
def test_mocker_valkey_authoritative_admission_across_three_frontends(
    request,
    runtime_services_dynamic_ports,
    predownload_tokenizers,
    tmp_path,
):
    """Module admission caps a shared 3-FE fleet and releases cancelled streams."""

    server, module = _valkey_module_test_paths()
    valkey_ports = allocate_ports(2, 15000)
    request.addfinalizer(lambda: deallocate_ports(valkey_ports))
    valkey_url_list = [f"valkey://127.0.0.1:{port}" for port in valkey_ports]
    valkey_urls = ",".join(valkey_url_list)
    valkey_index_scope = "mocker-authoritative-router"
    primary_url = valkey_urls.split(",", maxsplit=1)[0]
    config = router_valkey_config(
        valkey_url_list,
        valkey_index_scope,
        authoritative_admission=True,
    )

    with (
        ValkeyModuleProcess(
            request,
            port=valkey_ports[0],
            data_dir=tmp_path / "valkey-authority-0",
            server=str(server),
            module=str(module),
            require_replica=True,
        ),
        ValkeyModuleProcess(
            request,
            port=valkey_ports[1],
            data_dir=tmp_path / "valkey-authority-1",
            server=str(server),
            module=str(module),
            replica_of=valkey_ports[0],
        ),
        MockerProcess(
            request,
            mocker_args={
                "block_size": BLOCK_SIZE,
                "kv_bytes_per_token": 128,
                # Long output streams remain active long enough to observe the
                # module's global capacity fence before client cancellation.
                "speedup_ratio": 0.01,
                "num_gpu_blocks": 4096,
                "max_num_seqs": 1,
                "max_num_batched_tokens": 16384,
            },
            num_mockers=4,
            request_plane="nats",
            extra_env={"DYN_ROUTER_VALKEY_CONFIG": config},
        ) as mockers,
    ):
        frontend_ports = allocate_frontend_ports(request, 3)
        index_key = valkey_index_key(
            mockers.namespace,
            mockers.component_name,
            valkey_index_scope,
            BLOCK_SIZE,
        )

        async def wait_for_module_count(command: bytes, expected: int) -> None:
            observed = -1
            for _ in range(100):
                if command == b"DYNKV.STATS":
                    stats = await asyncio.to_thread(
                        _valkey_integer_array_command,
                        primary_url,
                        command,
                        index_key.encode(),
                    )
                    observed = stats[1] if len(stats) == 3 else -1
                else:
                    observed = await asyncio.to_thread(
                        _valkey_integer_command,
                        primary_url,
                        command,
                        index_key.encode(),
                    )
                if observed == expected:
                    return
                await asyncio.sleep(0.1)
            raise AssertionError(
                f"{command.decode()} never reached {expected}; last value={observed}"
            )

        with contextlib.ExitStack() as stack:
            for port in frontend_ports:
                stack.enter_context(
                    FrontendRouterProcess(
                        request,
                        block_size=BLOCK_SIZE,
                        frontend_port=port,
                        namespace=mockers.namespace,
                        store_backend="etcd",
                        request_plane="nats",
                        min_initial_workers=mockers.num_workers,
                        router_valkey_config=config,
                        event_plane="nats",
                    )
                )

            async def run() -> None:
                await asyncio.gather(
                    *(
                        wait_for_frontend_ready(
                            frontend_url=f"http://localhost:{port}",
                            expected_num_workers=mockers.num_workers,
                            timeout=120,
                            engine_workers=mockers,
                            store_backend="etcd",
                            request_plane="nats",
                        )
                        for port in frontend_ports
                    )
                )
                # REGISTER_WORKER is replicated before any worker starts
                # publishing; do not convert that asynchronous startup into a
                # spurious no-capacity assertion.
                await wait_for_module_count(b"DYNKV.STATS", 4)

                request_payload = {
                    **TEST_PAYLOAD,
                    "stream": True,
                    "max_tokens": 4096,
                    "min_tokens": 4096,
                    "ignore_eos": True,
                }
                timeout = aiohttp.ClientTimeout(total=90)
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    responses = await asyncio.gather(
                        *(
                            session.post(
                                f"http://localhost:{frontend_ports[index % len(frontend_ports)]}/v1/chat/completions",
                                json=request_payload,
                            )
                            for index in range(4)
                        )
                    )
                    try:
                        assert [response.status for response in responses] == [200] * 4
                        await wait_for_module_count(b"DYNKV.ADMISSION_STATS", 4)

                        async with session.post(
                            f"http://localhost:{frontend_ports[1]}/v1/chat/completions",
                            json=request_payload,
                        ) as rejected:
                            body = await rejected.text()
                            assert rejected.status == 529, body
                    finally:
                        for response in responses:
                            response.close()

                    await wait_for_module_count(b"DYNKV.ADMISSION_STATS", 0)

            asyncio.run(run())
