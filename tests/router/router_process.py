# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import select
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

from tests.utils.managed_process import ManagedProcess


class SlotTrackerProcess(ManagedProcess):
    """Manages a standalone dynamo.slot_tracker process."""

    def __init__(self, request, port: int):
        super().__init__(
            command=[
                sys.executable,
                "-m",
                "dynamo.slot_tracker",
                "--port",
                str(port),
            ],
            timeout=10,
            display_output=False,
            health_check_ports=[port],
            health_check_urls=[f"http://localhost:{port}/health"],
            log_dir=request.node.name,
            terminate_all_matching_process_names=False,
            display_name="dynamo-slot-tracker",
        )
        self.port = port


class FrontendRouterProcess(ManagedProcess):
    """Manages a dynamo.frontend process with configurable --router-mode.

    Supports all router modes (round-robin, random, kv, direct) and all
    KV-specific options (block size, thresholds, durable events, disagg).
    block_size is only sent to the CLI when router_mode is "kv".
    """

    def __init__(
        self,
        request,
        block_size: int,
        frontend_port: int,
        namespace: str,
        store_backend: str = "etcd",
        blocks_threshold: float | str | None = None,
        tokens_threshold: int | str | None = None,
        tokens_threshold_frac: float | str | None = None,
        router_queue_threshold: float | str | None = None,
        request_plane: str = "nats",
        durable_kv_events: bool = False,
        router_mode: str = "kv",
        min_initial_workers: int | None = None,
        router_aic_config: dict[str, str | int] | None = None,
        serve_indexer: bool = False,
        use_remote_indexer: bool = False,
        router_valkey_config: str | None = None,
        valkey_urls: str | None = None,
        valkey_index_scope: str | None = None,
        valkey_connection_pool_size: int | None = None,
        valkey_required_replica_acks: int | None = None,
        valkey_sentinel_urls: str | None = None,
        valkey_sentinel_master_name: str | None = None,
        valkey_sentinel_quorum: int | None = None,
        valkey_allow_degraded_writes: bool = False,
        valkey_worker_events: bool = False,
        valkey_authoritative_admission: bool = False,
        valkey_admission_lease_ms: int | None = None,
        router_replica_sync: bool = False,
        event_plane: str | None = None,
        session_affinity_ttl_secs: int | None = None,
    ):
        legacy_valkey_options = (
            valkey_urls,
            valkey_index_scope,
            valkey_connection_pool_size,
            valkey_required_replica_acks,
            valkey_sentinel_urls,
            valkey_sentinel_master_name,
            valkey_sentinel_quorum,
            valkey_admission_lease_ms,
        )
        legacy_valkey_bools = (
            valkey_allow_degraded_writes,
            valkey_worker_events,
            valkey_authoritative_admission,
        )
        if router_valkey_config is not None and (
            any(value is not None for value in legacy_valkey_options)
            or any(legacy_valkey_bools)
        ):
            raise ValueError(
                "router_valkey_config is mutually exclusive with legacy Valkey options"
            )
        json_authoritative_admission = False
        if router_valkey_config is not None:
            try:
                parsed_valkey_config = json.loads(router_valkey_config)
            except json.JSONDecodeError as error:
                raise ValueError("router_valkey_config must be valid JSON") from error
            if not isinstance(parsed_valkey_config, dict):
                raise ValueError("router_valkey_config must be a JSON object")
            json_authoritative_admission = (
                parsed_valkey_config.get("authoritative_admission") is True
            )
        effective_authoritative_admission = (
            valkey_authoritative_admission or json_authoritative_admission
        )
        if effective_authoritative_admission and router_replica_sync:
            raise ValueError(
                "valkey_authoritative_admission is mutually exclusive with router_replica_sync"
            )
        if valkey_authoritative_admission and (
            not valkey_worker_events
            or valkey_urls is None
            or valkey_index_scope is None
        ):
            raise ValueError(
                "valkey_authoritative_admission requires direct Valkey worker events, URLs, and index scope"
            )
        command = [
            sys.executable,
            "-m",
            "dynamo.frontend",
            "--router-mode",
            router_mode,
            "--http-port",
            str(frontend_port),
            "--discovery-backend",
            store_backend,
            "--namespace",
            namespace,
        ]

        if router_mode == "kv":
            command.extend(["--kv-cache-block-size", str(block_size)])

        if blocks_threshold is not None:
            command.extend(["--active-decode-blocks-threshold", str(blocks_threshold)])

        if tokens_threshold is not None:
            command.extend(["--active-prefill-tokens-threshold", str(tokens_threshold)])

        if tokens_threshold_frac is not None:
            command.extend(
                ["--active-prefill-tokens-threshold-frac", str(tokens_threshold_frac)]
            )

        if router_queue_threshold is not None:
            command.extend(["--router-queue-threshold", str(router_queue_threshold)])

        if durable_kv_events:
            command.append("--router-durable-kv-events")

        if serve_indexer:
            command.append("--serve-indexer")

        if use_remote_indexer:
            command.append("--use-remote-indexer")

        if router_valkey_config is not None:
            command.extend(["--router-valkey-config", router_valkey_config])

        if session_affinity_ttl_secs is not None:
            command.extend(
                ["--router-session-affinity-ttl-secs", str(session_affinity_ttl_secs)]
            )

        if valkey_urls is not None:
            command.extend(["--router-valkey-urls", valkey_urls])

        if valkey_index_scope is not None:
            command.extend(["--router-valkey-index-scope", valkey_index_scope])

        if valkey_connection_pool_size is not None:
            command.extend(
                [
                    "--router-valkey-connection-pool-size",
                    str(valkey_connection_pool_size),
                ]
            )

        if valkey_required_replica_acks is not None:
            command.extend(
                [
                    "--router-valkey-required-replica-acks",
                    str(valkey_required_replica_acks),
                ]
            )

        if valkey_sentinel_urls is not None:
            command.extend(["--router-valkey-sentinel-urls", valkey_sentinel_urls])

        if valkey_sentinel_master_name is not None:
            command.extend(
                ["--router-valkey-sentinel-master-name", valkey_sentinel_master_name]
            )

        if valkey_sentinel_quorum is not None:
            command.extend(
                ["--router-valkey-sentinel-quorum", str(valkey_sentinel_quorum)]
            )

        if valkey_allow_degraded_writes:
            command.append("--router-valkey-allow-degraded-writes")

        if valkey_worker_events:
            command.append("--router-valkey-worker-events")

        if (
            valkey_urls is not None or router_valkey_config is not None
        ) and not effective_authoritative_admission:
            # The Valkey module currently stores the device-tier index only.
            # Keep all Valkey test topologies explicit about disabling the
            # unsupported host/disk cache scoring tiers.
            command.extend(
                [
                    "--router-host-cache-hit-weight",
                    "0",
                    "--router-disk-cache-hit-weight",
                    "0",
                ]
            )

        if valkey_authoritative_admission:
            command.extend(
                [
                    # Phase one module admission is immediate-only.  Keep
                    # these explicit even when CLI defaults change.
                    "--router-queue-threshold",
                    "None",
                    "--no-router-track-prefill-tokens",
                    "--router-host-cache-hit-weight",
                    "0",
                    "--router-disk-cache-hit-weight",
                    "0",
                    "--shared-cache-multiplier",
                    "0",
                ]
            )
            command.append("--router-valkey-authoritative-admission")

        if valkey_admission_lease_ms is not None:
            command.extend(
                [
                    "--router-valkey-admission-lease-ms",
                    str(valkey_admission_lease_ms),
                ]
            )

        if router_replica_sync:
            command.append("--router-replica-sync")

        if router_aic_config is not None:
            command.extend(
                [
                    "--router-track-prefill-tokens",
                    "--router-prefill-load-model",
                    "aic",
                    "--aic-backend",
                    str(router_aic_config["aic_backend"]),
                    "--aic-system",
                    str(router_aic_config["aic_system"]),
                    "--aic-model-path",
                    str(router_aic_config["aic_model_path"]),
                    "--aic-tp-size",
                    str(router_aic_config.get("aic_tp_size", 1)),
                ]
            )
            if "aic_backend_version" in router_aic_config:
                command.extend(
                    [
                        "--aic-backend-version",
                        str(router_aic_config["aic_backend_version"]),
                    ]
                )

        env = os.environ.copy()
        env["DYN_REQUEST_PLANE"] = request_plane
        if event_plane is not None:
            env["DYN_EVENT_PLANE"] = event_plane
        if event_plane == "zmq" and request_plane != "nats":
            env.pop("NATS_SERVER", None)
        if min_initial_workers is not None:
            env["DYN_ROUTER_MIN_INITIAL_WORKERS"] = str(min_initial_workers)

        super().__init__(
            command=command,
            env=env,
            timeout=60,
            display_output=True,
            health_check_ports=[frontend_port],
            health_check_urls=[
                (f"http://localhost:{frontend_port}/v1/models", self._check_ready)
            ],
            log_dir=request.node.name,
            terminate_all_matching_process_names=False,
            display_name=f"dynamo-frontend-{router_mode}-{frontend_port}",
        )
        self.port = frontend_port
        self.router_mode = router_mode

    def _check_ready(self, response):
        """Check if KV, random, round-robin, or direct router is ready"""
        return response.status_code == 200

    def __exit__(self, exc_type, exc_val, exc_tb):
        super().__exit__(exc_type, exc_val, exc_tb)


class ValkeyModuleProcess(ManagedProcess):
    """Manages one module-loaded Valkey instance for router integration tests."""

    def __init__(
        self,
        request,
        *,
        port: int,
        data_dir: Path,
        server: str,
        module: str,
        replica_of: int | None = None,
        require_replica: bool = False,
    ):
        data_dir.mkdir(parents=True, exist_ok=True)
        command = [
            server,
            "--port",
            str(port),
            "--bind",
            "127.0.0.1",
            "--dir",
            str(data_dir),
            "--appendonly",
            "yes",
            "--appendfsync",
            "everysec",
            "--repl-diskless-sync-delay",
            "0",
            "--loadmodule",
            module,
        ]
        if require_replica:
            command.extend(
                ["--min-replicas-to-write", "1", "--min-replicas-max-lag", "5"]
            )
        if replica_of is not None:
            command.extend(
                [
                    "--replicaof",
                    "127.0.0.1",
                    str(replica_of),
                    "--replica-read-only",
                    "yes",
                    "--replica-serve-stale-data",
                    "no",
                ]
            )
        super().__init__(
            command=command,
            timeout=30,
            display_output=True,
            health_check_ports=[port],
            log_dir=request.node.name,
            terminate_all_matching_process_names=False,
            display_name=f"dynkv-valkey-{port}",
        )
        self.port = port
        self.valkey_data_dir = data_dir


class ValkeySentinelProcess(ManagedProcess):
    """Manages one Valkey Sentinel that can monitor multiple master sets.

    The same three Sentinel processes can monitor the router and tokenizer
    Valkey data planes independently. Each master set keeps its own election;
    the default quorum of two requires a witness majority for either failover.
    """

    def __init__(
        self,
        request,
        *,
        port: int,
        data_dir: Path,
        server: str,
        master_port: int,
        master_name: str = "dynkv-primary",
        master_host: str = "127.0.0.1",
        quorum: int = 2,
        down_after_ms: int = 1_000,
        failover_timeout_ms: int = 10_000,
        parallel_syncs: int = 1,
        additional_masters: tuple[tuple[str, int], ...] = (),
        cli: str | None = None,
    ):
        if quorum < 1:
            raise ValueError("Sentinel quorum must be at least one")
        if down_after_ms < 1 or failover_timeout_ms < 1:
            raise ValueError("Sentinel timeouts must be positive")
        if parallel_syncs < 1:
            raise ValueError("Sentinel parallel_syncs must be at least one")
        monitored_masters = ((master_name, master_port), *additional_masters)
        master_names = [name for name, _ in monitored_masters]
        if len(set(master_names)) != len(master_names):
            raise ValueError("Sentinel master names must be unique")
        for monitored_name, monitored_port in monitored_masters:
            self._validate_config_token("master_name", monitored_name)
            if not 1 <= monitored_port <= 65535:
                raise ValueError("Sentinel master ports must be in 1..=65535")
        self._validate_config_token("master_host", master_host)

        data_dir.mkdir(parents=True, exist_ok=True)
        config_path = data_dir / f"sentinel-{port}.conf"
        config_lines = [
            f"port {port}",
            "bind 127.0.0.1",
            "protected-mode no",
            "daemonize no",
            f"dir {data_dir.resolve()}",
        ]
        for monitored_name, monitored_port in monitored_masters:
            config_lines.extend(
                [
                    f"sentinel monitor {monitored_name} {master_host} {monitored_port} {quorum}",
                    f"sentinel down-after-milliseconds {monitored_name} {down_after_ms}",
                    f"sentinel failover-timeout {monitored_name} {failover_timeout_ms}",
                    f"sentinel parallel-syncs {monitored_name} {parallel_syncs}",
                ]
            )
        config_path.write_text(
            "\n".join([*config_lines, ""]),
            encoding="utf-8",
        )

        inferred_cli = Path(server).with_name("valkey-cli")
        self.cli = cli or (str(inferred_cli) if inferred_cli.is_file() else None)
        self.port = port
        self.master_name = master_name
        self.master_names = tuple(master_names)
        self.sentinel_config_path = config_path
        self.sentinel_data_dir = data_dir

        health_check_funcs = [self._sentinel_ready] if self.cli is not None else []
        super().__init__(
            command=[server, str(config_path), "--sentinel"],
            timeout=30,
            display_output=True,
            health_check_ports=[port],
            health_check_funcs=health_check_funcs,
            log_dir=request.node.name,
            terminate_all_matching_process_names=False,
            display_name=f"valkey-sentinel-{port}",
        )

    @staticmethod
    def _validate_config_token(name: str, value: str) -> None:
        if not value or any(character.isspace() for character in value):
            raise ValueError(f"Sentinel {name} must be one non-empty config token")

    def _sentinel_ready(self, remaining_timeout: float = 2.0) -> bool:
        try:
            timeout = max(0.1, min(remaining_timeout, 2.0))
            return all(
                self.get_master_addr(master_name=name, timeout=timeout) is not None
                for name in self.master_names
            )
        except (OSError, subprocess.SubprocessError, ValueError):
            return False

    def run_sentinel_command(
        self, *parts: str, timeout: float = 2.0
    ) -> subprocess.CompletedProcess[str]:
        """Run a command against this Sentinel with the matching Valkey CLI."""

        if self.cli is None:
            raise RuntimeError(
                "valkey-cli was not supplied and could not be inferred from valkey-server"
            )
        return subprocess.run(
            [self.cli, "--raw", "-h", "127.0.0.1", "-p", str(self.port), *parts],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

    def get_master_addr(
        self, *, master_name: str | None = None, timeout: float = 2.0
    ) -> tuple[str, int] | None:
        """Return Sentinel's current primary address, or ``None`` if unresolved."""

        master_name = master_name or self.master_name
        if master_name not in self.master_names:
            raise ValueError(f"Sentinel does not monitor master {master_name!r}")
        result = self.run_sentinel_command(
            "SENTINEL",
            "GET-MASTER-ADDR-BY-NAME",
            master_name,
            timeout=timeout,
        )
        if result.returncode != 0:
            return None
        fields = result.stdout.splitlines()
        if len(fields) != 2:
            return None
        return fields[0], int(fields[1])

    def wait_for_quorum(
        self,
        *,
        sentinel_count: int = 3,
        master_name: str | None = None,
        timeout: float = 15.0,
    ) -> None:
        """Wait until this Sentinel sees a usable failover majority."""

        if sentinel_count < 1 or timeout <= 0:
            raise ValueError("sentinel_count and timeout must be positive")
        master_name = master_name or self.master_name
        if master_name not in self.master_names:
            raise ValueError(f"Sentinel does not monitor master {master_name!r}")
        deadline = time.monotonic() + timeout
        last_output = ""
        while time.monotonic() < deadline:
            result = self.run_sentinel_command(
                "SENTINEL",
                "CKQUORUM",
                master_name,
                timeout=min(2.0, max(0.1, deadline - time.monotonic())),
            )
            last_output = (result.stdout + result.stderr).strip()
            if result.returncode == 0 and last_output.startswith("OK"):
                known = self.run_sentinel_command(
                    "SENTINEL",
                    "SENTINELS",
                    master_name,
                    timeout=min(2.0, max(0.1, deadline - time.monotonic())),
                )
                # Each peer is returned as a map whose first field is ``name``.
                if known.returncode == 0 and known.stdout.splitlines().count(
                    "name"
                ) >= (sentinel_count - 1):
                    return
            time.sleep(0.1)
        raise TimeoutError(
            f"Sentinel on port {self.port} did not reach a {sentinel_count}-node "
            f"quorum within {timeout:.1f}s: {last_output}"
        )


class ValkeyPrimaryProxy:
    """Stable TCP endpoint whose upstream can change between connections.

    A connection keeps the upstream selected when it was accepted. Calling
    :meth:`switch_target` only changes the destination used by later
    connections, which models a stable primary service without disrupting
    commands already in flight on an established socket.
    """

    def __init__(
        self,
        *,
        listen_port: int,
        target_port: int,
        listen_host: str = "127.0.0.1",
        target_host: str = "127.0.0.1",
        connect_timeout: float = 2.0,
    ):
        if not 0 <= listen_port <= 65535:
            raise ValueError("listen_port must be between 0 and 65535")
        if not 1 <= target_port <= 65535:
            raise ValueError("target_port must be between 1 and 65535")
        if connect_timeout <= 0:
            raise ValueError("connect_timeout must be positive")

        self.listen_host = listen_host
        self.listen_port = listen_port
        self.connect_timeout = connect_timeout
        self._target = (target_host, target_port)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._listener: socket.socket | None = None
        self._accept_thread: threading.Thread | None = None
        self._tunnel_threads: set[threading.Thread] = set()
        self._tunnels: set[tuple[socket.socket, socket.socket]] = set()

    @property
    def target(self) -> tuple[str, int]:
        with self._lock:
            return self._target

    @property
    def active_tunnels(self) -> int:
        with self._lock:
            return len(self._tunnels)

    @property
    def running(self) -> bool:
        thread = self._accept_thread
        return thread is not None and thread.is_alive()

    def start(self) -> "ValkeyPrimaryProxy":
        if self._listener is not None or self._accept_thread is not None:
            raise RuntimeError("ValkeyPrimaryProxy is already running")

        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind((self.listen_host, self.listen_port))
            listener.listen()
            listener.settimeout(0.2)
        except Exception:
            listener.close()
            raise

        self.listen_port = listener.getsockname()[1]
        self._stop.clear()
        self._listener = listener
        self._accept_thread = threading.Thread(
            target=self._accept_loop,
            name=f"valkey-primary-proxy-{self.listen_port}",
            daemon=True,
        )
        self._accept_thread.start()
        return self

    def switch_target(
        self, target_port: int, target_host: str = "127.0.0.1"
    ) -> tuple[str, int]:
        """Switch new connections and return the previously selected target."""

        if not 1 <= target_port <= 65535:
            raise ValueError("target_port must be between 1 and 65535")
        with self._lock:
            previous = self._target
            self._target = (target_host, target_port)
            return previous

    def _accept_loop(self) -> None:
        listener = self._listener
        assert listener is not None
        while not self._stop.is_set():
            try:
                client, _ = listener.accept()
            except socket.timeout:
                continue
            except OSError:
                if self._stop.is_set():
                    break
                continue

            with self._lock:
                target = self._target
            try:
                upstream = socket.create_connection(
                    target, timeout=self.connect_timeout
                )
                client.settimeout(None)
                upstream.settimeout(None)
                client.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                upstream.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            except OSError:
                client.close()
                continue
            if self._stop.is_set():
                self._close_socket(client)
                self._close_socket(upstream)
                break

            thread = threading.Thread(
                target=self._relay,
                args=(client, upstream),
                name=f"valkey-primary-tunnel-{self.listen_port}",
                daemon=True,
            )
            with self._lock:
                self._tunnels.add((client, upstream))
                self._tunnel_threads.add(thread)
            thread.start()

    def _relay(self, client: socket.socket, upstream: socket.socket) -> None:
        peers = {client: upstream, upstream: client}
        try:
            while not self._stop.is_set():
                try:
                    readable, _, _ = select.select(tuple(peers), (), (), 0.2)
                except (OSError, ValueError):
                    break
                for source in readable:
                    try:
                        payload = source.recv(64 * 1024)
                        if not payload:
                            return
                        peers[source].sendall(payload)
                    except OSError:
                        return
        finally:
            self._close_socket(client)
            self._close_socket(upstream)
            current = threading.current_thread()
            with self._lock:
                self._tunnels.discard((client, upstream))
                self._tunnel_threads.discard(current)

    @staticmethod
    def _close_socket(sock: socket.socket) -> None:
        try:
            sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        sock.close()

    def close(self, *, timeout: float = 5.0) -> None:
        """Stop accepting and close every active tunnel before returning."""

        self._stop.set()
        listener, self._listener = self._listener, None
        if listener is not None:
            self._close_socket(listener)

        deadline = time.monotonic() + timeout
        accept_thread = self._accept_thread
        if (
            accept_thread is not None
            and accept_thread is not threading.current_thread()
        ):
            accept_thread.join(max(0.0, deadline - time.monotonic()))

        with self._lock:
            tunnels = tuple(self._tunnels)
            tunnel_threads = tuple(self._tunnel_threads)
        for client, upstream in tunnels:
            self._close_socket(client)
            self._close_socket(upstream)

        for thread in tunnel_threads:
            if thread is not None and thread is not threading.current_thread():
                thread.join(max(0.0, deadline - time.monotonic()))

        threads = [accept_thread, *tunnel_threads]
        alive = [
            thread.name
            for thread in threads
            if thread is not None and thread.is_alive()
        ]
        if alive:
            raise RuntimeError(f"TCP proxy threads did not stop: {', '.join(alive)}")
        self._accept_thread = None

    def __enter__(self) -> "ValkeyPrimaryProxy":
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


# Backward-compatible alias so existing callers that import KVRouterProcess
# continue to work without changes.
KVRouterProcess = FrontendRouterProcess
