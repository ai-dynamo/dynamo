# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import contextlib
import io
import itertools
import os
import socket
import threading
import time
import unittest
from socketserver import ThreadingMixIn
from unittest.mock import MagicMock, patch

import power_agent
from power_agent import PowerAgent


def _bare_agent() -> PowerAgent:
    """A PowerAgent without `__init__`'s NVML / K8s dependencies."""
    agent = object.__new__(PowerAgent)
    agent._actuator = MagicMock()
    agent.node_name = "node-under-test"
    agent.safe_default_watts = 500
    agent.metrics = MagicMock()
    agent._cycle_state = (None, False)
    return agent


@contextlib.contextmanager
def _readyz_server(agent: PowerAgent):
    """Run the agent's real readiness server on an ephemeral port.

    Patches `READYZ_PORT` rather than hardcoding 8081 so the test is
    parallel-safe and cannot collide with anything already bound on the host.
    `_start_readyz_server` reads the module constant on each call.
    """
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    with patch.object(power_agent, "READYZ_PORT", port):
        server = agent._start_readyz_server()
    try:
        yield port
    finally:
        # Stop the daemon thread's serve_forever and release the listener, so
        # the suite does not accumulate one bound port per test.
        server.shutdown()
        server.server_close()


def _http_get_readyz(
    port: int, path: str = "/readyz", host: str = "127.0.0.1"
) -> tuple[int, str]:
    """Issue a real HTTP GET against the readiness server."""
    with socket.create_connection((host, port), timeout=5) as sock:
        sock.sendall(
            f"GET {path} HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n".encode()
        )
        chunks = []
        while True:
            chunk = sock.recv(4096)
            if not chunk:
                break
            chunks.append(chunk)
    raw = b"".join(chunks).decode("utf-8")
    head, _, body = raw.partition("\r\n\r\n")
    status = int(head.split("\r\n", 1)[0].split(" ")[1])
    return status, body


def _get_readyz(agent: PowerAgent, path: str = "/readyz") -> tuple[str, str]:
    """Drive the WSGI app directly. Returns (status, body).

    Calling the app rather than issuing a socket request keeps these tests off
    the network for everything except the bind-collision case, which is
    inherently about the socket.
    """
    captured: dict[str, str] = {}

    def start_response(status, headers):
        captured["status"] = status
        captured["headers"] = headers

    chunks = agent._readyz_app({"PATH_INFO": path}, start_response)
    return captured["status"], b"".join(chunks).decode("utf-8")


class TestReadinessEndpoint(unittest.TestCase):
    def test_not_ready_before_the_first_successful_cycle(self):
        agent = _bare_agent()

        status, body = _get_readyz(agent)

        self.assertTrue(status.startswith("503"))
        self.assertEqual(
            body, "not-ready last_good_cycle_age_s=none last_cycle_ok=false\n"
        )

    def test_ready_immediately_after_a_successful_cycle(self):
        agent = _bare_agent()
        agent._cycle_state = (time.monotonic(), True)

        status, body = _get_readyz(agent)

        self.assertTrue(status.startswith("200"))
        self.assertTrue(body.startswith("ok last_good_cycle_age_s="))

    def test_not_ready_once_the_staleness_bound_elapses(self):
        """The bound is what catches a HUNG cycle, which nothing else can: the
        probe's failureThreshold cannot see it, because nothing is failing —
        the timestamp is simply not advancing."""
        agent = _bare_agent()
        agent._cycle_state = (time.monotonic() - power_agent.STALE_AFTER_S - 1, True)

        status, body = _get_readyz(agent)

        self.assertTrue(status.startswith("503"))
        self.assertTrue(body.startswith("not-ready last_good_cycle_age_s="))

    def test_a_healthy_cycle_may_last_twice_the_reconcile_interval(self):
        """`run()` sleeps AFTER each cycle, so the timestamp refreshes every
        RECONCILE_INTERVAL_S plus the cycle's own duration. Asserted so the
        budget cannot be silently changed by a future interval or bound edit.
        """
        agent = _bare_agent()
        interval = power_agent.RECONCILE_INTERVAL_S

        # A cycle taking just under 2x the interval still leaves the agent
        # Ready at the moment the next one would publish...
        agent._cycle_state = (time.monotonic() - (interval + 2 * interval - 1), True)
        self.assertTrue(_get_readyz(agent)[0].startswith("200"))

        # ...and one exceeding it exposes a 503.
        agent._cycle_state = (time.monotonic() - (interval + 2 * interval + 1), True)
        self.assertTrue(_get_readyz(agent)[0].startswith("503"))

    def test_non_readyz_paths_are_404(self):
        agent = _bare_agent()
        agent._cycle_state = (time.monotonic(), True)

        status, body = _get_readyz(agent, path="/metrics")

        self.assertTrue(status.startswith("404"))
        self.assertEqual(body, "")

    def test_wall_clock_steps_do_not_extend_or_expire_readiness(self):
        """Readiness is measured on CLOCK_MONOTONIC. An NTP step or a
        container-start clock jump must not make a stale agent look fresh (or
        the reverse)."""
        agent = _bare_agent()
        agent._cycle_state = (time.monotonic(), True)

        with patch.object(power_agent.time, "time", return_value=time.time() + 86400):
            self.assertTrue(_get_readyz(agent)[0].startswith("200"))


class TestReadinessTimestamp(unittest.TestCase):
    """`run()` assigns `_cycle_state` exactly once per cycle."""

    def setUp(self):
        power_agent._shutdown.clear()

    def tearDown(self):
        power_agent._shutdown.clear()

    def _run_one_cycle(self, reconcile):
        agent = _bare_agent()
        agent.reconcile_once = reconcile
        agent._start_readyz_server = MagicMock()
        with patch.object(power_agent.signal, "signal"), patch.object(
            power_agent, "_shutdown_cleanup"
        ):
            agent.run()
        return agent

    def test_successful_cycle_publishes_a_fresh_timestamp(self):
        def reconcile():
            power_agent._shutdown.set()
            return True

        agent = self._run_one_cycle(reconcile)

        self.assertIsNotNone(agent._cycle_state[0])
        self.assertTrue(_get_readyz(agent)[0].startswith("200"))

    def test_failed_cycle_is_not_ready(self):
        def reconcile():
            power_agent._shutdown.set()
            return False

        agent = self._run_one_cycle(reconcile)

        self.assertFalse(agent._cycle_state[1])
        self.assertTrue(_get_readyz(agent)[0].startswith("503"))

    def test_failed_cycle_after_success_keeps_the_last_good_age(self):
        """A failure after a success must not look like startup: the body
        reports the latest-cycle failure and the age of the real last good
        cycle. Only startup reports `age=none`."""
        agent = _bare_agent()
        agent._start_readyz_server = MagicMock()
        results = iter([True, False])

        def reconcile():
            ok = next(results)
            if not ok:
                power_agent._shutdown.set()
            return ok

        agent.reconcile_once = reconcile
        with patch.object(power_agent.signal, "signal"), patch.object(
            power_agent, "_shutdown_cleanup"
        ), patch.object(power_agent._shutdown, "wait"), patch.object(
            power_agent.time, "monotonic", side_effect=[100.0, 105.0]
        ):
            agent.run()

        with patch.object(power_agent.time, "monotonic", return_value=105.0):
            status, body = _get_readyz(agent)

        self.assertTrue(status.startswith("503"))
        self.assertEqual(
            body, "not-ready last_good_cycle_age_s=5.0 last_cycle_ok=false\n"
        )

    def test_raising_cycle_is_not_ready_and_does_not_exit(self):
        """An unexpected exception must take the pod NotReady on the NEXT poll
        rather than after the staleness bound — and must NOT terminate the
        process: a NotReady agent that keeps reconciling can recover, while a
        restart drops in-memory GPU ownership."""
        agent = _bare_agent()
        agent._cycle_state = (time.monotonic(), True)  # a previously good cycle
        agent._start_readyz_server = MagicMock()

        def reconcile():
            power_agent._shutdown.set()
            raise RuntimeError("NVML exploded")

        agent.reconcile_once = reconcile
        with patch.object(power_agent.signal, "signal"), patch.object(
            power_agent, "_shutdown_cleanup"
        ) as cleanup:
            agent.run()  # must not raise

        self.assertFalse(agent._cycle_state[1])
        self.assertTrue(_get_readyz(agent)[0].startswith("503"))
        cleanup.assert_called_once()


class TestReadinessServer(unittest.TestCase):
    def setUp(self):
        power_agent._shutdown.clear()

    def tearDown(self):
        power_agent._shutdown.clear()

    def test_serves_with_prometheus_disabled(self):
        """Readiness must not be optional, so it is deliberately NOT served
        from the Prometheus server, which `--prometheus-port=0` disables.

        This one goes over a real socket: driving `_readyz_app` directly would
        not exercise the separate server at all, and the separateness IS the
        property under test.
        """
        agent = _bare_agent()
        # prometheus_port=0 → PowerAgentMetrics starts no HTTP server at all.
        agent.metrics = power_agent.PowerAgentMetrics(0)
        agent._cycle_state = (time.monotonic(), True)

        with _readyz_server(agent) as port:
            status, body = _http_get_readyz(port)

        self.assertEqual(status, 200)
        self.assertTrue(body.startswith("ok last_good_cycle_age_s="))

    def test_serves_over_ipv6_when_pod_ip_is_ipv6(self):
        """kubelet probes the primary pod IP, so an IPv6 pod needs an IPv6 listener."""
        try:
            with socket.socket(socket.AF_INET6) as probe:
                probe.bind(("::1", 0))
        except OSError:
            self.skipTest("IPv6 loopback unavailable")
        agent = _bare_agent()
        agent._cycle_state = (time.monotonic(), True)

        with patch.dict(os.environ, {"POD_IP": "::1"}):
            with _readyz_server(agent) as port:
                status, _ = _http_get_readyz(port, host="::1")

        self.assertEqual(status, 200)

    def test_bind_collision_is_fatal_at_startup(self):
        """A bind failure must propagate out of `run()` uncaught so the process
        exits and the pod enters CrashLoopBackOff. Starting the server inside
        `run()`'s `try` would instead send a never-enforcing agent through the
        full `_shutdown_cleanup` restore sweep on the way out.

        The pre-bind uses the WILDCARD address, not loopback: a loopback-only
        collision with a wildcard bind is kernel-dependent. The PORT is
        ephemeral rather than the literal 8081, so the test cannot collide with
        a real listener on the host and is safe to run in parallel;
        `_start_readyz_server` reads `READYZ_PORT` on each call.
        """
        agent = _bare_agent()
        agent.reconcile_once = MagicMock(return_value=True)

        occupier = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            occupier.bind(("", 0))
            occupier.listen(1)
            port = occupier.getsockname()[1]

            with patch.object(power_agent, "READYZ_PORT", port), patch.object(
                power_agent.signal, "signal"
            ), patch.object(power_agent, "_shutdown_cleanup") as cleanup:
                with self.assertRaises(OSError):
                    agent.run()

            cleanup.assert_not_called()
        finally:
            occupier.close()

    def test_probe_traffic_emits_no_per_request_log_line(self):
        """`WSGIRequestHandler` inherits `BaseHTTPRequestHandler.log_message`,
        which writes one stderr line per request — ~8,600 lines per node per day
        at the probe's `periodSeconds: 10`."""
        handler = object.__new__(power_agent._QuietWSGIHandler)
        captured = io.StringIO()

        with patch("sys.stderr", captured):
            for _ in range(5):
                handler.log_message('"%s" %s %s', "GET /readyz HTTP/1.1", "200", "-")

        self.assertEqual(captured.getvalue(), "")

    def test_handler_bounds_both_idle_and_total_connection_time(self):
        """Two distinct bounds, and the second is not redundant.

        `timeout` is per-operation — `settimeout` resets on every byte received
        — so it bounds only how long a connection may go SILENT. A client
        trickling its request line one byte at a time never trips it.
        `max_connection_seconds` is the absolute bound that does.
        """
        self.assertIsNotNone(power_agent._QuietWSGIHandler.timeout)
        self.assertGreater(power_agent._QuietWSGIHandler.timeout, 0)
        self.assertGreater(power_agent._QuietWSGIHandler.max_connection_seconds, 0)

    def test_server_handles_connections_concurrently(self):
        """The structural fix. A single-threaded server services one connection
        at a time, so no per-connection timeout can stop a slow client from
        delaying the probes queued behind it."""
        self.assertTrue(
            issubclass(power_agent._ThreadingWSGIServer, ThreadingMixIn),
            "readiness server must not serialize connections",
        )
        self.assertTrue(power_agent._ThreadingWSGIServer.daemon_threads)

    def test_slow_drip_client_does_not_block_a_later_probe(self):
        """Why `timeout` alone is not a fix.

        This client stays under the idle timeout INDEFINITELY by sending one
        byte at a time — a request line followed by an endless run of padding
        headers — so `settimeout` never fires and the request never completes.
        On a single-threaded server it holds the accept loop for as long as it
        likes and every probe behind it times out.

        Verified to discriminate: against a plain `WSGIServer` the probe does
        not complete at all while this client keeps dripping.
        """
        agent = _bare_agent()
        agent._cycle_state = (time.monotonic(), True)

        stop = threading.Event()

        def drip(sock):
            # Never completes the request, and never pauses long enough to look
            # idle: each byte lands well inside the 0.3s idle timeout.
            preamble = b"GET /readyz HTTP/1.1\r\nHost: localhost\r\n"
            stream = itertools.chain(preamble, itertools.cycle(b"X-Pad: 0\r\n"))
            for byte in stream:
                if stop.is_set():
                    return
                try:
                    sock.sendall(bytes([byte]))
                except OSError:
                    return
                stop.wait(0.05)

        with patch.object(power_agent._QuietWSGIHandler, "timeout", 0.3):
            with _readyz_server(agent) as port:
                dripper = socket.create_connection(("127.0.0.1", port), timeout=5)
                thread = threading.Thread(target=drip, args=(dripper,), daemon=True)
                thread.start()
                try:
                    # Give the dripper time to be accepted and to keep the
                    # connection alive well past the idle timeout, which is what
                    # made the previous fix insufficient.
                    time.sleep(0.6)
                    start = time.monotonic()
                    status, _ = _http_get_readyz(port)
                    elapsed = time.monotonic() - start
                finally:
                    stop.set()
                    dripper.close()
                    thread.join(timeout=5)

        self.assertEqual(status, 200)
        # Served on its own thread, not queued behind the dripper.
        self.assertLess(elapsed, 2.0)

    def test_watchdog_closes_a_connection_that_outlives_its_bound(self):
        """The absolute bound is enforced, so slow-drip connections (and their
        threads) cannot accumulate without limit."""
        agent = _bare_agent()
        agent._cycle_state = (time.monotonic(), True)

        with patch.object(
            power_agent._QuietWSGIHandler, "max_connection_seconds", 0.3
        ), patch.object(power_agent._QuietWSGIHandler, "timeout", 5):
            with _readyz_server(agent) as port:
                sock = socket.create_connection(("127.0.0.1", port), timeout=5)
                try:
                    sock.sendall(b"GET /readyz HTTP/1.1\r\n")  # deliberately partial
                    # The watchdog closes the socket, so the read returns EOF
                    # rather than hanging until the idle timeout.
                    sock.settimeout(5)
                    self.assertEqual(sock.recv(4096), b"")
                finally:
                    sock.close()

    def test_server_thread_is_a_daemon(self):
        """So it does not hold the process open during shutdown. Readiness keeps
        serving through the termination grace period, which is what
        `reconcile_once`'s shutdown-fold rule is for."""
        agent = _bare_agent()
        started: list[threading.Thread] = []
        real_thread = threading.Thread

        def capture(*args, **kwargs):
            t = real_thread(*args, **kwargs)
            started.append(t)
            return t

        with patch.object(power_agent, "make_server") as make_server:
            with patch.object(power_agent.threading, "Thread", side_effect=capture):
                agent._start_readyz_server()

        make_server.assert_called_once()
        args, kwargs = make_server.call_args
        self.assertEqual(args[1], power_agent.READYZ_PORT)
        self.assertIs(kwargs["handler_class"], power_agent._QuietWSGIHandler)
        self.assertIs(kwargs["server_class"], power_agent._ThreadingWSGIServer)
        self.assertEqual(len(started), 1)
        self.assertTrue(started[0].daemon)


if __name__ == "__main__":
    unittest.main()
