<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Registry Go client

Package `pkg/client` is the Go client of the Multi-Cluster Services
registry (`mcs-registry`), import path
`github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/pkg/client`. Its
consumer in this module is `mcs-controller`.

A `Client` holds one gRPC connection for unary calls and watches. On top of
it the package provides:

- typed sub-clients: `Clusters()`, `ServiceExports()`, `ServiceImports()`
  and `ServerInfo()`;
- endpoint failover with DNS re-resolution and a return to the primary URL;
- `WatchLoop`, a reconnecting watch with snapshot and resume handling;
- `NextGeneration` and `Generation` for writer generations, and helpers that
  read registry error details;
- `dynamo_mcs_client_*` Prometheus metrics.

`pkg/fakeregistry` is an in-memory registry for tests of code that uses
this package (see Testing).

## Creating a client

```go
c, err := client.New(ctx, client.Config{
	URLs:       []string{"https://mcs.example.internal", "https://mcs.us-west-2.example.internal"},
	CAFile:     "/etc/mcs/ca.crt",
	UserAgent:  "my-controller",
	Registerer: prometheus.DefaultRegisterer,
})
if err != nil {
	return err
}
defer c.Close()
```

`New` validates the configuration and returns at once; the client connects
in the background and calls wait for the connection within their deadline.

| Field | Default | Meaning |
| :---- | :---- | :---- |
| `URLs` | required | Ordered registry URLs. One scheme for all; ports default to 443 (https) and 80 (http). |
| `CAFile`, `TLS` | system roots | CA bundle, or a base `tls.Config`. `CAFile` and `TLS.RootCAs` are exclusive. |
| `AllowInsecureHTTP` | false | Permit `http://` URLs. |
| `ConnectTimeout` | 5 s | One connection attempt or DNS lookup. |
| `RPCTimeout` | 10 s | Deadline of unary calls whose context has none; negative disables. |
| `PrimaryRetryInterval` | 5 m | Probe and return to the primary URL; negative disables. |
| `Backoff` | 200 ms to 30 s, x2, jitter 0.5 | Unary retries, connection attempts, watch reconnects. |
| `MaxAttempts` | 4 | Attempts of a unary call failing with UNAVAILABLE; 1 disables retries; at most 5. |
| `UserAgent` | `global-discovery-mcs-client-go` | gRPC user agent prefix. |
| `Registerer` | nil | Receives the metrics; nil registers nothing. |
| `Resolver` | `net.DefaultResolver` | Resolves URL host names on every reconnect. |
| `Dialer` | TCP | Replaces the TCP dialer (tests, bufconn). |
| `DialOptions` | none | Extra gRPC dial options, for example an OpenTelemetry stats handler. |
| `Logger` | `slog.Default()` | Failover and watch logs. |

`Close` stops the background work and closes the connection. Stop watch
loops first; they keep retrying against a closed client otherwise.

## Failover

Every consumer takes an ordered list of URLs, normally a global DNS name
first and the per-region names after it.

- A per-client gRPC resolver (a unique scheme per `Client`, registered with
  `grpc.WithResolvers`) resolves every URL host name in order with
  `Config.Resolver` when the connection starts and whenever gRPC asks for a
  re-resolution, which it does after every lost or failed connection. DNS
  failover therefore applies on the next reconnect. Re-resolutions are
  spaced by at least one second.
- Each address carries its URL's host as the TLS server name and HTTP/2
  authority, so one connection can verify certificates of several names.
- The service config selects `pick_first` without shuffling: the client
  connects to the first address that answers, in URL order, and stays there
  while it is healthy. Address families are not interleaved.
- After each new connection the client calls `GetServerInfo` and appends the
  advertised `endpoints[].grpc_url` values that are valid, of the client's
  scheme and not configured already, after the configured URLs.
- A URL whose endpoint answers a unary call (after the gRPC retries) or a
  watch with `UNAVAILABLE` is left out of the address list for at least
  `ConnectTimeout`, growing with `Backoff` on consecutive failures, so
  `pick_first` drops the connection at once and moves on to the next URL
  instead of reconnecting to a replica that is up but not serving. When
  every URL is in its backoff, all are kept. A successful call to the URL
  after its backoff resets the failure count.
- Every `PrimaryRetryInterval` a client connected to another URL re-resolves
  and probes the primary URL over a separate connection with the gRPC health
  service (`grpc.health.v1.Health/Check` for each registry service, which the
  registry sets to `NOT_SERVING` while its view is stale, its store is
  unreachable or it is shutting down). When every service reports `SERVING`
  the client moves the connection to the primary and restarts its
  `WatchLoop` streams there. Calls in flight finish on the old connection. A
  primary within its `UNAVAILABLE` backoff is not probed.
- All replicas serve the same data, so any replica will do. A watch that
  reconnects to another replica receives a fresh snapshot.

`OnReconnect(func(url string))` registers a hook that runs after every
connection the client establishes after its first one. Hooks run one at a
time on a background goroutine; reconnects during a hook coalesce into one
call with the latest URL. `CurrentURL()` and `URLs()` report the state.

## Retries and deadlines

Unary calls without a deadline get `RPCTimeout`. The service config adds a
gRPC retry policy on UNAVAILABLE for every unary registry method with
`MaxAttempts` and `Backoff`; it applies to `Conn()` users too. Other codes
are not retried. Watches are not retried by gRPC; `WatchLoop` reconnects
them.

## Writers: generations and errors

Writes carry a generation that must exceed the stored one. Use
`NextGeneration(prev)`, `max(prev + 1, unix milliseconds)`, or a
`Generation` per record:

```go
gen := &client.Generation{}
req := &mcsv1.ReplaceEndpointsRequest{ /* ... */ Generation: gen.Next()}
_, err := c.ServiceExports().ReplaceEndpoints(ctx, req)
if cur, ok := client.StaleGeneration(err); ok {
	gen.Observe(cur)
	req.Generation = gen.Next()
	_, err = c.ServiceExports().ReplaceEndpoints(ctx, req)
}
```

`Reason(err)` returns the registry error reason; `IsLeaseEpochMismatch` and
`IsClusterNotRegistered` cover the lease cases: register again and resend
state.

## Watches

`WatchLoop` keeps one watch open:

```go
loop := client.NewWatchLoop(c, client.WatchLoopConfig[*mcsv1.ServiceImport]{
	Name:  client.WatchServiceImports,
	Watch: c.ServiceImports().Watcher(&mcsv1.WatchServiceImportsRequest{Clusterset: cs, IncludeEndpoints: true}),
	Handler: client.WatchHandler[*mcsv1.ServiceImport]{
		SnapshotStart: func() { seen = map[string]bool{} },
		Upsert:        apply,
		Delete:        remove,
		Synced:        func(string) { garbageCollect(seen) },
	},
	OnResync:    func() { log.Info("full resync") },
	IdleTimeout: client.DefaultWatchIdleTimeout,
})
go loop.Run(ctx)
```

- `SNAPSHOT_START` calls `SnapshotStart` (discard local state) and clears the
  resume revision, so a stream that breaks during a snapshot is never
  resumed from the revision before it.
- `UPSERT` and `DELETE` call `Upsert` and `Delete` and advance the revision.
- `SNAPSHOT_END` calls `Synced` with the revision; the state is complete.
- `BOOKMARK` calls `Bookmark` and advances the revision.
- A stream that ends or fails is reopened after `Backoff` (reset after a
  stream that delivered events) with the last revision as
  `resume_revision`. The registry replays the missed deltas or sends a new
  snapshot. A new snapshot after the loop had synced calls `OnResync` first.
- `IdleTimeout` reopens a stream that delivered nothing, not even a bookmark,
  for that long. The registry sends bookmarks every 5 s;
  `DefaultWatchIdleTimeout` is 30 s.
- Each reopen increments
  `dynamo_mcs_client_watch_reconnects_total{watch=Name}`.

## Metrics

Registered on `Config.Registerer` when it is set. Clients sharing a
registerer share the collectors.

| Metric | Type | Labels | Meaning |
| :---- | :---- | :---- | :---- |
| `dynamo_mcs_client_requests_total` | counter | `rpc`, `code` | Calls by RPC and gRPC code; a watch counts once, when it ends. Pre-initialised for every RPC and code. |
| `dynamo_mcs_client_request_duration_seconds` | histogram | `rpc` | Unary call duration, retries included. |
| `dynamo_mcs_client_endpoint_failovers_total` | counter | | New connections to a different URL than the previous one, the return to the primary included. |
| `dynamo_mcs_client_connected` | gauge | `url` | 1 for the URL in use, 0 for the other configured and advertised URLs. |
| `dynamo_mcs_client_watch_reconnects_total` | counter | `watch` | Watch reopens. Pre-initialised for `clusters` and `service_imports`. |

## Testing

`pkg/fakeregistry` implements the four services in memory with leases,
epochs, generations, merged imports and watches with revisions, bookmarks
and resume. It serves over bufconn or `127.0.0.1:0`:

```go
srv := fakeregistry.New()
run, _ := srv.ServeLocal()
defer run.Stop()
srv.PutExport("prod", "a", "llm", "frontend",
	fakeregistry.ServiceSpec(fakeregistry.TCPPort("http", 8000)),
	fakeregistry.WithPodNames(fakeregistry.ReadyEndpoints("10.0.0.1"), "frontend-0"))
c, _ := client.New(ctx, client.Config{URLs: []string{run.URL()}, AllowInsecureHTTP: true})
```

`FailNext(method, code)` injects failures, `CloseWatches` ends open watches,
`SetServing(false)` makes the gRPC health service report `NOT_SERVING` (the
failback probe then keeps the client away), `Running.Stop` drops a listener
and `ServeAddr` restarts it on the same address. See the package
documentation for the differences from `mcs-registry`.

## Limitations

- One scheme for all URLs: a client is either TLS or plaintext. Advertised
  fallbacks of the other scheme are ignored.
- The custom dialer bypasses gRPC's HTTPS proxy support; the registry is
  reached inside the network.
- No keepalive is configured by default, because the registry's enforcement
  policy is not known to the client; pass `grpc.WithKeepaliveParams` in
  `DialOptions` if needed (the registry accepts pings every 10 s or more).
  A `WatchLoop` `IdleTimeout` detects dead streams instead.
