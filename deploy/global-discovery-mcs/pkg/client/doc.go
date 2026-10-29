// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Package client is the Go client of the Multi-Cluster Services registry.
//
// A Client holds one gRPC connection for unary calls and watches, with:
//
//   - Endpoint failover. Config.URLs is an ordered list, normally the global
//     name first and per-region names after it. A per-client gRPC resolver
//     resolves every host name in URL order on start and on every
//     re-resolution gRPC asks for (after each lost or failed connection),
//     with the injected Config.Resolver, and sets each address's TLS server
//     name to its URL's host. pick_first, without shuffling, connects to
//     the first address that answers. The registry endpoints the server
//     advertises in GetServerInfo are appended after the configured URLs.
//     A URL whose endpoint answers a call or a watch with UNAVAILABLE is
//     left out of the address list for a backoff, so the client moves on
//     to the next URL. Every PrimaryRetryInterval a client connected
//     elsewhere probes the primary URL with the gRPC health service and,
//     when every registry service reports SERVING, moves its connection
//     and its WatchLoops back to it.
//   - Unary calls with a default deadline (RPCTimeout) and gRPC retries on
//     UNAVAILABLE (MaxAttempts, Backoff).
//   - Typed sub-clients: Clusters, ServiceExports, ServiceImports and
//     ServerInfo.
//   - OnReconnect hooks for writers that resend state after a reconnect.
//   - dynamo_mcs_client_* metrics when Config.Registerer is set.
//
// Helpers built on it:
//
//   - WatchLoop: keeps one watch open, reconnects with backoff, resumes from
//     the last revision, and maps SNAPSHOT_START, UPSERT, DELETE,
//     SNAPSHOT_END and BOOKMARK to handler calls, with an OnResync callback
//     when the server sends a new snapshot.
//   - NextGeneration and Generation for writer generations,
//     max(previous + 1, unix milliseconds), and StaleGeneration and Reason
//     to read registry error details.
//
// pkg/fakeregistry is an in-memory registry for tests of code using this
// package.
package client
