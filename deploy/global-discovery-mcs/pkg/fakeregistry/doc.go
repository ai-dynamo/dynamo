// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Package fakeregistry is an in-memory registry for tests.
//
// It implements the four gRPC services (ClusterService,
// ServiceExportService, ServiceImportService and ServerInfoService) and the
// gRPC health service with the semantics clients rely on, so that client,
// controller and integration tests can run against a real gRPC endpoint
// without a store or a server process:
//
//   - Cluster leases with server-minted epochs. Heartbeats with another
//     epoch fail with ABORTED and reason LEASE_EPOCH_MISMATCH. Leases are
//     evaluated at read time against the server clock (see WithClock) with
//     the default 2 s skew allowance.
//   - Writer generations. A PutServiceExport or ReplaceEndpoints without a
//     generation (0) gets one the server assigns, max(stored + 1, unix
//     milliseconds), returned in the response; one whose generation does
//     not exceed the stored one fails with FAILED_PRECONDITION, reason
//     STALE_GENERATION, ErrorInfo metadata current_generation and a
//     PreconditionFailure detail, like the server.
//   - Merged ServiceImports, simplified: the oldest export of a cluster with
//     an active lease wins (export creation time, then cluster name), ports
//     are unioned by (port, protocol) and no conflicts are reported.
//   - Watches for clusters and service imports: a snapshot (SNAPSHOT_START,
//     one UPSERT per object, SNAPSHOT_END), then UPSERT and DELETE deltas
//     and periodic BOOKMARKs, all carrying revisions of the form
//     <view_epoch>:<seq>. Every change is broadcast to the watchers of its
//     kind and clusterset; filters are applied per watcher, and an object
//     that stops matching a watcher's filter is sent to it as a DELETE. A
//     resume_revision of this server that is still in the history replays
//     the missed deltas; any other revision yields a new snapshot.
//
// Fault injection: FailNext makes the next call of a method fail with a
// given code, CloseWatches ends every open watch, SetServing(false) makes
// the health service report NOT_SERVING, and Stop on a Running listener
// drops its connections. CallCount and WatcherCount let tests observe
// traffic.
//
// Usage:
//
//	srv := fakeregistry.New()
//	run := srv.ServeBufconn()          // or srv.ServeLocal() on 127.0.0.1:0
//	defer run.Stop()
//	conn, _ := grpc.NewClient("passthrough:///fakeregistry",
//		grpc.WithContextDialer(run.Dialer()),
//		grpc.WithTransportCredentials(insecure.NewCredentials()))
//
// Errors carry an ErrorInfo with the registry's domain and reason,
// INVALID_ARGUMENT (with a BadRequest) and NOT_FOUND included.
//
// Seeding helpers (EnsureCluster, PutExport, SetEndpoints, DeleteExport,
// RemoveCluster) write state directly, with ServiceSpec, TCPPort,
// ReadyEndpoints, WithHostnames and WithPodNames building typical export
// specs and endpoints. They broadcast to watchers like the RPCs do.
//
// Differences from mcs-registry: one replica and no store, so writes are
// visible at once; no conflict detection (Conflict is never set); IP
// families come from the winning export; beyond required fields requests
// are not validated. PutServiceExport and ReplaceEndpoints require a
// registered cluster (FAILED_PRECONDITION, CLUSTER_NOT_REGISTERED).
package fakeregistry
