// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Package store defines the persistence contract of the registry: one
// interface per aggregate (clusters, exports, endpoint sets) composed into
// Store, the record types they persist, and typed errors.
//
// Every record belongs to a clusterset. Records are written by exactly one
// logical writer (the controller of a cluster), and every write is
// conditional:
//
//   - PutCluster mints a new lease epoch max(stored+1, now in unix ms).
//   - RenewClusterLease requires the current epoch.
//   - PutExport and PutEndpointSet require a generation greater than the
//     stored one and report the stored one in ErrStaleGeneration.
//   - Deletes are idempotent; DeleteCluster cascades to everything the
//     cluster owns.
//
// Leases are never expired by the store. The server evaluates them at read
// time from Lease.RenewedAt and Lease.TTL. Timestamps passed in as now are
// server time.
//
// Implementations: memory (development, tests, single replica) and dynamodb
// (one Global Table). storetest.Run is the conformance suite both must pass.
package store

import (
	"context"
	"time"

	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/api"
)

// Store is the complete persistence contract of the server.
type Store interface {
	ClusterStore
	ExportStore
	EndpointStore

	// ListClusterSets returns the names of every clusterset that has had a
	// record written, sorted. A clusterset stays listed after its records
	// are deleted until compaction removes it.
	ListClusterSets(ctx context.Context) ([]string, error)
	// LoadClusterSet returns every record of one clusterset. An unknown
	// clusterset yields an empty snapshot. The snapshot may be eventually
	// consistent across records but each record is whole.
	LoadClusterSet(ctx context.Context, clusterSet string) (ClusterSetSnapshot, error)
	// Ping reports whether the backend is reachable. It returns
	// ErrUnavailable, possibly wrapped, when it is not.
	Ping(ctx context.Context) error
	// Close releases the store's resources. Calls after Close fail.
	Close() error
}

// ClusterStore persists cluster registrations and their leases.
type ClusterStore interface {
	// PutCluster registers or re-registers a cluster. It mints a new lease
	// epoch max(stored epoch + 1, now.UnixMilli()), sets Lease.RenewedAt and
	// UpdatedAt to now, increments Generation, keeps the RegisteredAt of an
	// existing registration (now for a new one) and stores rec.Lease.TTL.
	// It returns the stored record.
	PutCluster(ctx context.Context, rec ClusterRecord, now time.Time) (ClusterRecord, error)
	// RenewClusterLease sets Lease.RenewedAt to now when epoch equals the
	// stored lease epoch. It returns ErrNotFound for an unknown cluster and
	// ErrLeaseEpochMismatch for another epoch.
	RenewClusterLease(ctx context.Context, key ClusterKey, epoch uint64, now time.Time) (ClusterRecord, error)
	// GetCluster returns ErrNotFound for an unknown cluster.
	GetCluster(ctx context.Context, key ClusterKey) (ClusterRecord, error)
	// DeleteCluster removes the cluster and every export and endpoint set of
	// the cluster. It is idempotent: deleting an unknown cluster succeeds,
	// and records of the cluster written without a registration are removed
	// as well.
	DeleteCluster(ctx context.Context, key ClusterKey) error
}

// ExportStore persists ServiceExport records.
type ExportStore interface {
	// PutExport creates or replaces an export when rec.Generation is greater
	// than the stored generation, and returns ErrStaleGeneration with the
	// stored generation otherwise. The record is stored as given.
	PutExport(ctx context.Context, rec ExportRecord) error
	// DeleteExport removes the export and its endpoint set. It is
	// idempotent.
	DeleteExport(ctx context.Context, key ExportKey) error
}

// EndpointStore persists the endpoint set of each export. A set is only ever
// replaced as a whole.
type EndpointStore interface {
	// PutEndpointSet replaces the endpoint set when rec.Generation is
	// greater than the stored generation, and returns ErrStaleGeneration
	// with the stored generation otherwise. It returns ErrTooLarge for more
	// than MaxEndpointsPerSet endpoints or a set above the backend's item
	// size limit. An endpoint set may exist without its export.
	PutEndpointSet(ctx context.Context, rec EndpointSetRecord) error
	// DeleteEndpointSet removes the endpoint set. It is idempotent.
	DeleteEndpointSet(ctx context.Context, key ExportKey) error
}

// ChangeHinter is an optional Store extension that signals that a clusterset
// may have changed in another region, so the server can refresh it before
// its next tick. The DynamoDB store can implement it with DynamoDB Streams.
// No implementation exists in v1; the server polls with LoadClusterSet.
type ChangeHinter interface {
	// ChangeHints returns a channel of clusterset names. The channel is
	// closed when ctx is done.
	ChangeHints(ctx context.Context) (<-chan string, error)
}

// MaxEndpointsPerSet is the largest endpoint set PutEndpointSet accepts.
const MaxEndpointsPerSet = api.MaxEndpointsPerExport

// ClusterKey identifies a cluster record.
type ClusterKey struct {
	ClusterSet string
	Cluster    string
}

// ExportKey identifies an export record and its endpoint set.
type ExportKey struct {
	ClusterSet string
	Cluster    string
	Namespace  string
	Name       string
}

// ClusterKey returns the key of the exporting cluster.
func (k ExportKey) ClusterKey() ClusterKey {
	return ClusterKey{ClusterSet: k.ClusterSet, Cluster: k.Cluster}
}

// ClusterSetSnapshot is every record of one clusterset. Each slice is sorted
// by key.
type ClusterSetSnapshot struct {
	ClusterSet   string
	Clusters     []ClusterRecord
	Exports      []ExportRecord
	EndpointSets []EndpointSetRecord
}

// Lease is a server-minted liveness lease. It is evaluated at read time; see
// registry.LeaseExpired.
type Lease struct {
	// Epoch is the incarnation; 0 means no lease.
	Epoch     uint64
	TTL       time.Duration
	RenewedAt time.Time
}

// ExpiresAt returns RenewedAt plus TTL.
func (l Lease) ExpiresAt() time.Time {
	return l.RenewedAt.Add(l.TTL)
}

// ClusterRecord is a cluster registration.
type ClusterRecord struct {
	Key          ClusterKey
	Region       string
	Zones        []string
	Provider     string
	Labels       map[string]string
	AgentID      string
	AgentVersion string
	Lease        Lease
	// Generation is incremented by the store on every registration.
	Generation   uint64
	RegisteredAt time.Time
	UpdatedAt    time.Time
}

// ExportRecord is the export of one Service from one cluster.
type ExportRecord struct {
	Key        ExportKey
	Spec       ExportSpec
	Generation uint64
	UpdatedAt  time.Time
}

// EndpointSetRecord is the full endpoint set of one export.
type EndpointSetRecord struct {
	Key        ExportKey
	Endpoints  []Endpoint
	Generation uint64
	UpdatedAt  time.Time
}
