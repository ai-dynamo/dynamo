/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package selectionservice

import "strings"

// These structs mirror the runtime-free selection service worker API in
// lib/kv-router/src/services/selection/types.rs. Keep the JSON field names in
// sync with WorkerRequest and WorkerCatalogRecord there.

const (
	// Rust deserializes omitted model_name to "default"; normalize Go requests to
	// the same value so controller diffs compare against the catalog record.
	DefaultModelName = "default"
	// Rust deserializes omitted tenant_id to "default"; normalize Go requests to
	// the same value so controller diffs compare against the catalog record.
	DefaultTenantID = "default"
)

// NormalizeModelName applies the selection service model_name default.
func NormalizeModelName(modelName string) string {
	return normalizeDefaultString(modelName, DefaultModelName)
}

// NormalizeTenantID applies the selection service tenant_id default.
func NormalizeTenantID(tenantID string) string {
	return normalizeDefaultString(tenantID, DefaultTenantID)
}

func normalizeDefaultString(value string, defaultValue string) string {
	value = strings.TrimSpace(value)
	if value == "" {
		return defaultValue
	}
	return value
}

// WorkerRequest is the platform-neutral worker registration payload accepted by
// the runtime-free selection service POST /workers API.
type WorkerRequest struct {
	WorkerID                  uint64            `json:"worker_id"`
	ModelName                 string            `json:"model_name,omitempty"`
	TenantID                  string            `json:"tenant_id,omitempty"`
	Endpoint                  string            `json:"endpoint,omitempty"`
	KVEventsEndpoint          string            `json:"kv_events_endpoint,omitempty"`
	KVEventsEndpoints         map[uint32]string `json:"kv_events_endpoints,omitempty"`
	ReplayEndpoint            string            `json:"replay_endpoint,omitempty"`
	BlockSize                 uint32            `json:"block_size,omitempty"`
	DataParallelStartRank     uint32            `json:"data_parallel_start_rank,omitempty"`
	DataParallelSize          uint32            `json:"data_parallel_size,omitempty"`
	MaxNumBatchedTokens       uint64            `json:"max_num_batched_tokens,omitempty"`
	TotalKVBlocks             uint64            `json:"total_kv_blocks,omitempty"`
	StableRoutingID           string            `json:"stable_routing_id,omitempty"`
	IsEagle                   *bool             `json:"is_eagle,omitempty"`
	Taints                    []string          `json:"taints,omitempty"`
	TopologyDomains           map[string]string `json:"topology_domains,omitempty"`
	KVTransferDomain          string            `json:"kv_transfer_domain,omitempty"`
	KVTransferEnforcement     string            `json:"kv_transfer_enforcement,omitempty"`
	KVTransferPreferredWeight *float32          `json:"kv_transfer_preferred_weight,omitempty"`
	Metadata                  map[string]string `json:"metadata,omitempty"`
}

type WorkerLifecycle string

const (
	WorkerLifecycleIncomplete    WorkerLifecycle = "incomplete"
	WorkerLifecycleSchedulable   WorkerLifecycle = "schedulable"
	WorkerLifecycleDraining      WorkerLifecycle = "draining"
	WorkerLifecycleUnschedulable WorkerLifecycle = "unschedulable"
)

// WorkerRecord is the selection service worker catalog record returned by
// GET/POST/PATCH/DELETE /workers APIs.
type WorkerRecord struct {
	WorkerID                  uint64            `json:"worker_id"`
	ModelName                 string            `json:"model_name,omitempty"`
	TenantID                  string            `json:"tenant_id,omitempty"`
	Lifecycle                 WorkerLifecycle   `json:"lifecycle,omitempty"`
	Endpoint                  string            `json:"endpoint,omitempty"`
	KVEventsEndpoint          string            `json:"kv_events_endpoint,omitempty"`
	KVEventsEndpoints         map[uint32]string `json:"kv_events_endpoints,omitempty"`
	ReplayEndpoint            string            `json:"replay_endpoint,omitempty"`
	BlockSize                 *uint32           `json:"block_size,omitempty"`
	DataParallelStartRank     *uint32           `json:"data_parallel_start_rank,omitempty"`
	DataParallelSize          *uint32           `json:"data_parallel_size,omitempty"`
	MaxNumBatchedTokens       *uint64           `json:"max_num_batched_tokens,omitempty"`
	TotalKVBlocks             *uint64           `json:"total_kv_blocks,omitempty"`
	StableRoutingID           string            `json:"stable_routing_id,omitempty"`
	IsEagle                   *bool             `json:"is_eagle,omitempty"`
	Taints                    []string          `json:"taints,omitempty"`
	TopologyDomains           map[string]string `json:"topology_domains,omitempty"`
	KVTransferDomain          string            `json:"kv_transfer_domain,omitempty"`
	KVTransferEnforcement     string            `json:"kv_transfer_enforcement,omitempty"`
	KVTransferPreferredWeight *float32          `json:"kv_transfer_preferred_weight,omitempty"`
	Metadata                  map[string]string `json:"metadata,omitempty"`
	NotSchedulableReasons     []string          `json:"not_schedulable_reasons,omitempty"`
}
