// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Package storefactory opens the store backend that the configuration
// selects. It is the seam where store implementations plug into mcs-registry
// and mcs-admin: the DynamoDB store adds its opener to openers (or calls
// Register from its own package) when it is built into the binaries.
package storefactory

import (
	"context"
	"fmt"
	"sort"
	"sync"

	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/internal/config"
	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/internal/store"
	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/internal/store/memory"
)

// Opener opens a store from the configuration (--store, --dynamodb-table,
// --dynamodb-endpoint, --region).
type Opener func(ctx context.Context, cfg *config.Config) (store.Store, error)

var (
	mu      sync.RWMutex
	openers = map[string]Opener{
		config.StoreMemory: func(context.Context, *config.Config) (store.Store, error) { return memory.New(), nil },
	}
)

// Register makes a backend available under kind, replacing any previous
// opener of that kind.
func Register(kind string, open Opener) {
	mu.Lock()
	defer mu.Unlock()
	openers[kind] = open
}

// Kinds returns the registered backend kinds, sorted.
func Kinds() []string {
	mu.RLock()
	defer mu.RUnlock()
	kinds := make([]string, 0, len(openers))
	for k := range openers {
		kinds = append(kinds, k)
	}
	sort.Strings(kinds)
	return kinds
}

// Open opens the backend named by cfg.Store.
func Open(ctx context.Context, cfg *config.Config) (store.Store, error) {
	mu.RLock()
	open, ok := openers[cfg.Store]
	mu.RUnlock()
	if ok {
		return open(ctx, cfg)
	}
	return nil, fmt.Errorf("--store=%s: unknown store; available: %v", cfg.Store, Kinds())
}
