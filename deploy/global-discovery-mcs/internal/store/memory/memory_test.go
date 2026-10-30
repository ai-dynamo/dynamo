// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package memory

import (
	"context"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/internal/store"
	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/internal/store/storetest"
)

func TestConformance(t *testing.T) {
	storetest.Run(t, func(*testing.T) store.Store { return New() })
}

func TestCloseMakesStoreUnavailable(t *testing.T) {
	s := New()
	ctx := context.Background()
	key := store.ClusterKey{ClusterSet: "cs", Cluster: "c1"}
	_, err := s.PutCluster(ctx, store.ClusterRecord{Key: key}, time.Now())
	require.NoError(t, err)

	require.NoError(t, s.Close())
	require.NoError(t, s.Close(), "Close is idempotent")

	assert.ErrorIs(t, s.Ping(ctx), store.ErrUnavailable)
	_, err = s.GetCluster(ctx, key)
	assert.ErrorIs(t, err, store.ErrUnavailable)
	_, err = s.PutCluster(ctx, store.ClusterRecord{Key: key}, time.Now())
	assert.ErrorIs(t, err, store.ErrUnavailable)
	_, err = s.ListClusterSets(ctx)
	assert.ErrorIs(t, err, store.ErrUnavailable)
	_, err = s.LoadClusterSet(ctx, "cs")
	assert.ErrorIs(t, err, store.ErrUnavailable)
}
