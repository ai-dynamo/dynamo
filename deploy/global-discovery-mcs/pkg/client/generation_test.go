// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package client

import (
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"google.golang.org/genproto/googleapis/rpc/errdetails"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/api"
	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/internal/registry"
)

func TestNextGeneration(t *testing.T) {
	now := time.Date(2026, 10, 26, 12, 0, 0, 0, time.UTC)
	ms := uint64(now.UnixMilli())
	tests := []struct {
		name string
		prev uint64
		want uint64
	}{
		{"fresh writer takes the clock", 0, ms},
		{"older generation takes the clock", ms - 1000, ms},
		{"equal to the clock moves past it", ms, ms + 1},
		{"ahead of the clock stays monotonic", ms + 5000, ms + 5001},
	}
	for _, tt := range tests {
		got := nextGeneration(tt.prev, now)
		assert.Equal(t, tt.want, got, tt.name)
		assert.Equal(t, registry.NextGeneration(tt.prev, now), got, "matches the server rule: %s", tt.name)
	}
	assert.Equal(t, uint64(1), nextGeneration(0, time.Unix(0, 0)))
	assert.GreaterOrEqual(t, NextGeneration(0), uint64(time.Now().Add(-time.Minute).UnixMilli()))
}

func TestGeneration(t *testing.T) {
	var g Generation
	first := g.Next()
	assert.Greater(t, g.Next(), first)
	g.Observe(first + 1_000_000)
	assert.Equal(t, first+1_000_000, g.Last())
	assert.Equal(t, first+1_000_001, g.Next())
	g.Observe(1)
	assert.Equal(t, first+1_000_001, g.Last(), "observing an older generation changes nothing")

	var wg sync.WaitGroup
	seen := sync.Map{}
	for range 8 {
		wg.Go(func() {
			for range 100 {
				_, dup := seen.LoadOrStore(g.Next(), true)
				assert.False(t, dup)
			}
		})
	}
	wg.Wait()
}

func TestErrorHelpers(t *testing.T) {
	stale := status.New(codes.FailedPrecondition, "stale")
	stale, _ = stale.WithDetails(&errdetails.ErrorInfo{
		Reason: api.ReasonStaleGeneration, Domain: api.ErrorDomain,
		Metadata: map[string]string{api.MetadataCurrentGeneration: "42"},
	})
	cur, ok := StaleGeneration(stale.Err())
	assert.True(t, ok)
	assert.Equal(t, uint64(42), cur)
	assert.Equal(t, api.ReasonStaleGeneration, Reason(stale.Err()))

	other := status.New(codes.FailedPrecondition, "x")
	other, _ = other.WithDetails(&errdetails.ErrorInfo{Reason: api.ReasonStaleGeneration, Domain: "example.com"})
	_, ok = StaleGeneration(other.Err())
	assert.False(t, ok, "another domain's reason is not a registry reason")

	epoch, _ := status.New(codes.Aborted, "epoch").WithDetails(&errdetails.ErrorInfo{Reason: api.ReasonLeaseEpochMismatch, Domain: api.ErrorDomain})
	assert.True(t, IsLeaseEpochMismatch(epoch.Err()))
	assert.False(t, IsClusterNotRegistered(epoch.Err()))
	notReg, _ := status.New(codes.NotFound, "nope").WithDetails(&errdetails.ErrorInfo{Reason: api.ReasonClusterNotRegistered, Domain: api.ErrorDomain})
	assert.True(t, IsClusterNotRegistered(notReg.Err()))

	assert.Empty(t, Reason(nil))
	assert.Empty(t, Reason(assert.AnError))
	_, ok = StaleGeneration(status.Error(codes.Unavailable, "down"))
	assert.False(t, ok)
}
