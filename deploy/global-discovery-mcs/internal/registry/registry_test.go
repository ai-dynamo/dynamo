// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package registry

import (
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
)

var now = time.Date(2026, 10, 26, 12, 0, 0, 0, time.UTC)

func TestLeaseExpired(t *testing.T) {
	ttl := 30 * time.Second
	skew := 2 * time.Second
	tests := []struct {
		name    string
		renewed time.Time
		now     time.Time
		skew    time.Duration
		want    bool
	}{
		{"just renewed", now, now, skew, false},
		{"within ttl", now, now.Add(29 * time.Second), skew, false},
		{"at expires_at", now, now.Add(ttl), skew, false},
		{"within skew", now, now.Add(ttl + skew - time.Millisecond), skew, false},
		{"at expires_at plus skew", now, now.Add(ttl + skew), skew, false},
		{"past expires_at plus skew", now, now.Add(ttl + skew + time.Nanosecond), skew, true},
		{"no skew allowance", now, now.Add(ttl + time.Nanosecond), 0, true},
		{"renewed in the future of the evaluating replica", now.Add(5 * time.Second), now, skew, false},
		{"long expired", now, now.Add(time.Hour), skew, true},
	}
	for _, tt := range tests {
		assert.Equal(t, tt.want, LeaseExpired(tt.renewed, ttl, tt.now, tt.skew), tt.name)
		want := LeaseStateActive
		if tt.want {
			want = LeaseStateExpired
		}
		assert.Equal(t, want, EvaluateLease(tt.renewed, ttl, tt.now, tt.skew), tt.name)
	}
}

func TestLeaseExpiresAt(t *testing.T) {
	assert.Equal(t, now.Add(30*time.Second), LeaseExpiresAt(now, 30*time.Second))
}

func TestClampLeaseTTL(t *testing.T) {
	tests := []struct {
		requested, def, want time.Duration
	}{
		{0, DefaultClusterLeaseTTL, 30 * time.Second},
		{-time.Second, time.Minute, time.Minute},
		{time.Second, DefaultClusterLeaseTTL, MinLeaseTTL},
		{10 * time.Second, DefaultClusterLeaseTTL, 10 * time.Second},
		{45 * time.Second, DefaultClusterLeaseTTL, 45 * time.Second},
		{5 * time.Minute, DefaultClusterLeaseTTL, 5 * time.Minute},
		{time.Hour, DefaultClusterLeaseTTL, MaxLeaseTTL},
	}
	for _, tt := range tests {
		assert.Equal(t, tt.want, ClampLeaseTTL(tt.requested, tt.def), "requested %s", tt.requested)
	}
}

func TestHeartbeatInterval(t *testing.T) {
	assert.Equal(t, 10*time.Second, HeartbeatInterval(DefaultClusterLeaseTTL))
	assert.Equal(t, 20*time.Second, HeartbeatInterval(time.Minute))
}

func TestNextGeneration(t *testing.T) {
	ms := uint64(now.UnixMilli())
	assert.Equal(t, ms, NextGeneration(0, now), "a restarted writer takes the clock")
	assert.Equal(t, ms, NextGeneration(ms-1, now))
	assert.Equal(t, ms+1, NextGeneration(ms, now), "two writes in one millisecond")
	assert.Equal(t, ms+1001, NextGeneration(ms+1000, now), "the clock went backwards")
	assert.Equal(t, uint64(1), NextGeneration(0, time.Time{}))
}
