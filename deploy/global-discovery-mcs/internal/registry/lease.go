// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package registry

import (
	"time"
)

// LeaseExpiresAt returns the time a lease renewed at renewedAt expires.
func LeaseExpiresAt(renewedAt time.Time, ttl time.Duration) time.Time {
	return renewedAt.Add(ttl)
}

// LeaseExpired reports whether a lease renewed at renewedAt with the given
// TTL has expired at now, allowing skew for clock differences between the
// replica that renewed it and the one evaluating it. A lease is expired once
// now is strictly after renewedAt + ttl + skew.
func LeaseExpired(renewedAt time.Time, ttl time.Duration, now time.Time, skew time.Duration) bool {
	return now.After(renewedAt.Add(ttl).Add(skew))
}

// EvaluateLease returns the LeaseState of a lease at now.
func EvaluateLease(renewedAt time.Time, ttl time.Duration, now time.Time, skew time.Duration) LeaseState {
	if LeaseExpired(renewedAt, ttl, now, skew) {
		return LeaseStateExpired
	}
	return LeaseStateActive
}

// ClampLeaseTTL returns the TTL a registration gets: def when requested is
// zero or negative, otherwise requested bounded to [MinLeaseTTL, MaxLeaseTTL].
func ClampLeaseTTL(requested, def time.Duration) time.Duration {
	switch {
	case requested <= 0:
		return def
	case requested < MinLeaseTTL:
		return MinLeaseTTL
	case requested > MaxLeaseTTL:
		return MaxLeaseTTL
	}
	return requested
}

// HeartbeatInterval returns how often a writer renews a lease of the given
// TTL: a third of it, so that two heartbeats may be lost before expiry.
func HeartbeatInterval(ttl time.Duration) time.Duration {
	return ttl / 3
}

// NextGeneration returns the generation a writer uses for its next write:
// max(prev + 1, now in unix milliseconds). It stays monotonic across writer
// restarts (prev unknown, so 0) and across region failover, which keeps
// DynamoDB Global Table last-writer-wins in generation order.
func NextGeneration(prev uint64, now time.Time) uint64 {
	next := prev + 1
	if ms := now.UnixMilli(); ms > 0 && uint64(ms) > next {
		next = uint64(ms)
	}
	return next
}
