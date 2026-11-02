// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Package registry holds the registry domain rules that do not depend on
// storage or transport: lease evaluation, lease TTL bounds, writer generations
// and DNS label checks.
package registry

import (
	"time"
)

// Lease defaults and bounds.
const (
	// DefaultClusterLeaseTTL applies when RegisterCluster names no TTL.
	DefaultClusterLeaseTTL = 30 * time.Second
	// MinLeaseTTL and MaxLeaseTTL bound every requested TTL.
	MinLeaseTTL = 10 * time.Second
	MaxLeaseTTL = 5 * time.Minute
	// DefaultLeaseSkew is the clock skew allowance between replicas and
	// writers before a lease counts as expired.
	DefaultLeaseSkew = 2 * time.Second
)

// LeaseState is the state of a lease evaluated at read time.
type LeaseState string

const (
	LeaseStateActive  LeaseState = "ACTIVE"
	LeaseStateExpired LeaseState = "EXPIRED"
)
