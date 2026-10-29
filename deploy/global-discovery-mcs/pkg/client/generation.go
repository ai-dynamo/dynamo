// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package client

import (
	"sync"
	"time"
)

// NextGeneration returns the generation of a writer's next write:
// max(prev + 1, now in unix milliseconds). Starting from 0 after a restart
// still exceeds every generation written before, and the order survives a
// failover to another region, which keeps DynamoDB Global Table
// last-writer-wins in generation order.
func NextGeneration(prev uint64) uint64 {
	return nextGeneration(prev, time.Now())
}

func nextGeneration(prev uint64, now time.Time) uint64 {
	next := prev + 1
	if ms := now.UnixMilli(); ms > 0 && uint64(ms) > next {
		next = uint64(ms)
	}
	return next
}

// Generation is the generation counter of one record a writer owns. The
// zero value is ready to use and safe for concurrent use.
//
//	gen := &client.Generation{}
//	req.Generation = gen.Next()
//	if _, err := c.ServiceExports().ReplaceEndpoints(ctx, req); err != nil {
//		if cur, ok := client.StaleGeneration(err); ok {
//			gen.Observe(cur)
//			req.Generation = gen.Next() // retry once with current+1 or later
//		}
//	}
type Generation struct {
	mu   sync.Mutex
	last uint64
}

// Next returns a generation above every one returned or observed before.
func (g *Generation) Next() uint64 {
	g.mu.Lock()
	defer g.mu.Unlock()
	g.last = NextGeneration(g.last)
	return g.last
}

// Observe records a generation stored by the server, for example the
// current generation of a STALE_GENERATION error, so that Next exceeds it.
func (g *Generation) Observe(current uint64) {
	g.mu.Lock()
	defer g.mu.Unlock()
	g.last = max(g.last, current)
}

// Last returns the last generation returned or observed.
func (g *Generation) Last() uint64 {
	g.mu.Lock()
	defer g.mu.Unlock()
	return g.last
}
