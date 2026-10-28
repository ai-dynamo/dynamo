//go:build !clustertest

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package enginegroup

import (
	"context"

	domain "github.com/ai-dynamo/dynamo/deploy/operator/internal/enginegroup"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/enginegroup/workload"
)

// physicalCapacityTestProvider has no provider-native resources, labels, or ownership conventions.
type physicalCapacityTestProvider struct {
	observation workload.Observation
	targets     []domain.CapacityTarget
	prepared    int
}

func (p *physicalCapacityTestProvider) Prepare(context.Context) error {
	p.prepared++
	return nil
}

func (p *physicalCapacityTestProvider) Observe(context.Context, domain.GroupID) (workload.Observation, error) {
	return p.observation, nil
}

func (p *physicalCapacityTestProvider) Apply(_ context.Context, _ domain.GroupID, target domain.CapacityTarget) (domain.ApplyResult, error) {
	p.targets = append(p.targets, target)
	p.observation.AppliedRevision = target.ControlRevision
	return domain.ApplyResult{}, nil
}
