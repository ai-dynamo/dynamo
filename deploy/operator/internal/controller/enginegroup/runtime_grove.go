/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package enginegroup

import (
	"context"
	"fmt"
	"net/http"

	api "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

// groveRuntimeProvider selects backend-owned control for one physical world.
// Client and HTTP must not be nil.
type groveRuntimeProvider struct {
	Client client.Client
	HTTP   *http.Client
}

func (p *groveRuntimeProvider) Resolve(ctx context.Context, group *api.DynamoGraphDeploymentEngineGroup) (Runtime, error) {
	// Never let one backend consume another backend's identities or control protocol.
	switch group.Labels[consts.KubeLabelDynamoEngineGroupRuntime] {
	default:
		return Runtime{}, fmt.Errorf("%w: unsupported Engine Group runtime %q", ErrRuntimeUnavailable,
			group.Labels[consts.KubeLabelDynamoEngineGroupRuntime])
	}
}
