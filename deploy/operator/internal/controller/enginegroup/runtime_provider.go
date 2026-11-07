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

// runtimeProvider connects one workload provider to the selected engine integration.
// Client, HTTP and Workload must not be nil. Provider-native objects never enter engine resolution.
type runtimeProvider struct {
	Client   client.Client
	HTTP     *http.Client
	Workload WorkloadProvider
}

var _ RuntimeProvider = (*runtimeProvider)(nil)

func (p *runtimeProvider) Resolve(ctx context.Context, group *api.DynamoGraphDeploymentEngineGroup) (Runtime, error) {
	// Never let one backend consume another backend's identities or control protocol.
	switch group.Labels[consts.KubeLabelDynamoEngineGroupRuntime] {
	case consts.KubeLabelDynamoEngineGroupSGLang:
		world, err := p.Workload.Resolve(ctx, group)
		if err != nil {
			return Runtime{}, err
		}
		return (&sglangRuntimeResolver{client: p.Client, httpClient: p.HTTP}).Resolve(ctx, group, world)
	default:
		return Runtime{}, fmt.Errorf("%w: unsupported Engine Group runtime %q", ErrRuntimeUnavailable,
			group.Labels[consts.KubeLabelDynamoEngineGroupRuntime])
	}
}
