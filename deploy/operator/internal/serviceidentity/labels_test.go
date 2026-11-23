// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package serviceidentity_test

import (
	"testing"

	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	componentdeployment "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller/component"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/controller/grove"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
)

func TestServicesPreserveUnresolvedComponentType(t *testing.T) {
	for _, tc := range []struct {
		name         string
		resolvedType string
		authored     map[string]string
		want         string
		wantPresent  bool
	}{
		{name: "authored type", authored: map[string]string{consts.KubeLabelDynamoComponentType: "worker"}, want: "worker", wantPresent: true},
		{name: "absent type", authored: map[string]string{}},
		{name: "resolved type wins", resolvedType: "frontend", authored: map[string]string{consts.KubeLabelDynamoComponentType: "worker"}, want: "frontend", wantPresent: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Log("Render ordinary and auxiliary Services from the same optional component type")
			params := componentdeployment.ServiceParams{
				ServiceName: "graph-component", Namespace: "default", DGDName: "graph",
				ComponentName: "component", ComponentType: tc.resolvedType, DynamoNamespace: "default-graph", Labels: tc.authored,
				IsK8sDiscovery: true,
			}
			ordinary, err := componentdeployment.RenderService(params)
			require.NoError(t, err)

			t.Log("Preserve an authored type or its absence when the type is unresolved")
			for _, service := range []*corev1.Service{ordinary, grove.RenderElasticEPLeaderService(&v1beta1.DynamoComponentDeploymentSharedSpec{}, params, true)} {
				value, present := service.Labels[consts.KubeLabelDynamoComponentType]
				require.Equal(t, tc.wantPresent, present)
				require.Equal(t, tc.want, value)
				selectorType, present := service.Spec.Selector[consts.KubeLabelDynamoComponentType]
				require.True(t, present, "an empty selector requirement must not become an absent requirement")
				require.Equal(t, tc.resolvedType, selectorType, "preserve existing Service selector semantics")
			}
		})
	}
}
