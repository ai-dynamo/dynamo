// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package shared_test

import (
	"maps"
	"testing"

	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/controller/shared"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
)

func TestRenderServicePreservesUnresolvedComponentType(t *testing.T) {
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
			t.Log("Render the base Service from an optional resolved component type")
			service := shared.RenderService(shared.ServiceParams{
				ServiceName: "graph-component", Namespace: "default", DGDName: "graph",
				ComponentName: "component", ComponentType: tc.resolvedType, DynamoNamespace: "default-graph", Labels: tc.authored,
			})

			t.Log("Preserve authored type metadata while retaining the exact Pod selector contract")
			value, present := service.Labels[consts.KubeLabelDynamoComponentType]
			require.Equal(t, tc.wantPresent, present)
			require.Equal(t, tc.want, value)
			selectorType, present := service.Spec.Selector[consts.KubeLabelDynamoComponentType]
			require.True(t, present, "an empty selector requirement must not become an absent requirement")
			require.Equal(t, tc.resolvedType, selectorType)
		})
	}
}

func TestRenderServiceGraphIdentity(t *testing.T) {
	for _, tc := range []struct {
		name        string
		dgdName     string
		authored    map[string]string
		want        string
		wantPresent bool
	}{
		{name: "resolved graph wins", dgdName: "graph", authored: map[string]string{consts.KubeLabelDynamoGraphDeploymentName: "wrong-graph"}, want: "graph", wantPresent: true},
		{name: "standalone preserves authored graph", authored: map[string]string{consts.KubeLabelDynamoGraphDeploymentName: "authored-graph"}, want: "authored-graph", wantPresent: true},
		{name: "standalone does not fabricate graph"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Log("Render the resolved graph identity, including standalone components")
			service := shared.RenderService(shared.ServiceParams{
				ServiceName: "Qwen3-0.6B", Namespace: "default", DGDName: tc.dgdName,
				ComponentName: "frontend", ComponentType: "frontend", DynamoNamespace: "default-graph", Labels: tc.authored,
			})

			t.Log("Keep canonical identity and selector metadata while leaving ports to the lowering")
			value, present := service.Labels[consts.KubeLabelDynamoGraphDeploymentName]
			require.Equal(t, tc.wantPresent, present)
			require.Equal(t, tc.want, value)
			require.Equal(t, "qwen3-0-6b", service.Name)
			require.Equal(t, "default", service.Namespace)
			require.Equal(t, "frontend", service.Labels[consts.KubeLabelDynamoComponent])
			require.Equal(t, "frontend", service.Labels[consts.KubeLabelDynamoComponentType])
			require.Equal(t, "default-graph", service.Labels[consts.KubeLabelDynamoNamespace])
			require.Equal(t, map[string]string{
				consts.KubeLabelDynamoComponent:     "frontend",
				consts.KubeLabelDynamoComponentType: "frontend",
				consts.KubeLabelDynamoNamespace:     "default-graph",
			}, service.Spec.Selector)
			require.Empty(t, service.Spec.Ports)
		})
	}
}

func TestRenderServiceDiscovery(t *testing.T) {
	for _, enabled := range []bool{false, true} {
		t.Run(map[bool]string{false: "disabled", true: "enabled"}[enabled], func(t *testing.T) {
			t.Log("Render the explicit Kubernetes discovery setting")
			service := shared.RenderService(shared.ServiceParams{IsK8sDiscovery: enabled})

			t.Log("Expose Kubernetes discovery labels only when requested")
			backend, backendPresent := service.Labels[consts.KubeLabelDynamoDiscoveryBackend]
			discoveryEnabled, enabledPresent := service.Labels[consts.KubeLabelDynamoDiscoveryEnabled]
			require.Equal(t, enabled, backendPresent)
			require.Equal(t, enabled, enabledPresent)
			if enabled {
				require.Equal(t, consts.DiscoveryBackendKubernetes, backend)
				require.Equal(t, consts.KubeLabelValueTrue, discoveryEnabled)
			}
		})
	}
}

func TestRenderServiceDoesNotAliasInputs(t *testing.T) {
	t.Log("Render twice from authored metadata that contains stale component identity")
	params := shared.ServiceParams{
		ServiceName: "graph-frontend", Namespace: "default", DGDName: "graph",
		ComponentName: "frontend", ComponentType: "frontend", DynamoNamespace: "default-graph",
		Labels: map[string]string{
			consts.KubeLabelDynamoComponent: "wrong-component", consts.KubeLabelDynamoNamespace: "wrong-namespace", "user-label": "original",
		},
		Annotations: map[string]string{"user-annotation": "original"},
	}
	previousLabels, previousAnnotations := maps.Clone(params.Labels), maps.Clone(params.Annotations)
	service := shared.RenderService(params)
	other := shared.RenderService(params)
	require.Equal(t, previousLabels, params.Labels)
	require.Equal(t, previousAnnotations, params.Annotations)

	t.Log("Extend one lowering's rendered metadata and spec without changing inputs or another rendering")
	service.Labels["user-label"] = "changed"
	service.Labels["lowering-label"] = "owned"
	service.Annotations["user-annotation"] = "changed"
	service.Spec.Selector[consts.KubeLabelDynamoComponent] = "leader"
	service.Spec.ClusterIP = corev1.ClusterIPNone
	service.Spec.Ports = []corev1.ServicePort{{Name: "http", Port: 8000}}
	require.Equal(t, previousLabels, params.Labels)
	require.Equal(t, previousAnnotations, params.Annotations)
	require.Equal(t, "original", other.Labels["user-label"])
	require.NotContains(t, other.Labels, "lowering-label")
	require.Equal(t, "original", other.Annotations["user-annotation"])
	require.Equal(t, "frontend", other.Spec.Selector[consts.KubeLabelDynamoComponent])
	require.Empty(t, other.Spec.ClusterIP)
	require.Empty(t, other.Spec.Ports)

	t.Log("Extending a rendering with absent authored metadata also has writable maps")
	standalone := shared.RenderService(shared.ServiceParams{})
	standalone.Labels["lowering-label"] = "owned"
	standalone.Annotations["lowering-annotation"] = "owned"
	require.Equal(t, "owned", standalone.Labels["lowering-label"])
	require.Equal(t, "owned", standalone.Annotations["lowering-annotation"])
}
