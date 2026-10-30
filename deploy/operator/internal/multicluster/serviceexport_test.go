/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package multicluster

import (
	"slices"
	"testing"

	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
)

func TestIsPool(t *testing.T) {
	for _, tc := range []struct {
		name        string
		annotations map[string]string
		want        bool
	}{
		{name: "no annotations"},
		{name: "marked", annotations: map[string]string{commonconsts.KubeAnnotationGlobalRouterPool: "true"}, want: true},
		{name: "marked in another case", annotations: map[string]string{commonconsts.KubeAnnotationGlobalRouterPool: "True"}, want: true},
		{name: "marked false", annotations: map[string]string{commonconsts.KubeAnnotationGlobalRouterPool: "false"}},
		{name: "not a boolean", annotations: map[string]string{commonconsts.KubeAnnotationGlobalRouterPool: "yes"}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Log("Read the Global Router pool mark")
			if got := IsPool(tc.annotations); got != tc.want {
				t.Errorf("IsPool(%v) = %v, want %v", tc.annotations, got, tc.want)
			}
		})
	}
}

func TestMaxServiceNameLength(t *testing.T) {
	for _, tc := range []struct {
		implementation configv1alpha1.MCSImplementation
		want           int
	}{
		{implementation: configv1alpha1.MCSImplementationStandard, want: 63},
		{implementation: configv1alpha1.MCSImplementationKarmada, want: 55},
	} {
		t.Run(string(tc.implementation), func(t *testing.T) {
			t.Log("Look up the longest Service name the implementation exports with its cluster")
			if got := MaxServiceNameLength(tc.implementation); got != tc.want {
				t.Errorf("MaxServiceNameLength(%q) = %d, want %d", tc.implementation, got, tc.want)
			}
		})
	}
}

func TestServiceNames(t *testing.T) {
	for _, tc := range []struct {
		dgdName string
		want    []string
	}{
		{dgdName: "qwen", want: []string{"qwen-pool-relay", "qwen-frontend"}},
		{dgdName: "qwen3-0.6b", want: []string{"qwen3-0-6b-pool-relay", "qwen3-0-6b-frontend"}},
	} {
		t.Run(tc.dgdName, func(t *testing.T) {
			t.Log("Name the pool's Services as the operator names its own")
			if got := ServiceNames(tc.dgdName); !slices.Equal(got, tc.want) {
				t.Errorf("ServiceNames(%q) = %v, want %v", tc.dgdName, got, tc.want)
			}
		})
	}
}

func TestRenderServiceExport(t *testing.T) {
	t.Log("Render the export of pool qwen's PoolRelay Service")
	export := RenderServiceExport("dynamo", "qwen-pool-relay", "qwen")
	if got := export.GroupVersionKind(); got != ServiceExportGVK {
		t.Errorf("GroupVersionKind = %v, want %v", got, ServiceExportGVK)
	}
	if export.GetNamespace() != "dynamo" || export.GetName() != "qwen-pool-relay" {
		t.Errorf("export = %s/%s, want dynamo/qwen-pool-relay", export.GetNamespace(), export.GetName())
	}
	if got := export.GetLabels()[commonconsts.KubeLabelDynamoGraphDeploymentName]; got != "qwen" {
		t.Errorf("DGD label = %q, want qwen", got)
	}
}
