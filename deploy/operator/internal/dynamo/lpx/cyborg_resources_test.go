/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	"testing"

	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
)

func TestEffectiveCyborgGPUCount(t *testing.T) {
	t.Log("Define the exact LPX Cyborg scalar resource contract")
	tests := []struct {
		name, request, limit, wantErr string
		resourceName                  corev1.ResourceName
		want                          int
	}{
		{name: "absent"},
		{name: "MIG does not satisfy classic inventory", resourceName: "nvidia.com/mig-3g.20gb", limit: "1"},
		{name: "limits only", limit: "2", want: 2},
		{name: "equal request and limit", request: "2", limit: "2", want: 2},
		{name: "request without limit", request: "1", wantErr: "requires an equal limit"},
		{name: "request does not equal limit", request: "1", limit: "2", wantErr: "must equal its limit"},
		{name: "fractional limit", limit: "500m", wantErr: "Cyborg GPU limit: GPU resource"},
		{name: "fractional request", request: "500m", limit: "1", wantErr: "Cyborg GPU request: GPU resource"},
		{name: "zero limit", limit: "0", wantErr: "Cyborg GPU limit"},
		{name: "negative request", request: "-1", limit: "1", wantErr: "Cyborg GPU request"},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Log("Resolve the effective exact nvidia.com/gpu count")
			got, err := EffectiveCyborgGPUCount(testCyborgGPUResources(test.resourceName, test.request, test.limit))
			if test.wantErr != "" {
				t.Log("Reject the invalid scalar resource relationship")
				require.ErrorContains(t, err, test.wantErr)
				return
			}

			t.Log("Return the Kubernetes-effective limit count")
			require.NoError(t, err)
			require.Equal(t, test.want, got)
		})
	}
}

func testCyborgGPUResources(name corev1.ResourceName, request, limit string) corev1.ResourceRequirements {
	// Default the resource name and materialize only the maps selected by the table case.
	name = corev1.ResourceName(defaultTestString(string(name), commonconsts.KubeResourceGPUNvidia))
	resources := corev1.ResourceRequirements{}
	if request != "" {
		resources.Requests = corev1.ResourceList{name: resource.MustParse(request)}
	}
	if limit != "" {
		resources.Limits = corev1.ResourceList{name: resource.MustParse(limit)}
	}

	return resources
}
