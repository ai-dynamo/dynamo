/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	"encoding/json"
	"strings"
	"testing"

	lpxv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/internal/thirdparty/lpxscheduler/v1alpha1"
	"github.com/stretchr/testify/require"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
)

func TestValidateRequestSizeMatchesLPXBoundary(t *testing.T) {
	t.Parallel()

	t.Log("Build a request close enough to the limit that valid annotation padding can reach the exact boundary")
	request := &lpxv1alpha1.LPUPipelineRequest{
		ObjectMeta: metav1.ObjectMeta{
			Name:        "request",
			Namespace:   "namespace",
			Annotations: map[string]string{"scheduling.lpu.nvidia.com/size-proof": ""},
		},
		Spec: lpxv1alpha1.LPUPipelineRequestSpec{
			AllocationMetadata: runtime.RawExtension{Raw: []byte(`{}`)},
			ExecutionBackend:   lpxv1alpha1.ExecutionBackendNodeLocal,
			NodeLocal:          &lpxv1alpha1.NodeLocalRequest{Model: "default"},
			Partitions: []lpxv1alpha1.PartitionRequest{{
				ID: "partition-000", Ordinal: 0, CompilerPartitionID: 0,
				Extent: &[]int64{16, 1, 1, 1},
			}},
			PodGangRef:   lpxv1alpha1.NamespacedName{Name: "agents"},
			TargetFamily: lpxv1alpha1.TargetFamilyHx16x8x2x3,
			WorkloadMode: lpxv1alpha1.WorkloadModeV3HxLPUOnly,
		},
	}
	connections := make([]lpxv1alpha1.HxLogicalConnection, 256)
	for index := range connections {
		connections[index] = lpxv1alpha1.HxLogicalConnection{
			FromLogicalDevice: int64(index / 16),
			ToLogicalDevice:   int64(index % 16),
		}
	}
	lanes := []int64{4, 2, 1}
	baseBytes := 0
	for len(request.Spec.PropSyncConnectors) < maxLPXPartitions-1 {
		baseBytes = requestWithRequirementEchoBytes(request)
		if baseBytes >= maxLPXPlannedRequestObjectBytes-200*1024 {
			break
		}
		request.Spec.PropSyncConnectors = append(
			request.Spec.PropSyncConnectors,
			lpxv1alpha1.PropSyncConnectorRequest{
				FromPartitionID: "partition-000",
				ToPartitionID:   "partition-000",
				Requirement: lpxv1alpha1.PropSyncConnectorRequirement{
					Kind:                         lpxv1alpha1.PropSyncConnectorKindHxPropSyncV1,
					Connections:                  &connections,
					AcceptableLaneMultiplicities: &lanes,
				},
			},
		)
	}

	t.Log("Pad the request annotation to the exact inclusive scheduler budget")
	baseBytes = requestWithRequirementEchoBytes(request)
	require.Less(t, baseBytes, maxLPXPlannedRequestObjectBytes)
	padding := maxLPXPlannedRequestObjectBytes - baseBytes
	require.LessOrEqual(t, padding, 200*1024)
	request.Annotations["scheduling.lpu.nvidia.com/size-proof"] = strings.Repeat("x", padding)

	t.Log("Accept equality and reject the first byte beyond LPX's inclusive limit")
	require.NoError(t, ValidateRequestSize(request))
	request.Annotations["scheduling.lpu.nvidia.com/size-proof"] += "x"
	require.ErrorContains(t, ValidateRequestSize(request), "maximum")
}

func TestWorkloadDigestIsIndependentOfBuildLocator(t *testing.T) {
	t.Parallel()

	t.Log("Acquire equal compiler contents under distinct local build paths")
	first := normalizeTestSnapshot(t, acquireTestSnapshot(t, writeV2CompilerFixture(t)))
	second := normalizeTestSnapshot(t, acquireTestSnapshot(t, writeV2CompilerFixture(t)))
	require.NotEqual(t, first.build.Path, second.build.Path)
	require.Equal(t, first.contentID, second.contentID)

	t.Log("Project either immutable snapshot through the same intent")
	firstProjection := projectTestBuild(t, first, PipelineSingle, `{"prop_sync":true}`)
	secondProjection := projectTestBuild(t, second, PipelineSingle, `{"prop_sync":true}`)

	t.Log("Produce the same projection digest independent of build locator")
	require.Equal(t, firstProjection.Digest(), secondProjection.Digest())
}

func TestCanonicalModelSettingsRetainRawAndNormalizeRuntimeNumbers(t *testing.T) {
	t.Log("Decode canonical model settings while retaining their exact raw bytes")
	raw, object := canonicalModelSettings(json.RawMessage(
		`{"exponent":1e3,"float":1.0,"integer":8192,"nested":{"array":[128001,1.5,9223372036854775808]}}`,
	))
	require.Equal(t, json.RawMessage(`{"exponent":1e3,"float":1.0,"integer":8192,"nested":{"array":[128001,1.5,9223372036854775808]}}`), raw)
	require.Equal(t, json.Number("8192"), object["integer"])

	t.Log("Normalize representable integers while retaining floating-point runtime values")
	merged, err := mergeRuntimeSettingOverride(nil, object, "settings")
	require.NoError(t, err)
	normalized := merged.(map[string]any)
	require.Equal(t, int64(8192), normalized["integer"])
	require.Equal(t, []any{int64(128001), float64(1.5), float64(9223372036854775808)}, normalized["nested"].(map[string]any)["array"])

	t.Log("Normalize explicit null settings to a present empty object")
	raw, object = canonicalModelSettings(json.RawMessage("null"))
	require.Nil(t, raw)
	require.NotNil(t, object)
	require.Empty(t, object)
}
