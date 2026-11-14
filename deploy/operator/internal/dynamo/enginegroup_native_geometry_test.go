/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package dynamo

import (
	"encoding/json"
	"os"
	"testing"

	"github.com/ai-dynamo/dynamo/deploy/operator/internal/enginegroup"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestEngineGroupNativeGeometryParity(t *testing.T) {
	t.Log("load the declarations also exercised through each native configuration pipeline")
	payload, err := os.ReadFile("testdata/enginegroup-native-geometry.json")
	require.NoError(t, err)
	var cases map[string][]struct {
		Name                      string
		Args                      []string
		GPUs                      int64 `json:"gpus"`
		Initial, Maximum, Members int32
	}
	require.NoError(t, json.Unmarshal(payload, &cases))

	for backend, declarations := range cases {
		for _, declaration := range declarations {
			t.Run(backend+"/"+declaration.Name, func(t *testing.T) {
				t.Log("resolve exactly the native fixture arguments, without rewriting the command")
				var geometry enginegroup.ResolvedProfileGeometry
				var initial, maximum, members int32
				switch backend {
				case "vllm":
					profile, resolveErr := ResolveVLLMElasticEPProfile(VLLMProfileGeometrySource{
						Command: []string{"python3", "-m", "vllm.entrypoints.openai.api_server"}, Args: declaration.Args,
						InitialReplicas: declaration.Initial, MainContainerGPUs: declaration.GPUs,
						DedicatedMainGPUAllocation: true, WorkloadRevisionDigest: "native-parity",
					})
					require.NoError(t, resolveErr)
					geometry, initial, maximum, members = profile.Geometry, profile.InitialReplicas, profile.MaximumReplicas, profile.NativeMembersPerReplica
				case "sglang":
					profile, resolveErr := ResolveSGLangElasticEPProfile(SGLangProfileGeometrySource{
						Command: []string{"python3", "-m", "sglang.launch_server"}, Args: declaration.Args,
						InitialReplicas: declaration.Initial, MainContainerGPUs: declaration.GPUs,
						DedicatedMainGPUAllocation: true, WorkloadRevisionDigest: "native-parity",
					})
					require.NoError(t, resolveErr)
					geometry, initial, maximum, members = profile.Geometry, profile.InitialReplicas, profile.MaximumReplicas, profile.NativeMembersPerReplica
				default:
					t.Fatalf("unhandled native backend %q", backend)
				}

				t.Log("preserve complete allocation counts and native ownership from the shared contract")
				assert.Equal(t, declaration.Initial, initial)
				assert.Equal(t, declaration.Maximum, maximum)
				assert.Equal(t, declaration.Members, members)
				assert.Equal(t, declaration.GPUs, geometry.GPUsPerReplica)
				assert.Equal(t, int32(1), geometry.PodsPerReplica)
			})
		}
	}
}
