package checkpoint

import (
	"testing"

	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestComputeIdentityHash(t *testing.T) {
	tests := []struct {
		name          string
		identity      nvidiacomv1alpha1.DynamoCheckpointIdentity
		expectError   bool
		expectedHash  string // Only set for deterministic checks
		otherIdentity *nvidiacomv1alpha1.DynamoCheckpointIdentity
		shouldMatch   bool
	}{
		{
			name: "basic identity produces deterministic hash",
			identity: nvidiacomv1alpha1.DynamoCheckpointIdentity{
				Model:            "meta-llama/Llama-2-7b-hf",
				BackendFramework: "vllm",
			},
			expectError:  false,
			expectedHash: "96429b2725761a09", // Known hash for this specific identity
		},
		{
			name: "identity with all fields produces deterministic hash",
			identity: nvidiacomv1alpha1.DynamoCheckpointIdentity{
				Model:                "meta-llama/Llama-2-13b-hf",
				BackendFramework:     "sglang",
				DynamoVersion:        "0.4.2",
				TensorParallelSize:   2,
				PipelineParallelSize: 1,
				Dtype:                "float16",
				MaxModelLen:          4096,
				ExtraParameters: map[string]string{
					"gpu_memory_utilization": "0.9",
				},
			},
			expectError:  false,
			expectedHash: "f4ba65bccbb8e4fb", // Known hash for this specific identity
		},
		{
			name: "same identity produces same hash",
			identity: nvidiacomv1alpha1.DynamoCheckpointIdentity{
				Model:            "meta-llama/Llama-2-7b-hf",
				BackendFramework: "vllm",
			},
			otherIdentity: &nvidiacomv1alpha1.DynamoCheckpointIdentity{
				Model:            "meta-llama/Llama-2-7b-hf",
				BackendFramework: "vllm",
			},
			expectError: false,
			shouldMatch: true,
		},
		{
			name: "different models produce different hashes",
			identity: nvidiacomv1alpha1.DynamoCheckpointIdentity{
				Model:            "meta-llama/Llama-2-7b-hf",
				BackendFramework: "vllm",
			},
			otherIdentity: &nvidiacomv1alpha1.DynamoCheckpointIdentity{
				Model:            "meta-llama/Llama-2-13b-hf",
				BackendFramework: "vllm",
			},
			expectError: false,
			shouldMatch: false,
		},
		{
			name: "different frameworks produce different hashes",
			identity: nvidiacomv1alpha1.DynamoCheckpointIdentity{
				Model:            "meta-llama/Llama-2-7b-hf",
				BackendFramework: "vllm",
			},
			otherIdentity: &nvidiacomv1alpha1.DynamoCheckpointIdentity{
				Model:            "meta-llama/Llama-2-7b-hf",
				BackendFramework: "sglang",
			},
			expectError: false,
			shouldMatch: false,
		},
		{
			name: "normalization: zero vs unset numeric fields",
			identity: nvidiacomv1alpha1.DynamoCheckpointIdentity{
				Model:              "meta-llama/Llama-2-7b-hf",
				BackendFramework:   "vllm",
				TensorParallelSize: 0,
				MaxModelLen:        0,
			},
			otherIdentity: &nvidiacomv1alpha1.DynamoCheckpointIdentity{
				Model:            "meta-llama/Llama-2-7b-hf",
				BackendFramework: "vllm",
				// TensorParallelSize and MaxModelLen omitted (defaults to 0)
			},
			expectError: false,
			shouldMatch: true,
		},
		{
			name: "normalization: empty vs nil map",
			identity: nvidiacomv1alpha1.DynamoCheckpointIdentity{
				Model:            "meta-llama/Llama-2-7b-hf",
				BackendFramework: "vllm",
				ExtraParameters:  map[string]string{},
			},
			otherIdentity: &nvidiacomv1alpha1.DynamoCheckpointIdentity{
				Model:            "meta-llama/Llama-2-7b-hf",
				BackendFramework: "vllm",
				ExtraParameters:  nil,
			},
			expectError: false,
			shouldMatch: true,
		},
		{
			name: "extra parameters order should not matter",
			identity: nvidiacomv1alpha1.DynamoCheckpointIdentity{
				Model:            "meta-llama/Llama-2-7b-hf",
				BackendFramework: "vllm",
				ExtraParameters: map[string]string{
					"param_a": "value1",
					"param_b": "value2",
				},
			},
			otherIdentity: &nvidiacomv1alpha1.DynamoCheckpointIdentity{
				Model:            "meta-llama/Llama-2-7b-hf",
				BackendFramework: "vllm",
				ExtraParameters: map[string]string{
					"param_b": "value2",
					"param_a": "value1",
				},
			},
			expectError: false,
			shouldMatch: true,
		},
		{
			name: "different extra parameters produce different hashes",
			identity: nvidiacomv1alpha1.DynamoCheckpointIdentity{
				Model:            "meta-llama/Llama-2-7b-hf",
				BackendFramework: "vllm",
				ExtraParameters: map[string]string{
					"gpu_memory_utilization": "0.9",
				},
			},
			otherIdentity: &nvidiacomv1alpha1.DynamoCheckpointIdentity{
				Model:            "meta-llama/Llama-2-7b-hf",
				BackendFramework: "vllm",
				ExtraParameters: map[string]string{
					"gpu_memory_utilization": "0.8",
				},
			},
			expectError: false,
			shouldMatch: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			hash1, err1 := ComputeIdentityHash(tt.identity)

			if tt.expectError {
				require.Error(t, err1)
				return
			}

			require.NoError(t, err1)
			assert.NotEmpty(t, hash1, "hash should not be empty")
			assert.Len(t, hash1, 16, "hash should be 16 characters (64 bits)")
			// Verify it's hex
			assert.Regexp(t, "^[0-9a-f]{16}$", hash1, "hash should be 16 hex characters")

			// If we have an expected hash, check it
			if tt.expectedHash != "" {
				assert.Equal(t, tt.expectedHash, hash1)
			}

			// If we have another identity to compare, compute its hash
			if tt.otherIdentity != nil {
				hash2, err2 := ComputeIdentityHash(*tt.otherIdentity)
				require.NoError(t, err2)

				if tt.shouldMatch {
					assert.Equal(t, hash1, hash2, "hashes should match")
				} else {
					assert.NotEqual(t, hash1, hash2, "hashes should differ")
				}
			}
		})
	}
}
