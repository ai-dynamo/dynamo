/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package checkpoint

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"

	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
)

// normalizedIdentity is the canonical form used for hash computation
// Only fields that affect checkpoint equivalence are included
type normalizedIdentity struct {
	Model                string            `json:"model"`
	BackendFramework     string            `json:"backendFramework"`
	DynamoVersion        string            `json:"dynamoVersion,omitempty"`
	TensorParallelSize   int32             `json:"tensorParallelSize"`
	PipelineParallelSize int32             `json:"pipelineParallelSize"`
	Dtype                string            `json:"dtype,omitempty"`
	MaxModelLen          int32             `json:"maxModelLen,omitempty"`
	ExtraParameters      map[string]string `json:"extraParameters,omitempty"`
}

// ComputeIdentityHash computes a deterministic hash from a DynamoCheckpointIdentity
// The hash is computed by:
// 1. Normalizing all fields
// 2. Serializing to JSON (with sorted keys)
// 3. Computing SHA256 hash
// 4. Returning first 16 characters of hex encoding (64 bits)
//
// 16 hex characters (64 bits) provides excellent collision resistance:
// - 1% collision probability at ~500 million configs
// - 50% collision probability at ~4 billion configs
// This is a perfect balance between readability and safety.
func ComputeIdentityHash(identity nvidiacomv1alpha1.DynamoCheckpointIdentity) (string, error) {
	normalized := normalizeIdentity(identity)

	// Serialize to JSON (Go's json.Marshal sorts map keys)
	data, err := json.Marshal(normalized)
	if err != nil {
		// This should never happen with our controlled types, but bubble up error if it does
		return "", fmt.Errorf("failed to marshal identity for hashing: %w", err)
	}

	// Compute SHA256 hash
	hash := sha256.Sum256(data)

	// Return first 16 characters of hex encoding (64 bits)
	// Provides excellent collision resistance while remaining readable
	return hex.EncodeToString(hash[:])[:16], nil
}

func normalizeIdentity(identity nvidiacomv1alpha1.DynamoCheckpointIdentity) normalizedIdentity {
	// Apply defaults for TP/PP if not set
	tp := identity.TensorParallelSize
	if tp == 0 {
		tp = 1
	}
	pp := identity.PipelineParallelSize
	if pp == 0 {
		pp = 1
	}

	// ExtraParameters - ensure non-nil for consistent JSON
	extraParams := identity.ExtraParameters
	if extraParams == nil {
		extraParams = make(map[string]string)
	}

	return normalizedIdentity{
		Model:                identity.Model,
		BackendFramework:     identity.BackendFramework,
		DynamoVersion:        identity.DynamoVersion,
		TensorParallelSize:   tp,
		PipelineParallelSize: pp,
		Dtype:                identity.Dtype,
		MaxModelLen:          identity.MaxModelLen,
		ExtraParameters:      extraParams,
	}
}
