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

	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
)

// normalizedIdentity is the canonical form used for hash computation
// Only fields that affect checkpoint equivalence are included
type normalizedIdentity struct {
	Model                string            `json:"model"`
	Framework            string            `json:"framework"`
	FrameworkVersion     string            `json:"frameworkVersion,omitempty"`
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
// 4. Returning first 12 characters of hex encoding
func ComputeIdentityHash(identity nvidiacomv1alpha1.DynamoCheckpointIdentity) string {
	normalized := normalizeIdentity(identity)

	// Serialize to JSON (Go's json.Marshal sorts map keys)
	data, err := json.Marshal(normalized)
	if err != nil {
		// This should never happen with our controlled types
		return ""
	}

	// Compute SHA256 hash
	hash := sha256.Sum256(data)

	// Return first 12 characters of hex encoding
	return hex.EncodeToString(hash[:])[:12]
}

// ComputeServiceIdentityHash computes a hash from a ServiceCheckpointIdentity
func ComputeServiceIdentityHash(identity nvidiacomv1alpha1.ServiceCheckpointIdentity) string {
	// Convert ServiceCheckpointIdentity to DynamoCheckpointIdentity (same fields)
	return ComputeIdentityHash(nvidiacomv1alpha1.DynamoCheckpointIdentity(identity))
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
		Framework:            identity.Framework,
		FrameworkVersion:     identity.FrameworkVersion,
		TensorParallelSize:   tp,
		PipelineParallelSize: pp,
		Dtype:                identity.Dtype,
		MaxModelLen:          identity.MaxModelLen,
		ExtraParameters:      extraParams,
	}
}

// GetTarPath returns the full path to the checkpoint tar file
func GetTarPath(basePath, hash string) string {
	return basePath + "/" + hash + ".tar"
}

// IdentitiesMatch checks if two checkpoint identities would produce the same hash
func IdentitiesMatch(a, b nvidiacomv1alpha1.DynamoCheckpointIdentity) bool {
	return ComputeIdentityHash(a) == ComputeIdentityHash(b)
}
