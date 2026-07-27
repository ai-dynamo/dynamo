/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

package dgdrutil

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"

	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
)

// SpecFingerprint returns a stable SHA-256 fingerprint of a DGDR spec.
func SpecFingerprint(spec *nvidiacomv1beta1.DynamoGraphDeploymentRequestSpec) (string, error) {
	data, err := json.Marshal(spec)
	if err != nil {
		return "", fmt.Errorf("marshal DGDR spec: %w", err)
	}
	return fmt.Sprintf("%x", sha256.Sum256(data)), nil
}

// IsRuntimeVersionOverrideRepair reports whether current only adds an override to the observed spec.
func IsRuntimeVersionOverrideRepair(
	current *nvidiacomv1beta1.DynamoGraphDeploymentRequestSpec,
	observedFingerprint string,
) (bool, error) {
	if observedFingerprint == "" || current.RuntimeVersionOverride == "" {
		return false, nil
	}

	observedCandidate := current.DeepCopy()
	observedCandidate.RuntimeVersionOverride = ""
	fingerprint, err := SpecFingerprint(observedCandidate)
	if err != nil {
		return false, err
	}
	return fingerprint == observedFingerprint, nil
}
