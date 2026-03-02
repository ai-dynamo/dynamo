/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

package featuregate

import (
	semver "github.com/Masterminds/semver/v3"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

// OperatorOriginFeatureGate represents a feature gated on the operator version
// that originally created the DGD (nvidia.com/dynamo-operator-origin-version).
//
// The origin version is stamped by the mutating webhook at CREATE time and never
// changes afterwards. This allows the operator to introduce new default behaviors
// for newly created resources while preserving backward compatibility for existing ones.
//
// When the annotation is absent (pre-upgrade DGD), IsEnabled returns false
// to preserve backward compatibility.
type OperatorOriginFeatureGate struct {
	Name             string         // Human-readable feature name (for logging)
	MinOriginVersion semver.Version // Minimum origin version required (semver)
}

// IsEnabled returns true if the origin version in annotations meets or exceeds
// the gate's MinOriginVersion threshold.
//
// Returns false when:
//   - annotations is nil (no metadata)
//   - origin version annotation is absent (pre-upgrade DGD)
//   - origin version is not valid semver
//   - origin version < MinOriginVersion
func (fg OperatorOriginFeatureGate) IsEnabled(annotations map[string]string) bool {
	logger := log.Log.WithName("featuregate").WithValues("feature", fg.Name)

	originVersion, exists := annotations[consts.KubeAnnotationDynamoOperatorOriginVersion]
	if !exists {
		logger.V(1).Info("No operator origin version annotation, feature disabled (backward compat)")
		return false
	}

	version, err := semver.NewVersion(originVersion)
	if err != nil {
		logger.Info("Invalid origin version, feature disabled",
			"version", originVersion, "error", err.Error())
		return false
	}

	enabled := version.Compare(&fg.MinOriginVersion) >= 0

	logger.V(1).Info("Feature gate evaluated",
		"originVersion", originVersion,
		"threshold", fg.MinOriginVersion,
		"enabled", enabled)

	return enabled
}
