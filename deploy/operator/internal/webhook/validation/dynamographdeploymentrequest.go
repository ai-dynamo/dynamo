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

package validation

import (
	"encoding/json"
	"errors"
	"fmt"

	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"sigs.k8s.io/controller-runtime/pkg/webhook/admission"
)

// DynamoGraphDeploymentRequestValidator validates DynamoGraphDeploymentRequest resources.
// This validator can be used by both webhooks and controllers for consistent validation.
type DynamoGraphDeploymentRequestValidator struct {
	request               *nvidiacomv1beta1.DynamoGraphDeploymentRequest
	isClusterWideOperator bool
	gpuDiscoveryEnabled   bool
}

// NewDynamoGraphDeploymentRequestValidator creates a new validator for DynamoGraphDeploymentRequest.
// isClusterWide indicates whether the operator has cluster-wide permissions.
// gpuDiscoveryEnabled indicates whether Helm provisioned node read access for the operator.
func NewDynamoGraphDeploymentRequestValidator(request *nvidiacomv1beta1.DynamoGraphDeploymentRequest, isClusterWide bool, gpuDiscoveryEnabled bool) *DynamoGraphDeploymentRequestValidator {
	return &DynamoGraphDeploymentRequestValidator{
		request:               request,
		isClusterWideOperator: isClusterWide,
		gpuDiscoveryEnabled:   gpuDiscoveryEnabled,
	}
}

// Validate performs stateless validation on the DynamoGraphDeploymentRequest.
// Returns warnings and error.
func (v *DynamoGraphDeploymentRequestValidator) Validate() (admission.Warnings, error) {
	var err error

	// Disallow searchStrategy: thorough with backend: auto.
	// "thorough" sweeps more configurations and requires a concrete backend to be selected;
	// "auto" defers backend selection and is only compatible with the "rapid" search strategy.
	if v.request.Spec.SearchStrategy == nvidiacomv1beta1.SearchStrategyThorough &&
		v.request.Spec.Backend == nvidiacomv1beta1.BackendTypeAuto {
		err = errors.Join(err, fmt.Errorf(
			"spec.searchStrategy %q is incompatible with spec.backend %q: set spec.backend to a specific backend (sglang, trtllm, or vllm)",
			nvidiacomv1beta1.SearchStrategyThorough,
			nvidiacomv1beta1.BackendTypeAuto,
		))
	}

	// Validate GPU hardware information is available (last, so other errors are collected first).
	if gpuErr := v.validateGPUHardwareInfo(); gpuErr != nil {
		err = errors.Join(err, gpuErr)
	}

	return nil, err
}

// validateGPUHardwareInfo ensures GPU hardware information will be available for profiling.
// Returns an error at admission time if GPU discovery is disabled and no manual hardware config is provided.
// Also validates consistency of GPU range fields.
func (v *DynamoGraphDeploymentRequestValidator) validateGPUHardwareInfo() error {
	// Check if manual hardware config is provided via typed spec.hardware fields.
	var hasManualHardwareConfig bool
	if hw := v.request.Spec.Hardware; hw != nil {
		hasManualHardwareConfig = hw.GPUSKU != "" || hw.VRAMMB != nil || hw.NumGPUsPerNode != nil
	}

	if hasManualHardwareConfig {
		return nil
	}

	// No manual hardware config provided. Cluster-wide operators always have GPU discovery via node
	// permissions. Namespace-scoped operators rely on Helm-provisioned GPU discovery (gpuDiscovery.enabled).
	if v.isClusterWideOperator || v.gpuDiscoveryEnabled {
		return nil
	}

	return errors.New("GPU hardware configuration required: GPU discovery is disabled (set dynamo-operator.gpuDiscovery.enabled=true in Helm values, or provide hardware config in spec.hardware)")
}

// ValidateUpdate performs stateful validation comparing old and new DynamoGraphDeploymentRequest.
// Returns warnings and error.
func (v *DynamoGraphDeploymentRequestValidator) ValidateUpdate(old *nvidiacomv1beta1.DynamoGraphDeploymentRequest) (admission.Warnings, error) {
	// Reject spec changes when the resource is in a non-editable lifecycle phase.
	// During Profiling, Deploying, or Deployed the controller is actively reconciling
	// the resource and spec mutations would conflict with in-flight operations.
	phase := old.Status.Phase
	immutablePhases := map[nvidiacomv1beta1.DGDRPhase]bool{
		nvidiacomv1beta1.DGDRPhaseProfiling: true,
		nvidiacomv1beta1.DGDRPhaseDeploying: true,
		nvidiacomv1beta1.DGDRPhaseDeployed:  true,
	}

	if immutablePhases[phase] {
		// Compare specs — if they differ, reject the update.
		oldSpec := old.Spec
		newSpec := v.request.Spec
		if !specEqual(oldSpec, newSpec) {
			return nil, fmt.Errorf("spec updates are forbidden while the resource is in phase %q; delete and recreate the resource to change its spec", phase)
		}
	}

	return nil, nil
}

// specEqual performs a JSON-round-trip comparison of two DynamoGraphDeploymentRequestSpec values.
func specEqual(a, b nvidiacomv1beta1.DynamoGraphDeploymentRequestSpec) bool {
	aj, err1 := json.Marshal(a)
	bj, err2 := json.Marshal(b)
	if err1 != nil || err2 != nil {
		return false
	}
	return string(aj) == string(bj)
}
