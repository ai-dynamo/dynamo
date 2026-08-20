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
	"context"
	"encoding/json"
	"fmt"
	"sort"

	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/features"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/runtimeversion"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	k8sruntime "k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"sigs.k8s.io/controller-runtime/pkg/webhook/admission"
)

// DynamoGraphDeploymentRequestValidator validates v1beta1 DynamoGraphDeploymentRequest resources.
type DynamoGraphDeploymentRequestValidator struct{}

// NewDynamoGraphDeploymentRequestValidator creates a DynamoGraphDeploymentRequest validator.
func NewDynamoGraphDeploymentRequestValidator() *DynamoGraphDeploymentRequestValidator {
	return &DynamoGraphDeploymentRequestValidator{}
}

// dynamoGraphDeploymentRequestValidation carries DGDR-specific request state.
// API values, paths, and accumulated errors remain explicit validator arguments.
type dynamoGraphDeploymentRequestValidation struct {
	ctx      context.Context
	warnings admission.Warnings
}

func (v *dynamoGraphDeploymentRequestValidation) warn(message string) {
	v.warnings = append(v.warnings, message)
}

var deprecatedDGDRDGDComponentNames = map[string]string{
	"VllmWorker":          "worker",
	"VllmPrefillWorker":   "prefill",
	"VllmDecodeWorker":    "worker for aggregate deployments or decode for disaggregated deployments",
	"TRTLLMPrefillWorker": "prefill",
	"TRTLLMDecodeWorker":  "decode",
	"SGLangPrefillWorker": "prefill",
	"SGLangDecodeWorker":  "worker for aggregate deployments or decode for disaggregated deployments",
	"SglangPrefillWorker": "prefill",
	"SglangDecodeWorker":  "worker for aggregate deployments or decode for disaggregated deployments",
}

// Validate performs stateless validation on request. ctx and request must not be nil.
func (v *DynamoGraphDeploymentRequestValidator) Validate(
	ctx context.Context,
	request *nvidiacomv1beta1.DynamoGraphDeploymentRequest,
) (admission.Warnings, error) {
	validation := &dynamoGraphDeploymentRequestValidation{ctx: ctx}
	allErrs := validation.validateDynamoGraphDeploymentRequest(request)
	return validation.warnings, invalidDynamoGraphDeploymentRequestError(request, allErrs)
}

// ValidateUpdate validates newRequest against oldRequest. ctx, oldRequest, and newRequest must not be nil.
func (v *DynamoGraphDeploymentRequestValidator) ValidateUpdate(
	ctx context.Context,
	oldRequest *nvidiacomv1beta1.DynamoGraphDeploymentRequest,
	newRequest *nvidiacomv1beta1.DynamoGraphDeploymentRequest,
) (admission.Warnings, error) {
	validation := &dynamoGraphDeploymentRequestValidation{ctx: ctx}
	allErrs := validation.validateDynamoGraphDeploymentRequestUpdate(newRequest, oldRequest)
	validation.warnDeprecatedDGDOverrideComponentNames(&newRequest.Spec)
	return validation.warnings, invalidDynamoGraphDeploymentRequestError(newRequest, allErrs)
}

// validateDynamoGraphDeploymentRequest validates request. request must not be nil.
func (v *dynamoGraphDeploymentRequestValidation) validateDynamoGraphDeploymentRequest(
	request *nvidiacomv1beta1.DynamoGraphDeploymentRequest,
) field.ErrorList {
	return v.validateDynamoGraphDeploymentRequestSpec(&request.Spec, field.NewPath("spec"))
}

// validateDynamoGraphDeploymentRequestSpec validates spec. spec and fldPath must not be nil.
func (v *dynamoGraphDeploymentRequestValidation) validateDynamoGraphDeploymentRequestSpec(
	spec *nvidiacomv1beta1.DynamoGraphDeploymentRequestSpec,
	fldPath *field.Path,
) field.ErrorList {
	allErrs := field.ErrorList{}

	if !features.MustGateFrom(v.ctx).Enabled(features.GPUDiscovery) && !hasManualDGDRHardware(spec.Hardware) {
		allErrs = append(allErrs, field.Required(
			fldPath.Child("hardware"),
			"GPU hardware configuration is required when GPU discovery is disabled",
		))
	}

	if spec.SearchStrategy == nvidiacomv1beta1.SearchStrategyThorough &&
		spec.Backend == nvidiacomv1beta1.BackendTypeAuto {
		allErrs = append(allErrs, field.Invalid(
			fldPath.Child("searchStrategy"),
			spec.SearchStrategy,
			fmt.Sprintf("is incompatible with spec.backend %q; set spec.backend to a specific backend (sglang, trtllm, or vllm)", spec.Backend),
		))
	}

	if dgdrRuntimeVersionOverrideRequired(spec) {
		allErrs = append(allErrs, field.Required(
			fldPath.Child("runtimeVersionOverride"),
			"is required when spec.image has no parseable semantic-version tag",
		))
	}

	v.warnDeprecatedDGDOverrideComponentNames(spec)

	return allErrs
}

func (v *dynamoGraphDeploymentRequestValidation) warnDeprecatedDGDOverrideComponentNames(
	spec *nvidiacomv1beta1.DynamoGraphDeploymentRequestSpec,
) {
	if spec.Overrides == nil || spec.Overrides.DGD == nil {
		return
	}

	override, ok := rawExtensionToUnstructured(spec.Overrides.DGD)
	if !ok {
		return
	}

	type deprecatedName struct {
		name string
		path string
	}
	deprecated := make([]deprecatedName, 0)
	switch override.GetAPIVersion() {
	case nvidiacomv1alpha1.GroupVersion.String():
		services, _, err := unstructured.NestedMap(override.Object, "spec", "services")
		if err != nil {
			return
		}
		for name := range services {
			if _, found := deprecatedDGDRDGDComponentNames[name]; found {
				deprecated = append(deprecated, deprecatedName{
					name: name,
					path: "spec.overrides.dgd.spec.services." + name,
				})
			}
		}
	case nvidiacomv1beta1.GroupVersion.String():
		components, _, err := unstructured.NestedSlice(override.Object, "spec", "components")
		if err != nil {
			return
		}
		for _, value := range components {
			component, ok := value.(map[string]interface{})
			if !ok {
				continue
			}
			name, _ := component["name"].(string)
			if _, found := deprecatedDGDRDGDComponentNames[name]; found {
				deprecated = append(deprecated, deprecatedName{
					name: name,
					path: fmt.Sprintf("spec.overrides.dgd.spec.components[name=%s]", name),
				})
			}
		}
	default:
		return
	}

	sort.Slice(deprecated, func(i, j int) bool {
		return deprecated[i].path < deprecated[j].path
	})
	for _, item := range deprecated {
		v.warn(fmt.Sprintf(
			"%s: generated component name %q is deprecated; use %s",
			item.path,
			item.name,
			deprecatedDGDRDGDComponentNames[item.name],
		))
	}
}

func rawExtensionToUnstructured(extension *k8sruntime.RawExtension) (*unstructured.Unstructured, bool) {
	if len(extension.Raw) > 0 {
		object := map[string]interface{}{}
		if err := json.Unmarshal(extension.Raw, &object); err != nil {
			return nil, false
		}
		return &unstructured.Unstructured{Object: object}, true
	}
	if extension.Object == nil {
		return nil, false
	}
	object, err := k8sruntime.DefaultUnstructuredConverter.ToUnstructured(extension.Object)
	if err != nil {
		return nil, false
	}
	return &unstructured.Unstructured{Object: object}, true
}

// validateDynamoGraphDeploymentRequestUpdate validates an update. newRequest and oldRequest must not be nil.
func (v *dynamoGraphDeploymentRequestValidation) validateDynamoGraphDeploymentRequestUpdate(
	newRequest *nvidiacomv1beta1.DynamoGraphDeploymentRequest,
	oldRequest *nvidiacomv1beta1.DynamoGraphDeploymentRequest,
) field.ErrorList {
	return v.validateDynamoGraphDeploymentRequestSpecUpdate(
		&newRequest.Spec,
		&oldRequest.Spec,
		field.NewPath("spec"),
		oldRequest.Status.Phase,
	)
}

// validateDynamoGraphDeploymentRequestSpecUpdate validates a spec update.
// newSpec, oldSpec, and fldPath must not be nil; oldPhase comes from the owning old resource.
func (v *dynamoGraphDeploymentRequestValidation) validateDynamoGraphDeploymentRequestSpecUpdate(
	newSpec *nvidiacomv1beta1.DynamoGraphDeploymentRequestSpec,
	oldSpec *nvidiacomv1beta1.DynamoGraphDeploymentRequestSpec,
	fldPath *field.Path,
	oldPhase nvidiacomv1beta1.DGDRPhase,
) field.ErrorList {
	allErrs := field.ErrorList{}

	gpuDiscoveryEnabled := features.MustGateFrom(v.ctx).Enabled(features.GPUDiscovery)
	newRequiresHardware := !gpuDiscoveryEnabled && !hasManualDGDRHardware(newSpec.Hardware)
	oldRequiresHardware := !gpuDiscoveryEnabled && !hasManualDGDRHardware(oldSpec.Hardware)
	if newRequiresHardware && !oldRequiresHardware {
		allErrs = append(allErrs, field.Required(
			fldPath.Child("hardware"),
			"GPU hardware configuration is required when GPU discovery is disabled",
		))
	}

	newHasIncompatibleSearch := newSpec.SearchStrategy == nvidiacomv1beta1.SearchStrategyThorough &&
		newSpec.Backend == nvidiacomv1beta1.BackendTypeAuto
	oldHasIncompatibleSearch := oldSpec.SearchStrategy == nvidiacomv1beta1.SearchStrategyThorough &&
		oldSpec.Backend == nvidiacomv1beta1.BackendTypeAuto
	if newHasIncompatibleSearch && !oldHasIncompatibleSearch {
		allErrs = append(allErrs, field.Invalid(
			fldPath.Child("searchStrategy"),
			newSpec.SearchStrategy,
			fmt.Sprintf("is incompatible with spec.backend %q; set spec.backend to a specific backend (sglang, trtllm, or vllm)", newSpec.Backend),
		))
	}

	// Revalidate legacy image compatibility when autoApply creates a DGD.
	autoApplyActivated := oldSpec.AutoApply != nil && !*oldSpec.AutoApply &&
		(newSpec.AutoApply == nil || *newSpec.AutoApply)
	if dgdrRuntimeVersionOverrideRequired(newSpec) &&
		(newSpec.Image != oldSpec.Image ||
			newSpec.RuntimeVersionOverride != oldSpec.RuntimeVersionOverride ||
			autoApplyActivated) {
		allErrs = append(allErrs, field.Required(
			fldPath.Child("runtimeVersionOverride"),
			"is required when spec.image has no parseable semantic-version tag",
		))
	}

	if isImmutableDGDRPhase(oldPhase) &&
		!apiequality.Semantic.DeepEqual(newSpec, oldSpec) &&
		!isDGDRDeferredRuntimeVersionUpdate(newSpec, oldSpec, oldPhase) {
		allErrs = append(allErrs, field.Forbidden(
			fldPath,
			fmt.Sprintf("updates are forbidden while the resource is in phase %q; delete and recreate the resource to change its spec", oldPhase),
		))
	}

	return allErrs
}

func dgdrRuntimeVersionOverrideRequired(spec *nvidiacomv1beta1.DynamoGraphDeploymentRequestSpec) bool {
	if spec.Image == "" || spec.RuntimeVersionOverride != "" {
		return false
	}
	_, err := runtimeversion.ParseImageVersion(spec.Image)
	return err != nil
}

// isDGDRDeferredRuntimeVersionUpdate permits choosing the runtime version before deferred DGD creation.
func isDGDRDeferredRuntimeVersionUpdate(
	newSpec *nvidiacomv1beta1.DynamoGraphDeploymentRequestSpec,
	oldSpec *nvidiacomv1beta1.DynamoGraphDeploymentRequestSpec,
	oldPhase nvidiacomv1beta1.DGDRPhase,
) bool {
	if oldSpec.AutoApply == nil || *oldSpec.AutoApply ||
		(oldPhase != nvidiacomv1beta1.DGDRPhaseProfiling &&
			oldPhase != nvidiacomv1beta1.DGDRPhaseReady) {
		return false
	}

	// autoApply may only be activated once the generated snapshot is Ready.
	autoApplyActivated := newSpec.AutoApply == nil || *newSpec.AutoApply
	if autoApplyActivated && oldPhase != nvidiacomv1beta1.DGDRPhaseReady {
		return false
	}

	// Require every other field to remain unchanged.
	newWithoutDeferredFields := newSpec.DeepCopy()
	newWithoutDeferredFields.AutoApply = oldSpec.AutoApply
	newWithoutDeferredFields.RuntimeVersionOverride = oldSpec.RuntimeVersionOverride
	return apiequality.Semantic.DeepEqual(newWithoutDeferredFields, oldSpec)
}
