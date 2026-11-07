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

package validation

import (
	"encoding/json"
	"errors"
	"strings"
	"unicode/utf8"

	nvidiacomv1beta2 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta2"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/aisimulate"
	"github.com/santhosh-tekuri/jsonschema/v6"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/util/validation"
	"k8s.io/apimachinery/pkg/util/validation/field"
)

// validateNativeSearchInput validates typed fields and the merged AISimulate input.
// spec and fldPath must not be nil. checkInputSchema=false skips only lowering and schema validation.
func (v *sharedValidation) validateNativeSearchInput(spec *nvidiacomv1beta2.NativeSearchInput, fldPath *field.Path, checkInputSchema bool) field.ErrorList {
	allErrs := field.ErrorList{}
	if spec.Hardware != nil {
		allErrs = append(allErrs, v.validateHardwareSpecV1beta2(spec.Hardware, fldPath.Child("hardware"))...)
	}
	if spec.Workload != nil {
		allErrs = append(allErrs, validateWorkloadSpec(spec.Workload, fldPath.Child("workload"))...)
	}
	if spec.AISim != nil {
		allErrs = append(allErrs, validateAISimSpec(spec.AISim, fldPath.Child("aisim"))...)
	}
	if spec.Overrides != nil {
		allErrs = append(allErrs, v.validateOverridesSpecV1beta2(spec.Overrides, fldPath.Child("overrides"))...)
	}
	if !checkInputSchema {
		return allErrs
	}

	// Validate the final merge independently of typed rules; overrides remain authoritative.
	input, err := aisimulate.Lower(spec)
	if err != nil {
		return append(allErrs, field.Invalid(fldPath, "<redacted>", "cannot be lowered to the public AISimulate recommendation input: "+err.Error()))
	}
	if err := aisimulate.ValidateInput(input); err != nil {
		var schemaErr *jsonschema.ValidationError
		if !errors.As(err, &schemaErr) {
			return append(allErrs, field.InternalError(fldPath, err))
		}
		allErrs = append(allErrs, inputSchemaErrors(schemaErr, fldPath)...)
	}
	return allErrs
}

// validateNativeSearchInputUpdate preserves metadata/status updates across upstream schema changes.
// newSpec, oldSpec, and fldPath must not be nil. Only the upstream schema check is ratcheted;
// typed API and operator-owned execution-field invariants still apply to unchanged specs.
// CRD admission serializes RawExtension objects with canonical JSON map-key ordering.
func (v *sharedValidation) validateNativeSearchInputUpdate(newSpec, oldSpec *nvidiacomv1beta2.NativeSearchInput, fldPath *field.Path) field.ErrorList {
	return v.validateNativeSearchInput(newSpec, fldPath, !apiequality.Semantic.DeepEqual(newSpec, oldSpec))
}

// validateOverridesSpecV1beta2 validates the partial resource envelope. spec and fldPath must not be nil.
func (v *sharedValidation) validateOverridesSpecV1beta2(
	spec *nvidiacomv1beta2.OverridesSpec,
	fldPath *field.Path,
) field.ErrorList {
	if spec.DGD == nil {
		return nil
	}

	// Nested DGD values are checked on the complete generated deployment in the publisher.
	var override map[string]json.RawMessage
	path := fldPath.Child("dgd")
	if err := json.Unmarshal(spec.DGD.Raw, &override); err != nil || override == nil {
		return field.ErrorList{field.Invalid(path, "<redacted>", "must be a partial DGD object")}
	}
	allErrs := field.ErrorList{}
	for _, member := range []struct{ name, expected string }{
		{"apiVersion", "nvidia.com/v1beta1"}, {"kind", "DynamoGraphDeployment"},
	} {
		var value string
		if err := json.Unmarshal(override[member.name], &value); err != nil || value != member.expected {
			allErrs = append(allErrs, field.NotSupported(path.Child(member.name), value, []string{member.expected}))
		}
	}
	if _, found := override["status"]; found {
		allErrs = append(allErrs, field.Forbidden(path.Child("status"), "DGD status cannot be overridden"))
	}
	return allErrs
}

// validateHardwareSpecV1beta2 validates hardware. spec and fldPath must not be nil.
func (v *sharedValidation) validateHardwareSpecV1beta2(
	spec *nvidiacomv1beta2.HardwareSpec,
	fldPath *field.Path,
) field.ErrorList {
	if spec.GPU == nil {
		return nil
	}
	return v.validateGPUHardwareSpecV1beta2(spec.GPU, fldPath.Child("gpu"))
}

// validateGPUHardwareSpecV1beta2 validates gpu. spec and fldPath must not be nil.
func (v *sharedValidation) validateGPUHardwareSpecV1beta2(
	spec *nvidiacomv1beta2.GPUHardwareSpec,
	fldPath *field.Path,
) field.ErrorList {
	if spec.SKU != nil && !aisimulate.IsKnownHardwareSKU(*spec.SKU) {
		v.warnf(
			"%s: %q is not known to the AISimulate release bundled with this operator; the value is passed through unchanged and may work with an external AISimulate systems catalog",
			fldPath.Child("sku"), *spec.SKU,
		)
	}
	return nil
}

// validateAISimSpec validates native AISimulate overrides. input and fldPath must not be nil.
func validateAISimSpec(
	input *nvidiacomv1beta2.AISimSpec,
	fldPath *field.Path,
) field.ErrorList {
	// Use the lowering package's decoder as the single interpretation of RawExtension.
	overrides, err := aisimulate.InputOverrides(&nvidiacomv1beta2.NativeSearchInput{AISim: input})
	if err != nil || overrides == nil {
		// The lowering validation reports malformed override documents once.
		return nil
	}

	dgdrRunValue, found := overrides["dgdr_run"]
	if !found {
		return nil
	}
	dgdrRun, ok := dgdrRunValue.(map[string]any)
	if !ok {
		return field.ErrorList{field.Forbidden(
			fldPath.Child("inputOverrides", "dgdr_run"),
			"must be an object because the operator owns required execution fields within it",
		)}
	}

	// Search overrides remain authoritative; only execution values coupled to the
	// controller-owned Job and shared volume are reserved.
	reserved := []string{
		"derive_runtime_images",
		"is_pareto",
		"max_candidates",
		"name",
		"namespace",
		"num_gpus_per_node",
		"pareto_objectives",
		"runtime_image",
		"runtime_version_override",
		"snapshot_dir",
	}
	allErrs := field.ErrorList{}
	for _, key := range reserved {
		if _, found := dgdrRun[key]; found {
			allErrs = append(allErrs, field.Forbidden(
				fldPath.Child("inputOverrides", "dgdr_run", key),
				"is managed by the operator and cannot be overridden",
			))
		}
	}
	return allErrs
}

// maxTraceClaimSubPathLength is the CRD maxLength of
// spec.workload.trace.source.persistentVolumeClaim.subPath.
const maxTraceClaimSubPathLength = 1024

// validateWorkloadSpec validates workload, a type that Requests and Runs share. The CRD schema and
// CEL enforce the same rules on every v1beta2 write; this recursion reaches a Request whose v1beta1
// carrier was stored without admission and that CEL ratcheting then lets through. workload and
// fldPath must not be nil.
func validateWorkloadSpec(workload *nvidiacomv1beta2.WorkloadSpec, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	if workload.Trace == nil {
		return allErrs
	}
	if workload.Static != nil {
		allErrs = append(allErrs, field.Forbidden(fldPath, "static and trace are mutually exclusive"))
	}
	return append(allErrs, validateTraceWorkloadSpec(workload.Trace, fldPath.Child("trace"))...)
}

// validateTraceWorkloadSpec validates trace. Its format is not checked here: the CRD schema
// defaults and enumerates it, and lowering rejects a format it cannot translate. trace and fldPath
// must not be nil.
func validateTraceWorkloadSpec(trace *nvidiacomv1beta2.TraceWorkloadSpec, fldPath *field.Path) field.ErrorList {
	allErrs := validateTraceSourceSpec(&trace.Source, fldPath.Child("source"))
	if trace.Replay != nil {
		allErrs = append(allErrs, validateTraceReplaySpec(trace.Replay, fldPath.Child("replay"))...)
	}
	return allErrs
}

// validateTraceSourceSpec validates source. source and fldPath must not be nil.
func validateTraceSourceSpec(source *nvidiacomv1beta2.TraceSourceSpec, fldPath *field.Path) field.ErrorList {
	if source.PersistentVolumeClaim == nil {
		return field.ErrorList{field.Required(fldPath.Child("persistentVolumeClaim"), "")}
	}
	return validateTracePVCSource(source.PersistentVolumeClaim, fldPath.Child("persistentVolumeClaim"))
}

// validateTracePVCSource validates claim: a PersistentVolumeClaim name and one nameable file inside
// it. claim and fldPath must not be nil.
func validateTracePVCSource(claim *nvidiacomv1beta2.TracePVCSource, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	for _, msg := range validation.IsDNS1123Subdomain(claim.ClaimName) {
		allErrs = append(allErrs, field.Invalid(fldPath.Child("claimName"), claim.ClaimName, msg))
	}

	// The subPath names one file below the claim root: relative, with only named segments.
	subPath := fldPath.Child("subPath")
	if utf8.RuneCountInString(claim.SubPath) > maxTraceClaimSubPathLength {
		allErrs = append(allErrs, field.TooLongCharacters(subPath, claim.SubPath, maxTraceClaimSubPathLength))
	}
	if !traceClaimPathHasOnlyNamedSegments(claim.SubPath) {
		allErrs = append(allErrs, field.Invalid(subPath, claim.SubPath,
			"subPath must be relative to the claim root without empty, '.' or '..' segments"))
	}
	if strings.ContainsRune(claim.SubPath, 0) {
		allErrs = append(allErrs, field.Invalid(subPath, claim.SubPath, "subPath must not contain a NUL character"))
	}
	return allErrs
}

// validateTraceReplaySpec validates replay. replay and fldPath must not be nil.
func validateTraceReplaySpec(replay *nvidiacomv1beta2.TraceReplaySpec, fldPath *field.Path) field.ErrorList {
	if replay.ArrivalSpeedupRatio != nil && *replay.ArrivalSpeedupRatio <= 0 {
		return field.ErrorList{field.Invalid(fldPath.Child("arrivalSpeedupRatio"), *replay.ArrivalSpeedupRatio,
			"must be greater than 0")}
	}
	return nil
}
