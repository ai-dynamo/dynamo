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
	"strings"
	"unicode/utf8"

	nvidiacomv1beta2 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta2"
	"k8s.io/apimachinery/pkg/util/validation"
	"k8s.io/apimachinery/pkg/util/validation/field"
)

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
