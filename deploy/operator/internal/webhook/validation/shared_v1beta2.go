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

// maxTraceClaimPathLength is the CRD maxLength of spec.workload.trace.source.pvc.path.
const maxTraceClaimPathLength = 1024

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

// validateTraceWorkloadSpec validates trace. trace and fldPath must not be nil.
func validateTraceWorkloadSpec(trace *nvidiacomv1beta2.TraceWorkloadSpec, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}

	// The format has no default: AISimulate and the DEP default to mooncake, which is not admitted.
	switch trace.Format {
	case nvidiacomv1beta2.TraceFormatDynamo:
	case "":
		allErrs = append(allErrs, field.Required(fldPath.Child("format"), ""))
	default:
		allErrs = append(allErrs, field.NotSupported(fldPath.Child("format"), trace.Format,
			[]nvidiacomv1beta2.TraceFormat{nvidiacomv1beta2.TraceFormatDynamo}))
	}

	allErrs = append(allErrs, validateTraceSourceSpec(&trace.Source, fldPath.Child("source"))...)
	if trace.Replay != nil {
		allErrs = append(allErrs, validateTraceReplaySpec(trace.Replay, fldPath.Child("replay"))...)
	}
	return allErrs
}

// validateTraceSourceSpec validates source. source and fldPath must not be nil.
func validateTraceSourceSpec(source *nvidiacomv1beta2.TraceSourceSpec, fldPath *field.Path) field.ErrorList {
	if source.PVC == nil {
		return field.ErrorList{field.Required(fldPath.Child("pvc"), "")}
	}
	return validateTracePVCSource(source.PVC, fldPath.Child("pvc"))
}

// validateTracePVCSource validates pvc: a PersistentVolumeClaim name and one nameable file inside
// it. pvc and fldPath must not be nil.
func validateTracePVCSource(pvc *nvidiacomv1beta2.TracePVCSource, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	for _, msg := range validation.IsDNS1123Subdomain(pvc.ClaimName) {
		allErrs = append(allErrs, field.Invalid(fldPath.Child("claimName"), pvc.ClaimName, msg))
	}

	// The path names one file below the claim root: relative, with only named segments.
	pathPath := fldPath.Child("path")
	if utf8.RuneCountInString(pvc.Path) > maxTraceClaimPathLength {
		allErrs = append(allErrs, field.TooLongCharacters(pathPath, pvc.Path, maxTraceClaimPathLength))
	}
	if !traceClaimPathHasOnlyNamedSegments(pvc.Path) {
		allErrs = append(allErrs, field.Invalid(pathPath, pvc.Path,
			"path must be relative to the claim root without empty, '.' or '..' segments"))
	}
	if strings.ContainsRune(pvc.Path, 0) {
		allErrs = append(allErrs, field.Invalid(pathPath, pvc.Path, "path must not contain a NUL character"))
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
