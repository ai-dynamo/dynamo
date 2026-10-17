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

package dgdrinput

import (
	"fmt"
	"sort"

	v1beta2 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta2"
	"k8s.io/apimachinery/pkg/util/validation/field"
)

// Conflicts rejects typed objective fields that disagree with spec.aisimulate.parameters.goal.
//
// Only explicitly supplied typed values count. The API server applies the CRD defaults
// before admission, so the webhook cannot see which values the caller wrote. A typed value
// is therefore treated as explicit when it is one the schema never fills in: a metric
// (admission skips its default when the parameters choose goal.target), a Pareto mode, or
// an SLA that differs from the schema default. A caller who writes the default value
// explicitly is treated as not having written it, so the native value wins through the
// merge. That keeps defaults from creating false conflicts, at the price of not catching
// that one case.
func Conflicts(spec *v1beta2.DynamoGraphDeploymentRequestSpec) field.ErrorList {
	objective := spec.Objective
	patch, err := Parameters(spec)
	goal := asMap(patch["goal"])
	if err != nil || objective == nil || goal == nil {
		return nil
	}

	var errs field.ErrorList
	path := field.NewPath("spec", "objective")
	nativePath := "spec.aisimulate.parameters.goal."

	target, hasTarget := goal["target"].(string)
	if objective.Metric != nil && hasTarget && target != nativeMetric(*objective.Metric) {
		errs = append(errs, field.Invalid(path.Child("metric"), string(*objective.Metric),
			fmt.Sprintf("conflicts with %starget %q", nativePath, target)))
	}
	if objective.Mode == v1beta2.ObjectiveModePareto {
		if hasTarget && target != nativePareto {
			errs = append(errs, field.Invalid(path.Child("mode"), string(objective.Mode),
				fmt.Sprintf("conflicts with %starget %q", nativePath, target)))
		}
		if native, ok := goal["pareto_objectives"].([]any); ok && !sameStrings(native, nativeMetrics(objective.Metrics)) {
			errs = append(errs, field.Invalid(path.Child("metrics"), objective.Metrics,
				"conflicts with "+nativePath+"pareto_objectives"))
		}
	}
	errs = append(errs, slaConflicts(objective.SLA, asMap(goal["sla"]), path.Child("sla"), nativePath+"sla")...)
	return errs
}

func slaConflicts(typed *v1beta2.ObjectiveSLASpec, native map[string]any, path *field.Path, nativePath string) field.ErrorList {
	if typed == nil || native == nil || isDefaultSLA(typed) {
		return nil
	}
	var errs field.ErrorList
	bounds := []struct {
		key   string
		typed *float64
	}{
		{"ttft_ms", typed.TTFTMs},
		{"itl_ms", typed.ITLMs},
		{"e2e_ms", typed.E2EMs},
	}
	disabled := typed.TTFTMs == nil && typed.ITLMs == nil && typed.E2EMs == nil
	for _, bound := range bounds {
		value, ok := number(native[bound.key])
		switch {
		case !ok:
		case disabled:
			errs = append(errs, field.Invalid(path, "{}", fmt.Sprintf("disables SLA bounds but %s.%s is %v", nativePath, bound.key, value)))
		case bound.typed != nil && *bound.typed != value:
			errs = append(errs, field.Invalid(path, *bound.typed, fmt.Sprintf("conflicts with %s.%s %v", nativePath, bound.key, value)))
		}
	}
	return errs
}

func isDefaultSLA(sla *v1beta2.ObjectiveSLASpec) bool {
	return sla.E2EMs == nil &&
		sla.TTFTMs != nil && *sla.TTFTMs == DefaultTTFTMs &&
		sla.ITLMs != nil && *sla.ITLMs == DefaultITLMs
}

func sameStrings(a, b []any) bool {
	toSet := func(values []any) []string {
		out := make([]string, 0, len(values))
		for _, value := range values {
			s, _ := value.(string)
			out = append(out, s)
		}
		sort.Strings(out)
		return out
	}
	left, right := toSet(a), toSet(b)
	if len(left) != len(right) {
		return false
	}
	for i := range left {
		if left[i] != right[i] {
			return false
		}
	}
	return true
}
