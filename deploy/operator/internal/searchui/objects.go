/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package searchui

import (
	"encoding/json"
	"strconv"
	"time"

	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
)

// The UI reads Runs and Candidates as unstructured objects so it keeps working across the
// publisher versions found in the wild: status.aisim.input was status.aisimulate.input
// before #17950, and Candidate evaluation parameters moved from spec to status with the
// DGD-override work. Each accessor below tries the current field first, then the older one.

var (
	gvkRequest   = schema.GroupVersionKind{Group: "nvidia.com", Version: "v1beta2", Kind: "DynamoGraphDeploymentRequest"}
	gvkRun       = schema.GroupVersionKind{Group: "nvidia.com", Version: "v1beta2", Kind: "DynamoGraphDeploymentRun"}
	gvkCandidate = schema.GroupVersionKind{Group: "nvidia.com", Version: "v1beta2", Kind: "DynamoGraphDeploymentCandidate"}
	gvkDGD       = schema.GroupVersionKind{Group: "nvidia.com", Version: "v1beta1", Kind: "DynamoGraphDeployment"}
)

const (
	labelCandidateID  = "nvidia.com/dgdr-candidate-id"
	labelRunUID       = "nvidia.com/dgdr-run-uid"
	annotationFromDGD = "nvidia.com/dgdr-candidate"
	conditionValid    = "Valid"
)

func str(obj map[string]any, fields ...string) string {
	v, _, _ := unstructured.NestedString(obj, fields...)
	return v
}

func nestedMap(obj map[string]any, fields ...string) map[string]any {
	v, found, err := unstructured.NestedMap(obj, fields...)
	if !found || err != nil {
		return nil
	}
	return v
}

func nestedSlice(obj map[string]any, fields ...string) []any {
	v, found, err := unstructured.NestedSlice(obj, fields...)
	if !found || err != nil {
		return nil
	}
	return v
}

// number reads an integral or floating JSON number, however the decoder typed it.
func number(obj map[string]any, fields ...string) (float64, bool) {
	v, found, err := unstructured.NestedFieldNoCopy(obj, fields...)
	if !found || err != nil {
		return 0, false
	}
	return toFloat(v)
}

func toFloat(v any) (float64, bool) {
	switch n := v.(type) {
	case float64:
		return n, true
	case float32:
		return float64(n), true
	case int64:
		return float64(n), true
	case int32:
		return float64(n), true
	case int:
		return float64(n), true
	case json.Number:
		f, err := n.Float64()
		return f, err == nil
	case string:
		f, err := strconv.ParseFloat(n, 64)
		return f, err == nil
	}
	return 0, false
}

func parseTime(s string) *time.Time {
	if s == "" {
		return nil
	}
	t, err := time.Parse(time.RFC3339, s)
	if err != nil {
		return nil
	}
	return &t
}

type condition struct {
	Type    string `json:"type"`
	Status  string `json:"status"`
	Reason  string `json:"reason,omitempty"`
	Message string `json:"message,omitempty"`
	Time    string `json:"lastTransitionTime,omitempty"`
}

func conditions(obj map[string]any) []condition {
	var out []condition
	for _, item := range nestedSlice(obj, "status", "conditions") {
		m, ok := item.(map[string]any)
		if !ok {
			continue
		}
		out = append(out, condition{
			Type: str(m, "type"), Status: str(m, "status"), Reason: str(m, "reason"),
			Message: str(m, "message"), Time: str(m, "lastTransitionTime"),
		})
	}
	return out
}

func ownerOfKind(obj map[string]any, kind string) (name, uid string) {
	for _, item := range nestedSlice(obj, "metadata", "ownerReferences") {
		m, ok := item.(map[string]any)
		if ok && str(m, "kind") == kind {
			return str(m, "name"), str(m, "uid")
		}
	}
	return "", ""
}

// --- Run accessors ---

func runRequestName(run map[string]any) string {
	name, _ := ownerOfKind(run, gvkRequest.Kind)
	return name
}

func runAISimInput(run map[string]any) map[string]any {
	if m := nestedMap(run, "status", "aisim", "input"); m != nil {
		return m
	}
	return nestedMap(run, "status", "aisimulate", "input")
}

func runBackends(run map[string]any) []string {
	var out []string
	for _, b := range nestedSlice(run, "spec", "backends") {
		if s, ok := b.(string); ok {
			out = append(out, s)
		}
	}
	return out
}

func runHardwareSKU(run map[string]any) string {
	if sku := str(run, "spec", "hardware", "gpu", "sku"); sku != "" {
		return sku
	}
	return str(runAISimInput(run), "engine", "hardware")
}

// --- Candidate accessors ---

func candidateRunUID(cand map[string]any) string {
	if uid := str(cand, "metadata", "labels", labelRunUID); uid != "" {
		return uid
	}
	_, uid := ownerOfKind(cand, gvkRun.Kind)
	return uid
}

func candidateRunName(cand map[string]any) string {
	name, _ := ownerOfKind(cand, gvkRun.Kind)
	return name
}

func candidateParameters(cand map[string]any) map[string]any {
	if m := nestedMap(cand, "status", "parameters"); m != nil {
		return m
	}
	return nestedMap(cand, "spec", "parameters")
}

func candidateMetrics(cand map[string]any) map[string]any {
	return nestedMap(cand, "status", "metrics")
}

// candidateDGDSpec is the deployable DGD spec: the candidate spec without the legacy
// inline evaluation parameters.
func candidateDGDSpec(cand map[string]any) map[string]any {
	spec := nestedMap(cand, "spec")
	if spec == nil {
		return nil
	}
	delete(spec, "parameters")
	return spec
}

// validity is "True", "False" or "NotEvaluated" (no Valid condition: the publisher that
// wrote this Candidate predates DGD validation).
func candidateValidity(cand map[string]any) (status, reason, message string) {
	for _, c := range conditions(cand) {
		if c.Type == conditionValid {
			return c.Status, c.Reason, c.Message
		}
	}
	return "NotEvaluated", "", ""
}

func candidateBackend(cand map[string]any) string {
	if b := str(candidateParameters(cand), "backend"); b != "" {
		return b
	}
	return str(cand, "spec", "backendFramework")
}
