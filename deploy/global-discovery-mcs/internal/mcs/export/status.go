// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package export

import (
	"regexp"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	mcsv1 "github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/api/gen/go/dynamo/mcs/v1"
)

// conditionReason is the pattern metav1.Condition requires of reasons.
var conditionReason = regexp.MustCompile(`^[A-Za-z]([A-Za-z0-9_,:]*[A-Za-z0-9_])?$`)

// ConditionFromRegistry converts a registry condition. Reasons the Kubernetes
// API would reject become "Unknown"; a missing transition time is left zero so
// that meta.SetStatusCondition stamps it.
func ConditionFromRegistry(c *mcsv1.Condition) metav1.Condition {
	out := metav1.Condition{Type: c.GetType(), Reason: c.GetReason(), Message: c.GetMessage()}
	switch c.GetStatus() {
	case mcsv1.ConditionStatus_CONDITION_STATUS_TRUE:
		out.Status = metav1.ConditionTrue
	case mcsv1.ConditionStatus_CONDITION_STATUS_FALSE:
		out.Status = metav1.ConditionFalse
	default:
		out.Status = metav1.ConditionUnknown
	}
	if !conditionReason.MatchString(out.Reason) || len(out.Reason) > 1024 {
		out.Reason = "Unknown"
	}
	if len(out.Message) > 32768 {
		out.Message = out.Message[:32768]
	}
	if t := c.GetLastTransitionAt(); t != nil {
		out.LastTransitionTime = metav1.NewTime(t.AsTime())
	}
	return out
}

// registryCondition returns the condition of type t in a registry cluster
// status, or nil.
func registryCondition(st *mcsv1.ClusterStatus, t string) *metav1.Condition {
	for _, c := range st.GetConditions() {
		if c.GetType() == t {
			out := ConditionFromRegistry(c)
			return &out
		}
	}
	return nil
}

// equalConditions compares condition lists ignoring order.
func equalConditions(a, b []metav1.Condition) bool {
	if len(a) != len(b) {
		return false
	}
	byType := make(map[string]metav1.Condition, len(a))
	for _, c := range a {
		byType[c.Type] = c
	}
	for _, c := range b {
		o, ok := byType[c.Type]
		if !ok || o.Status != c.Status || o.Reason != c.Reason || o.Message != c.Message ||
			o.ObservedGeneration != c.ObservedGeneration || !o.LastTransitionTime.Equal(&c.LastTransitionTime) {
			return false
		}
	}
	return true
}
