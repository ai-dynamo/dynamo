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

// PLACEHOLDER -- not the real v1beta2 API (and deliberately not under api/, so it is not a served
// or converted API version). This package exists only so the DGDR(v2)
// run publisher (internal/dgdrrunpublisher) has something to compile against before the
// real v1beta2 types exist on this branch. The real, reviewed types are PRs #13603
// ("feat: add api for dgdr v1beta2") and #13744 ("feat: dgdr v1beta2 conversion"),
// still feature-gated/inactive when this file was written. The shapes below follow the
// review of #15554: a candidate is an immutable evaluated point (spec = the DGD fields
// plus the resolved evaluation parameters, status = metrics + evaluation conditions)
// and ordering lives on the run's status, never on the candidate. Delete this package when the real types land;
// only import paths change.
//
// Deliberately NOT included: conversion webhooks, the full Request type, and
// deepcopy-gen output (DeepCopyObject is hand-written here).
package placeholderapi

import (
	v1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
)

// GroupVersion is the group version used to register these placeholder objects.
var GroupVersion = schema.GroupVersion{Group: "nvidia.com", Version: "v1beta2"}

// AddToScheme registers the placeholder types with the given scheme.
func AddToScheme(scheme *runtime.Scheme) error {
	scheme.AddKnownTypes(GroupVersion,
		&DynamoGraphDeploymentCandidate{},
		&DynamoGraphDeploymentCandidateList{},
		&DynamoGraphDeploymentRun{},
		&DynamoGraphDeploymentRunList{},
	)
	metav1.AddToGroupVersion(scheme, GroupVersion)
	return nil
}

// DynamoGraphDeploymentCandidateSpec is the immutable evaluated point: the flat DGD
// fields plus the resolved evaluation parameters.
type DynamoGraphDeploymentCandidateSpec struct {
	v1beta1.DynamoGraphDeploymentSpec `json:",inline"`

	// Parameters are the resolved evaluation parameters of the point.
	// +optional
	Parameters *runtime.RawExtension `json:"parameters,omitempty"`
}

// DynamoGraphDeploymentCandidateStatus records evaluation results. It is populated once
// after creation and never updated afterwards; it carries no rank.
type DynamoGraphDeploymentCandidateStatus struct {
	// Metrics are the evaluation metrics of the point.
	// +optional
	Metrics *runtime.RawExtension `json:"metrics,omitempty"`

	// Conditions describes evaluation and materialization.
	// +optional
	Conditions []metav1.Condition `json:"conditions,omitempty"`
}

// DynamoGraphDeploymentCandidate is one bounded, user-visible search result.
type DynamoGraphDeploymentCandidate struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	Spec   DynamoGraphDeploymentCandidateSpec   `json:"spec"`
	Status DynamoGraphDeploymentCandidateStatus `json:"status,omitempty"`
}

type DynamoGraphDeploymentCandidateList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []DynamoGraphDeploymentCandidate `json:"items"`
}

// CandidateRef is an ordered reference from a run to one of its candidates.
type CandidateRef struct {
	Name string `json:"name"`
}

// RunProgress is the Sweeper progress reported by the snapshot.
type RunProgress struct {
	Round     int32 `json:"round"`
	Evaluated int32 `json:"evaluated"`
}

// DynamoGraphDeploymentRunSpec is intentionally empty in this placeholder: the real
// spec is an alias of the request spec (see #13603).
type DynamoGraphDeploymentRunSpec struct{}

// DynamoGraphDeploymentRunStatus holds everything that changes while a search runs.
// Rank is the order of CandidateRefs (best first for scalar searches). The terminal
// Completed condition is derived by the run controller from the Job result, not
// written by the publisher.
type DynamoGraphDeploymentRunStatus struct {
	// Message is a human-readable summary of the latest state or failure.
	// +optional
	Message string `json:"message,omitempty"`

	// Progress is the latest Sweeper progress.
	// +optional
	Progress *RunProgress `json:"progress,omitempty"`

	// LastProgressTime is when the Sweeper last published new state.
	// +optional
	LastProgressTime *metav1.Time `json:"lastProgressTime,omitempty"`

	// CandidateRefs lists the run's candidates in rank order.
	// +optional
	CandidateRefs []CandidateRef `json:"candidateRefs,omitempty"`

	// Conditions describes the run.
	// +optional
	Conditions []metav1.Condition `json:"conditions,omitempty"`
}

// DynamoGraphDeploymentRun is one execution of a search request.
type DynamoGraphDeploymentRun struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	Spec   DynamoGraphDeploymentRunSpec   `json:"spec,omitempty"`
	Status DynamoGraphDeploymentRunStatus `json:"status,omitempty"`
}

type DynamoGraphDeploymentRunList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []DynamoGraphDeploymentRun `json:"items"`
}

// -- hand-written runtime.Object plumbing (deepcopy-gen output in the real PR) --

func (in *DynamoGraphDeploymentCandidateSpec) DeepCopyInto(out *DynamoGraphDeploymentCandidateSpec) {
	*out = *in
	in.DynamoGraphDeploymentSpec.DeepCopyInto(&out.DynamoGraphDeploymentSpec)
	if in.Parameters != nil {
		out.Parameters = in.Parameters.DeepCopy()
	}
}

func (in *DynamoGraphDeploymentCandidateStatus) DeepCopyInto(out *DynamoGraphDeploymentCandidateStatus) {
	*out = *in
	if in.Metrics != nil {
		out.Metrics = in.Metrics.DeepCopy()
	}
	if in.Conditions != nil {
		out.Conditions = make([]metav1.Condition, len(in.Conditions))
		for i := range in.Conditions {
			in.Conditions[i].DeepCopyInto(&out.Conditions[i])
		}
	}
}

func (in *DynamoGraphDeploymentCandidate) DeepCopyInto(out *DynamoGraphDeploymentCandidate) {
	*out = *in
	out.TypeMeta = in.TypeMeta
	in.ObjectMeta.DeepCopyInto(&out.ObjectMeta)
	in.Spec.DeepCopyInto(&out.Spec)
	in.Status.DeepCopyInto(&out.Status)
}

func (in *DynamoGraphDeploymentCandidate) DeepCopy() *DynamoGraphDeploymentCandidate {
	if in == nil {
		return nil
	}
	out := new(DynamoGraphDeploymentCandidate)
	in.DeepCopyInto(out)
	return out
}

func (in *DynamoGraphDeploymentCandidate) DeepCopyObject() runtime.Object { return in.DeepCopy() }

func (in *DynamoGraphDeploymentCandidateList) DeepCopyInto(out *DynamoGraphDeploymentCandidateList) {
	*out = *in
	out.TypeMeta = in.TypeMeta
	out.ListMeta = in.ListMeta
	if in.Items != nil {
		out.Items = make([]DynamoGraphDeploymentCandidate, len(in.Items))
		for i := range in.Items {
			in.Items[i].DeepCopyInto(&out.Items[i])
		}
	}
}

func (in *DynamoGraphDeploymentCandidateList) DeepCopy() *DynamoGraphDeploymentCandidateList {
	if in == nil {
		return nil
	}
	out := new(DynamoGraphDeploymentCandidateList)
	in.DeepCopyInto(out)
	return out
}

func (in *DynamoGraphDeploymentCandidateList) DeepCopyObject() runtime.Object { return in.DeepCopy() }

func (in *DynamoGraphDeploymentRunStatus) DeepCopyInto(out *DynamoGraphDeploymentRunStatus) {
	*out = *in
	if in.Progress != nil {
		progress := *in.Progress
		out.Progress = &progress
	}
	if in.LastProgressTime != nil {
		out.LastProgressTime = in.LastProgressTime.DeepCopy()
	}
	if in.CandidateRefs != nil {
		out.CandidateRefs = append([]CandidateRef(nil), in.CandidateRefs...)
	}
	if in.Conditions != nil {
		out.Conditions = make([]metav1.Condition, len(in.Conditions))
		for i := range in.Conditions {
			in.Conditions[i].DeepCopyInto(&out.Conditions[i])
		}
	}
}

func (in *DynamoGraphDeploymentRun) DeepCopyInto(out *DynamoGraphDeploymentRun) {
	*out = *in
	out.TypeMeta = in.TypeMeta
	in.ObjectMeta.DeepCopyInto(&out.ObjectMeta)
	out.Spec = in.Spec
	in.Status.DeepCopyInto(&out.Status)
}

func (in *DynamoGraphDeploymentRun) DeepCopy() *DynamoGraphDeploymentRun {
	if in == nil {
		return nil
	}
	out := new(DynamoGraphDeploymentRun)
	in.DeepCopyInto(out)
	return out
}

func (in *DynamoGraphDeploymentRun) DeepCopyObject() runtime.Object { return in.DeepCopy() }

func (in *DynamoGraphDeploymentRunList) DeepCopyInto(out *DynamoGraphDeploymentRunList) {
	*out = *in
	out.TypeMeta = in.TypeMeta
	out.ListMeta = in.ListMeta
	if in.Items != nil {
		out.Items = make([]DynamoGraphDeploymentRun, len(in.Items))
		for i := range in.Items {
			in.Items[i].DeepCopyInto(&out.Items[i])
		}
	}
}

func (in *DynamoGraphDeploymentRunList) DeepCopy() *DynamoGraphDeploymentRunList {
	if in == nil {
		return nil
	}
	out := new(DynamoGraphDeploymentRunList)
	in.DeepCopyInto(out)
	return out
}

func (in *DynamoGraphDeploymentRunList) DeepCopyObject() runtime.Object { return in.DeepCopy() }
