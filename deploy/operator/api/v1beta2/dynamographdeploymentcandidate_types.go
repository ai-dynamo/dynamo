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

// PLACEHOLDER -- not the real v1beta2 API. This package exists only so the DGDC
// reconciler in this PR (internal/controller/dgdc_reconciler.go) has something to
// compile against before the real v1beta2 types exist on this branch. The real,
// reviewed types are PRs #13603 ("feat: add api for dgdr v1beta2") and #13744
// ("feat: dgdr v1beta2 conversion") by ashnamehrotra, still feature-gated/inactive at
// the time this file was written. DynamoGraphDeploymentCandidateStatus below is a
// field-for-field match of that PR's real type (same JSON shape, same Rank semantics),
// so once #13603/#13744 merge, delete this whole package and the reconciler needs no
// changes beyond its import path -- it was written against this shape specifically so
// that swap is mechanical.
//
// Deliberately NOT included here (out of scope for a placeholder): the
// DynamoGraphDeploymentRun type, the v1beta1<->v1beta2 DynamoGraphDeploymentRequest
// conversion webhook, and deepcopy-gen output -- DeepCopyObject below is written by
// hand instead of generated, since this package will not outlive the real PRs
// landing.
package v1beta2

import (
	v1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
)

// GroupVersion is the group version used to register these placeholder objects.
// Matches the real PR's intended group version exactly, so nothing about the
// reconciler's use of the typed client needs to change once the placeholder is
// deleted.
var GroupVersion = schema.GroupVersion{Group: "nvidia.com", Version: "v1beta2"}

// DynamoGraphDeploymentCandidateGVK is the v1beta2 DynamoGraphDeploymentCandidate kind.
var DynamoGraphDeploymentCandidateGVK = GroupVersion.WithKind("DynamoGraphDeploymentCandidate")

// AddToScheme registers the placeholder types with the given scheme.
func AddToScheme(scheme *runtime.Scheme) error {
	scheme.AddKnownTypes(GroupVersion,
		&DynamoGraphDeploymentCandidate{},
		&DynamoGraphDeploymentCandidateList{},
	)
	metav1.AddToGroupVersion(scheme, GroupVersion)
	return nil
}

// DynamoGraphDeploymentCandidateStatus describes simulation and materialization,
// never deployment health. Field-for-field identical to PR #13744's real type.
type DynamoGraphDeploymentCandidateStatus struct {
	// Rank is the one-based scalar ordering and is absent for Pareto searches.
	// +optional
	Rank *int32 `json:"rank,omitempty"`

	// Conditions describes evaluation and materialization.
	// +optional
	Conditions []metav1.Condition `json:"conditions,omitempty"`

	// Experimental contains Sweeper-version-specific diagnostics and round-trips
	// without a nested CRD schema.
	// +optional
	Experimental *runtime.RawExtension `json:"experimental,omitempty"`
}

// DynamoGraphDeploymentCandidate is one bounded, user-visible search result. Its
// spec is exactly the v1beta1 DynamoGraphDeploymentSpec schema, matching PR #13744.
type DynamoGraphDeploymentCandidate struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	Spec   v1beta1.DynamoGraphDeploymentSpec    `json:"spec"`
	Status DynamoGraphDeploymentCandidateStatus `json:"status,omitempty"`
}

// DynamoGraphDeploymentCandidateList contains a list of
// DynamoGraphDeploymentCandidate resources.
type DynamoGraphDeploymentCandidateList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []DynamoGraphDeploymentCandidate `json:"items"`
}

// -- hand-written runtime.Object plumbing (deepcopy-gen output in the real PR) --

func (in *DynamoGraphDeploymentCandidateStatus) DeepCopyInto(out *DynamoGraphDeploymentCandidateStatus) {
	*out = *in
	if in.Rank != nil {
		rank := *in.Rank
		out.Rank = &rank
	}
	if in.Conditions != nil {
		out.Conditions = make([]metav1.Condition, len(in.Conditions))
		for i := range in.Conditions {
			in.Conditions[i].DeepCopyInto(&out.Conditions[i])
		}
	}
	if in.Experimental != nil {
		out.Experimental = in.Experimental.DeepCopy()
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

func (in *DynamoGraphDeploymentCandidate) DeepCopyObject() runtime.Object {
	return in.DeepCopy()
}

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

func (in *DynamoGraphDeploymentCandidateList) DeepCopyObject() runtime.Object {
	return in.DeepCopy()
}
