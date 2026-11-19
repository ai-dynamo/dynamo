/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package validation_test

import (
	"testing"

	v1beta2 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta2"
	"k8s.io/apimachinery/pkg/runtime"
)

func TestDynamoGraphDeploymentCandidateValidator_Validate(t *testing.T) {
	candidate := func(spec string) *v1beta2.DynamoGraphDeploymentCandidate {
		return &v1beta2.DynamoGraphDeploymentCandidate{
			Name: "candidate", Namespace: "default",
			Spec: runtime.RawExtension{Raw: []byte(spec)},
		}
	}
	tests := []struct {
		name       string
		oldSpec    string
		spec       string
		wantErrors []string
	}{
		{name: "invalid DGD content stays unpruned", spec: `{"components":"wrong-type","unknown":9007199254740993}`},
		{name: "unchanged opaque spec permits metadata update", oldSpec: `{"unknown":{"a":1,"b":2}}`, spec: `{"unknown":{"b":2,"a":1}}`},
		{name: "unknown field mutation is immutable", oldSpec: `{"unknown":1}`, spec: `{"unknown":2}`, wantErrors: []string{"spec: Forbidden: spec is immutable"}},
		{name: "large integers are distinct", oldSpec: `{"unknown":9007199254740992}`, spec: `{"unknown":9007199254740993}`, wantErrors: []string{"spec: Forbidden: spec is immutable"}},
		{name: "removing an opaque field is immutable", oldSpec: `{"unknown":true}`, spec: `{}`, wantErrors: []string{"spec: Forbidden: spec is immutable"}},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			current := candidate(test.spec)
			current.Labels = map[string]string{"updated": "true"}
			admission := admissionTestCase{object: current, wantWebhookErrors: test.wantErrors}
			if test.oldSpec != "" {
				admission.oldObject = candidate(test.oldSpec)
			}
			got := runAdmissionTest(t, admission)
			if got != nil && len(test.wantErrors) == 0 {
				if _, found := got.Object["spec"]; !found {
					t.Fatal("opaque spec was pruned")
				}
			}
		})
	}
}
