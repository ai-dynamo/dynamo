/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package compatibility

import (
	"testing"

	semver "github.com/Masterminds/semver/v3"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/runtimeversion"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

type testResource struct {
	metav1.Object
	runtime *runtimeversion.Version
	optedIn bool
}

func (r testResource) RuntimeVersion() *runtimeversion.Version {
	return r.runtime
}

func TestGateEvaluate(t *testing.T) {
	t.Log("define the compatibility thresholds")
	minimumOrigin := semver.MustParse("1.4.0")
	minimumRuntime := runtimeversion.Version{Major: 1, Minor: 4, Patch: 0}

	t.Log("define a gate whose opt-in can bypass only the origin constraint")
	gate := Gate[testResource]{
		Name:              "TestFeature",
		MinOriginVersion:  minimumOrigin,
		MinRuntimeVersion: &minimumRuntime,
		OptIn: func(resource testResource) bool {
			return resource.optedIn
		},
	}

	t.Log("define the runtime, origin, and opt-in decision matrix")
	tests := []struct {
		name    string
		runtime *runtimeversion.Version
		origin  string
		optedIn bool
		want    Decision
	}{
		{
			name:    "unknown runtime is disabled",
			origin:  "1.4.0",
			optedIn: true,
			want: Decision{
				Status: DecisionDisabled,
				Reason: ReasonRuntimeVersionUnsupported,
			},
		},
		{
			name:    "older runtime is disabled despite opt-in",
			runtime: &runtimeversion.Version{Major: 1, Minor: 3, Patch: 9},
			origin:  "1.4.0",
			optedIn: true,
			want: Decision{
				Status: DecisionDisabled,
				Reason: ReasonRuntimeVersionUnsupported,
			},
		},
		{
			name:    "supported runtime and new origin are enabled",
			runtime: &runtimeversion.Version{Major: 1, Minor: 4, Patch: 0},
			origin:  "1.4.0",
			want: Decision{
				Status: DecisionEnabled,
				Reason: ReasonConstraintsSatisfied,
			},
		},
		{
			name:    "supported runtime and older origin are pending",
			runtime: &runtimeversion.Version{Major: 1, Minor: 4, Patch: 0},
			origin:  "1.3.0",
			want: Decision{
				Status: DecisionPending,
				Reason: ReasonOriginVersionUnsupported,
			},
		},
		{
			name:    "supported runtime and unknown origin are pending",
			runtime: &runtimeversion.Version{Major: 1, Minor: 4, Patch: 0},
			want: Decision{
				Status: DecisionPending,
				Reason: ReasonOriginVersionUnsupported,
			},
		},
		{
			name:    "invalid origin is pending",
			runtime: &runtimeversion.Version{Major: 1, Minor: 4, Patch: 0},
			origin:  "not-a-version",
			want: Decision{
				Status: DecisionPending,
				Reason: ReasonOriginVersionUnsupported,
			},
		},
		{
			name:    "opt-in bypasses an older origin",
			runtime: &runtimeversion.Version{Major: 1, Minor: 4, Patch: 0},
			origin:  "1.3.0",
			optedIn: true,
			want: Decision{
				Status: DecisionEnabled,
				Reason: ReasonExplicitOptIn,
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Log("build a resource with Kubernetes metadata and a resolved runtime")
			metadata := &metav1.ObjectMeta{}
			if tt.origin != "" {
				metadata.Annotations = map[string]string{
					consts.KubeAnnotationDynamoOperatorOriginVersion: tt.origin,
				}
			}
			resource := testResource{
				Object:  metadata,
				runtime: tt.runtime,
				optedIn: tt.optedIn,
			}

			t.Log("evaluate the resource against both compatibility constraints")
			got := gate.Evaluate(resource)

			t.Log("compare the typed status and reason")
			if got != tt.want {
				t.Fatalf("Evaluate() = %+v, want %+v", got, tt.want)
			}
		})
	}
}

func TestGateEvaluateOptionalConstraints(t *testing.T) {
	t.Log("evaluate a gate without version constraints")
	unconstrained := Gate[testResource]{Name: "Unconstrained"}.Evaluate(testResource{
		Object: &metav1.ObjectMeta{},
	})

	t.Log("verify an unconstrained gate is enabled")
	if !unconstrained.Enabled() {
		t.Fatalf("unconstrained gate = %+v, want enabled", unconstrained)
	}

	t.Log("evaluate a gate with only a runtime constraint")
	runtimeOnly := Gate[testResource]{
		Name:              "RuntimeOnly",
		MinRuntimeVersion: &runtimeversion.Version{Major: 1, Minor: 4, Patch: 0},
	}.Evaluate(testResource{
		Object:  &metav1.ObjectMeta{},
		runtime: &runtimeversion.Version{Major: 1, Minor: 4, Patch: 0},
	})

	t.Log("verify the runtime-only gate is enabled at its threshold")
	if !runtimeOnly.Enabled() {
		t.Fatalf("runtime-only gate = %+v, want enabled", runtimeOnly)
	}

	t.Log("evaluate a gate with only an origin constraint")
	originOnly := Gate[testResource]{
		Name:             "OriginOnly",
		MinOriginVersion: semver.MustParse("1.0.0"),
	}.Evaluate(testResource{
		Object: &metav1.ObjectMeta{
			Annotations: map[string]string{
				consts.KubeAnnotationDynamoOperatorOriginVersion: "2.0.0",
			},
		},
	})

	t.Log("verify the origin-only gate is enabled above its threshold")
	if !originOnly.Enabled() {
		t.Fatalf("origin-only gate = %+v, want enabled", originOnly)
	}
}

func TestCanaryHealthChecksPolicy(t *testing.T) {
	t.Log("create the central canary gate with a concrete opt-in predicate")
	gate := CanaryHealthChecks(func(resource testResource) bool {
		return resource.optedIn
	})

	t.Log("verify the central origin and runtime thresholds")
	if gate.MinOriginVersion.String() != "1.4.0" {
		t.Fatalf("MinOriginVersion = %s, want 1.4.0", gate.MinOriginVersion)
	}
	if gate.MinRuntimeVersion.String() != "1.4.0" {
		t.Fatalf("MinRuntimeVersion = %s, want 1.4.0", gate.MinRuntimeVersion)
	}
}
