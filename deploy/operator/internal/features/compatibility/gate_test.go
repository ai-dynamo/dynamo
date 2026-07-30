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
)

type testResource struct {
	runtime *runtimeversion.Version
	origin  *semver.Version
	optedIn bool
}

func (r testResource) RuntimeVersion() *runtimeversion.Version {
	return r.runtime
}

func (r testResource) OriginVersion() *semver.Version {
	return r.origin
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
		name     string
		resource testResource
		want     Decision
	}{
		{
			name: "unknown runtime is disabled",
			resource: testResource{
				origin:  semver.MustParse("1.4.0"),
				optedIn: true,
			},
			want: Decision{
				Status: DecisionDisabled,
				Reason: ReasonRuntimeVersionUnsupported,
			},
		},
		{
			name: "older runtime is disabled despite opt-in",
			resource: testResource{
				runtime: &runtimeversion.Version{Major: 1, Minor: 3, Patch: 9},
				origin:  semver.MustParse("1.4.0"),
				optedIn: true,
			},
			want: Decision{
				Status: DecisionDisabled,
				Reason: ReasonRuntimeVersionUnsupported,
			},
		},
		{
			name: "supported runtime and new origin are enabled",
			resource: testResource{
				runtime: &runtimeversion.Version{Major: 1, Minor: 4, Patch: 0},
				origin:  semver.MustParse("1.4.0"),
			},
			want: Decision{
				Status: DecisionEnabled,
				Reason: ReasonConstraintsSatisfied,
			},
		},
		{
			name: "supported runtime and older origin are pending",
			resource: testResource{
				runtime: &runtimeversion.Version{Major: 1, Minor: 4, Patch: 0},
				origin:  semver.MustParse("1.3.0"),
			},
			want: Decision{
				Status: DecisionPending,
				Reason: ReasonOriginVersionUnsupported,
			},
		},
		{
			name: "supported runtime and unknown origin are pending",
			resource: testResource{
				runtime: &runtimeversion.Version{Major: 1, Minor: 4, Patch: 0},
			},
			want: Decision{
				Status: DecisionPending,
				Reason: ReasonOriginVersionUnsupported,
			},
		},
		{
			name: "opt-in bypasses an older origin",
			resource: testResource{
				runtime: &runtimeversion.Version{Major: 1, Minor: 4, Patch: 0},
				origin:  semver.MustParse("1.3.0"),
				optedIn: true,
			},
			want: Decision{
				Status: DecisionEnabled,
				Reason: ReasonExplicitOptIn,
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Log("evaluate the resource against both compatibility constraints")
			got := gate.Evaluate(tt.resource)

			t.Log("compare the typed status and reason")
			if got != tt.want {
				t.Fatalf("Evaluate() = %+v, want %+v", got, tt.want)
			}
		})
	}
}

func TestGateEvaluateOptionalConstraints(t *testing.T) {
	t.Log("define an origin version above every test threshold")
	newOrigin := semver.MustParse("2.0.0")

	t.Log("evaluate a gate without version constraints")
	unconstrained := Gate[testResource]{Name: "Unconstrained"}.Evaluate(testResource{})

	t.Log("verify an unconstrained gate is enabled")
	if !unconstrained.Enabled() {
		t.Fatalf("unconstrained gate = %+v, want enabled", unconstrained)
	}

	t.Log("evaluate a gate with only a runtime constraint")
	runtimeOnly := Gate[testResource]{
		Name:              "RuntimeOnly",
		MinRuntimeVersion: &runtimeversion.Version{Major: 1, Minor: 4, Patch: 0},
	}.Evaluate(testResource{
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
	}.Evaluate(testResource{origin: newOrigin})

	t.Log("verify the origin-only gate is enabled above its threshold")
	if !originOnly.Enabled() {
		t.Fatalf("origin-only gate = %+v, want enabled", originOnly)
	}
}

func TestVersionsFromAnnotations(t *testing.T) {
	t.Log("define annotation parsing cases")
	tests := []struct {
		name        string
		annotations map[string]string
		want        string
	}{
		{
			name: "missing origin annotation",
		},
		{
			name: "invalid origin annotation",
			annotations: map[string]string{
				consts.KubeAnnotationDynamoOperatorOriginVersion: "not-a-version",
			},
		},
		{
			name: "valid origin annotation",
			annotations: map[string]string{
				consts.KubeAnnotationDynamoOperatorOriginVersion: "1.4.0",
			},
			want: "1.4.0",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Log("resolve the version carrier from annotations")
			versions := VersionsFromAnnotations(tt.annotations)

			t.Log("compare the resolved origin version")
			if tt.want == "" {
				if versions.OriginVersion() != nil {
					t.Fatalf("OriginVersion() = %v, want nil", versions.OriginVersion())
				}

				return
			}
			if versions.OriginVersion() == nil || versions.OriginVersion().String() != tt.want {
				t.Fatalf("OriginVersion() = %v, want %s", versions.OriginVersion(), tt.want)
			}
		})
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
