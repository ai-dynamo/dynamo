/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package features

import (
	"context"
	"strings"
	"testing"
	"time"

	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
	"sigs.k8s.io/controller-runtime/pkg/client/interceptor"
)

// testPodCliqueSetCRD returns a trimmed podcliquesets.grove.io CRD whose
// served v1alpha1 clique items carry rollingUpdate only when withMarker is set.
func testPodCliqueSetCRD(withMarker bool) *apiextensionsv1.CustomResourceDefinition {
	cliqueItem := apiextensionsv1.JSONSchemaProps{
		Type: "object",
		Properties: map[string]apiextensionsv1.JSONSchemaProps{
			"name": {Type: "string"},
			"spec": {Type: "object"},
		},
	}
	if withMarker {
		cliqueItem.Properties["rollingUpdate"] = apiextensionsv1.JSONSchemaProps{Type: "object"}
	}
	return &apiextensionsv1.CustomResourceDefinition{
		ObjectMeta: metav1.ObjectMeta{Name: grovePodCliqueSetCRDName},
		Spec: apiextensionsv1.CustomResourceDefinitionSpec{
			Group: "grove.io",
			Versions: []apiextensionsv1.CustomResourceDefinitionVersion{{
				Name:    groveAPIVersion,
				Served:  true,
				Storage: true,
				Schema: &apiextensionsv1.CustomResourceValidation{OpenAPIV3Schema: &apiextensionsv1.JSONSchemaProps{
					Type: "object",
					Properties: map[string]apiextensionsv1.JSONSchemaProps{
						"spec": {Type: "object", Properties: map[string]apiextensionsv1.JSONSchemaProps{
							"template": {Type: "object", Properties: map[string]apiextensionsv1.JSONSchemaProps{
								"cliques": {Type: "array", Items: &apiextensionsv1.JSONSchemaPropsOrArray{Schema: &cliqueItem}},
							}},
						}},
					},
				}},
			}},
		},
	}
}

func newGroveCompatibilityTestScheme(t *testing.T) *runtime.Scheme {
	t.Helper()
	scheme := runtime.NewScheme()
	if err := apiextensionsv1.AddToScheme(scheme); err != nil {
		t.Fatalf("add apiextensions scheme: %v", err)
	}
	return scheme
}

func TestGroveCompatibilityProbeCheck(t *testing.T) {
	forbidden := apierrors.NewForbidden(
		schema.GroupResource{Group: "apiextensions.k8s.io", Resource: "customresourcedefinitions"},
		grovePodCliqueSetCRDName,
		nil,
	)
	tests := []struct {
		name         string
		objects      []client.Object
		getErr       error
		wantState    GroveCompatibilityState
		wantContains []string
	}{
		{
			name:      "CRD with per-clique rollingUpdate is compatible",
			objects:   []client.Object{testPodCliqueSetCRD(true)},
			wantState: GroveCompatible,
		},
		{
			name:         "CRD without per-clique rollingUpdate is incompatible",
			objects:      []client.Object{testPodCliqueSetCRD(false)},
			wantState:    GroveIncompatible,
			wantContains: []string{"incompatible", "v0.1.0-alpha.14", "Upgrade the Grove controller and matching CRDs"},
		},
		{
			name:         "missing CRD is unverifiable",
			wantState:    GroveUnverifiable,
			wantContains: []string{"could not be verified", "CRD was not found", "v0.1.0-alpha.14"},
		},
		{
			name:         "forbidden CRD read is unverifiable",
			getErr:       forbidden,
			wantState:    GroveUnverifiable,
			wantContains: []string{"could not be verified", "lacks permission", "v0.1.0-alpha.14"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Log("Build a reader that serves the fixture CRD or the injected error")
			builder := fake.NewClientBuilder().
				WithScheme(newGroveCompatibilityTestScheme(t)).
				WithObjects(tt.objects...)
			if tt.getErr != nil {
				builder = builder.WithInterceptorFuncs(interceptor.Funcs{
					Get: func(context.Context, client.WithWatch, client.ObjectKey, client.Object, ...client.GetOption) error {
						return tt.getErr
					},
				})
			}
			probe := NewGroveCompatibilityProbe(builder.Build())

			t.Log("Probe Grove compatibility")
			got := probe.Check(t.Context())

			t.Log("Verify the state and the actionable message")
			if got.State != tt.wantState {
				t.Fatalf("Check().State = %q, want %q (message %q)", got.State, tt.wantState, got.Message)
			}
			if tt.wantState == GroveCompatible && got.Message != "" {
				t.Errorf("Check().Message = %q, want empty for a compatible Grove", got.Message)
			}
			for _, want := range tt.wantContains {
				if !strings.Contains(got.Message, want) {
					t.Errorf("Check().Message = %q, want it to contain %q", got.Message, want)
				}
			}
		})
	}
}

func TestGroveCompatibilityProbeRechecksAfterTTL(t *testing.T) {
	t.Log("Start with an old Grove CRD and a controllable clock")
	reader := fake.NewClientBuilder().
		WithScheme(newGroveCompatibilityTestScheme(t)).
		WithObjects(testPodCliqueSetCRD(false)).
		Build()
	probe := NewGroveCompatibilityProbe(reader)
	now := time.Unix(1000, 0)
	probe.now = func() time.Time { return now }
	if got := probe.Check(t.Context()).State; got != GroveIncompatible {
		t.Fatalf("initial Check().State = %q, want %q", got, GroveIncompatible)
	}

	t.Log("Upgrade the CRD and verify the cached result is reused within the TTL")
	upgraded := testPodCliqueSetCRD(true)
	current := &apiextensionsv1.CustomResourceDefinition{}
	if err := reader.Get(t.Context(), client.ObjectKey{Name: grovePodCliqueSetCRDName}, current); err != nil {
		t.Fatalf("get CRD: %v", err)
	}
	current.Spec = upgraded.Spec
	if err := reader.Update(t.Context(), current); err != nil {
		t.Fatalf("update CRD: %v", err)
	}
	now = now.Add(groveCompatibilityTTL / 2)
	if got := probe.Check(t.Context()).State; got != GroveIncompatible {
		t.Fatalf("cached Check().State = %q, want %q", got, GroveIncompatible)
	}

	t.Log("Advance past the TTL and verify the upgrade is observed")
	now = now.Add(groveCompatibilityTTL)
	if got := probe.Check(t.Context()).State; got != GroveCompatible {
		t.Fatalf("rechecked Check().State = %q, want %q", got, GroveCompatible)
	}
}

func TestNilGroveCompatibilityProbeIsCompatible(t *testing.T) {
	var probe *GroveCompatibilityProbe
	if got := probe.Check(t.Context()).State; got != GroveCompatible {
		t.Fatalf("nil probe Check().State = %q, want %q", got, GroveCompatible)
	}
}
