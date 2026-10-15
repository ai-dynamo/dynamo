// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package features

import (
	"context"
	"errors"
	"strings"
	"testing"
	"time"

	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
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

// testGroveOperatorDeployment returns a grove-operator Deployment labelled as
// the Grove chart labels it, running image.
func testGroveOperatorDeployment(image string) *appsv1.Deployment {
	return &appsv1.Deployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:      groveControllerName,
			Namespace: "grove-system",
			Labels:    map[string]string{"app.kubernetes.io/name": groveControllerName},
		},
		Spec: appsv1.DeploymentSpec{Template: corev1.PodTemplateSpec{Spec: corev1.PodSpec{
			Containers: []corev1.Container{{Name: groveControllerName, Image: image}},
		}}},
	}
}

func newGroveCompatibilityTestScheme(t *testing.T) *runtime.Scheme {
	t.Helper()
	scheme := runtime.NewScheme()
	if err := apiextensionsv1.AddToScheme(scheme); err != nil {
		t.Fatalf("add apiextensions scheme: %v", err)
	}
	if err := appsv1.AddToScheme(scheme); err != nil {
		t.Fatalf("add apps scheme: %v", err)
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
		listErr      error
		wantState    GroveCompatibilityState
		wantContains []string
	}{
		{
			name:      "new controller and CRD with per-clique rollingUpdate is compatible",
			objects:   []client.Object{testPodCliqueSetCRD(true), testGroveOperatorDeployment("ghcr.io/ai-dynamo/grove/grove-operator:v0.1.0-alpha.14")},
			wantState: GroveCompatible,
		},
		{
			name:         "CRD with per-clique rollingUpdate and no controller Deployment is compatible",
			objects:      []client.Object{testPodCliqueSetCRD(true)},
			wantState:    GroveCompatible,
			wantContains: []string{"controller version could not be identified"},
		},
		{
			name:         "old controller tag with a new CRD is incompatible",
			objects:      []client.Object{testPodCliqueSetCRD(true), testGroveOperatorDeployment("ghcr.io/ai-dynamo/grove/grove-operator:v0.1.0-alpha.11")},
			wantState:    GroveIncompatible,
			wantContains: []string{"External Grove v0.1.0-alpha.11 is incompatible", "v0.1.0-alpha.14", "Upgrade the Grove controller and matching CRDs"},
		},
		{
			name:         "unparsable controller tag with a new CRD is compatible",
			objects:      []client.Object{testPodCliqueSetCRD(true), testGroveOperatorDeployment("registry.example.com/grove-operator:v0.1.0-dev")},
			wantState:    GroveCompatible,
			wantContains: []string{"controller version could not be identified"},
		},
		{
			name:         "forbidden Deployment list keeps the CRD verdict",
			objects:      []client.Object{testPodCliqueSetCRD(true)},
			listErr:      apierrors.NewForbidden(schema.GroupResource{Group: "apps", Resource: "deployments"}, "", nil),
			wantState:    GroveCompatible,
			wantContains: []string{"controller version could not be identified"},
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
			funcs := interceptor.Funcs{}
			if tt.getErr != nil {
				funcs.Get = func(context.Context, client.WithWatch, client.ObjectKey, client.Object, ...client.GetOption) error {
					return tt.getErr
				}
			}
			if tt.listErr != nil {
				funcs.List = func(context.Context, client.WithWatch, client.ObjectList, ...client.ListOption) error {
					return tt.listErr
				}
			}
			probe := NewGroveCompatibilityProbe(builder.WithInterceptorFuncs(funcs).Build())

			t.Log("Probe Grove compatibility")
			got, err := probe.Check(t.Context())
			if err != nil {
				t.Fatalf("Check() error = %v", err)
			}

			t.Log("Verify the state and the actionable message")
			if got.State != tt.wantState {
				t.Fatalf("Check().State = %q, want %q (message %q)", got.State, tt.wantState, got.Message)
			}
			if tt.wantState == GroveCompatible && len(tt.wantContains) == 0 && got.Message != "" {
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
	if got := mustCheck(t, probe).State; got != GroveIncompatible {
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
	if got := mustCheck(t, probe).State; got != GroveIncompatible {
		t.Fatalf("cached Check().State = %q, want %q", got, GroveIncompatible)
	}

	t.Log("Advance past the TTL and verify the upgrade is observed")
	now = now.Add(groveCompatibilityTTL)
	if got := mustCheck(t, probe).State; got != GroveCompatible {
		t.Fatalf("rechecked Check().State = %q, want %q", got, GroveCompatible)
	}
}

func TestNilGroveCompatibilityProbeIsCompatible(t *testing.T) {
	var probe *GroveCompatibilityProbe
	if got := mustCheck(t, probe).State; got != GroveCompatible {
		t.Fatalf("nil probe Check().State = %q, want %q", got, GroveCompatible)
	}
}

func TestGroveCompatibilityProbeDoesNotCacheTransientErrors(t *testing.T) {
	t.Log("Serve a new CRD through a reader whose CRD reads fail transiently")
	transient := apierrors.NewServiceUnavailable("etcd leader changed")
	failing := true
	reader := fake.NewClientBuilder().
		WithScheme(newGroveCompatibilityTestScheme(t)).
		WithObjects(testPodCliqueSetCRD(true)).
		WithInterceptorFuncs(interceptor.Funcs{
			Get: func(ctx context.Context, c client.WithWatch, key client.ObjectKey, obj client.Object, opts ...client.GetOption) error {
				if failing {
					return transient
				}
				return c.Get(ctx, key, obj, opts...)
			},
		}).
		Build()
	probe := NewGroveCompatibilityProbe(reader)
	now := time.Unix(1000, 0)
	probe.now = func() time.Time { return now }

	t.Log("Verify a transient error with no previous verdict is returned, not reported as Unverifiable")
	got, err := probe.Check(t.Context())
	if !errors.Is(err, transient) {
		t.Fatalf("Check() = (%+v, %v), want the transient error", got, err)
	}
	if got.State == GroveUnverifiable {
		t.Fatalf("Check().State = %q, want no verdict for a transient error", got.State)
	}

	t.Log("Verify the error was not cached: the next call within the TTL probes again")
	failing = false
	if got := mustCheck(t, probe).State; got != GroveCompatible {
		t.Fatalf("Check().State after recovery = %q, want %q", got, GroveCompatible)
	}

	t.Log("Verify a transient error after the TTL keeps the previous verdict")
	failing = true
	now = now.Add(2 * groveCompatibilityTTL)
	if got := mustCheck(t, probe).State; got != GroveCompatible {
		t.Fatalf("Check().State on transient error = %q, want previous %q", got, GroveCompatible)
	}
}

func TestGroveControllerTagParsing(t *testing.T) {
	tests := []struct {
		image  string
		wantN  int
		wantOK bool
	}{
		{image: "ghcr.io/ai-dynamo/grove/grove-operator:v0.1.0-alpha.11", wantN: 11, wantOK: true},
		{image: "ghcr.io/ai-dynamo/grove/grove-operator:v0.1.0-alpha.14-rc1", wantN: 14, wantOK: true},
		{image: "localhost:5000/grove-operator:v0.1.0-alpha.9@sha256:abc", wantN: 9, wantOK: true},
		{image: "localhost:5000/grove-operator", wantOK: false},
		{image: "ghcr.io/ai-dynamo/grove/grove-operator@sha256:abc", wantOK: false},
		{image: "ghcr.io/ai-dynamo/grove/grove-operator:v0.1.0-dev", wantOK: false},
		{image: "grove-operator:custom", wantOK: false},
	}
	for _, tt := range tests {
		n, ok := groveAlphaNumber(imageTag(tt.image))
		if ok != tt.wantOK || n != tt.wantN {
			t.Errorf("groveAlphaNumber(imageTag(%q)) = (%d, %v), want (%d, %v)", tt.image, n, ok, tt.wantN, tt.wantOK)
		}
	}
}

func mustCheck(t *testing.T, probe *GroveCompatibilityProbe) GroveCompatibility {
	t.Helper()
	got, err := probe.Check(t.Context())
	if err != nil {
		t.Fatalf("Check() error = %v", err)
	}
	return got
}
