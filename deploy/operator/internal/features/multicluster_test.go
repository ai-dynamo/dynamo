// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package features

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/rest"
	ctrl "sigs.k8s.io/controller-runtime"
	metricsserver "sigs.k8s.io/controller-runtime/pkg/metrics/server"
)

func TestNewResolvesMultiClusterExport(t *testing.T) {
	t.Log("Serve discovery for a cluster whose only API group is multicluster.x-k8s.io, with ServiceExports at v1alpha1")
	mcs := metav1.GroupVersionForDiscovery{GroupVersion: "multicluster.x-k8s.io/v1alpha1", Version: "v1alpha1"}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var response any
		switch r.URL.Path {
		case "/api":
			response = metav1.APIVersions{Kind: "APIVersions", Versions: []string{"v1"}}
		case "/apis":
			response = metav1.APIGroupList{
				Kind: "APIGroupList", APIVersion: "v1",
				Groups: []metav1.APIGroup{{
					Name: "multicluster.x-k8s.io", Versions: []metav1.GroupVersionForDiscovery{mcs}, PreferredVersion: mcs,
				}},
			}
		case "/apis/multicluster.x-k8s.io/v1alpha1":
			response = metav1.APIResourceList{
				Kind: "APIResourceList", APIVersion: "v1",
				GroupVersion: mcs.GroupVersion,
				APIResources: []metav1.APIResource{{Name: "serviceexports", Namespaced: true, Kind: "ServiceExport"}},
			}
		default:
			http.NotFound(w, r)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		if err := json.NewEncoder(w).Encode(response); err != nil {
			t.Errorf("write discovery response: %v", err)
		}
	}))
	defer server.Close()
	manager, err := ctrl.NewManager(&rest.Config{Host: server.URL}, ctrl.Options{
		Metrics: metricsserver.Options{BindAddress: "0"},
	})
	if err != nil {
		t.Fatal(err)
	}

	for _, tc := range []struct {
		implementation configv1alpha1.MCSImplementation
		wantVersion    string
	}{
		{implementation: configv1alpha1.MCSImplementationNone},
		{implementation: configv1alpha1.MCSImplementationStandard, wantVersion: "v1alpha1"},
		{implementation: configv1alpha1.MCSImplementationKarmada, wantVersion: "v1alpha1"},
	} {
		t.Logf("Resolve the gates with mcsImplementation %q: a configured implementation turns the gate on", tc.implementation)
		gates, err := New(t.Context(), manager, &configv1alpha1.OperatorConfiguration{
			MultiCluster: configv1alpha1.MultiClusterConfiguration{MCSImplementation: tc.implementation},
		})
		if err != nil {
			t.Fatalf("New() with %q: %v", tc.implementation, err)
		}
		wantGate := tc.implementation != configv1alpha1.MCSImplementationNone
		if gates.MultiClusterExport != wantGate || gates.ServiceExportVersion != tc.wantVersion {
			t.Errorf("New() with %q: multiClusterExport %v at %q, want %v at %q",
				tc.implementation, gates.MultiClusterExport, gates.ServiceExportVersion, wantGate, tc.wantVersion)
		}
	}
}

func TestResolveMultiClusterExport(t *testing.T) {
	const (
		mcsPath    = "/apis/multicluster.x-k8s.io/v1alpha1"
		missingAPI = "multiCluster.mcsImplementation is set but the multicluster.x-k8s.io/v1alpha1 ServiceExport API was not detected in the cluster"
	)
	tests := []struct {
		name     string
		status   int
		resource string
		wantErr  string
	}{
		{name: "ServiceExport API present", status: http.StatusOK, resource: "serviceexports"},
		{name: "group version absent", status: http.StatusNotFound, wantErr: missingAPI},
		{name: "only ServiceImport served", status: http.StatusOK, resource: "serviceimports", wantErr: missingAPI},
		{name: "discovery forbidden", status: http.StatusForbidden, wantErr: "discover multicluster.x-k8s.io/v1alpha1 API resources: Forbidden"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Log("Serve the multicluster.x-k8s.io/v1alpha1 discovery path")
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.URL.Path != mcsPath {
					t.Errorf("unexpected discovery path %q", r.URL.Path)
					http.NotFound(w, r)
					return
				}
				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(tt.status)
				var response any = metav1.Status{Kind: "Status", APIVersion: "v1", Status: metav1.StatusFailure, Code: int32(tt.status), Message: http.StatusText(tt.status)}
				if tt.status == http.StatusOK {
					response = metav1.APIResourceList{
						Kind:         "APIResourceList",
						APIVersion:   "v1",
						GroupVersion: strings.TrimPrefix(r.URL.Path, "/apis/"),
						APIResources: []metav1.APIResource{{Name: tt.resource}},
					}
				}
				if err := json.NewEncoder(w).Encode(response); err != nil {
					t.Errorf("write discovery response: %v", err)
				}
			}))
			defer server.Close()

			t.Log("Require the ServiceExport resource, return its version, and preserve discovery failures")
			version, err := resolveMultiClusterExport(t.Context(), &rest.Config{Host: server.URL})
			wantVersion := ""
			if tt.wantErr == "" {
				wantVersion = "v1alpha1"
			}
			if version != wantVersion || (err == nil) != (tt.wantErr == "") {
				t.Fatalf("resolveMultiClusterExport() = %q, %v; want %q, error=%q", version, err, wantVersion, tt.wantErr)
			}
			if err != nil && err.Error() != tt.wantErr {
				t.Errorf("resolveMultiClusterExport() error = %q, want %q", err, tt.wantErr)
			}
		})
	}
}
