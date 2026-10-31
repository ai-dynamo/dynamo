// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package features

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/rest"
)

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
