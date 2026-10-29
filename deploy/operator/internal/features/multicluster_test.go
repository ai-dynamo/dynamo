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
		mcsPathPrefix = "/apis/multicluster.x-k8s.io/"
		missingAPI    = "multiCluster.mcsImplementation is set but neither the multicluster.x-k8s.io/v1alpha1 " +
			"nor the multicluster.x-k8s.io/v1beta1 ServiceExport API was detected in the cluster"
	)
	type discovery struct {
		status   int
		resource string
	}
	serviceExports := discovery{status: http.StatusOK, resource: "serviceexports"}
	serviceImports := discovery{status: http.StatusOK, resource: "serviceimports"}
	forbidden := discovery{status: http.StatusForbidden}
	tests := []struct {
		name string
		// versions maps each multicluster.x-k8s.io version the cluster serves to
		// its discovery response; any other version is absent.
		versions    map[string]discovery
		wantVersion string
		wantErr     string
	}{
		{
			name:        "both versions served",
			versions:    map[string]discovery{"v1alpha1": serviceExports, "v1beta1": serviceExports},
			wantVersion: "v1alpha1",
		},
		{
			name:        "only v1alpha1 served",
			versions:    map[string]discovery{"v1alpha1": serviceExports},
			wantVersion: "v1alpha1",
		},
		{
			name:        "only v1beta1 served",
			versions:    map[string]discovery{"v1beta1": serviceExports},
			wantVersion: "v1beta1",
		},
		{
			name:    "no version served",
			wantErr: missingAPI,
		},
		{
			name:     "only ServiceImport served",
			versions: map[string]discovery{"v1alpha1": serviceImports, "v1beta1": serviceImports},
			wantErr:  missingAPI,
		},
		{
			name:        "v1alpha1 serves only ServiceImport",
			versions:    map[string]discovery{"v1alpha1": serviceImports, "v1beta1": serviceExports},
			wantVersion: "v1beta1",
		},
		{
			name:     "v1alpha1 discovery forbidden",
			versions: map[string]discovery{"v1alpha1": forbidden, "v1beta1": serviceExports},
			wantErr:  "discover multicluster.x-k8s.io/v1alpha1 API resources: Forbidden",
		},
		{
			name:     "v1beta1 discovery forbidden",
			versions: map[string]discovery{"v1beta1": forbidden},
			wantErr:  "discover multicluster.x-k8s.io/v1beta1 API resources: Forbidden",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Log("Serve the multicluster.x-k8s.io discovery path of each version")
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				version, found := strings.CutPrefix(r.URL.Path, mcsPathPrefix)
				if !found {
					t.Errorf("unexpected discovery path %q", r.URL.Path)
					http.NotFound(w, r)
					return
				}
				response, served := tt.versions[version]
				if !served {
					response = discovery{status: http.StatusNotFound}
				}
				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(response.status)
				var body any = metav1.Status{
					Kind: "Status", APIVersion: "v1", Status: metav1.StatusFailure,
					Code: int32(response.status), Message: http.StatusText(response.status),
				}
				if response.status == http.StatusOK {
					body = metav1.APIResourceList{
						Kind:         "APIResourceList",
						APIVersion:   "v1",
						GroupVersion: strings.TrimPrefix(r.URL.Path, "/apis/"),
						APIResources: []metav1.APIResource{{Name: response.resource}},
					}
				}
				if err := json.NewEncoder(w).Encode(body); err != nil {
					t.Errorf("write discovery response: %v", err)
				}
			}))
			defer server.Close()

			t.Log("Resolve the preferred served ServiceExport version and preserve discovery failures")
			version, err := resolveMultiClusterExport(t.Context(), &rest.Config{Host: server.URL})
			if version != tt.wantVersion {
				t.Errorf("resolveMultiClusterExport() version = %q, want %q", version, tt.wantVersion)
			}
			if (err == nil) != (tt.wantErr == "") || (err != nil && err.Error() != tt.wantErr) {
				t.Errorf("resolveMultiClusterExport() error = %v, want %q", err, tt.wantErr)
			}
		})
	}
}
