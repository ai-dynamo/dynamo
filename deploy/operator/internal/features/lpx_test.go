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

func TestResolveLPX(t *testing.T) {
	const (
		lprPath    = "/apis/scheduling.lpu.nvidia.com/v1alpha1"
		missingLPR = "LPX is explicitly enabled in config but the scheduling.lpu.nvidia.com/v1alpha1 LpuPipelineRequest API was not detected in the cluster"
	)
	tests := []struct {
		name, path        string
		status            int
		resource, wantErr string
	}{
		{name: "LPR API present"},
		{name: "LPR group version absent", path: lprPath, status: http.StatusNotFound, wantErr: missingLPR},
		{name: "LPR resource absent", path: lprPath, status: http.StatusOK, wantErr: missingLPR},
		{name: "LPR subresource alone is insufficient", path: lprPath, status: http.StatusOK, resource: "lpupipelinerequests/status", wantErr: missingLPR},
		{name: "LPR discovery forbidden", path: lprPath, status: http.StatusForbidden, wantErr: "discover scheduling.lpu.nvidia.com/v1alpha1 API resources: Forbidden"},
		{name: "LPR discovery fails", path: lprPath, status: http.StatusInternalServerError, wantErr: "discover scheduling.lpu.nvidia.com/v1alpha1 API resources: Internal Server Error"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Log("Serve the exact LPX API discovery path")
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.URL.Path != lprPath {
					t.Errorf("unexpected discovery path %q", r.URL.Path)
					http.NotFound(w, r)
					return
				}
				resource := "lpupipelinerequests"
				status := http.StatusOK
				if tt.path != "" {
					status, resource = tt.status, tt.resource
				}
				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(status)
				var response any = metav1.Status{TypeMeta: metav1.TypeMeta{Kind: "Status", APIVersion: "v1"}, Status: metav1.StatusFailure, Code: int32(status), Message: http.StatusText(status)}
				if status == http.StatusOK {
					resources := []metav1.APIResource{}
					if resource != "" {
						resources = append(resources, metav1.APIResource{Name: resource})
					}
					response = metav1.APIResourceList{TypeMeta: metav1.TypeMeta{Kind: "APIResourceList", APIVersion: "v1"}, GroupVersion: strings.TrimPrefix(r.URL.Path, "/apis/"), APIResources: resources}
				}
				if err := json.NewEncoder(w).Encode(response); err != nil {
					t.Errorf("write discovery response: %v", err)
				}
			}))
			defer server.Close()

			t.Log("Require the LPR API and preserve discovery failures")
			enabled, err := resolveLPX(t.Context(), &rest.Config{Host: server.URL})
			if enabled != (tt.wantErr == "") || (err == nil) != (tt.wantErr == "") {
				t.Fatalf("resolveLPX() = %v, %v; want enabled=%v, error=%q", enabled, err, tt.wantErr == "", tt.wantErr)
			}
			if err != nil && err.Error() != tt.wantErr {
				t.Errorf("resolveLPX() error = %q, want %q", err, tt.wantErr)
			}
		})
	}
}
