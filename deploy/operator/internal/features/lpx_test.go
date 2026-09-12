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
		lprPath        = "/apis/scheduling.lpu.nvidia.com/v1alpha1"
		podGangPath    = "/apis/scheduler.grove.io/v1alpha1"
		missingLPR     = "LPX is explicitly enabled in config but the scheduling.lpu.nvidia.com/v1alpha1 LpuPipelineRequest API was not detected in the cluster"
		missingPodGang = "LPX is explicitly enabled in config but the scheduler.grove.io/v1alpha1 PodGang API was not detected in the cluster"
	)
	tests := []struct {
		name, path        string
		status            int
		resource, wantErr string
	}{
		{name: "both APIs present"},
		{name: "LPR resource absent", path: lprPath, status: http.StatusOK, wantErr: missingLPR},
		{name: "PodGang group version absent", path: podGangPath, status: http.StatusNotFound, wantErr: missingPodGang},
		{name: "PodGang resource absent", path: podGangPath, status: http.StatusOK, wantErr: missingPodGang},
		{name: "PodGang subresource alone is insufficient", path: podGangPath, status: http.StatusOK, resource: "podgangs/status", wantErr: missingPodGang},
		{name: "PodGang discovery forbidden", path: podGangPath, status: http.StatusForbidden, wantErr: "discover scheduler.grove.io/v1alpha1 API resources: Forbidden"},
		{name: "PodGang discovery fails", path: podGangPath, status: http.StatusInternalServerError, wantErr: "discover scheduler.grove.io/v1alpha1 API resources: Internal Server Error"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Log("Serve the exact LPX and PodGang API discovery paths")
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				resource := "lpupipelinerequests"
				switch r.URL.Path {
				case lprPath:
				case podGangPath:
					resource = "podgangs"
				default:
					t.Errorf("unexpected discovery path %q", r.URL.Path)
					http.NotFound(w, r)
					return
				}
				status := http.StatusOK
				if r.URL.Path == tt.path {
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

			t.Log("Require both APIs and preserve discovery failures")
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
