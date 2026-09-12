/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package controller

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestScaleElasticEP_ContractChecks(t *testing.T) {
	tests := []struct {
		name       string
		reply      map[string]any
		httpStatus int
		wantErr    bool
		wantBusy   bool
	}{
		{
			name:  "ok and echoed size matches -> success",
			reply: map[string]any{"status": "ok", "new_data_parallel_size": 3},
		},
		{
			name:    "ok but echoed a different size -> failure (endpoint takes a target, not a delta)",
			reply:   map[string]any{"status": "ok", "new_data_parallel_size": 2},
			wantErr: true,
		},
		{
			name:     "busy -> requeue signal, not a hard failure",
			reply:    map[string]any{"status": "busy"},
			wantErr:  true,
			wantBusy: true,
		},
		{
			name:    "error status -> failure",
			reply:   map[string]any{"status": "error", "message": "boom"},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.URL.Path != epScaleRoute {
					t.Errorf("unexpected path %q", r.URL.Path)
				}
				var body map[string]any
				_ = json.NewDecoder(r.Body).Decode(&body)
				if body["new_data_parallel_size"] == nil {
					t.Error("request must carry new_data_parallel_size")
				}
				if tt.httpStatus != 0 {
					w.WriteHeader(tt.httpStatus)
				}
				_ = json.NewEncoder(w).Encode(tt.reply)
			}))
			defer srv.Close()

			err := scaleElasticEP(context.Background(), srv.Client(), srv.URL, 3)
			if (err != nil) != tt.wantErr {
				t.Fatalf("scaleElasticEP err = %v, wantErr = %v", err, tt.wantErr)
			}
			if tt.wantBusy && !errors.Is(err, errEngineBusy) {
				t.Errorf("busy reply should surface errEngineBusy so the caller requeues; got %v", err)
			}
		})
	}
}

func TestReadEPCapacity(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != epCapacityRoute {
			t.Errorf("unexpected path %q", r.URL.Path)
		}
		_ = json.NewEncoder(w).Encode(map[string]any{
			"status": "ok", "data_parallel_size": 2, "tensor_parallel_size": 1,
			"data_parallel_backend": "ray",
			"nodes":                 []map[string]any{{"node_ip": "10.0.0.1", "total_gpus": 4}},
		})
	}))
	defer srv.Close()

	cap, err := readEPCapacity(context.Background(), srv.Client(), srv.URL)
	if err != nil {
		t.Fatalf("readEPCapacity: %v", err)
	}
	if cap.DataParallelSize != 2 || cap.DataParallelBackend != "ray" || len(cap.Nodes) != 1 {
		t.Errorf("unexpected capacity: %+v", cap)
	}
}

func TestElasticEPLeaderBaseURL(t *testing.T) {
	got := elasticEPLeaderBaseURL("Leader", "tzulingk-ft-tests")
	want := "http://leader-ray.tzulingk-ft-tests.svc:9090"
	if got != want {
		t.Errorf("elasticEPLeaderBaseURL = %q, want %q", got, want)
	}
}
