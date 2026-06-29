/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package selectionservice

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"net/url"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

const testWorkerPath = "/workers/7"

func TestClientUpsertsAndDeletesWorkers(t *testing.T) {
	var posted WorkerRequest
	var deletedPath string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.Method == http.MethodPost && r.URL.Path == workersPath:
			if err := json.NewDecoder(r.Body).Decode(&posted); err != nil {
				t.Fatalf("decode posted worker: %v", err)
			}
			w.WriteHeader(http.StatusCreated)
			_, _ = w.Write([]byte(`{
				"worker_id": 7,
				"model_name": "qwen",
				"tenant_id": "tenant-a",
				"lifecycle": "schedulable",
				"endpoint": "http://worker:30000",
				"kv_events_endpoint": "tcp://worker:5557",
				"kv_events_endpoints": {"0": "tcp://worker:5557"},
				"replay_endpoint": "tcp://worker:5560",
				"block_size": 16,
				"data_parallel_start_rank": 0,
				"data_parallel_size": 1,
				"max_num_batched_tokens": 4096,
				"total_kv_blocks": 1024,
				"stable_routing_id": "pod-uid",
				"is_eagle": true,
				"taints": ["dynamo.topology/zone=zone-a"],
				"topology_domains": {"zone": "zone-a"},
				"kv_transfer_domain": "zone",
				"kv_transfer_enforcement": "preferred",
				"kv_transfer_preferred_weight": 0.75,
				"metadata": {"adapter": "external-sglang"},
				"not_schedulable_reasons": ["example"]
			}`))
		case r.Method == http.MethodGet && r.URL.Path == workersPath:
			if r.URL.Query().Get("model_name") != "qwen" || r.URL.Query().Get("tenant_id") != "tenant-a" {
				t.Fatalf("query = %s, want model_name and tenant_id", r.URL.RawQuery)
			}
			_, _ = w.Write([]byte(`[{"worker_id":7,"model_name":"qwen","tenant_id":"tenant-a","metadata":{"adapter":"external-sglang"}}]`))
		case r.Method == http.MethodDelete && r.URL.Path == testWorkerPath:
			deletedPath = r.URL.Path
			_, _ = w.Write([]byte(`{"worker_id":7}`))
		default:
			http.NotFound(w, r)
		}
	}))
	defer server.Close()

	client, err := NewClientWithHTTPClient(server.URL, server.Client())
	require.NoError(t, err)
	upserted, err := client.UpsertWorker(context.Background(), WorkerRequest{WorkerID: 7, ModelName: "qwen"})
	if err != nil {
		t.Fatalf("UpsertWorker() error = %v", err)
	}
	if upserted.WorkerID != 7 ||
		upserted.Lifecycle != WorkerLifecycleSchedulable ||
		upserted.KVEventsEndpoints[0] != "tcp://worker:5557" ||
		upserted.TopologyDomains["zone"] != "zone-a" ||
		upserted.Metadata["adapter"] != "external-sglang" ||
		len(upserted.NotSchedulableReasons) != 1 {
		t.Fatalf("upserted worker = %#v, want selection service catalog record", upserted)
	}
	if posted.WorkerID != 7 || posted.ModelName != "qwen" {
		t.Fatalf("posted worker = %#v", posted)
	}
	workers, err := client.ListWorkers(context.Background(), "qwen", "tenant-a")
	if err != nil {
		t.Fatalf("ListWorkers() error = %v", err)
	}
	if len(workers) != 1 || workers[0].Metadata["adapter"] != "external-sglang" {
		t.Fatalf("workers = %#v, want metadata round trip", workers)
	}
	deleted, err := client.DeleteWorker(context.Background(), 7, true)
	if err != nil {
		t.Fatalf("DeleteWorker() error = %v", err)
	}
	if deleted.WorkerID != 7 {
		t.Fatalf("deleted worker = %#v, want worker ID 7", deleted)
	}
	if deletedPath != testWorkerPath {
		t.Fatalf("deleted path = %q", deletedPath)
	}
}

func TestClientNormalizesWorkerKeyDefaults(t *testing.T) {
	var posted WorkerRequest
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost || r.URL.Path != workersPath {
			http.NotFound(w, r)
			return
		}
		if err := json.NewDecoder(r.Body).Decode(&posted); err != nil {
			t.Fatalf("decode posted worker: %v", err)
		}
		if posted.ModelName != DefaultModelName || posted.TenantID != DefaultTenantID {
			t.Fatalf("posted worker key = model:%q tenant:%q, want defaults", posted.ModelName, posted.TenantID)
		}
		w.WriteHeader(http.StatusCreated)
		if err := json.NewEncoder(w).Encode(WorkerRecord{
			WorkerID:  posted.WorkerID,
			ModelName: posted.ModelName,
			TenantID:  posted.TenantID,
		}); err != nil {
			t.Fatalf("encode worker record: %v", err)
		}
	}))
	defer server.Close()

	client, err := NewClientWithHTTPClient(server.URL, server.Client())
	require.NoError(t, err)
	upserted, err := client.UpsertWorker(context.Background(), WorkerRequest{
		WorkerID:  8,
		ModelName: " ",
	})
	require.NoError(t, err)
	if upserted.ModelName != DefaultModelName || upserted.TenantID != DefaultTenantID {
		t.Fatalf("upserted worker key = model:%q tenant:%q, want defaults", upserted.ModelName, upserted.TenantID)
	}
}

func TestClientBuildsURLsWithBasePathAndQuery(t *testing.T) {
	client, err := NewClientWithHTTPClient("http://selector.default.svc/api/", nil)
	require.NoError(t, err)
	got, err := client.requestURL(workersPath, url.Values{
		"model_name": []string{"qwen/qwen3"},
		"tenant_id":  []string{"tenant-a"},
	})
	if err != nil {
		t.Fatalf("requestURL() error = %v", err)
	}
	want := "http://selector.default.svc/api/workers?model_name=qwen%2Fqwen3&tenant_id=tenant-a"
	if got != want {
		t.Fatalf("requestURL() = %q, want %q", got, want)
	}
}

func TestClientWrapsHTTPErrors(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "boom", http.StatusBadGateway)
	}))
	defer server.Close()

	client, err := NewClientWithHTTPClient(server.URL, server.Client())
	require.NoError(t, err)
	_, err = client.UpsertWorker(context.Background(), WorkerRequest{WorkerID: 7})
	require.ErrorContains(t, err, "HTTP 502")
}

func TestClientAllowsMissingDeleteWhenRequestedAndWrapsTimeout(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == testWorkerPath {
			http.NotFound(w, r)
			return
		}
		time.Sleep(25 * time.Millisecond)
		_, _ = w.Write([]byte(`{}`))
	}))
	defer server.Close()

	client, err := NewClientWithHTTPClient(server.URL, server.Client())
	require.NoError(t, err)
	if _, err := client.DeleteWorker(context.Background(), 7, true); err != nil {
		t.Fatalf("DeleteWorker(missingOK=true) error = %v", err)
	}

	timeoutClient, err := NewClientWithHTTPClient(server.URL, &http.Client{Timeout: time.Millisecond})
	require.NoError(t, err)
	_, err = timeoutClient.UpsertWorker(context.Background(), WorkerRequest{WorkerID: 8})
	require.ErrorContains(t, err, "failed")
}
