/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package sglang

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

func TestMetadataClientFetchesSGLangEndpoints(t *testing.T) {
	var paths []string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		paths = append(paths, r.URL.Path)
		if got := r.Header.Get("Authorization"); got != "Bearer test-key" {
			t.Fatalf("Authorization = %q, want bearer api key", got)
		}
		switch r.URL.Path {
		case "/v1/models":
			_, _ = w.Write(fixtureBytes(t, "models.json"))
		case "/model_info":
			_, _ = w.Write(fixtureBytes(t, "model_info_generation.json"))
		case "/server_info":
			_, _ = w.Write(fixtureBytes(t, "server_info_qwen3_huggingface_legacy_kv_events_config.json"))
		default:
			http.NotFound(w, r)
		}
	}))
	defer server.Close()

	client, err := NewMetadataClientWithHTTPClient(server.URL, "test-key", server.Client())
	require.NoError(t, err)
	metadata, err := client.Fetch(context.Background())
	require.NoError(t, err)
	require.Equal(t, []string{"/v1/models", "/model_info", "/server_info"}, paths)
	if len(metadata.Models.Data) != 1 ||
		metadata.Models.Data[0].ID != testQwen3ModelName ||
		metadata.ModelInfo.ModelPath == "" ||
		metadata.ServerInfo.ServedModelName != testQwen3ModelName {
		t.Fatalf("unexpected metadata: %#v", metadata)
	}
}

func TestMetadataClientBuildsURLsWithBasePath(t *testing.T) {
	client, err := NewMetadataClientWithHTTPClient("http://worker.default.svc/api/", "", nil)
	require.NoError(t, err)
	got, err := client.requestURL("/v1/models")
	require.NoError(t, err)
	want := "http://worker.default.svc/api/v1/models"
	if got != want {
		t.Fatalf("requestURL() = %q, want %q", got, want)
	}
}

func TestMetadataClientWrapsHTTPAndJSONErrors(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/v1/models":
			http.Error(w, "boom", http.StatusBadGateway)
		default:
			_, _ = w.Write([]byte(`{bad json`))
		}
	}))
	defer server.Close()

	client, err := NewMetadataClientWithHTTPClient(server.URL, "", server.Client())
	require.NoError(t, err)
	var models ModelsResponse
	err = client.getJSON(context.Background(), "/v1/models", &models)
	require.ErrorContains(t, err, "GET /v1/models")
	require.ErrorContains(t, err, "HTTP 502")
	require.ErrorContains(t, err, "boom")
	var modelInfo ModelInfo
	err = client.getJSON(context.Background(), "/model_info", &modelInfo)
	require.ErrorContains(t, err, "GET /model_info")
	require.ErrorContains(t, err, "JSON")
}

func TestMetadataClientWrapsTimeout(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		time.Sleep(25 * time.Millisecond)
		_, _ = w.Write([]byte(`{}`))
	}))
	defer server.Close()

	client, err := NewMetadataClientWithHTTPClient(server.URL, "", &http.Client{Timeout: time.Millisecond})
	require.NoError(t, err)
	var models ModelsResponse
	err = client.getJSON(context.Background(), "/v1/models", &models)
	require.ErrorContains(t, err, "failed")
}
