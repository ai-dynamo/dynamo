/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package searchui

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/ai-dynamo/dynamo/deploy/operator/internal/searchui/history"
	"github.com/go-logr/logr"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
)

// Fixtures: one Request, one Run in the current API shape, and three Candidates — one in
// the current shape (status.parameters, Valid=True), one in the pre-#17950 shape
// (spec.parameters, no conditions), and one that the publisher later removed.

func fixtureRequest() map[string]any {
	return map[string]any{
		"apiVersion": "nvidia.com/v1beta2", "kind": "DynamoGraphDeploymentRequest",
		"metadata": map[string]any{"name": "pareto-vllm-h100", "namespace": "dynamo-system", "uid": "req-uid"},
		"spec":     map[string]any{"model": map[string]any{"name": "Qwen/Qwen3-8B"}, "objective": map[string]any{"mode": "Pareto"}},
		"status":   map[string]any{"recentRunRefs": []any{map[string]any{"name": "pareto-vllm-h100-1"}}},
	}
}

func fixtureRun(legacyInput bool) map[string]any {
	input := map[string]any{"engine": map[string]any{"hardware": "h100_sxm", "model": "Qwen/Qwen3-8B"}}
	status := map[string]any{
		"phase":          "Succeeded",
		"startTime":      "2026-10-29T14:30:33Z",
		"completionTime": "2026-10-29T14:30:59Z",
		"jobRef":         map[string]any{"name": "pareto-vllm-h100-1-aisimulate"},
		"candidateRefs":  []any{map[string]any{"name": "run-1-aaaa-point-new"}, map[string]any{"name": "run-1-aaaa-point-old"}},
		"progress": map[string]any{
			"candidates":  map[string]any{"published": int64(2), "paretoFront": int64(2)},
			"evaluations": map[string]any{"scheduled": int64(12), "feasible": int64(9), "infeasible": int64(2), "failed": int64(1), "cacheHits": int64(3)},
			"branches":    []any{map[string]any{"name": "vllm", "rounds": map[string]any{"completed": int64(4), "total": int64(6)}}},
		},
		"conditions": []any{map[string]any{"type": "Completed", "status": "True", "reason": "SearchSucceeded"}},
	}
	if legacyInput {
		status["aisimulate"] = map[string]any{"input": input}
	} else {
		status["aisim"] = map[string]any{"input": input}
	}
	return map[string]any{
		"apiVersion": "nvidia.com/v1beta2", "kind": "DynamoGraphDeploymentRun",
		"metadata": map[string]any{
			"name": "pareto-vllm-h100-1", "namespace": "dynamo-system", "uid": "run-uid-1",
			"ownerReferences": []any{map[string]any{"kind": "DynamoGraphDeploymentRequest", "name": "pareto-vllm-h100", "uid": "req-uid"}},
			"managedFields":   []any{map[string]any{"manager": "noise"}},
		},
		"spec": map[string]any{
			"model": map[string]any{"name": "Qwen/Qwen3-8B"}, "backends": []any{"vllm"},
			"objective": map[string]any{"mode": "Pareto"}, "recommendation": map[string]any{"maxCandidates": int64(5)},
		},
		"status": status,
	}
}

func dgdSpec() map[string]any {
	return map[string]any{"components": []any{map[string]any{"name": "worker", "type": "worker", "replicas": int64(1)}}}
}

func fixtureCandidate(name string, current bool, params map[string]any) map[string]any {
	metrics := map[string]any{"score": 650.6, "usedGpus": int64(2), "mean_e2e_latency_ms": 786.9,
		"objectives": map[string]any{"throughput_per_gpu": 650.6, "throughput_per_user": 189.1}}
	obj := map[string]any{
		"apiVersion": "nvidia.com/v1beta2", "kind": "DynamoGraphDeploymentCandidate",
		"metadata": map[string]any{
			"name": name, "namespace": "dynamo-system", "uid": "uid-" + name,
			"labels":          map[string]any{labelRunUID: "run-uid-1", labelCandidateID: "evaluated-point-1"},
			"ownerReferences": []any{map[string]any{"kind": "DynamoGraphDeploymentRun", "name": "pareto-vllm-h100-1", "uid": "run-uid-1"}},
		},
	}
	if current {
		obj["spec"] = dgdSpec()
		obj["status"] = map[string]any{"parameters": params, "metrics": metrics,
			"conditions": []any{map[string]any{"type": "Valid", "status": "True", "reason": "ValidationSucceeded"}}}
	} else {
		spec := dgdSpec()
		spec["parameters"] = params
		obj["spec"] = spec
		obj["status"] = map[string]any{"metrics": metrics}
	}
	return obj
}

func aggParams(tp int64) map[string]any {
	return map[string]any{"deployment_mode": "agg", "tp": tp, "pp": int64(1), "attention_dp": int64(1), "replicas": int64(1), "backend": "vllm", "used_gpus": tp}
}

func populatedIndex(t *testing.T, store history.Store) *Index {
	t.Helper()
	ctx := context.Background()
	ix := NewIndex(store)
	ix.UpsertRequest(fixtureRequest())
	if err := ix.UpsertRun(ctx, fixtureRun(true)); err != nil {
		t.Fatal(err)
	}
	for _, c := range []map[string]any{
		fixtureCandidate("run-1-aaaa-point-new", true, aggParams(2)),
		fixtureCandidate("run-1-aaaa-point-old", false, aggParams(1)),
		fixtureCandidate("run-1-aaaa-point-gone", false, map[string]any{"deployment_mode": "disagg", "prefill_tp": int64(1), "decode_tp": int64(2), "decode_replicas": int64(2)}),
	} {
		if err := ix.UpsertCandidate(ctx, c); err != nil {
			t.Fatal(err)
		}
	}
	if err := ix.DeleteCandidate(ctx, "dynamo-system", "run-1-aaaa-point-gone"); err != nil {
		t.Fatal(err)
	}
	return ix
}

func TestIndexViewsAcrossObjectShapes(t *testing.T) {
	ix := populatedIndex(t, nil)

	t.Log("Requests group their runs and the legacy aisimulate input is read")
	requests := ix.ListRequests("")
	if len(requests) != 1 || requests[0].Name != "pareto-vllm-h100" || len(requests[0].Runs) != 1 || !requests[0].Live {
		t.Fatalf("ListRequests = %+v", requests)
	}
	detail, ok := ix.RunDetail("dynamo-system", "pareto-vllm-h100-1")
	if !ok {
		t.Fatal("run not found")
	}
	if detail.Request != "pareto-vllm-h100" || detail.HardwareSKU != "h100_sxm" || detail.AISimInput == nil {
		t.Fatalf("run view = %+v", detail.RunView)
	}
	if detail.Evaluations == nil || detail.Evaluations.CacheHits != 3 || len(detail.Rounds) != 1 || detail.Rounds[0].Completed != 4 {
		t.Fatalf("progress not surfaced: %+v %+v", detail.Evaluations, detail.Rounds)
	}
	if detail.Duration != "26s" || detail.Published != 2 || detail.ParetoFront != 2 {
		t.Fatalf("duration/published = %q %d %d", detail.Duration, detail.Published, detail.ParetoFront)
	}
	if len(detail.Samples) != 1 {
		t.Fatalf("expected one status sample, got %d", len(detail.Samples))
	}

	t.Log("Candidates are ranked by candidateRefs, with the removed one retained as superseded")
	if len(detail.Candidates) != 3 {
		t.Fatalf("candidates = %d", len(detail.Candidates))
	}
	first, second, gone := detail.Candidates[0], detail.Candidates[1], detail.Candidates[2]
	if first.Name != "run-1-aaaa-point-new" || first.Rank == nil || *first.Rank != 1 || first.Valid != "True" || first.Topology != "agg TP2" {
		t.Fatalf("first = %+v", first)
	}
	if second.Name != "run-1-aaaa-point-old" || second.Rank == nil || *second.Rank != 2 || second.Valid != "NotEvaluated" || second.Topology != "agg TP1" {
		t.Fatalf("second = %+v", second)
	}
	if gone.Name != "run-1-aaaa-point-gone" || gone.Rank != nil || !gone.Retired || gone.RetiredReason != "superseded" || gone.Topology != "disagg P1/D2/x2" {
		t.Fatalf("gone = %+v", gone)
	}
	if first.Objectives["throughput_per_gpu"] != 650.6 || first.UsedGPUs == nil || *first.UsedGPUs != 2 || first.Parameters["tp"] != int64(2) {
		t.Fatalf("metrics/params = %+v", first)
	}

	t.Log("Candidate detail strips legacy inline parameters from the DGD spec")
	cd, ok := ix.CandidateDetail("dynamo-system", "run-1-aaaa-point-old")
	if !ok {
		t.Fatal("candidate not found")
	}
	if _, leaked := cd.Spec["parameters"]; leaked || cd.AllParams["tp"] != int64(1) {
		t.Fatalf("candidate detail = %+v", cd)
	}

	t.Log("A deleted run is retained and its candidates report 'run deleted'")
	if err := ix.DeleteRun(context.Background(), "run-uid-1"); err != nil {
		t.Fatal(err)
	}
	detail, _ = ix.RunDetail("dynamo-system", "pareto-vllm-h100-1")
	if !detail.Retained || detail.Candidates[0].Rank != nil || detail.Candidates[0].RetiredReason != "run deleted" {
		t.Fatalf("after delete = %+v", detail.Candidates[0])
	}
}

func TestHistoryReloadsRetainedObjects(t *testing.T) {
	store, err := history.NewLocal(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	ix := populatedIndex(t, store)
	if err := ix.DeleteRun(context.Background(), "run-uid-1"); err != nil {
		t.Fatal(err)
	}

	t.Log("A fresh index loads the deleted run, its candidates and samples from the store")
	reloaded := NewIndex(store)
	if err := reloaded.Load(context.Background()); err != nil {
		t.Fatal(err)
	}
	detail, ok := reloaded.RunDetail("dynamo-system", "pareto-vllm-h100-1")
	if !ok || !detail.Retained || len(detail.Candidates) != 3 || len(detail.Samples) != 1 || detail.DeletedAt == "" {
		t.Fatalf("reloaded detail = %+v", detail.RunView)
	}
	if ns := reloaded.Namespaces(); len(ns) != 1 || ns[0] != "dynamo-system" {
		t.Fatalf("namespaces = %v", ns)
	}
}

func TestServerEndpointsAndDGDCreation(t *testing.T) {
	ix := populatedIndex(t, nil)
	cl := fake.NewClientBuilder().WithScheme(runtime.NewScheme()).Build()
	srv := httptest.NewServer(NewServer(ix, NewLogManager(nil, nil, logr.Discard()), cl))
	defer srv.Close()

	get := func(path string) (int, string) {
		resp, err := http.Get(srv.URL + path)
		if err != nil {
			t.Fatal(err)
		}
		defer resp.Body.Close()
		var sb strings.Builder
		buf := make([]byte, 4096)
		for {
			n, err := resp.Body.Read(buf)
			sb.Write(buf[:n])
			if err != nil {
				break
			}
		}
		return resp.StatusCode, sb.String()
	}
	post := func(body DGDRequest) (int, string) {
		data, _ := json.Marshal(body)
		resp, err := http.Post(srv.URL+"/api/v1/dgd", "application/json", strings.NewReader(string(data)))
		if err != nil {
			t.Fatal(err)
		}
		defer resp.Body.Close()
		var sb strings.Builder
		buf := make([]byte, 4096)
		for {
			n, err := resp.Body.Read(buf)
			sb.Write(buf[:n])
			if err != nil {
				break
			}
		}
		return resp.StatusCode, sb.String()
	}

	t.Log("Read endpoints")
	for _, path := range []string{"/api/v1/namespaces", "/api/v1/requests?namespace=dynamo-system", "/api/v1/runs",
		"/api/v1/runs/dynamo-system/pareto-vllm-h100-1", "/api/v1/candidates/dynamo-system/run-1-aaaa-point-new", "/", "/healthz"} {
		if code, body := get(path); code >= 300 {
			t.Fatalf("GET %s = %d %s", path, code, body)
		}
	}
	if code, _ := get("/api/v1/runs/dynamo-system/missing"); code != http.StatusNotFound {
		t.Fatalf("missing run = %d", code)
	}
	if code, body := get("/api/v1/runs/dynamo-system/pareto-vllm-h100-1/logs"); code != http.StatusNotFound {
		t.Fatalf("logs without history = %d %s", code, body)
	}
	if _, body := get("/"); !strings.Contains(body, "Dynamo Search") {
		t.Fatal("index page not served")
	}

	t.Log("DGD creation: validation errors, dry run, create, conflict")
	req := DGDRequest{CandidateNamespace: "dynamo-system", CandidateName: "run-1-aaaa-point-old", Namespace: "inference", Name: "qwen-prod"}
	if code, _ := post(DGDRequest{CandidateNamespace: "dynamo-system", CandidateName: "nope", Name: "x"}); code != http.StatusNotFound {
		t.Fatalf("unknown candidate = %d", code)
	}
	bad := req
	bad.Name = "Not_Valid"
	if code, _ := post(bad); code != http.StatusBadRequest {
		t.Fatalf("bad name = %d", code)
	}
	dry := req
	dry.DryRun = true
	code, body := post(dry)
	if code != http.StatusOK || !strings.Contains(body, "validated now") {
		t.Fatalf("dry run = %d %s", code, body)
	}
	if code, body := post(req); code != http.StatusCreated {
		t.Fatalf("create = %d %s", code, body)
	}
	if code, body := post(req); code != http.StatusConflict || !strings.Contains(body, "already exists") {
		t.Fatalf("second create = %d %s", code, body)
	}

	t.Log("The created DGD carries only the deployable spec and a provenance annotation")
	created := &unstructured.Unstructured{}
	created.SetGroupVersionKind(gvkDGD)
	if err := cl.Get(context.Background(), namespacedName("inference", "qwen-prod"), created); err != nil {
		t.Fatal(err)
	}
	if _, leaked := created.Object["spec"].(map[string]any)["parameters"]; leaked {
		t.Fatal("evaluation parameters leaked into the DGD")
	}
	if created.GetAnnotations()[annotationFromDGD] != "dynamo-system/run-1-aaaa-point-old" {
		t.Fatalf("annotations = %v", created.GetAnnotations())
	}

	t.Log("An invalid candidate is refused before any API call")
	invalid := fixtureCandidate("run-1-aaaa-point-bad", true, aggParams(4))
	invalid["status"].(map[string]any)["conditions"] = []any{map[string]any{"type": "Valid", "status": "False", "reason": "ValidationFailed", "message": "boom"}}
	if err := ix.UpsertCandidate(context.Background(), invalid); err != nil {
		t.Fatal(err)
	}
	if code, body := post(DGDRequest{CandidateNamespace: "dynamo-system", CandidateName: "run-1-aaaa-point-bad", Namespace: "inference", Name: "bad"}); code != http.StatusConflict || !strings.Contains(body, "ValidationFailed") {
		t.Fatalf("invalid candidate = %d %s", code, body)
	}
}
