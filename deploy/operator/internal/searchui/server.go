/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Package searchui implements the Dynamo Search UI: a read-only browser view of
// DynamoGraphDeploymentRequest, Run and Candidate resources with UI-local history, plus one
// separately authorized action that creates an independent DGD from a Candidate.
package searchui

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"net/http"
	"slices"
	"sync"

	"github.com/ai-dynamo/dynamo/deploy/operator/internal/searchui/history"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

// Server is the HTTP handler.
type Server struct {
	index  *Index
	logs   *LogManager
	client client.Client
	mux    *http.ServeMux

	subMu sync.Mutex
	subs  map[chan struct{}]struct{}
}

// NewServer wires the routes. client may be nil to disable DGD creation.
func NewServer(index *Index, logs *LogManager, cl client.Client) *Server {
	s := &Server{index: index, logs: logs, client: cl, subs: map[chan struct{}]struct{}{}}
	mux := http.NewServeMux()
	mux.HandleFunc("GET /healthz", func(w http.ResponseWriter, _ *http.Request) { w.WriteHeader(http.StatusNoContent) })
	mux.HandleFunc("GET /api/v1/namespaces", s.handleNamespaces)
	mux.HandleFunc("GET /api/v1/requests", s.handleRequests)
	mux.HandleFunc("GET /api/v1/runs", s.handleRuns)
	mux.HandleFunc("GET /api/v1/runs/{ns}/{name}", s.handleRunDetail)
	mux.HandleFunc("GET /api/v1/runs/{ns}/{name}/logs", s.handleLogs)
	mux.HandleFunc("GET /api/v1/runs/{ns}/{name}/logs/stream", s.handleLogStream)
	mux.HandleFunc("GET /api/v1/candidates/{ns}/{name}", s.handleCandidateDetail)
	mux.HandleFunc("GET /api/v1/events", s.handleEvents)
	mux.HandleFunc("POST /api/v1/dgd", s.handleCreateDGD)
	static, err := fs.Sub(staticFS, "static")
	if err != nil {
		panic(err)
	}
	mux.Handle("GET /", http.FileServer(http.FS(static)))
	s.mux = mux
	return s
}

// ServeHTTP implements http.Handler.
func (s *Server) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Cache-Control", "no-store")
	s.mux.ServeHTTP(w, r)
}

// Notify wakes every /api/v1/events subscriber.
func (s *Server) Notify() {
	s.subMu.Lock()
	defer s.subMu.Unlock()
	for ch := range s.subs {
		select {
		case ch <- struct{}{}:
		default:
		}
	}
}

func (s *Server) handleNamespaces(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, s.index.Namespaces())
}

func (s *Server) handleRequests(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, s.index.ListRequests(r.URL.Query().Get("namespace")))
}

func (s *Server) handleRuns(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, s.index.ListRuns(r.URL.Query().Get("namespace")))
}

func (s *Server) handleRunDetail(w http.ResponseWriter, r *http.Request) {
	detail, ok := s.index.RunDetail(r.PathValue("ns"), r.PathValue("name"))
	if !ok {
		http.Error(w, "run not found", http.StatusNotFound)
		return
	}
	writeJSON(w, detail)
}

func (s *Server) handleCandidateDetail(w http.ResponseWriter, r *http.Request) {
	detail, ok := s.index.CandidateDetail(r.PathValue("ns"), r.PathValue("name"))
	if !ok {
		http.Error(w, "candidate not found", http.StatusNotFound)
		return
	}
	writeJSON(w, detail)
}

func (s *Server) logContainer(r *http.Request) (RunView, string, error) {
	run, ok := s.index.RunByName(r.PathValue("ns"), r.PathValue("name"))
	if !ok {
		return RunView{}, "", failf(http.StatusNotFound, "run not found")
	}
	container := r.URL.Query().Get("container")
	if container == "" {
		container = LogContainers[0]
	}
	if slices.Contains(LogContainers, container) {
		return run, container, nil
	}
	return RunView{}, "", failf(http.StatusBadRequest, "unknown container %q", container)
}

func (s *Server) handleLogs(w http.ResponseWriter, r *http.Request) {
	run, container, err := s.logContainer(r)
	if err != nil {
		writeError(w, err)
		return
	}
	data, live, err := s.logs.Get(r.Context(), run.Namespace, run.UID, container)
	if errors.Is(err, history.ErrNotFound) {
		http.Error(w, "no retained log for this run; its Job pod was gone before the UI observed it", http.StatusNotFound)
		return
	}
	if err != nil {
		writeError(w, err)
		return
	}
	w.Header().Set("Content-Type", "text/plain; charset=utf-8")
	w.Header().Set("X-Log-Live", fmt.Sprint(live))
	_, _ = w.Write(data)
}

// handleLogStream sends the retained log, then follows it, as SSE events carrying JSON
// strings.
func (s *Server) handleLogStream(w http.ResponseWriter, r *http.Request) {
	run, container, err := s.logContainer(r)
	if err != nil {
		writeError(w, err)
		return
	}
	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "streaming not supported", http.StatusInternalServerError)
		return
	}
	snapshot, updates, cancel, streaming := s.logs.Subscribe(run.UID, container)
	if !streaming {
		data, _, err := s.logs.Get(r.Context(), run.Namespace, run.UID, container)
		if err != nil && !errors.Is(err, history.ErrNotFound) {
			writeError(w, err)
			return
		}
		snapshot = data
	} else {
		defer cancel()
	}
	sseHeaders(w)
	sendEvent(w, "chunk", string(snapshot))
	flusher.Flush()
	if !streaming {
		sendEvent(w, "end", "retained")
		flusher.Flush()
		return
	}
	for {
		select {
		case <-r.Context().Done():
			return
		case chunk, open := <-updates:
			if !open {
				sendEvent(w, "end", "stream closed")
				flusher.Flush()
				return
			}
			sendEvent(w, "chunk", string(chunk))
			flusher.Flush()
		}
	}
}

// handleEvents tells the browser when anything changed; it re-fetches what it shows.
func (s *Server) handleEvents(w http.ResponseWriter, r *http.Request) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "streaming not supported", http.StatusInternalServerError)
		return
	}
	ch := make(chan struct{}, 1)
	s.subMu.Lock()
	s.subs[ch] = struct{}{}
	s.subMu.Unlock()
	defer func() {
		s.subMu.Lock()
		delete(s.subs, ch)
		s.subMu.Unlock()
	}()
	sseHeaders(w)
	sendEvent(w, "hello", "connected")
	flusher.Flush()
	for {
		select {
		case <-r.Context().Done():
			return
		case <-ch:
			sendEvent(w, "update", "changed")
			flusher.Flush()
		}
	}
}

func (s *Server) handleCreateDGD(w http.ResponseWriter, r *http.Request) {
	if s.client == nil {
		http.Error(w, "DGD creation is disabled on this server", http.StatusNotImplemented)
		return
	}
	var req DGDRequest
	if err := json.NewDecoder(http.MaxBytesReader(w, r.Body, 1<<20)).Decode(&req); err != nil {
		http.Error(w, "bad request: "+err.Error(), http.StatusBadRequest)
		return
	}
	if req.Namespace == "" {
		req.Namespace = req.CandidateNamespace
	}
	cand, ok := s.index.CandidateObject(req.CandidateNamespace, req.CandidateName)
	if !ok {
		http.Error(w, "candidate not found", http.StatusNotFound)
		return
	}
	result, err := CreateDGD(r.Context(), s.client, cand, req)
	if err != nil {
		writeError(w, err)
		return
	}
	status := http.StatusCreated
	if result.DryRun {
		status = http.StatusOK
	}
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(result)
}

func sseHeaders(w http.ResponseWriter) {
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("X-Accel-Buffering", "no")
}

func sendEvent(w http.ResponseWriter, event, data string) {
	payload, _ := json.Marshal(data)
	_, _ = fmt.Fprintf(w, "event: %s\ndata: %s\n\n", event, payload)
}

func writeJSON(w http.ResponseWriter, v any) {
	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(v); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
	}
}

func writeError(w http.ResponseWriter, err error) {
	if se, ok := errors.AsType[*statusError](err); ok {
		http.Error(w, se.msg, se.status)
		return
	}
	if errors.Is(err, context.Canceled) {
		return
	}
	http.Error(w, err.Error(), http.StatusInternalServerError)
}
