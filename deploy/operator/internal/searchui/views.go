/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package searchui

import (
	"fmt"
	"sort"
	"strings"
	"time"
)

// RequestView is one DynamoGraphDeploymentRequest with its Runs.
type RequestView struct {
	Namespace  string      `json:"namespace"`
	Name       string      `json:"name"`
	Model      string      `json:"model,omitempty"`
	Mode       string      `json:"mode,omitempty"`
	ActiveRun  string      `json:"activeRun,omitempty"`
	Live       bool        `json:"live"`
	Conditions []condition `json:"conditions,omitempty"`
	Runs       []RunView   `json:"runs"`
}

// RunView is the list and header representation of one Run.
type RunView struct {
	Namespace        string      `json:"namespace"`
	Name             string      `json:"name"`
	UID              string      `json:"uid"`
	Request          string      `json:"request,omitempty"`
	Phase            string      `json:"phase"`
	Model            string      `json:"model,omitempty"`
	Backends         []string    `json:"backends,omitempty"`
	HardwareSKU      string      `json:"hardwareSku,omitempty"`
	Mode             string      `json:"mode,omitempty"`
	Metric           string      `json:"metric,omitempty"`
	MaxCandidates    int64       `json:"maxCandidates,omitempty"`
	Published        int64       `json:"published"`
	ParetoFront      int64       `json:"paretoFront,omitempty"`
	Evaluations      *EvalView   `json:"evaluations,omitempty"`
	Rounds           []RoundView `json:"rounds,omitempty"`
	StartTime        string      `json:"startTime,omitempty"`
	CompletionTime   string      `json:"completionTime,omitempty"`
	LastProgressTime string      `json:"lastProgressTime,omitempty"`
	Duration         string      `json:"duration,omitempty"`
	JobName          string      `json:"jobName,omitempty"`
	Conditions       []condition `json:"conditions,omitempty"`
	Retained         bool        `json:"retained"`
	FirstSeen        string      `json:"firstSeen,omitempty"`
	DeletedAt        string      `json:"deletedAt,omitempty"`
}

// EvalView mirrors status.progress.evaluations.
type EvalView struct {
	Scheduled   int64 `json:"scheduled"`
	Running     int64 `json:"running"`
	Feasible    int64 `json:"feasible"`
	Infeasible  int64 `json:"infeasible"`
	Failed      int64 `json:"failed"`
	Unsupported int64 `json:"unsupported"`
	CacheHits   int64 `json:"cacheHits"`
}

// RoundView mirrors one status.progress.branches entry.
type RoundView struct {
	Name      string `json:"name"`
	Completed int64  `json:"completed"`
	Total     int64  `json:"total"`
}

// SampleView is one point of the Run status timeline.
type SampleView struct {
	Time        string `json:"time"`
	Phase       string `json:"phase,omitempty"`
	Published   int64  `json:"published"`
	ParetoFront int64  `json:"paretoFront,omitempty"`
}

// CandidateView is the list representation of one Candidate.
type CandidateView struct {
	Namespace     string             `json:"namespace"`
	Name          string             `json:"name"`
	ID            string             `json:"id,omitempty"`
	RunUID        string             `json:"runUID,omitempty"`
	RunName       string             `json:"runName,omitempty"`
	Rank          *int               `json:"rank,omitempty"`
	Backend       string             `json:"backend,omitempty"`
	Valid         string             `json:"valid"`
	ValidReason   string             `json:"validReason,omitempty"`
	ValidMessage  string             `json:"validMessage,omitempty"`
	Retired       bool               `json:"retired"`
	RetiredReason string             `json:"retiredReason,omitempty"`
	Topology      string             `json:"topology,omitempty"`
	UsedGPUs      *float64           `json:"usedGpus,omitempty"`
	Score         *float64           `json:"score,omitempty"`
	Objectives    map[string]float64 `json:"objectives,omitempty"`
	Parameters    map[string]any     `json:"parameters,omitempty"`
	Summary       map[string]float64 `json:"summary,omitempty"`
	FirstSeen     string             `json:"firstSeen,omitempty"`
	LastSeen      string             `json:"lastSeen,omitempty"`
}

// CandidateDetail adds the full payloads.
type CandidateDetail struct {
	CandidateView
	Spec       map[string]any `json:"spec,omitempty"`
	AllParams  map[string]any `json:"allParameters,omitempty"`
	Metrics    map[string]any `json:"metrics,omitempty"`
	Conditions []condition    `json:"conditions,omitempty"`
}

// RunDetail adds the effective input, candidates and timeline.
type RunDetail struct {
	RunView
	AISimInput map[string]any  `json:"aisimInput,omitempty"`
	Candidates []CandidateView `json:"candidates"`
	Samples    []SampleView    `json:"samples,omitempty"`
}

// searchDimensions are the evaluation parameters worth plotting as search-space axes,
// in display order. Absent ones are dropped per candidate.
var searchDimensions = []string{
	"deployment_mode", "strategy", "tp", "pp", "attention_dp", "moe_tp", "moe_ep",
	"prefill_tp", "decode_tp", "replicas", "prefill_replicas", "decode_replicas",
	"agg_max_num_seqs", "agg_max_num_batched_tokens", "prefill_max_num_seqs",
	"decode_max_num_seqs", "concurrency", "used_gpus", "hardware_sku", "backend",
}

var summaryMetrics = []string{
	"mean_ttft_ms", "mean_itl_ms", "mean_e2e_latency_ms", "p99_e2e_latency_ms",
	"output_throughput_tok_s", "request_throughput_rps", "gpu_hours",
}

func formatTime(t *time.Time) string {
	if t == nil {
		return ""
	}
	return t.UTC().Format(time.RFC3339)
}

func intField(obj map[string]any, fields ...string) int64 {
	v, _ := number(obj, fields...)
	return int64(v)
}

func runView(entry *runEntry) RunView {
	obj := entry.env.Object
	status := nestedMap(obj, "status")
	v := RunView{
		Namespace:        entry.env.Namespace,
		Name:             entry.env.Name,
		UID:              entry.env.UID,
		Request:          runRequestName(obj),
		Phase:            str(obj, "status", "phase"),
		Model:            str(obj, "spec", "model", "name"),
		Backends:         runBackends(obj),
		HardwareSKU:      runHardwareSKU(obj),
		Mode:             str(obj, "spec", "objective", "mode"),
		Metric:           str(obj, "spec", "objective", "metric"),
		MaxCandidates:    intField(obj, "spec", "recommendation", "maxCandidates"),
		Published:        intField(status, "progress", "candidates", "published"),
		ParetoFront:      intField(status, "progress", "candidates", "paretoFront"),
		StartTime:        str(status, "startTime"),
		CompletionTime:   str(status, "completionTime"),
		LastProgressTime: str(status, "lastProgressTime"),
		JobName:          str(status, "jobRef", "name"),
		Conditions:       conditions(obj),
		Retained:         !entry.live,
		FirstSeen:        formatTime(&entry.env.FirstSeen),
		DeletedAt:        formatTime(entry.env.DeletedAt),
	}
	if v.Phase == "" {
		v.Phase = "Pending"
	}
	if evals := nestedMap(status, "progress", "evaluations"); evals != nil {
		v.Evaluations = &EvalView{
			Scheduled: intField(evals, "scheduled"), Running: intField(evals, "running"),
			Feasible: intField(evals, "feasible"), Infeasible: intField(evals, "infeasible"),
			Failed: intField(evals, "failed"), Unsupported: intField(evals, "unsupported"),
			CacheHits: intField(evals, "cacheHits"),
		}
	}
	for _, item := range nestedSlice(status, "progress", "branches") {
		branch, ok := item.(map[string]any)
		if !ok {
			continue
		}
		v.Rounds = append(v.Rounds, RoundView{
			Name: str(branch, "name"), Completed: intField(branch, "rounds", "completed"), Total: intField(branch, "rounds", "total"),
		})
	}
	if start := parseTime(v.StartTime); start != nil {
		end := time.Now()
		if completion := parseTime(v.CompletionTime); completion != nil {
			end = *completion
		} else if v.Phase == "Succeeded" || v.Phase == "Failed" {
			if last := parseTime(v.LastProgressTime); last != nil {
				end = *last
			}
		}
		v.Duration = end.Sub(*start).Round(time.Second).String()
	}
	return v
}

func topology(p map[string]any) string {
	if p == nil {
		return ""
	}
	n := func(key string) (float64, bool) { return number(p, key) }
	part := func(prefix, key string, hideOne bool) string {
		v, ok := n(key)
		if !ok || (hideOne && v <= 1) {
			return ""
		}
		return fmt.Sprintf("%s%g", prefix, v)
	}
	join := func(parts ...string) string {
		var out []string
		for _, s := range parts {
			if s != "" {
				out = append(out, s)
			}
		}
		return strings.Join(out, "/")
	}
	mode := str(p, "deployment_mode")
	switch mode {
	case "disagg":
		prefill := join(part("P", "prefill_tp", false), part("PP", "prefill_pp", true), part("x", "prefill_replicas", true))
		decode := join(part("D", "decode_tp", false), part("PP", "decode_pp", true), part("x", "decode_replicas", true))
		if prefill == "" {
			prefill = part("P", "tp", false)
		}
		if decode == "" {
			decode = part("D", "tp", false)
		}
		return strings.TrimSpace("disagg " + join(prefill, decode))
	default:
		label := mode
		if label == "" {
			label = "agg"
		}
		core := join(part("TP", "tp", false), part("PP", "pp", true), part("DP", "attention_dp", true), part("EP", "moe_ep", true))
		if replicas, ok := n("replicas"); ok && replicas > 1 {
			core += fmt.Sprintf(" ×%g", replicas)
		}
		return strings.TrimSpace(label + " " + core)
	}
}

func candidateView(entry *candidateEntry, run *runEntry, rank *int) CandidateView {
	obj := entry.env.Object
	params := candidateParameters(obj)
	metrics := candidateMetrics(obj)
	valid, reason, message := candidateValidity(obj)
	// A Candidate whose Run is gone is retained history even before garbage collection
	// removes the object itself.
	runGone := run == nil || !run.live
	v := CandidateView{
		Namespace:    entry.env.Namespace,
		Name:         entry.env.Name,
		ID:           str(obj, "metadata", "labels", labelCandidateID),
		RunUID:       entry.env.RunUID,
		RunName:      candidateRunName(obj),
		Rank:         rank,
		Backend:      candidateBackend(obj),
		Valid:        valid,
		ValidReason:  reason,
		ValidMessage: message,
		Retired:      !entry.live || runGone,
		Topology:     topology(params),
		FirstSeen:    formatTime(&entry.env.FirstSeen),
		LastSeen:     formatTime(&entry.env.LastSeen),
	}
	if v.RunName == "" && run != nil {
		v.RunName = run.env.Name
	}
	switch {
	case runGone:
		v.RetiredReason = "run deleted"
	case !entry.live:
		v.RetiredReason = "superseded"
	}
	if gpus, ok := number(metrics, "usedGpus"); ok {
		v.UsedGPUs = &gpus
	} else if gpus, ok := number(params, "used_gpus"); ok {
		v.UsedGPUs = &gpus
	}
	if score, ok := number(metrics, "score"); ok {
		v.Score = &score
	}
	if objectives := nestedMap(metrics, "objectives"); objectives != nil {
		v.Objectives = map[string]float64{}
		for key, raw := range objectives {
			if f, ok := toFloat(raw); ok {
				v.Objectives[key] = f
			}
		}
	}
	if params != nil {
		v.Parameters = map[string]any{}
		for _, key := range searchDimensions {
			if value, ok := params[key]; ok && value != nil {
				v.Parameters[key] = value
			}
		}
	}
	if metrics != nil {
		v.Summary = map[string]float64{}
		for _, key := range summaryMetrics {
			if f, ok := number(metrics, key); ok {
				v.Summary[key] = f
			}
		}
	}
	return v
}

// rankedCandidates orders a run's candidates: the live projection in candidateRefs
// order first (ranked), then everything else by first observation.
func (ix *Index) rankedCandidates(run *runEntry) []CandidateView {
	all := ix.candidatesOfRun(run.env.UID)
	byName := make(map[string]*candidateEntry, len(all))
	for _, c := range all {
		byName[c.env.Name] = c
	}
	var out []CandidateView
	seen := map[string]bool{}
	for i, item := range nestedSlice(run.env.Object, "status", "candidateRefs") {
		ref, ok := item.(map[string]any)
		if !ok {
			continue
		}
		name := str(ref, "name")
		c, ok := byName[name]
		if !ok || !c.live || !run.live {
			continue
		}
		rank := i + 1
		out = append(out, candidateView(c, run, &rank))
		seen[name] = true
	}
	for _, c := range all {
		if !seen[c.env.Name] {
			out = append(out, candidateView(c, run, nil))
		}
	}
	return out
}

func sortRuns(runs []RunView) {
	sort.Slice(runs, func(i, j int) bool {
		if runs[i].StartTime != runs[j].StartTime {
			return runs[i].StartTime > runs[j].StartTime
		}
		return runs[i].Name > runs[j].Name
	})
}

// --- Index read API ---

func (ix *Index) ListRuns(namespace string) []RunView {
	ix.mu.RLock()
	defer ix.mu.RUnlock()
	var out []RunView
	for _, entry := range ix.runs {
		if namespace == "" || entry.env.Namespace == namespace {
			out = append(out, runView(entry))
		}
	}
	sortRuns(out)
	if out == nil {
		out = []RunView{}
	}
	return out
}

func (ix *Index) ListRequests(namespace string) []RequestView {
	ix.mu.RLock()
	defer ix.mu.RUnlock()
	views := map[string]*RequestView{}
	keyOf := func(ns, name string) string { return ns + "/" + name }
	for key, obj := range ix.requests {
		if namespace != "" && key.Namespace != namespace {
			continue
		}
		views[keyOf(key.Namespace, key.Name)] = &RequestView{
			Namespace: key.Namespace, Name: key.Name, Live: true,
			Model:      str(obj, "spec", "model", "name"),
			Mode:       str(obj, "spec", "objective", "mode"),
			ActiveRun:  str(obj, "status", "activeRunRef", "name"),
			Conditions: conditions(obj),
		}
	}
	for _, entry := range ix.runs {
		if namespace != "" && entry.env.Namespace != namespace {
			continue
		}
		rv := runView(entry)
		request := rv.Request
		if request == "" {
			request = "(no request)"
		}
		key := keyOf(entry.env.Namespace, request)
		view, ok := views[key]
		if !ok {
			view = &RequestView{Namespace: entry.env.Namespace, Name: request, Model: rv.Model, Mode: rv.Mode}
			views[key] = view
		}
		view.Runs = append(view.Runs, rv)
	}
	out := make([]RequestView, 0, len(views))
	for _, view := range views {
		sortRuns(view.Runs)
		if view.Runs == nil {
			view.Runs = []RunView{}
		}
		out = append(out, *view)
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].Namespace != out[j].Namespace {
			return out[i].Namespace < out[j].Namespace
		}
		return out[i].Name < out[j].Name
	})
	return out
}

func (ix *Index) RunDetail(namespace, name string) (RunDetail, bool) {
	ix.mu.RLock()
	defer ix.mu.RUnlock()
	entry, ok := ix.run(namespace, name)
	if !ok {
		return RunDetail{}, false
	}
	detail := RunDetail{
		RunView:    runView(entry),
		AISimInput: runAISimInput(entry.env.Object),
		Candidates: ix.rankedCandidates(entry),
	}
	if detail.Candidates == nil {
		detail.Candidates = []CandidateView{}
	}
	for _, sample := range ix.samples[entry.env.UID] {
		detail.Samples = append(detail.Samples, SampleView{
			Time:        formatTime(&sample.Time),
			Phase:       str(sample.Status, "phase"),
			Published:   intField(sample.Status, "progress", "candidates", "published"),
			ParetoFront: intField(sample.Status, "progress", "candidates", "paretoFront"),
		})
	}
	return detail, true
}

func (ix *Index) CandidateDetail(namespace, name string) (CandidateDetail, bool) {
	ix.mu.RLock()
	defer ix.mu.RUnlock()
	entry, ok := ix.candidates[namespacedName(namespace, name)]
	if !ok {
		return CandidateDetail{}, false
	}
	run := ix.runs[entry.env.RunUID]
	var rank *int
	if run != nil && run.live && entry.live {
		for i, item := range nestedSlice(run.env.Object, "status", "candidateRefs") {
			if ref, ok := item.(map[string]any); ok && str(ref, "name") == name {
				r := i + 1
				rank = &r
				break
			}
		}
	}
	obj := entry.env.Object
	return CandidateDetail{
		CandidateView: candidateView(entry, run, rank),
		Spec:          candidateDGDSpec(obj),
		AllParams:     candidateParameters(obj),
		Metrics:       candidateMetrics(obj),
		Conditions:    conditions(obj),
	}, true
}

// CandidateObject returns a copy of the stored object for DGD creation.
func (ix *Index) CandidateObject(namespace, name string) (map[string]any, bool) {
	ix.mu.RLock()
	defer ix.mu.RUnlock()
	entry, ok := ix.candidates[namespacedName(namespace, name)]
	if !ok {
		return nil, false
	}
	return deepCopy(entry.env.Object), true
}

// RunByUID returns run coordinates for log lookups.
func (ix *Index) RunByName(namespace, name string) (RunView, bool) {
	ix.mu.RLock()
	defer ix.mu.RUnlock()
	entry, ok := ix.run(namespace, name)
	if !ok {
		return RunView{}, false
	}
	return runView(entry), true
}

func deepCopy(obj map[string]any) map[string]any {
	return deepCopyValue(obj).(map[string]any)
}

func deepCopyValue(v any) any {
	switch t := v.(type) {
	case map[string]any:
		out := make(map[string]any, len(t))
		for k, vv := range t {
			out[k] = deepCopyValue(vv)
		}
		return out
	case []any:
		out := make([]any, len(t))
		for i, vv := range t {
			out[i] = deepCopyValue(vv)
		}
		return out
	default:
		return v
	}
}
