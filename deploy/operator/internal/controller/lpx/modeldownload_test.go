/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	"context"
	"fmt"
	"net/url"
	"slices"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx"
)

const (
	modelDownloadTestBuildID       = "gs://bucket/model/build"
	modelDownloadTestSecondBuildID = "gs://bucket/model/second-build"
	modelDownloadTestLocalBuildID  = "file:///models/local-build"
)

func TestEnsureModelsDownloaded(t *testing.T) {
	t.Log("Own a mixed graph with reversed build order")
	pending := newModelDownloadDGD(modelDownloadTestSecondBuildID, modelDownloadTestBuildID)
	pending.Spec.Components = append([]v1beta1.DynamoComponentDeploymentSharedSpec{{
		ComponentName: "worker", ComponentType: v1beta1.ComponentTypeWorker,
	}}, pending.Spec.Components...)
	tests := []struct {
		name       string
		dgd        *v1beta1.DynamoGraphDeployment
		registry   lpx.ModelRegistry
		existing   []string
		wantReady  bool
		wantErr    string
		wantCalls  []string
		wantBuilds []string
	}{
		{
			name:      "local build does not call Model Express",
			dgd:       newModelDownloadDGD(modelDownloadTestLocalBuildID),
			registry:  newModelDownloadRegistry(t, nil, nil),
			wantReady: true,
		},
		{
			name:       "deduplicates the same build across draft and target",
			dgd:        newModelDownloadDGD(modelDownloadTestBuildID, modelDownloadTestBuildID),
			registry:   newModelDownloadRegistry(t, map[string]bool{modelDownloadTestBuildID: true}, nil),
			wantReady:  true,
			wantCalls:  []string{modelDownloadTestBuildID},
			wantBuilds: []string{modelDownloadTestBuildID},
		},
		{
			name:     "checks only uncached builds and drops unselected cached builds",
			dgd:      newModelDownloadDGD(modelDownloadTestBuildID, modelDownloadTestSecondBuildID),
			existing: []string{modelDownloadTestBuildID, "gs://bucket/removed/build"},
			registry: newModelDownloadRegistry(t, map[string]bool{
				modelDownloadTestBuildID:       true,
				modelDownloadTestSecondBuildID: true,
			}, nil),
			wantReady:  true,
			wantCalls:  []string{modelDownloadTestSecondBuildID},
			wantBuilds: []string{modelDownloadTestBuildID, modelDownloadTestSecondBuildID},
		},
		{
			name: "skips ordinary components, sorts remote builds and records partial progress",
			dgd:  pending,
			registry: newModelDownloadRegistry(t, map[string]bool{
				modelDownloadTestBuildID: true,
			}, nil),
			wantReady:  false,
			wantCalls:  []string{modelDownloadTestBuildID, modelDownloadTestSecondBuildID},
			wantBuilds: []string{modelDownloadTestBuildID},
		},
		{
			name:     "returns build URL resolution error",
			dgd:      newModelDownloadDGD("relative-build-id"),
			registry: newModelDownloadRegistry(t, nil, nil),
			wantErr:  "model registry URL is not configured",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Log("Check uncached selected builds without acquiring their compiler snapshots")
			builds, ready, err := ensureModelsDownloaded(t.Context(), tt.dgd, tt.registry, tt.existing)
			if tt.wantErr != "" {
				require.ErrorContains(t, err, tt.wantErr)
			} else {
				require.NoError(t, err)
			}
			require.Equal(t, tt.wantReady, ready)
			require.True(t, slices.Equal(builds, tt.wantBuilds), "downloaded builds = %v, want %v", builds, tt.wantBuilds)
			if registry, ok := tt.registry.(*fakeModelDownloadRegistry); ok {
				require.Equal(t, tt.wantCalls, registry.calls)
				require.Zero(t, registry.acquireBuildSnapshotCalls)
			}
		})
	}
}

func TestEnsureModelsDownloadedSharesDeadlineAcrossBuilds(t *testing.T) {
	t.Log("Give both builds a shared deadline and let the first download fail")
	dgd := newModelDownloadDGD(modelDownloadTestBuildID, modelDownloadTestSecondBuildID)
	registry := newModelDownloadRegistry(t,
		map[string]bool{modelDownloadTestBuildID: true, modelDownloadTestSecondBuildID: true},
		map[string]error{modelDownloadTestBuildID: fmt.Errorf("download failed: %w", context.DeadlineExceeded)},
	)
	ctx, cancel := context.WithTimeout(t.Context(), time.Hour)
	defer cancel()
	started := time.Now()

	t.Log("Continue checking later builds and retain their completed progress")
	builds, result, err := ensureModelsDownloaded(ctx, dgd, registry, nil)
	require.ErrorIs(t, err, context.DeadlineExceeded)
	require.ErrorContains(t, err, "download failed")
	require.Zero(t, result)
	require.Equal(t, []string{modelDownloadTestBuildID, modelDownloadTestSecondBuildID}, registry.calls)
	require.Equal(t, []string{modelDownloadTestSecondBuildID}, builds)
	require.Zero(t, registry.acquireBuildSnapshotCalls)

	t.Log("Each download receives a bounded share of the total check budget")
	require.Len(t, registry.deadlines, 2)
	for _, deadline := range registry.deadlines {
		require.WithinDuration(t, started.Add(modelDownloadCheckTimeout/2), deadline, time.Second)
	}
}

func TestRunningLPXModelDownloadRefresh(t *testing.T) {
	tests := []struct {
		name      string
		checkedAt time.Time
		ready     bool
		err       error
		wantCalls int
	}{
		{
			name:      "fresh check skips ModelExpress",
			checkedAt: time.Now(),
			ready:     true,
		},
		{
			name:      "stale check calls ModelExpress",
			checkedAt: time.Now().Add(-modelDownloadRefreshInterval),
			ready:     true,
			wantCalls: 1,
		},
		{
			name:      "stale check remains ready when ModelExpress is down",
			checkedAt: time.Now().Add(-modelDownloadRefreshInterval),
			err:       fmt.Errorf("ModelExpress unavailable"),
			wantCalls: 2,
		},
		{
			name:      "stale check remains ready while an evicted model redownloads",
			checkedAt: time.Now().Add(-modelDownloadRefreshInterval),
			wantCalls: 2,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Log("Refresh a running workload while preserving its downloaded builds")
			dgd := newModelDownloadDGD(modelDownloadTestBuildID)
			child := &v1alpha1.LPXGraphDeployment{ObjectMeta: metav1.ObjectMeta{Generation: 1}}
			child.Status.ModelDownload = &v1alpha1.ModelDownloadStatus{
				Builds: []string{modelDownloadTestBuildID}, LastCheckedAt: &metav1.Time{Time: tt.checkedAt},
			}
			child.Status.Conditions = []metav1.Condition{{Type: "Ready", Status: metav1.ConditionTrue}}
			child.Status.ObservedGeneration = child.Generation
			registry := newModelDownloadRegistry(t,
				map[string]bool{modelDownloadTestBuildID: tt.ready},
				map[string]error{modelDownloadTestBuildID: tt.err},
			)

			lpx := &graphReconciler{modelRegistry: registry}
			for range 2 {
				result, err := lpx.reconcileModelDownloads(t.Context(), child, dgd)
				require.NoError(t, err)
				require.Zero(t, result)
			}
			if len(registry.calls) != tt.wantCalls {
				t.Fatalf("ModelExpress calls = %d, want %d", len(registry.calls), tt.wantCalls)
			}

			t.Log("Only a successful refresh starts another 24-hour cache window")
			if tt.wantCalls > 0 && tt.ready {
				require.True(t, child.Status.ModelDownload.LastCheckedAt.After(tt.checkedAt))
			} else {
				require.Equal(t, tt.checkedAt, child.Status.ModelDownload.LastCheckedAt.Time)
			}

			if !slices.Equal(child.Status.ModelDownload.Builds, []string{modelDownloadTestBuildID}) {
				t.Fatalf("downloaded builds = %v, want preserved", child.Status.ModelDownload.Builds)
			}
		})
	}
}

func TestEnsureModelsDownloadedCheckTimeout(t *testing.T) {
	for _, parentTimeout := range []time.Duration{time.Hour, time.Second} {
		t.Run(parentTimeout.String(), func(t *testing.T) {
			t.Log("Bound the download RPC by both the check timeout and the caller's deadline")
			dgd := newModelDownloadDGD(modelDownloadTestBuildID)
			registry := newModelDownloadRegistry(t, map[string]bool{modelDownloadTestBuildID: true}, nil)
			started := time.Now()
			ctx, cancel := context.WithTimeout(t.Context(), parentTimeout)
			defer cancel()

			_, ready, err := ensureModelsDownloaded(ctx, dgd, registry, nil)
			require.NoError(t, err)
			require.True(t, ready)
			require.Len(t, registry.deadlines, 1)
			require.WithinDuration(t, started.Add(min(parentTimeout, modelDownloadCheckTimeout)), registry.deadlines[0], 100*time.Millisecond)
		})
	}
}

func TestModelDownloadCacheAcrossRevisions(t *testing.T) {
	for _, tc := range []struct {
		name       string
		build      string
		cacheAge   time.Duration
		observed   bool
		ready      bool
		wantCached bool
	}{
		{name: "unchanged build survives an input edit", build: modelDownloadTestBuildID, wantCached: true},
		{name: "new build is checked after an input edit", build: modelDownloadTestSecondBuildID},
		{name: "expired cache is checked after an input edit", build: modelDownloadTestBuildID, cacheAge: modelDownloadRefreshInterval},
		{name: "observed but not Ready still checks new builds", build: modelDownloadTestSecondBuildID, observed: true},
		{name: "old Ready receipt cannot hide a new build failure", build: modelDownloadTestSecondBuildID, ready: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Log("Retain a previous build's successful download while changing the desired input")
			dgd := newModelDownloadDGD(tc.build)
			child := &v1alpha1.LPXGraphDeployment{ObjectMeta: metav1.ObjectMeta{Generation: 1}}
			if tc.observed {
				child.Status.ObservedGeneration = child.Generation
			}
			if tc.ready {
				setReadyCondition(child, v1beta1.DGDStateSuccessful, "Previous input is ready")
			}
			checkedAt := metav1.NewTime(time.Now().Add(-tc.cacheAge))
			child.Status.ModelDownload = &v1alpha1.ModelDownloadStatus{
				Builds: []string{modelDownloadTestBuildID}, LastCheckedAt: &checkedAt,
			}
			registry := newModelDownloadRegistry(t, nil, map[string]error{tc.build: fmt.Errorf("ModelExpress unavailable")})
			r := &graphReconciler{modelRegistry: registry}

			t.Log("Reuse only fresh matching builds; new or expired builds must be checked successfully")
			result, err := r.reconcileModelDownloads(t.Context(), child, dgd)
			require.Zero(t, result)
			if tc.wantCached {
				require.NoError(t, err)
				require.Empty(t, registry.calls)
				require.Equal(t, &checkedAt, child.Status.ModelDownload.LastCheckedAt)
				require.Equal(t, []string{tc.build}, child.Status.ModelDownload.Builds)
			} else {
				require.ErrorContains(t, err, "ModelExpress unavailable")
				require.Equal(t, []string{tc.build}, registry.calls)
				require.Empty(t, child.Status.ModelDownload.Builds)
				require.Nil(t, child.Status.ModelDownload.LastCheckedAt)
			}
		})
	}
}

type fakeModelDownloadRegistry struct {
	lpx.ModelRegistry
	ready                     map[string]bool
	err                       map[string]error
	calls                     []string
	deadlines                 []time.Time
	acquireBuildSnapshotCalls int
}

func (r *fakeModelDownloadRegistry) AcquireBuildSnapshot(ctx context.Context, id string) (*lpx.BuildSnapshot, error) {
	r.acquireBuildSnapshotCalls++
	return r.ModelRegistry.AcquireBuildSnapshot(ctx, id)
}

func (r *fakeModelDownloadRegistry) EnsureDownloaded(ctx context.Context, buildURL url.URL) (bool, error) {
	build := buildURL.String()
	r.calls = append(r.calls, build)
	deadline, _ := ctx.Deadline()
	r.deadlines = append(r.deadlines, deadline)
	return r.ready[build], r.err[build]
}

func newModelDownloadRegistry(t *testing.T, ready map[string]bool, errs map[string]error) *fakeModelDownloadRegistry {
	t.Helper()

	registry, err := lpx.NewModelRegistry("", nil)
	require.NoError(t, err)
	return &fakeModelDownloadRegistry{ModelRegistry: registry, ready: ready, err: errs}
}

// newModelDownloadDGD supplies only the component-to-build mapping consumed by download checks.
func newModelDownloadDGD(buildIDs ...string) *v1beta1.DynamoGraphDeployment {
	dgd := &v1beta1.DynamoGraphDeployment{}
	for index, buildID := range buildIDs {
		dgd.Spec.Components = append(dgd.Spec.Components, v1beta1.DynamoComponentDeploymentSharedSpec{
			ComponentName: fmt.Sprintf("model-%d", index), ComponentType: v1beta1.ComponentTypeLPX,
			LPX: &v1beta1.LPXConfig{BuildID: buildID},
		})
	}
	return dgd
}
