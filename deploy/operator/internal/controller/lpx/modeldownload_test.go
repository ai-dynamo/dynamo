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
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/utils/ptr"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"

	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/features"
)

const (
	modelDownloadTestBuildID       = "gs://bucket/model/build"
	modelDownloadTestSecondBuildID = "gs://bucket/model/second-build"
	modelDownloadTestLocalBuildID  = "file:///models/local-build"
	modelDownloadTestDGDName       = "test-dgd"
	modelDownloadTestNamespace     = "default"
)

func TestEnsureModelsDownloaded(t *testing.T) {
	t.Log("Use the real registry to check missing Model Express configuration")
	withoutModelExpress, err := lpx.NewModelRegistry("", nil)
	require.NoError(t, err)

	t.Log("Own a mixed graph with reversed build order and a previously cached source")
	pending := newModelDownloadDGD(modelDownloadTestSecondBuildID, modelDownloadTestBuildID)
	pending.Spec.Components = append([]v1beta1.DynamoComponentDeploymentSharedSpec{{
		ComponentName: "worker", ComponentType: v1beta1.ComponentTypeWorker,
	}}, pending.Spec.Components...)
	cached := newModelDownloadDGD(modelDownloadTestBuildID, modelDownloadTestSecondBuildID)
	cached.Status.LPX = &v1beta1.DynamoGraphDeploymentLPXStatus{
		ModelDownload: &v1beta1.ModelDownloadStatus{Builds: []string{modelDownloadTestBuildID}},
	}
	tests := []struct {
		name       string
		dgd        *v1beta1.DynamoGraphDeployment
		registry   lpx.ModelRegistry
		wantReady  bool
		wantErr    string
		wantCalls  []string
		wantBuilds []string
	}{
		{
			name:     "returns Model Express client error for remote builds when Model Express is not configured",
			dgd:      newModelDownloadDGD(modelDownloadTestBuildID),
			registry: withoutModelExpress,
			wantErr:  "Model Express client is required for GCS model downloads",
		},
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
			name: "checks every remote build regardless of cached status",
			dgd:  cached,
			registry: newModelDownloadRegistry(t, map[string]bool{
				modelDownloadTestBuildID:       true,
				modelDownloadTestSecondBuildID: true,
			}, nil),
			wantReady:  true,
			wantCalls:  []string{modelDownloadTestBuildID, modelDownloadTestSecondBuildID},
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
			t.Log("Check every selected remote build without acquiring its snapshot")
			builds, ready, err := ensureModelsDownloaded(t.Context(), tt.dgd, tt.registry)
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

	t.Log("Continue checking later builds and retain their completed progress")
	builds, ready, err := ensureModelsDownloaded(ctx, dgd, registry)
	require.ErrorIs(t, err, context.DeadlineExceeded)
	require.ErrorContains(t, err, "download failed")
	require.False(t, ready)
	require.Equal(t, []string{modelDownloadTestBuildID, modelDownloadTestSecondBuildID}, registry.calls)
	require.Equal(t, []string{modelDownloadTestSecondBuildID}, builds)
	require.Zero(t, registry.acquireBuildSnapshotCalls)

	t.Log("Reserve time for the second build without extending the parent's budget")
	deadline, _ := ctx.Deadline()
	require.Len(t, registry.deadlines, 2)
	require.False(t, registry.deadlines[0].IsZero())
	require.True(t, registry.deadlines[0].Before(registry.deadlines[1]))
	require.Equal(t, deadline, registry.deadlines[1])
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
			child := newLPXTestDeployment(t, dgd)
			child.Status.ModelDownload = &v1beta1.ModelDownloadStatus{
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
				ready, err := lpx.reconcileModelDownloads(t.Context(), child, dgd, false)
				require.NoError(t, err)
				require.True(t, ready)
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

func TestInitialLPXModelDownloadChecksHaveDeadline(t *testing.T) {
	for _, forceCheck := range []bool{false, true} {
		t.Run(fmt.Sprintf("forceCheck=%t", forceCheck), func(t *testing.T) {
			dgd := newModelDownloadDGD(modelDownloadTestBuildID)
			child := newLPXTestDeployment(t, dgd)
			registry := newModelDownloadRegistry(t, map[string]bool{modelDownloadTestBuildID: true}, nil)
			lifecycle := &graphReconciler{modelRegistry: registry}
			started := time.Now()

			ready, err := lifecycle.reconcileModelDownloads(t.Context(), child, dgd, forceCheck)

			require.NoError(t, err)
			require.True(t, ready)
			require.Len(t, registry.deadlines, 1)
			require.False(t, registry.deadlines[0].IsZero())
			require.WithinDuration(t, started.Add(modelDownloadCheckTimeout), registry.deadlines[0], time.Second)
		})
	}
}

func TestModelDownloadSpecChangeRemainsFailClosed(t *testing.T) {
	t.Log("Keep the source's download status independent of the unobserved child")
	source := newModelDownloadDGD(modelDownloadTestBuildID)
	source.Status.LPX = &v1beta1.DynamoGraphDeploymentLPXStatus{
		ModelDownload: &v1beta1.ModelDownloadStatus{Builds: []string{modelDownloadTestBuildID}},
	}
	child := newLPXTestDeployment(t, source)
	child.Status.ObservedGeneration = child.Generation - 1
	child.Status.Conditions = []metav1.Condition{{Type: "Ready", Status: metav1.ConditionTrue}}
	child.Status.ModelDownload = source.Status.LPX.ModelDownload.DeepCopy()
	child.Status.ModelDownload.LastCheckedAt = ptr.To(metav1.Now())
	registry := newModelDownloadRegistry(t, nil, map[string]error{modelDownloadTestBuildID: fmt.Errorf("ModelExpress unavailable")})
	lifecycle := &graphReconciler{modelRegistry: registry}
	t.Log("Do not reuse a fresh download check for an unobserved LPX generation")
	ready, err := lifecycle.reconcileModelDownloads(t.Context(), child.DeepCopy(), source, false)
	require.ErrorContains(t, err, "ModelExpress unavailable")
	require.False(t, ready)
	t.Log("The child controller persists failed download progress without publishing a PCS")
	r := newLPXTestReconciler(t, registry, child, source)
	_, err = r.Reconcile(t.Context(), ctrl.Request{NamespacedName: client.ObjectKeyFromObject(child)})
	require.ErrorContains(t, err, "ModelExpress unavailable")
	updated := &v1alpha1.LPXGraphDeployment{}
	require.NoError(t, r.Get(t.Context(), client.ObjectKeyFromObject(child), updated))
	require.NotNil(t, updated.Status.ModelDownload)
	require.Empty(t, updated.Status.ModelDownload.Builds)
	require.False(t, meta.IsStatusConditionTrue(updated.Status.Conditions, "Ready"))
	require.Equal(t, []string{modelDownloadTestBuildID}, source.Status.LPX.ModelDownload.Builds)
}

func TestLPXDisabledRevisionPreservesObservationFence(t *testing.T) {
	t.Log("A new revision starts with old download observations and no scheduling deadline")
	source := newModelDownloadDGD(modelDownloadTestSecondBuildID)
	source.Generation++
	child := newLPXTestDeployment(t, source)
	child.Status.ObservedGeneration = child.Generation
	child.Status.ModelDownload = &v1beta1.ModelDownloadStatus{
		Builds: []string{modelDownloadTestBuildID}, LastCheckedAt: ptr.To(metav1.Now()),
	}
	child.Generation++
	previousGeneration := child.Status.ObservedGeneration
	require.Nil(t, lpxRequestDeadlineSeconds(source))
	registry := newModelDownloadRegistry(t, map[string]bool{modelDownloadTestSecondBuildID: true}, nil)
	r := newLPXTestReconciler(t, registry, child, source)
	r.runtimeConfig.Gate = features.Gates{LPX: true}
	require.NoError(t, r.Get(t.Context(), client.ObjectKeyFromObject(child), child))

	t.Log("Disabled reconciliation reports the new failure without marking the new revision as observed")
	for range 2 {
		_, err := r.Reconcile(t.Context(), ctrl.Request{NamespacedName: client.ObjectKeyFromObject(child)})
		require.NoError(t, err)
		require.NoError(t, r.Get(t.Context(), client.ObjectKeyFromObject(child), child))
		require.Equal(t, previousGeneration, child.Status.ObservedGeneration)
		require.Nil(t, child.Status.ModelDownload)
		require.Empty(t, registry.calls, "the early exit has not checked either build")
		failed := meta.FindStatusCondition(child.Status.Conditions, "Failed")
		require.Equal(t, child.Generation, failed.ObservedGeneration)
		require.Equal(t, "LPXUnavailable", failed.Reason)
	}

	t.Log("Resumed download checking cannot reuse the previous revision's cached success")
	r.runtimeConfig.Gate.Grove = true
	ready, err := r.reconcileModelDownloads(t.Context(), child, source, false)
	require.NoError(t, err)
	require.True(t, ready)
	require.Equal(t, []string{modelDownloadTestSecondBuildID}, registry.calls)
	require.Equal(t, []string{modelDownloadTestSecondBuildID}, child.Status.ModelDownload.Builds)
	require.NotNil(t, child.Status.ModelDownload.LastCheckedAt)
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

func newModelDownloadDGD(buildIDs ...string) *v1beta1.DynamoGraphDeployment {
	dgd := &v1beta1.DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:       modelDownloadTestDGDName,
			Namespace:  modelDownloadTestNamespace,
			Generation: 1,
			UID:        types.UID("test-dgd-uid"),
		},
	}
	// Each build belongs to a distinct public component; only the target serves.
	for i, buildID := range buildIDs {
		component := v1beta1.DynamoComponentDeploymentSharedSpec{
			ComponentName: "lpx-worker", ComponentType: v1beta1.ComponentTypeLPX, Replicas: ptr.To(int32(1)),
			LPX: &v1beta1.LPXConfig{BuildID: buildID},
			Roles: []v1beta1.ComponentRoleSpec{{Name: v1beta1.ComponentRoleLPXAgent, PodTemplate: &corev1.PodTemplateSpec{Spec: corev1.PodSpec{
				Containers: []corev1.Container{{Name: "main", Image: "lpu-runtime"}},
			}}}},
		}
		if i == len(buildIDs)-1 {
			component.Roles = append(component.Roles, v1beta1.ComponentRoleSpec{
				Name: v1beta1.ComponentRoleLPXConductor,
				PodTemplate: &corev1.PodTemplateSpec{Spec: corev1.PodSpec{
					Containers: []corev1.Container{{Name: "main", Image: "lpu-runtime", Command: []string{"/bin/nova"}}},
				}},
			})
		} else {
			component.ComponentName = "draft"
		}
		dgd.Spec.Components = append(dgd.Spec.Components, component)
	}
	return dgd
}
