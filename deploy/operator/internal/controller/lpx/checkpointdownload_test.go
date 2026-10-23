/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	"context"
	"fmt"
	"net/url"
	"os"
	"path/filepath"
	"testing"
	"time"

	"capnproto.org/go/capnp/v3"

	grovev1alpha1 "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"

	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx"
	commoncapnpv1 "github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx/manifest/common/v1"
	deploymentcapnpv1 "github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx/manifest/deployment/v1"
)

const (
	checkpointTestModel    = "openai/gpt-oss-20b"
	checkpointTestRevision = "6cee5e81ee83917806bbde320786a8fb61efebee"
	checkpointTestKey      = checkpointTestModel + "@" + checkpointTestRevision
	checkpointTestOtherKey = "openai/gpt-oss-120b@0000000000000000000000000000000000000000"

	checkpointTestModelExpressURL = "http://modelexpress.dynamo-system.svc.cluster.local:8001"
)

func TestEnsureCheckpointsDownloaded(t *testing.T) {
	tests := []struct {
		name           string
		ready          map[string]bool
		err            map[string]error
		existing       []string
		wantErr        string
		wantCalls      []string
		wantDownloaded []string
		wantPending    []string
	}{
		{
			name:           "downloads every checkpoint",
			ready:          map[string]bool{checkpointTestKey: true, checkpointTestOtherKey: true},
			wantCalls:      []string{checkpointTestOtherKey, checkpointTestKey},
			wantDownloaded: []string{checkpointTestOtherKey, checkpointTestKey},
		},
		{
			name:           "waits for a checkpoint download in progress",
			ready:          map[string]bool{checkpointTestOtherKey: true},
			wantCalls:      []string{checkpointTestOtherKey, checkpointTestKey},
			wantDownloaded: []string{checkpointTestOtherKey},
			wantPending:    []string{checkpointTestKey},
		},
		{
			name:           "reports a download error with the checkpoint identity",
			ready:          map[string]bool{checkpointTestOtherKey: true},
			err:            map[string]error{checkpointTestKey: fmt.Errorf("repository not found")},
			wantErr:        `ensure checkpoint "` + checkpointTestKey + `" is downloaded: repository not found`,
			wantCalls:      []string{checkpointTestOtherKey, checkpointTestKey},
			wantDownloaded: []string{checkpointTestOtherKey},
		},
		{
			name:           "reuses fresh checkpoint downloads",
			existing:       []string{checkpointTestOtherKey, checkpointTestKey},
			wantDownloaded: []string{checkpointTestOtherKey, checkpointTestKey},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Log("Select two checkpoints in key order")
			checkpoints := []*lpx.Checkpoint{
				{Model: "openai/gpt-oss-120b", Revision: "0000000000000000000000000000000000000000"},
				{Model: checkpointTestModel, Revision: checkpointTestRevision},
			}
			registry := newModelDownloadRegistry(t, tt.ready, tt.err)
			started := time.Now()

			t.Log("Check every uncached checkpoint within one shared budget")
			downloaded, pending, err := ensureCheckpointsDownloaded(t.Context(), checkpoints, registry, tt.existing)
			if tt.wantErr != "" {
				require.EqualError(t, err, tt.wantErr)
			} else {
				require.NoError(t, err)
			}
			require.Equal(t, tt.wantCalls, registry.checkpointCalls)
			require.Equal(t, tt.wantDownloaded, downloaded)
			require.Equal(t, tt.wantPending, pending)
			for _, deadline := range registry.deadlines {
				require.WithinDuration(t, started.Add(modelDownloadCheckTimeout/2), deadline, time.Second)
			}
		})
	}
}

func TestHybridCheckpointDownloadGatesRendering(t *testing.T) {
	t.Log("Deploy a hybrid build whose deployment contract names a checkpoint that is still downloading")
	child, dgd, baseRegistry := newLPXCheckpointTestDGD(t, deploymentcapnpv1.CompilationMode_lpx)
	registry := &fakeModelDownloadRegistry{ModelRegistry: baseRegistry}
	reconciler := newLPXCheckpointTestReconciler(t, registry, child, dgd)
	request := ctrl.Request{NamespacedName: client.ObjectKeyFromObject(child)}

	t.Log("Hold the workload and name the pending checkpoint in the Ready condition")
	result, err := reconciler.Reconcile(t.Context(), request)
	require.NoError(t, err)
	require.Equal(t, modelDownloadRequeueAfter, result.RequeueAfter)
	require.Equal(t, []string{checkpointTestKey}, registry.checkpointCalls)
	require.NoError(t, reconciler.Get(t.Context(), request.NamespacedName, child))
	ready := meta.FindStatusCondition(child.Status.Conditions, v1alpha1.LPXReadyCondition)
	require.NotNil(t, ready)
	require.Equal(t, metav1.ConditionFalse, ready.Status)
	require.Equal(t, checkpointDownloadPendingMessage+": "+checkpointTestKey, ready.Message)
	require.NotNil(t, child.Status.CheckpointDownload)
	require.Empty(t, child.Status.CheckpointDownload.Checkpoints)
	sets := &grovev1alpha1.PodCliqueSetList{}
	require.NoError(t, reconciler.List(t.Context(), sets))
	require.Empty(t, sets.Items)

	t.Log("Render the workload with the checkpoint snapshot once Model Express completes it")
	registry.ready = map[string]bool{checkpointTestKey: true}
	_, err = reconciler.Reconcile(t.Context(), request)
	require.NoError(t, err)
	require.NoError(t, reconciler.Get(t.Context(), request.NamespacedName, child))
	require.Equal(t, []string{checkpointTestKey}, child.Status.CheckpointDownload.Checkpoints)
	require.NotNil(t, child.Status.CheckpointDownload.LastCheckedAt)
	require.NoError(t, reconciler.List(t.Context(), sets))
	require.Len(t, sets.Items, 1)
	require.Equal(t, []string{"/models/models--openai--gpt-oss-20b/snapshots/" + checkpointTestRevision}, renderedWeightsPaths(sets.Items[0]))
}

func TestLPUOnlyBuildIgnoresItsCheckpoint(t *testing.T) {
	t.Log("Deploy an LPU-only build whose deployment contract names a checkpoint")
	child, dgd, baseRegistry := newLPXCheckpointTestDGD(t, deploymentcapnpv1.CompilationMode_lpuOnly)
	registry := &fakeModelDownloadRegistry{ModelRegistry: baseRegistry}
	reconciler := newLPXCheckpointTestReconciler(t, registry, child, dgd)

	t.Log("Render without a download because no Cyborg conductor reads the checkpoint")
	_, err := reconciler.Reconcile(t.Context(), ctrl.Request{NamespacedName: client.ObjectKeyFromObject(child)})
	require.NoError(t, err)
	require.Empty(t, registry.checkpointCalls)
	sets := &grovev1alpha1.PodCliqueSetList{}
	require.NoError(t, reconciler.List(t.Context(), sets))
	require.Len(t, sets.Items, 1)
	require.Empty(t, renderedWeightsPaths(sets.Items[0]))
}

func (r *fakeModelDownloadRegistry) EnsureCheckpointDownloaded(ctx context.Context, checkpoint *lpx.Checkpoint) (bool, error) {
	key := lpx.CheckpointKey(checkpoint)
	r.checkpointCalls = append(r.checkpointCalls, key)
	deadline, _ := ctx.Deadline()
	r.deadlines = append(r.deadlines, deadline)
	return r.ready[key], r.err[key]
}

func TestDownloadRefreshAt(t *testing.T) {
	buildCheck := metav1.NewTime(time.Date(2026, 10, 6, 12, 0, 0, 0, time.UTC))
	checkpointCheck := metav1.NewTime(buildCheck.Add(-time.Hour))
	tests := []struct {
		name          string
		status        v1alpha1.LPXGraphDeploymentStatus
		wantRefreshAt time.Time
		wantFound     bool
	}{
		{name: "no recorded check"},
		{
			name:          "build check only",
			status:        v1alpha1.LPXGraphDeploymentStatus{ModelDownload: &v1alpha1.ModelDownloadStatus{LastCheckedAt: &buildCheck}},
			wantRefreshAt: buildCheck.Add(modelDownloadRefreshInterval),
			wantFound:     true,
		},
		{
			name:          "checkpoint check with a local build",
			status:        v1alpha1.LPXGraphDeploymentStatus{CheckpointDownload: &v1alpha1.CheckpointDownloadStatus{LastCheckedAt: &checkpointCheck}},
			wantRefreshAt: checkpointCheck.Add(modelDownloadRefreshInterval),
			wantFound:     true,
		},
		{
			name: "earlier of both checks",
			status: v1alpha1.LPXGraphDeploymentStatus{
				ModelDownload:      &v1alpha1.ModelDownloadStatus{LastCheckedAt: &buildCheck},
				CheckpointDownload: &v1alpha1.CheckpointDownloadStatus{LastCheckedAt: &checkpointCheck},
			},
			wantRefreshAt: checkpointCheck.Add(modelDownloadRefreshInterval),
			wantFound:     true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Log("Schedule the Ready refresh from the earliest recorded download check")
			refreshAt, found := downloadRefreshAt(&tt.status)
			require.Equal(t, tt.wantFound, found)
			require.True(t, tt.wantRefreshAt.Equal(refreshAt), "refreshAt = %v, want %v", refreshAt, tt.wantRefreshAt)
		})
	}
}

func TestAdoptBuildCheckpoints(t *testing.T) {
	const modelExpressURL = checkpointTestModelExpressURL
	for _, test := range []struct {
		name            string
		modelExpressURL string
		observed        int64
		recorded        bool
		want            bool
	}{
		{name: "no Model Express ignores build checkpoints", observed: 1},
		{name: "new input adopts build checkpoints", modelExpressURL: modelExpressURL, observed: 1, want: true},
		{name: "operator upgrade leaves a running deployment alone", modelExpressURL: modelExpressURL, observed: 2},
		{name: "recorded adoption persists", modelExpressURL: modelExpressURL, observed: 2, recorded: true, want: true},
	} {
		t.Run(test.name, func(t *testing.T) {
			t.Log("Observe a deployment at generation 2")
			deployment := &v1alpha1.LPXGraphDeployment{Generation: 2}
			deployment.Status.ObservedGeneration = test.observed
			if test.recorded {
				deployment.Status.CheckpointDownload = &v1alpha1.CheckpointDownloadStatus{}
			}

			t.Log("Adopt build checkpoints only with Model Express, for new input or an existing record")
			markBuildCheckpointAdoption(test.modelExpressURL, deployment)
			require.Equal(t, test.want, adoptBuildCheckpoints(test.modelExpressURL, deployment))
		})
	}
}

func TestBuildCheckpointAdoptionSurvivesPendingBuildDownload(t *testing.T) {
	const modelExpressURL = checkpointTestModelExpressURL

	t.Log("Record adoption when a new deployment is first reconciled")
	deployment := &v1alpha1.LPXGraphDeployment{Generation: 1}
	markBuildCheckpointAdoption(modelExpressURL, deployment)

	t.Log("A reconcile that waits for the build download still observes the generation")
	deployment.Status.ObservedGeneration = deployment.Generation

	t.Log("Render after the build download with the build checkpoint adopted")
	markBuildCheckpointAdoption(modelExpressURL, deployment)
	require.True(t, adoptBuildCheckpoints(modelExpressURL, deployment))

	t.Log("A resolution without checkpoints keeps the record")
	r := &graphReconciler{}
	result, err := r.reconcileCheckpointDownloads(t.Context(), deployment, nil)
	require.NoError(t, err)
	require.Zero(t, result)
	require.True(t, adoptBuildCheckpoints(modelExpressURL, deployment))

	t.Log("A deployment that never recorded adoption still has no record")
	upgraded := &v1alpha1.LPXGraphDeployment{Generation: 1}
	upgraded.Status.ObservedGeneration = upgraded.Generation
	_, err = r.reconcileCheckpointDownloads(t.Context(), upgraded, nil)
	require.NoError(t, err)
	require.Nil(t, upgraded.Status.CheckpointDownload)
}

func TestReadyCheckpointRefreshGatesAChangedCheckpoint(t *testing.T) {
	for _, test := range []struct {
		name        string
		recorded    string
		wantRequeue time.Duration
		wantSets    int
	}{
		{name: "same checkpoint tolerates a pending refresh", recorded: checkpointTestKey, wantSets: 1},
		{name: "changed checkpoint stays gated", recorded: checkpointTestOtherKey, wantRequeue: modelDownloadRequeueAfter},
	} {
		t.Run(test.name, func(t *testing.T) {
			t.Log("Observe a Ready deployment whose stale record confirms one checkpoint")
			child, dgd, baseRegistry := newLPXCheckpointTestDGD(t, deploymentcapnpv1.CompilationMode_lpx)
			registry := &fakeModelDownloadRegistry{ModelRegistry: baseRegistry}
			reconciler := newLPXCheckpointTestReconciler(t, registry, child, dgd)
			request := ctrl.Request{NamespacedName: client.ObjectKeyFromObject(child)}
			require.NoError(t, reconciler.Get(t.Context(), request.NamespacedName, child))
			child.Status.ObservedGeneration = child.Generation
			child.Status.Conditions = []metav1.Condition{{
				Type: v1alpha1.LPXReadyCondition, Status: metav1.ConditionTrue, Reason: "Ready", LastTransitionTime: metav1.Now(),
			}}
			child.Status.CheckpointDownload = &v1alpha1.CheckpointDownloadStatus{
				Checkpoints:   []string{test.recorded},
				LastCheckedAt: &metav1.Time{Time: time.Now().Add(-modelDownloadRefreshInterval)},
			}
			require.NoError(t, reconciler.Status().Update(t.Context(), child))

			t.Log("Check the build's checkpoint while its download is pending")
			result, err := reconciler.Reconcile(t.Context(), request)
			require.NoError(t, err)
			require.Equal(t, []string{checkpointTestKey}, registry.checkpointCalls)
			require.Equal(t, test.wantRequeue, result.RequeueAfter)
			sets := &grovev1alpha1.PodCliqueSetList{}
			require.NoError(t, reconciler.List(t.Context(), sets))
			require.Len(t, sets.Items, test.wantSets)
		})
	}
}

func TestPendingCheckpointKeepsReleasingAgentPods(t *testing.T) {
	for _, test := range []struct {
		name        string
		err         error
		wantRequeue time.Duration
		wantErr     string
	}{
		{name: "download in progress", wantRequeue: modelDownloadRequeueAfter},
		{
			name:    "download error",
			err:     fmt.Errorf("model express unavailable"),
			wantErr: fmt.Sprintf("ensure checkpoint %q is downloaded: model express unavailable", checkpointTestKey),
		},
	} {
		t.Run(test.name, func(t *testing.T) {
			t.Log("Publish a hybrid workload whose checkpoint is not downloaded")
			ctx := t.Context()
			child, dgd, baseRegistry := newLPXCheckpointTestDGD(t, deploymentcapnpv1.CompilationMode_lpx)
			registry := &fakeModelDownloadRegistry{ModelRegistry: baseRegistry, err: map[string]error{checkpointTestKey: test.err}}
			r, selected := newPreparedLPXTestReconciler(t, registry, ctx, child, dgd)
			r.config.Infrastructure.ModelExpressURL = checkpointTestModelExpressURL
			objects := lpxMaterializedObjects(t, r, child, dgd, selected)
			createLPXTestObjects(t, ctx, r.Client, objects...)
			publishSelectedLPXForTest(t, ctx, r, child, selected)

			t.Log("Journal one Agent Pod for release")
			agentClique := getResource[*grovev1alpha1.PodClique](t, objects, selected.plan.Agents[0].CliqueName)
			pod := &corev1.Pod{
				Namespace: child.Namespace, Name: "released", UID: "released-uid",
				OwnerReferences: []metav1.OwnerReference{*metav1.NewControllerRef(agentClique, grovev1alpha1.SchemeGroupVersion.WithKind("PodClique"))}}
			sibling := pod.DeepCopy()
			sibling.Name, sibling.UID = "sibling", "sibling-uid"
			createLPXTestObjects(t, ctx, r.Client, pod, sibling)
			request := getTestPipelineRequest(t, ctx, r, child.Namespace, selected.requests[0].Name)
			request.Status = newLPXTestReleaseRequest(pod).Status
			require.NoError(t, r.Update(ctx, request))

			t.Log("Release the journaled Pod once while the checkpoint holds forward publication")
			result, err := r.Reconcile(ctx, ctrl.Request{NamespacedName: client.ObjectKeyFromObject(child)})
			if test.wantErr != "" {
				require.EqualError(t, err, test.wantErr)
			} else {
				require.NoError(t, err)
				require.Equal(t, test.wantRequeue, result.RequeueAfter)
			}
			require.Equal(t, []string{checkpointTestKey}, registry.checkpointCalls)
			require.True(t, apierrors.IsNotFound(r.Get(ctx, client.ObjectKeyFromObject(pod), &corev1.Pod{})))
			require.NoError(t, r.Get(ctx, client.ObjectKeyFromObject(sibling), &corev1.Pod{}))
		})
	}
}

func TestPendingCheckpointHoldsAnAddedWorkload(t *testing.T) {
	t.Log("Publish one hybrid workload whose checkpoint is downloaded")
	ctx := t.Context()
	_, dgd, baseRegistry := newLPXCheckpointTestDGD(t, deploymentcapnpv1.CompilationMode_lpx)
	dgd.Spec.Components[0].Replicas = new(int32(1))
	child := newLPXTestDeployment(t, dgd)
	registry := &fakeModelDownloadRegistry{ModelRegistry: baseRegistry, ready: map[string]bool{checkpointTestKey: true}}
	r, selected := newPreparedLPXTestReconciler(t, registry, ctx, child, dgd)
	r.config.Infrastructure.ModelExpressURL = checkpointTestModelExpressURL
	objects := lpxMaterializedObjects(t, r, child, dgd, selected)
	createLPXTestObjects(t, ctx, r.Client, objects...)
	publishSelectedLPXForTest(t, ctx, r, child, selected)
	key := ctrl.Request{NamespacedName: client.ObjectKeyFromObject(child)}
	_, err := r.Reconcile(ctx, key)
	require.NoError(t, err)
	before := findLPXTestPodCliqueSet(t, objects)

	t.Log("Add a second workload, which renames the scaling groups, while its checkpoint download is pending")
	require.NoError(t, r.Get(ctx, client.ObjectKeyFromObject(dgd), dgd))
	second := dgd.Spec.Components[0].DeepCopy()
	second.ComponentName = "second"
	dgd.Spec.Components = append(dgd.Spec.Components, *second)
	require.NoError(t, r.Update(ctx, dgd))
	require.NoError(t, r.Get(ctx, key.NamespacedName, child))
	child.Spec.InputRevision = mustLPXInputRevision(t, dgd)
	child.Generation++
	require.NoError(t, r.Update(ctx, child))
	child.Status.CheckpointDownload = &v1alpha1.CheckpointDownloadStatus{}
	require.NoError(t, r.Status().Update(ctx, child))
	registry.ready = nil

	t.Log("Hold the existing PodCliqueSet without deriving capacity for the new layout")
	result, err := r.Reconcile(ctx, key)
	require.NoError(t, err)
	require.Equal(t, modelDownloadRequeueAfter, result.RequeueAfter)
	live := &grovev1alpha1.PodCliqueSet{}
	require.NoError(t, r.Get(ctx, client.ObjectKeyFromObject(before), live))
	require.True(t, live.DeletionTimestamp.IsZero())
	require.Equal(t, before.Spec.Template.PodCliqueScalingGroupConfigs, live.Spec.Template.PodCliqueScalingGroupConfigs)
}

func TestPendingCheckpointReusesConfirmedCheckpoints(t *testing.T) {
	t.Log("Deploy two hybrid workloads whose builds name different checkpoints")
	const otherBuildID = "build-v1-other"
	root := t.TempDir()
	writeLPXCheckpointTestBuild(t, root, "build-v1", deploymentcapnpv1.CompilationMode_lpx, checkpointTestModel, checkpointTestRevision)
	writeLPXCheckpointTestBuild(t, root, otherBuildID, deploymentcapnpv1.CompilationMode_lpx, "openai/gpt-oss-120b", "0000000000000000000000000000000000000000")
	baseRegistry, err := lpx.NewModelRegistry((&url.URL{Scheme: lpx.BuildSchemeFile, Path: root}).String(), nil)
	require.NoError(t, err)
	dgd := loadTestDGD(t, lpx.PipelineHybrid, "build-v1")
	second := dgd.Spec.Components[0].DeepCopy()
	second.ComponentName = "second"
	second.LPX.BuildID = otherBuildID
	dgd.Spec.Components = append(dgd.Spec.Components, *second)
	child := newLPXTestDeployment(t, dgd)
	registry := &fakeModelDownloadRegistry{ModelRegistry: baseRegistry, ready: map[string]bool{checkpointTestKey: true}}
	reconciler := newLPXCheckpointTestReconciler(t, registry, child, dgd)
	request := ctrl.Request{NamespacedName: client.ObjectKeyFromObject(child)}

	t.Log("Record the confirmed checkpoint and its confirmation time while the other downloads")
	result, err := reconciler.Reconcile(t.Context(), request)
	require.NoError(t, err)
	require.Equal(t, modelDownloadRequeueAfter, result.RequeueAfter)
	require.NoError(t, reconciler.Get(t.Context(), request.NamespacedName, child))
	require.Equal(t, []string{checkpointTestKey}, child.Status.CheckpointDownload.Checkpoints)
	confirmedAt := child.Status.CheckpointDownload.LastCheckedAt
	require.NotNil(t, confirmedAt)

	t.Log("Check only the pending checkpoint on the next pass and keep the original confirmation time")
	registry.checkpointCalls = nil
	_, err = reconciler.Reconcile(t.Context(), request)
	require.NoError(t, err)
	require.Equal(t, []string{checkpointTestOtherKey}, registry.checkpointCalls)
	require.NoError(t, reconciler.Get(t.Context(), request.NamespacedName, child))
	require.True(t, confirmedAt.Equal(child.Status.CheckpointDownload.LastCheckedAt))
}

// newLPXCheckpointTestDGD returns a deployment of a deployment.v1 build that
// names the test checkpoint. The hybrid build adds a CUDA partition.
func newLPXCheckpointTestDGD(
	t *testing.T,
	mode deploymentcapnpv1.CompilationMode,
) (*v1alpha1.LPXGraphDeployment, *v1beta1.DynamoGraphDeployment, lpx.ModelRegistry) {
	t.Helper()
	const buildID = "build-v1"
	root := t.TempDir()
	writeLPXCheckpointTestBuild(t, root, buildID, mode, checkpointTestModel, checkpointTestRevision)
	registry, err := lpx.NewModelRegistry((&url.URL{Scheme: lpx.BuildSchemeFile, Path: root}).String(), nil)
	require.NoError(t, err)

	pipeline := lpx.PipelineSingle
	if mode == deploymentcapnpv1.CompilationMode_lpx {
		pipeline = lpx.PipelineHybrid
	}
	dgd := loadTestDGD(t, pipeline, buildID)
	return newLPXTestDeployment(t, dgd), dgd, registry
}

func writeLPXCheckpointTestBuild(
	t *testing.T,
	root, buildID string,
	mode deploymentcapnpv1.CompilationMode,
	model, revision string,
) {
	t.Helper()
	partitionIDs := []int{7, 8}

	t.Log("Encode a deployment.v1 build with one prop-sync chain and a checkpoint")
	_, segment := capnp.NewSingleSegmentMessage(nil)
	deployment, err := deploymentcapnpv1.NewRootDeployment(segment)
	require.NoError(t, err)
	deployment.SetContractRevision(deploymentcapnpv1.CurrentContractRevision)
	config, err := deployment.NewConfig()
	require.NoError(t, err)
	config.SetCompilationMode(mode)
	config.SetNumLpuNodes(uint32(len(partitionIDs) * 2))
	program, err := config.NewProgram()
	require.NoError(t, err)
	program.SetBatchSize(1)
	runtimeIO, err := config.NewRuntimeIo()
	require.NoError(t, err)
	runtimeIO.SetProtocol(deploymentcapnpv1.RuntimeIoProtocol_fpgaRoce)
	runtimeIO.SetIoFpgaCount(1)
	runtimeIO.SetFanoutFactor(1)
	chains, err := config.NewSelectedPropSyncChains(1)
	require.NoError(t, err)
	chainIDs, err := chains.At(0).NewPartitionIds(int32(len(partitionIDs)))
	require.NoError(t, err)
	for index, partitionID := range partitionIDs {
		chainIDs.Set(index, uint32(partitionID))
	}
	checkpoint, err := deployment.NewCheckpoint()
	require.NoError(t, err)
	checkpoint.SetProvider(deploymentcapnpv1.CheckpointProvider_huggingface)
	require.NoError(t, checkpoint.SetModel(model))
	require.NoError(t, checkpoint.SetRevision(revision))

	partitionCount := len(partitionIDs)
	if mode == deploymentcapnpv1.CompilationMode_lpx {
		partitionCount++
	}
	partitions, err := deployment.NewPartitions(int32(partitionCount))
	require.NoError(t, err)
	for index, partitionID := range partitionIDs {
		ref, err := partitions.At(index).NewPartition()
		require.NoError(t, err)
		ref.SetDeviceType(commoncapnpv1.DeviceType_lpu)
		ref.SetPartitionId(uint32(partitionID))
		detail, err := partitions.At(index).Detail().NewLpu()
		require.NoError(t, err)
		require.NoError(t, detail.SetPath(fmt.Sprintf("part-%d", partitionID)))
		require.NoError(t, detail.SetTopology("test-topology"))
		detail.SetArchitecture(deploymentcapnpv1.LpuArchitecture_lp20)
		detail.SetNumChips(16)
		detail.SetDevicesPerNode(8)
	}
	if mode == deploymentcapnpv1.CompilationMode_lpx {
		ref, err := partitions.At(len(partitionIDs)).NewPartition()
		require.NoError(t, err)
		ref.SetDeviceType(commoncapnpv1.DeviceType_cuda)
		ref.SetPartitionId(uint32(len(partitionIDs)))
	}

	t.Log("Publish the build in the file registry")
	payload, err := deployment.Message().Marshal()
	require.NoError(t, err)
	require.NoError(t, os.Mkdir(filepath.Join(root, buildID), 0o700))
	require.NoError(t, os.WriteFile(filepath.Join(root, buildID, "deployment.v1.capnp.bin"), payload, 0o600))
}

func newLPXCheckpointTestReconciler(
	t *testing.T,
	registry lpx.ModelRegistry,
	deployment *v1alpha1.LPXGraphDeployment,
	dgd *v1beta1.DynamoGraphDeployment,
) *graphReconciler {
	t.Helper()
	reconciler := newLPXTestReconciler(t, registry, deployment, dgd)
	reconciler.config.Infrastructure.ModelExpressURL = checkpointTestModelExpressURL
	return reconciler
}

func renderedWeightsPaths(set grovev1alpha1.PodCliqueSet) []string {
	var paths []string
	for _, clique := range set.Spec.Template.Cliques {
		for _, container := range clique.Spec.PodSpec.Containers {
			for _, variable := range container.Env {
				if variable.Name == "CYBORG_WEIGHTS_PATH" {
					paths = append(paths, variable.Value)
				}
			}
		}
	}
	return paths
}
