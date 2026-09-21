/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	"context"
	"errors"
	"fmt"
	"maps"
	"net/url"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"testing"
	"time"
	"unicode/utf8"

	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	commoncontroller "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx"
	manifestcapnpv2 "github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx/manifest/v2"
	lpxv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx/scheduler/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/features"
	grovecommon "github.com/ai-dynamo/grove/operator/api/common"
	groveconstants "github.com/ai-dynamo/grove/operator/api/common/constants"
	grovev1alpha1 "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	"github.com/stretchr/testify/require"
	autoscalingv1 "k8s.io/api/autoscaling/v1"
	corev1 "k8s.io/api/core/v1"
	resourcev1 "k8s.io/api/resource/v1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	k8sruntime "k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/tools/events"
	"k8s.io/utils/ptr"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
	"sigs.k8s.io/controller-runtime/pkg/client/interceptor"
)

const lpxTestOtherName = "other"

func newLPXTestScheme(t testing.TB) *k8sruntime.Scheme {
	t.Helper()
	scheme := k8sruntime.NewScheme()
	for _, add := range []func(*k8sruntime.Scheme) error{
		corev1.AddToScheme, resourcev1.AddToScheme, nvidiacomv1alpha1.AddToScheme, nvidiacomv1beta1.AddToScheme,
		grovev1alpha1.AddToScheme, lpxv1alpha1.AddToScheme,
	} {
		require.NoError(t, add(scheme))
	}
	return scheme
}

type snapshotFailureRegistry struct {
	lpx.ModelRegistry
	err error
}

type downloadOrderedLPXRegistry struct {
	lpx.ModelRegistry
	buildURL   url.URL
	downloaded bool
	calls      []string
}

func (r *snapshotFailureRegistry) AcquireBuildSnapshot(context.Context, string) (*lpx.BuildSnapshot, error) {
	return nil, r.err
}

func (r *downloadOrderedLPXRegistry) BuildURL(string) (*url.URL, error) {
	buildURL := r.buildURL
	return &buildURL, nil
}

func (r *downloadOrderedLPXRegistry) EnsureDownloaded(context.Context, url.URL) (bool, error) {
	firstDownload := !slices.Contains(r.calls, "download")
	r.calls = append(r.calls, "download")
	if firstDownload {
		return false, nil
	}
	r.downloaded = true
	return true, nil
}

func (r *downloadOrderedLPXRegistry) AcquireBuildSnapshot(
	ctx context.Context,
	buildID string,
) (*lpx.BuildSnapshot, error) {
	r.calls = append(r.calls, "snapshot")
	if !r.downloaded {
		return nil, errors.New("LPX snapshot acquired before Model Express download")
	}
	return r.ModelRegistry.AcquireBuildSnapshot(ctx, buildID)
}

func requirePreparedLPX(
	t *testing.T,
	reconciler *graphReconciler,
	ctx context.Context,
	dgd *nvidiacomv1alpha1.LPXGraphDeployment,
	source *nvidiacomv1beta1.DynamoGraphDeployment,
) (*lpxMaterializing, *lpxRejected) {
	t.Helper()
	desired, classification, err := reconciler.prepareLPXMaterializing(ctx, dgd, source, nil)
	require.NoError(t, err)
	if classification == nil {
		return desired, nil
	}
	rejected, ok := classification.(*lpxRejected)
	require.True(t, ok, "classification=%T", classification)
	return desired, rejected
}

func newPreparedLPXTestReconciler(
	t *testing.T,
	registry lpx.ModelRegistry,
	ctx context.Context,
	dgd *nvidiacomv1alpha1.LPXGraphDeployment,
	source *nvidiacomv1beta1.DynamoGraphDeployment,
) (*graphReconciler, *lpxMaterializing) {
	t.Helper()

	// Construct the fake client before preparing the selected LPX plan.
	reconciler := newLPXTestReconciler(t, registry, dgd, source)
	desired, rejected := requirePreparedLPX(t, reconciler, ctx, dgd, source)
	require.Nil(t, rejected)
	return reconciler, desired
}

func TestImplicitV2LPXConductorlessGroveIdentityPublishesRequest(t *testing.T) {
	t.Log("Publish the implicit hybrid runtime without interpreting stale selector annotations")
	ctx := t.Context()
	dgd, source, registry := newLPXTestDGD(t, lpx.PipelineLPX)
	source.Annotations[consts.KubeAnnotationLPXSchedulerBackend] = "unknown-scheduler"
	source.Annotations[consts.KubeAnnotationLPXExecutionBackend] = "unknown-execution"
	reconciler, desired := newPreparedLPXTestReconciler(t, registry, ctx, dgd, source)
	require.NotNil(t, desired)
	require.Equal(t, "unknown-scheduler", source.Annotations[consts.KubeAnnotationLPXSchedulerBackend])
	require.Equal(t, "unknown-execution", source.Annotations[consts.KubeAnnotationLPXExecutionBackend])
	require.Empty(t, desired.plan.ConductorTemplate)
	require.Empty(t, desired.plan.ConductorClique)
	require.NotEmpty(t, desired.plan.CyborgClique)

	objects := lpxMaterializedObjects(t, reconciler, dgd, source, desired)
	for _, object := range objects {
		require.NotEmpty(t, object.GetName())
	}
	group := findLPXTestScalingGroup(t, objects, desired.plan.LPXScalingGroup)
	require.NotContains(t, group.Spec.CliqueNames, "")
	cyborg := findLPXTestClique(t, objects, desired.plan.CyborgClique)
	require.NotContains(t, cyborg.Spec.StartsAfter, "")
	require.Equal(t, lpx.SchedulerName, cyborg.Spec.PodSpec.SchedulerName)
	createLPXTestObjects(t, ctx, reconciler.Client, objects...)

	classification := publishSelectedLPXForTest(t, ctx, reconciler, dgd, desired)
	require.IsType(t, &lpxOpen{}, classification)
	request := getLPXRequest(t, ctx, reconciler.Client, dgd.Namespace, desired.requests[0].requestName)
	require.NotNil(t, request.Spec.CyborgPodCliqueRef)
	require.Equal(t, desired.plan.CyborgClique, request.Spec.CyborgPodCliqueRef.Name)
}

func TestSelectedLPXColdCacheDownloadsBeforeSnapshot(t *testing.T) {
	t.Log("Build a selected LPX deployment backed by an initially cold Model Express cache")
	child, source, baseRegistry := newLPXTestDGD(t, lpx.PipelineSingle)
	registry := &downloadOrderedLPXRegistry{
		ModelRegistry: baseRegistry,
		buildURL: url.URL{
			Scheme: lpx.BuildSchemeGCS,
			Host:   "test-bucket",
			Path:   "/build-v2",
		},
	}
	child.Status.Conditions = []metav1.Condition{{Type: "Ready", Status: metav1.ConditionTrue, Reason: "Ready", ObservedGeneration: child.Generation}}
	child.Status.ObservedGeneration = child.Generation
	child.Status.ModelDownload = &nvidiacomv1beta1.ModelDownloadStatus{
		Builds:        []string{registry.buildURL.String()},
		LastCheckedAt: &metav1.Time{Time: time.Now()},
	}
	reconciler := newLPXTestReconciler(t, registry, child, source)
	request := ctrl.Request{NamespacedName: client.ObjectKeyFromObject(child)}

	t.Log("Force Model Express despite the fresh cached status and avoid acquiring a build snapshot")
	result, err := reconciler.Reconcile(t.Context(), request)
	require.NoError(t, err)
	require.Equal(t, modelDownloadRequeueAfter, result.RequeueAfter)
	require.Equal(t, []string{"download"}, registry.calls)
	require.NoError(t, reconciler.Get(t.Context(), request.NamespacedName, child))
	require.False(t, meta.IsStatusConditionTrue(child.Status.Conditions, "Ready"))
	require.NotNil(t, child.Status.ModelDownload)
	require.Empty(t, child.Status.ModelDownload.Builds)
	sets := &grovev1alpha1.PodCliqueSetList{}
	require.NoError(t, reconciler.List(t.Context(), sets))
	require.Empty(t, sets.Items)

	t.Log("Complete the download and reconcile the selected deployment again")
	_, err = reconciler.Reconcile(t.Context(), request)
	require.NoError(t, err)
	require.Equal(t, []string{"download", "download", "snapshot"}, registry.calls)
	require.NoError(t, reconciler.Get(t.Context(), request.NamespacedName, child))
	require.NotNil(t, child.Status.ModelDownload)
	require.Equal(t, []string{registry.buildURL.String()}, child.Status.ModelDownload.Builds)
}

func TestSelectedLPXSuccessorBuildDownloadsBeforePublication(t *testing.T) {
	t.Log("Publish the previous build before switching to an uncached successor")
	ctx := t.Context()
	child, source, baseRegistry := newLPXTestDGD(t, lpx.PipelineSingle)
	r, selected := newPreparedLPXTestReconciler(t, baseRegistry, ctx, child, source)
	objects := lpxMaterializedObjects(t, r, child, source, selected)
	createLPXTestObjects(t, ctx, r.Client, objects...)
	publishSelectedLPXForTest(t, ctx, r, child, selected)
	previous := getLPXRequest(t, ctx, r.Client, child.Namespace, selected.requests[0].requestName)
	const nextBuild = "next-build"
	registry := &downloadOrderedLPXRegistry{
		ModelRegistry: newLPXTestRegistryWithPartitionsAndMode(t, nextBuild, []int{7, 8}, manifestcapnpv2.CompilationMode_lpuOnly),
		buildURL:      url.URL{Scheme: lpx.BuildSchemeGCS, Host: "test-bucket", Path: "/" + nextBuild},
	}
	r.modelRegistry = registry
	require.NoError(t, r.Get(ctx, client.ObjectKeyFromObject(source), source))
	lpx.ServingComponent(source).LPX.BuildID = nextBuild
	source.Generation++
	require.NoError(t, r.Update(ctx, source))
	require.NoError(t, r.Get(ctx, client.ObjectKeyFromObject(child), child))
	child.Generation++
	var err error
	child.Spec.InputRevision, err = dynamo.LPXInputRevision(source, "")
	require.NoError(t, err)
	require.NoError(t, r.Update(ctx, child))

	t.Log("Start downloading despite snapshot unavailability, preserving the existing request")
	request := ctrl.Request{NamespacedName: client.ObjectKeyFromObject(child)}
	result, err := r.Reconcile(ctx, request)
	require.NoError(t, err)
	require.Equal(t, modelDownloadRequeueAfter, result.RequeueAfter)
	require.Equal(t, []string{"snapshot", "download"}, registry.calls)
	require.Equal(t, previous, getLPXRequest(t, ctx, r.Client, previous.Namespace, previous.Name))
	require.NoError(t, r.Get(ctx, request.NamespacedName, child))
	require.Equal(t, child.Generation, child.Status.ObservedGeneration)

	t.Log("Resolve the successor after download completes on the next reconciliation")
	_, err = r.Reconcile(ctx, request)
	require.NoError(t, err)
	require.Equal(t, []string{"snapshot", "download", "snapshot", "download", "snapshot"}, registry.calls)
	require.NoError(t, r.Get(ctx, request.NamespacedName, child))
	require.Equal(t, []string{registry.buildURL.String()}, child.Status.ModelDownload.Builds)
	require.NotNil(t, child.Status.ModelDownload.LastCheckedAt)
}

func TestNodeLocalSpecDecodePublishesOneRequestAndAgentCliquePerModelProjection(t *testing.T) {
	ctx := t.Context()
	dgd, source, registry := newLPXSpecDecodeTestDGD(t)

	reconciler, desired := newPreparedLPXTestReconciler(t, registry, ctx, dgd, source)
	require.Len(t, desired.requests, 3)
	require.Len(t, desired.plan.Agents, 3)
	objects := lpxMaterializedObjects(t, reconciler, dgd, source, desired)
	createLPXTestObjects(t, ctx, reconciler.Client, objects...)

	classification := publishSelectedLPXForTest(t, ctx, reconciler, dgd, desired)
	require.IsType(t, &lpxOpen{}, classification)
	requests, err := reconciler.listOwnedLPXRequests(ctx, dgd, findLPXTestPodCliqueSet(t, objects))
	require.NoError(t, err)
	require.Len(t, requests, 3)

	requestByModel := make(map[string]lpxv1alpha1.LPUPipelineRequest, len(requests))
	for _, request := range requests {
		model := request.Annotations[lpxModelAnnotation]
		requestByModel[model] = request
		require.Equal(t, string(dgd.UID), request.Labels[lpxOwnerUIDLabel])
		require.Equal(t, model, request.Spec.NodeLocal.Model)
		require.Equal(t, desired.plan.LPXScalingGroup, request.Spec.MaterializationTarget.PodCliqueScalingGroupRef.Name)
	}
	for _, projection := range desired.requests {
		request, found := requestByModel[projection.modelProjection.Model()]
		require.True(t, found)
		require.Equal(t, projection.modelProjection.Digest().String(), request.Annotations[lpx.WorkloadDigestAnnotation])
		require.Equal(t, projection.modelProjection.CompilerSnapshotDigest(), request.Annotations[lpxv1alpha1.CompilerSnapshotDigestAnnotation])
	}

	for index, expected := range desired.plan.Agents {
		projection := &desired.requests[index]
		clique := findLPXTestClique(t, objects, expected.CliqueName)
		require.Equal(t, int32(expected.Replicas), clique.Spec.Replicas)
		require.Equal(t, ptr.To(int32(expected.Replicas)), clique.Spec.MinAvailable)
		require.Equal(t, projection.modelProjection.Digest().String(), clique.Annotations[lpx.WorkloadDigestAnnotation])
		require.Equal(t, projection.modelProjection.CompilerSnapshotDigest(), clique.Annotations[lpxv1alpha1.CompilerSnapshotDigestAnnotation])
		require.Equal(t, projection.modelProjection.Model(), clique.Annotations[lpxv1alpha1.PodModelAnnotation])
		require.NotContains(t, clique.Annotations, lpxv1alpha1.PodPartitionIDAnnotation)
		require.NotContains(t, clique.Annotations, lpxv1alpha1.PodRankInPartitionAnnotation)
	}
	pcs := findLPXTestPodCliqueSet(t, objects)
	require.Equal(t, desired.workloadDigest, pcs.Annotations[lpx.WorkloadDigestAnnotation])

	targetRequest := requestByModel["target"]
	target := targetRequest.DeepCopy()
	target.Status = &lpxv1alpha1.LPUPipelineRequestStatus{
		Phase:              lpxv1alpha1.RequestPhaseUnsupported,
		ObservedGeneration: ptr.To(target.Generation),
		Diagnostics: []lpxv1alpha1.StatusDiagnostic{{
			Code: "TargetUnsupported", Subject: "model/target", Detail: "target projection is unsupported",
		}},
	}
	require.NoError(t, reconciler.Update(ctx, target))
	classification, err = reconcileSelectedLPXForTest(t, ctx, reconciler, dgd, desired)
	require.NoError(t, err)
	require.Equal(t, nvidiacomv1beta1.DGDStateFailed, lpxResult(classification).State)
	require.Equal(t, "LPXUnsupported", lpxResult(classification).Reason)
	require.Contains(t, lpxResult(classification).Message, "TargetUnsupported")
}

func TestLPXMaterializationRejectsExcessiveReplicas(t *testing.T) {
	for _, nativeScale := range []bool{false, true} {
		t.Run(fmt.Sprintf("nativeScale=%t", nativeScale), func(t *testing.T) {
			t.Log("Prepare a valid workload and its owned Grove scaling group")
			ctx := t.Context()
			child, source, registry := newLPXTestDGD(t, lpx.PipelineSingle)
			r, desired := newPreparedLPXTestReconciler(t, registry, ctx, child, source)
			objects := lpxMaterializedObjects(t, r, child, source, desired)
			pcs := findLPXTestPodCliqueSet(t, objects)
			group := findLPXTestScalingGroup(t, objects, desired.plan.LPXScalingGroup)

			t.Log("Exceed the engine limit through either the DGD or native Grove scale")
			lpx.ServingComponent(source).Replicas = ptr.To(int32(2497))
			if nativeScale {
				lpx.ServingComponent(source).Replicas = nil
				group.Spec.Replicas = 2497
			}
			createLPXTestObjects(t, ctx, r.Client, pcs, group)

			t.Log("Reject the count before constructing per-engine request state")
			selected, classification, err := r.prepareLPXMaterializing(ctx, child, source, pcs)
			require.NoError(t, err)
			require.Nil(t, selected)
			require.Equal(t, &lpxRejected{reason: "LPX replica count must be between 0 and 2496"}, classification)
		})
	}
}

func TestLPXImplicitReplicaObservationWaitsForMatchingPodCliqueSet(t *testing.T) {
	t.Log("Render the stable Grove identities for an externally managed replica count")
	ctx := t.Context()
	child, source, registry := newLPXTestDGD(t, lpx.PipelineSingle)
	lpx.ServingComponent(source).Replicas = nil
	base, desired := newPreparedLPXTestReconciler(t, registry, ctx, child, source)
	objects := lpxMaterializedObjects(t, base, child, source, desired)
	pcs := findLPXTestPodCliqueSet(t, objects)
	group := findLPXTestScalingGroup(t, objects, desired.plan.LPXScalingGroup)
	replacement := pcs.DeepCopy()
	replacement.UID = "replacement-pcs"

	for _, test := range []struct {
		name string
		pcs  *grovev1alpha1.PodCliqueSet
	}{
		{name: "PCS has already disappeared"},
		{name: "replacement PCS does not own the old group", pcs: replacement},
	} {
		t.Run(test.name, func(t *testing.T) {
			t.Log("Keep the old PCSG visible during background owner cleanup")
			seed := []client.Object{group.DeepCopy()}
			if test.pcs != nil {
				seed = append(seed, test.pcs.DeepCopy())
			}
			r := newLPXTestReconciler(t, registry, child.DeepCopy(), source.DeepCopy(), seed...)

			t.Log("Classify the transient owner mismatch as retirement, not reconciliation failure")
			selected, classification, err := r.prepareLPXMaterializing(ctx, child, source, test.pcs)
			require.NoError(t, err)
			require.Nil(t, selected)
			require.IsType(t, &lpxRetiring{}, classification)
		})
	}
}

func TestLPXRetainedReplicaPrecedence(t *testing.T) {
	for _, tc := range []struct {
		name     string
		retained *int32
		explicit *int32
		native   *int32
		want     int32
		rejected bool
	}{
		{name: "first deployment defaults to one", want: 1},
		{name: "retain scale out", retained: ptr.To(int32(3)), want: 3},
		{name: "retain zero", retained: ptr.To(int32(0)), want: 0},
		{name: "explicit scale wins", retained: ptr.To(int32(3)), explicit: ptr.To(int32(2)), want: 2},
		{name: "explicit zero wins", retained: ptr.To(int32(3)), explicit: ptr.To(int32(0)), want: 0},
		{name: "live scale wins", retained: ptr.To(int32(3)), native: ptr.To(int32(5)), want: 5},
		{name: "live zero wins", retained: ptr.To(int32(3)), native: ptr.To(int32(0)), want: 0},
		{name: "reject negative retained count", retained: ptr.To(int32(-1)), rejected: true},
		{name: "reject excessive retained count", retained: ptr.To(int32(2497)), rejected: true},
		{name: "live scale supersedes invalid retained count", retained: ptr.To(int32(-1)), native: ptr.To(int32(2)), want: 2},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Log("Prepare an independently owned workload with optional retained and native capacity")
			ctx := t.Context()
			child, source, registry := newLPXTestDGD(t, lpx.PipelineSingle)
			r, initial := newPreparedLPXTestReconciler(t, registry, ctx, child, source)
			child.Status.RetainedReplicas = tc.retained
			lpx.ServingComponent(source).Replicas = tc.explicit
			var pcs *grovev1alpha1.PodCliqueSet
			if tc.native != nil {
				objects := lpxMaterializedObjects(t, r, child, source, initial)
				pcs = findLPXTestPodCliqueSet(t, objects)
				group := findLPXTestScalingGroup(t, objects, initial.plan.LPXScalingGroup)
				group.Spec.Replicas = *tc.native
				createLPXTestObjects(t, ctx, r.Client, pcs, group)
			}

			t.Log("Resolve replicas and request fanout without changing the input objects")
			beforeChild, beforeSource := child.DeepCopy(), source.DeepCopy()
			selected, classification, err := r.prepareLPXMaterializing(ctx, child, source, pcs)
			require.NoError(t, err)
			if tc.rejected {
				require.Nil(t, selected)
				require.IsType(t, &lpxRejected{}, classification)
			} else {
				require.Nil(t, classification)
				require.Equal(t, tc.want, selected.plan.Replicas)
				require.Len(t, selected.requests, int(tc.want))
			}
			require.Equal(t, beforeChild, child)
			require.Equal(t, beforeSource, source)
		})
	}
}

func TestLPXPublishedWorkloadFailureRetirement(t *testing.T) {
	for _, test := range []struct {
		name          string
		snapshotError error
	}{
		{name: "inconsistent snapshot", snapshotError: fmt.Errorf("%w: compiler metadata changed during duplicate reads", lpx.ErrBuildSnapshotInconsistent)},
		{name: "transient snapshot", snapshotError: errors.New("temporary object-store timeout")},
		{name: "render failure"},
	} {
		t.Run(test.name, func(t *testing.T) {
			t.Log("Publish a request backed by the rendered LPX workload")
			ctx := t.Context()
			dgd, source, registry := newLPXTestDGD(t, lpx.PipelineSingle)
			reconciler, desired := newPreparedLPXTestReconciler(t, registry, ctx, dgd, source)
			createLPXTestObjects(t, ctx, reconciler.Client, lpxMaterializedObjects(t, reconciler, dgd, source, desired)...)
			classification := publishSelectedLPXForTest(t, ctx, reconciler, dgd, desired)
			require.IsType(t, &lpxOpen{}, classification)
			requestBefore := getLPXRequest(t, ctx, reconciler.Client, dgd.Namespace, desired.requests[0].requestName)
			key := client.ObjectKeyFromObject(dgd)
			pcsKey := client.ObjectKey{Namespace: dgd.Namespace, Name: desired.plan.PodCliqueSetName}
			pcsBefore := &grovev1alpha1.PodCliqueSet{}
			require.NoError(t, reconciler.Get(ctx, pcsKey, pcsBefore))

			t.Log("Fail the actual snapshot dependency or break cross-role model storage")
			message := "must use the Conductor model-storage mount path"
			if test.snapshotError != nil {
				reconciler.modelRegistry = &snapshotFailureRegistry{ModelRegistry: registry, err: test.snapshotError}
				message = test.snapshotError.Error()
			} else {
				require.NoError(t, reconciler.Get(ctx, client.ObjectKeyFromObject(source), source))
				conductor := source.Spec.Components[0].ComponentRole(nvidiacomv1beta1.ComponentRoleLPXConductor)
				for index := range conductor.PodTemplate.Spec.Containers[0].VolumeMounts {
					mount := &conductor.PodTemplate.Spec.Containers[0].VolumeMounts[index]
					if mount.Name == consts.ModelStorageVolumeName {
						mount.MountPath = "/different-model-storage"
					}
				}
				require.NoError(t, reconciler.Update(ctx, source))
				require.NoError(t, reconciler.Get(ctx, key, dgd))
				revision, revisionErr := dynamo.LPXInputRevision(source, "")
				require.NoError(t, revisionErr)
				dgd.Spec.InputRevision = revision
				require.NoError(t, reconciler.Update(ctx, dgd))
			}
			_, err := reconciler.Reconcile(ctx, ctrl.Request{NamespacedName: key})
			pcs := &grovev1alpha1.PodCliqueSet{}
			t.Log("Failed desired input must preserve the exact published request and PCS")
			require.Equal(t, requestBefore, getLPXRequest(t, ctx, reconciler.Client, dgd.Namespace, requestBefore.Name))
			require.NoError(t, reconciler.Get(ctx, pcsKey, pcs))
			require.Equal(t, pcsBefore, pcs)

			t.Log("Persist the actual actionable failure and retain its wrapped cause")
			require.ErrorContains(t, err, message)
			if test.snapshotError != nil {
				require.ErrorIs(t, err, test.snapshotError)
				require.ErrorIs(t, err, lpx.ErrBuildSnapshotAcquisition)
			}
			require.NoError(t, reconciler.Get(ctx, key, dgd))
			failed := meta.FindStatusCondition(dgd.Status.Conditions, "Failed")
			require.NotNil(t, failed)
			require.Equal(t, metav1.ConditionTrue, failed.Status)
			require.Equal(t, dgd.Generation, failed.ObservedGeneration)
			require.Equal(t, err.Error(), failed.Message)
		})
	}
}

func TestLPXPodCliqueSetMetadataSyncUsesSuppliedObservation(t *testing.T) {
	t.Log("Render initial root metadata and seed an exact-owned PCS with native Grove controls")
	ctx := t.Context()
	dgd, source, registry := newLPXTestDGD(t, lpx.PipelineSingle)
	source.Spec.Labels = map[string]string{"test.example/version": "before", "kai.scheduler/removed": "before"}
	source.Spec.Annotations = maps.Clone(source.Spec.Labels)
	prepare, selected := newPreparedLPXTestReconciler(t, registry, ctx, dgd, source)
	desired := renderLPXTestPodCliqueSet(t, ctx, prepare, dgd, source, selected)
	cached := desired.DeepCopy()
	require.NoError(t, ctrl.SetControllerReference(dgd, cached, prepare.Scheme()))
	cached.UID = "cached-pcs-uid"
	cached.ResourceVersion = "7"
	cached.Generation = 4
	hash, err := commoncontroller.GetSpecHash(cached, commoncontroller.WithPreservedListOrder())
	require.NoError(t, err)
	metav1.SetMetaDataAnnotation(&cached.ObjectMeta, commoncontroller.NvidiaAnnotationHashKey, hash)
	delete(cached.Annotations, commoncontroller.NvidiaAnnotationGenerationKey)
	cached.Finalizers = []string{groveconstants.FinalizerPodCliqueSet}
	cached.Annotations[groveconstants.AnnotationDisableManagedResourceProtection] = "true"
	cached.Annotations[groveconstants.AnnotationReconcileTrigger] = "previous-trigger"
	before := cached.DeepCopy()
	for _, metadata := range []map[string]string{desired.Labels, desired.Annotations} {
		metadata["test.example/version"] = "after"
		metadata["kai.scheduler/added"] = ""
		delete(metadata, "kai.scheduler/removed")
	}

	t.Log("Record writes and reject extra reads of the supplied observation")
	reconciler := newLPXTestReconciler(t, registry, dgd, source, cached)
	base, ok := reconciler.Client.(client.WithWatch)
	require.True(t, ok)
	updates := 0
	reconciler.Client = interceptor.NewClient(base, interceptor.Funcs{
		Get: func(ctx context.Context, delegated client.WithWatch, key client.ObjectKey, object client.Object, opts ...client.GetOption) error {
			return errors.New("PCS synchronization must use its supplied observation")
		},
		Update: func(ctx context.Context, delegated client.WithWatch, object client.Object, opts ...client.UpdateOption) error {
			updates++
			return delegated.Update(ctx, object, opts...)
		},
	})

	t.Log("Only bookkeeping changes; root metadata additions, edits and removals are ignored")
	modified, synced, err := commoncontroller.SyncObservedResource(
		ctx, reconciler, dgd, cached, desired, commoncontroller.WithPreservedListOrder(),
	)
	require.NoError(t, err)
	require.True(t, modified)
	require.Equal(t, 1, updates)
	want := before.DeepCopy()
	want.ResourceVersion = synced.ResourceVersion
	want.Annotations[commoncontroller.NvidiaAnnotationGenerationKey] = "4"
	require.Equal(t, want, synced)
	require.Equal(t, before, cached, "the supplied observation must not be mutated")

	t.Log("Root metadata differences alone are a no-op once bookkeeping is current")
	updates = 0
	modified, synced, err = commoncontroller.SyncObservedResource(
		ctx, reconciler, dgd, synced, desired, commoncontroller.WithPreservedListOrder(),
	)
	require.NoError(t, err)
	require.False(t, modified)
	require.Zero(t, updates)
	require.Equal(t, want, synced)

	t.Log("Template metadata still updates through Spec without changing root metadata or native controls")
	desired.Spec.Template.Cliques[0].Annotations = map[string]string{"test.example/template": "next"}
	modified, synced, err = commoncontroller.SyncObservedResource(
		ctx, reconciler, dgd, synced, desired, commoncontroller.WithPreservedListOrder(),
	)
	require.NoError(t, err)
	require.True(t, modified)
	require.Equal(t, 1, updates)
	want.Spec = desired.Spec
	want.ResourceVersion = synced.ResourceVersion
	want.Annotations[commoncontroller.NvidiaAnnotationHashKey], err = commoncontroller.GetSpecHash(desired, commoncontroller.WithPreservedListOrder())
	require.NoError(t, err)
	want.Annotations[commoncontroller.NvidiaAnnotationGenerationKey] = "5"
	require.True(t, apiequality.Semantic.DeepEqual(want, synced))

	t.Log("A concurrent spec edit conflicts instead of being overwritten from a stale observation")
	cached = synced.DeepCopy()
	concurrent := synced.DeepCopy()
	concurrent.Spec.Replicas++
	concurrent.Generation++
	require.NoError(t, base.Update(ctx, concurrent))
	desired.Spec.Replicas += 2
	updates = 0
	modified, synced, err = commoncontroller.SyncObservedResource(
		ctx, reconciler, dgd, cached, desired, commoncontroller.WithPreservedListOrder(),
	)
	require.True(t, apierrors.IsConflict(err), "concurrent spec change must trigger a fresh reconciliation: %v", err)
	require.Nil(t, synced)
	require.False(t, modified)
	require.Equal(t, 1, updates)
	stored := &grovev1alpha1.PodCliqueSet{}
	require.NoError(t, base.Get(ctx, client.ObjectKeyFromObject(concurrent), stored))
	require.Equal(t, concurrent, stored)
}

func TestLPXPodCliqueSetListOrder(t *testing.T) {
	for _, reorder := range []bool{false, true} {
		t.Run(fmt.Sprintf("reorder=%t", reorder), func(t *testing.T) {
			t.Log("Create an LPX PCS with two ordered init containers")
			ctx := t.Context()
			dgd, source, registry := newLPXTestDGD(t, lpx.PipelineSingle)
			reconciler, selected := newPreparedLPXTestReconciler(t, registry, ctx, dgd, source)
			desired := renderLPXTestPodCliqueSet(t, ctx, reconciler, dgd, source, selected)
			desired.Spec.Template.Cliques[0].Spec.PodSpec.InitContainers = []corev1.Container{
				{Name: "setup", Image: "busybox"}, {Name: "migrate", Image: "busybox"},
			}
			modified, observed, err := commoncontroller.SyncObservedResource(
				ctx, reconciler, dgd, nil, desired.DeepCopy(), commoncontroller.WithPreservedListOrder(),
			)
			require.NoError(t, err)
			require.True(t, modified)

			t.Log("Model API-defaulted live fields without changing the applied generation")
			observed.Generation = 1
			require.Nil(t, desired.Spec.Template.Cliques[0].Spec.PodSpec.EnableServiceLinks)
			observed.Spec.Template.Cliques[0].Spec.PodSpec.EnableServiceLinks = ptr.To(true)
			require.NoError(t, reconciler.Update(ctx, observed))
			before := observed.DeepCopy()
			if reorder {
				slices.Reverse(desired.Spec.Template.Cliques[0].Spec.PodSpec.InitContainers)
			}

			t.Log("Apply authored ordering changes but ignore unchanged desired specs")
			modified, synced, err := commoncontroller.SyncObservedResource(
				ctx, reconciler, dgd, observed, desired, commoncontroller.WithPreservedListOrder(),
			)
			require.NoError(t, err)
			require.Equal(t, reorder, modified)
			require.Equal(t, desired.Spec.Template.Cliques[0].Spec.PodSpec.InitContainers, synced.Spec.Template.Cliques[0].Spec.PodSpec.InitContainers)
			require.Equal(t, before, observed, "synchronization must not mutate its observation")
			if !reorder {
				require.Equal(t, before, synced, "API defaults must not cause a spec rewrite")
			}

			t.Log("The next reconciliation is a no-op")
			modified, _, err = commoncontroller.SyncObservedResource(
				ctx, reconciler, dgd, synced, desired, commoncontroller.WithPreservedListOrder(),
			)
			require.NoError(t, err)
			require.False(t, modified)
		})
	}
}

func TestLPXPublicationFencePrecedesPodCliqueSetWrite(t *testing.T) {
	t.Log("Prepare a render for publication and observe writes across the authoritative fence")
	ctx := t.Context()
	dgd, source, registry := newLPXTestDGD(t, lpx.PipelineSingle)
	reconciler, selected := newPreparedLPXTestReconciler(t, registry, ctx, dgd, source)
	desired := renderLPXTestPodCliqueSet(t, ctx, reconciler, dgd, source, selected)
	base, ok := reconciler.Client.(client.WithWatch)
	require.True(t, ok)
	writes, lists := 0, 0
	reconciler.Client = interceptor.NewClient(base, interceptor.Funcs{
		Create: func(ctx context.Context, delegated client.WithWatch, object client.Object, opts ...client.CreateOption) error {
			writes++
			return delegated.Create(ctx, object, opts...)
		},
		Update: func(ctx context.Context, delegated client.WithWatch, object client.Object, opts ...client.UpdateOption) error {
			writes++
			return delegated.Update(ctx, object, opts...)
		},
	})
	reader, ok := reconciler.apiReader.(client.WithWatch)
	require.True(t, ok)
	fenceErr := errors.New("fence list failed")
	failFence := true
	reconciler.apiReader = interceptor.NewClient(reader, interceptor.Funcs{
		List: func(ctx context.Context, delegated client.WithWatch, list client.ObjectList, opts ...client.ListOption) error {
			lists++
			if failFence {
				return fenceErr
			}
			return delegated.List(ctx, list, opts...)
		},
	})

	t.Log("A failed publication observation prevents initial creation")
	currents, classification, err := reconciler.reconcileLPXPublicationFence(ctx, dgd, selected, nil)
	require.ErrorIs(t, err, fenceErr)
	require.Nil(t, classification)
	require.Nil(t, currents)
	require.Zero(t, writes)
	require.Equal(t, 1, lists)
	require.True(t, apierrors.IsNotFound(base.Get(ctx, client.ObjectKeyFromObject(desired), &grovev1alpha1.PodCliqueSet{})))

	t.Log("Recover the observation and create the exact child-owned PCS")
	failFence = false
	writes, lists = 0, 0
	currents, classification, err = reconciler.reconcileLPXPublicationFence(ctx, dgd, selected, nil)
	require.NoError(t, err)
	require.Nil(t, classification)
	require.Empty(t, currents)
	modified, synced, err := commoncontroller.SyncObservedResource(
		ctx, reconciler, dgd, nil, desired.DeepCopy(), commoncontroller.WithPreservedListOrder(),
	)
	require.NoError(t, err)
	require.True(t, modified)
	require.Equal(t, 1, lists)
	require.Equal(t, 1, writes)
	require.True(t, apiequality.Semantic.DeepEqual(desired.Spec, synced.Spec))
	require.True(t, metav1.IsControlledBy(synced, dgd))
	require.Equal(t, desired.Labels, synced.Labels)
	require.Equal(t, desired.Annotations[lpx.WorkloadDigestAnnotation], synced.Annotations[lpx.WorkloadDigestAnnotation])
	hash, hashErr := commoncontroller.GetSpecHash(desired, commoncontroller.WithPreservedListOrder())
	require.NoError(t, hashErr)
	require.Equal(t, hash, synced.Annotations[commoncontroller.NvidiaAnnotationHashKey])
	require.Equal(t, "1", synced.Annotations[commoncontroller.NvidiaAnnotationGenerationKey])
	stored := &grovev1alpha1.PodCliqueSet{}
	require.NoError(t, base.Get(ctx, client.ObjectKeyFromObject(desired), stored))
	require.Equal(t, stored.ResourceVersion, synced.ResourceVersion)

	t.Log("A stale missing observation does not adopt an already-created PCS")
	modified, _, err = commoncontroller.SyncObservedResource(
		ctx, reconciler, dgd, nil, desired.DeepCopy(), commoncontroller.WithPreservedListOrder(),
	)
	require.True(t, apierrors.IsAlreadyExists(err), "create race must trigger a fresh reconciliation: %v", err)
	require.False(t, modified)

	t.Log("An in-place update does not require listing every scheduler request")
	live := synced
	desired = live.DeepCopy()
	desired.ResourceVersion, desired.UID = "", ""
	desired.Spec.Replicas++
	desired.Annotations[lpx.WorkloadDigestAnnotation] = "next"
	failFence = true
	writes, lists = 0, 0
	modified, synced, err = commoncontroller.SyncObservedResource(
		ctx, reconciler, dgd, live, desired, commoncontroller.WithPreservedListOrder(),
	)
	require.NoError(t, err)
	require.True(t, modified)
	require.Equal(t, live.UID, synced.UID)
	require.Equal(t, 1, writes)
	require.Zero(t, lists)
	require.NoError(t, base.Get(ctx, client.ObjectKeyFromObject(live), stored))
	require.True(t, apiequality.Semantic.DeepEqual(desired.Spec, stored.Spec))
	require.Equal(t, live.Annotations[lpx.WorkloadDigestAnnotation], stored.Annotations[lpx.WorkloadDigestAnnotation])
}

func TestLPXPublicationFenceReusesObservedPodCliqueSetForRequestOwnership(t *testing.T) {
	t.Log("Publish two requests owned by the already-observed PodCliqueSet")
	ctx := t.Context()
	child, source, registry := newLPXTestDGD(t, lpx.PipelineSingle)
	lpx.ServingComponent(source).Replicas = ptr.To(int32(2))
	r, desired := newPreparedLPXTestReconciler(t, registry, ctx, child, source)
	objects := lpxMaterializedObjects(t, r, child, source, desired)
	createLPXTestObjects(t, ctx, r.Client, objects...)
	publishSelectedLPXForTest(t, ctx, r, child, desired)
	pcs := findLPXTestPodCliqueSet(t, objects)
	pcsGets := 0
	r.apiReader = interceptor.NewClient(r.apiReader.(client.WithWatch), interceptor.Funcs{
		Get: func(ctx context.Context, delegated client.WithWatch, key client.ObjectKey, object client.Object, opts ...client.GetOption) error {
			if _, ok := object.(*grovev1alpha1.PodCliqueSet); ok {
				pcsGets++
			}
			return delegated.Get(ctx, key, object, opts...)
		},
	})

	t.Log("Classify every LPR against that observation without per-request API reads")
	currents, classification, err := r.reconcileLPXPublicationFence(ctx, child, desired, pcs)
	require.NoError(t, err)
	require.Nil(t, classification)
	require.Len(t, currents, len(desired.requests))
	require.Zero(t, pcsGets)
}

func TestLPXPublicationWaitsForRequestGarbageCollectionBeforeRecreatingPodCliqueSet(t *testing.T) {
	ctx := t.Context()
	dgd, source, registry := newLPXTestDGD(t, lpx.PipelineSingle)
	reconciler, desired := newPreparedLPXTestReconciler(t, registry, ctx, dgd, source)
	createLPXTestObjects(t, ctx, reconciler.Client, lpxMaterializedObjects(t, reconciler, dgd, source, desired)...)
	publishSelectedLPXForTest(t, ctx, reconciler, dgd, desired)

	pcs := &grovev1alpha1.PodCliqueSet{}
	require.NoError(t, reconciler.Get(ctx, client.ObjectKey{Namespace: dgd.Namespace, Name: desired.plan.PodCliqueSetName}, pcs))
	t.Log("An unrelated live PCS also has a request with this deployment's copied UID label")
	foreignPCS := pcs.DeepCopy()
	foreignPCS.Name, foreignPCS.UID, foreignPCS.ResourceVersion = lpxTestOtherName, "foreign-pcs", ""
	foreignPCS.OwnerReferences = nil
	require.NoError(t, reconciler.Create(ctx, foreignPCS))
	foreign := getLPXRequest(t, ctx, reconciler.Client, dgd.Namespace, desired.requests[0].requestName)
	foreign.Name, foreign.UID, foreign.ResourceVersion = "foreign-request", "foreign-request-uid", ""
	foreign.OwnerReferences = []metav1.OwnerReference{*metav1.NewControllerRef(foreignPCS, grovev1alpha1.SchemeGroupVersion.WithKind(lpxPodCliqueSetKind))}
	require.NoError(t, reconciler.Create(ctx, foreign))
	require.NoError(t, reconciler.Delete(ctx, pcs))

	t.Log("Keep the replacement PCS fenced while its predecessor's request remains persisted")
	currents, classification, err := reconciler.reconcileLPXPublicationFence(ctx, dgd, desired, nil)
	require.NoError(t, err)
	require.Len(t, currents, 1)
	require.IsType(t, &lpxRetiring{}, classification)

	t.Log("Permit idempotent replacement after garbage collection removes the previous request")
	request := getLPXRequest(t, ctx, reconciler.Client, dgd.Namespace, desired.requests[0].requestName)
	require.NoError(t, reconciler.Delete(ctx, request))
	currents, classification, err = reconciler.reconcileLPXPublicationFence(ctx, dgd, desired, nil)
	require.NoError(t, err)
	require.Empty(t, currents)
	require.Nil(t, classification)
	require.Equal(t, foreign, getLPXRequest(t, ctx, reconciler.Client, dgd.Namespace, foreign.Name))
	require.NoError(t, reconciler.Get(ctx, client.ObjectKeyFromObject(foreignPCS), &grovev1alpha1.PodCliqueSet{}))
}

func TestLPXScaleDownDeletesOnlyStaleRequests(t *testing.T) {
	ctx := t.Context()
	dgd, source, registry := newLPXTestDGD(t, lpx.PipelineSingle)
	source.Spec.Components[0].Replicas = ptr.To[int32](3)
	reconciler, desired := newPreparedLPXTestReconciler(t, registry, ctx, dgd, source)
	objects := lpxMaterializedObjects(t, reconciler, dgd, source, desired)
	createLPXTestObjects(t, ctx, reconciler.Client, objects...)
	publishSelectedLPXForTest(t, ctx, reconciler, dgd, desired)
	for _, projection := range desired.requests {
		request := getLPXRequest(t, ctx, reconciler.Client, dgd.Namespace, projection.requestName)
		request.Status = &lpxv1alpha1.LPUPipelineRequestStatus{
			Phase: lpxv1alpha1.RequestPhasePending, ObservedGeneration: ptr.To(request.Generation),
		}
		require.NoError(t, reconciler.Update(ctx, request))
	}
	pcs := findLPXTestPodCliqueSet(t, objects)
	pcsUID := pcs.UID
	retainedUID := getLPXRequest(t, ctx, reconciler.Client, dgd.Namespace, desired.requests[0].requestName).UID

	t.Log("Scale from three replicas to one while every request remains pending")
	scaledDown := withLPXTestReplicas(t, desired, 1)
	_, classification, err := reconciler.reconcileLPXPublicationFence(
		ctx, dgd, scaledDown, observedLPXTestPodCliqueSet(t, ctx, reconciler, dgd, scaledDown),
	)
	require.NoError(t, err)
	require.IsType(t, &lpxRetiring{}, classification)

	t.Log("Delete every stale LPR concurrently without replacing the shared PCS")
	currents, classification, err := reconciler.reconcileLPXPublicationFence(
		ctx, dgd, scaledDown, observedLPXTestPodCliqueSet(t, ctx, reconciler, dgd, scaledDown),
	)
	require.NoError(t, err)
	require.Nil(t, classification)
	require.Len(t, currents, 1)
	for _, projection := range desired.requests[1:] {
		requireLPXRequestNotFound(t, ctx, reconciler.Client, dgd.Namespace, projection.requestName)
	}
	retained := getLPXRequest(t, ctx, reconciler.Client, dgd.Namespace, desired.requests[0].requestName)
	require.Equal(t, retainedUID, retained.UID)
	storedPCS := &grovev1alpha1.PodCliqueSet{}
	require.NoError(t, reconciler.Get(ctx, client.ObjectKeyFromObject(pcs), storedPCS))
	require.Equal(t, pcsUID, storedPCS.UID)
}

func TestLPXScaleUpWaitsForTerminatingScaledDownRequest(t *testing.T) {
	ctx := t.Context()
	dgd, source, registry := newLPXTestDGD(t, lpx.PipelineSingle)
	source.Spec.Components[0].Replicas = ptr.To[int32](2)
	reconciler, desired := newPreparedLPXTestReconciler(t, registry, ctx, dgd, source)
	objects := lpxMaterializedObjects(t, reconciler, dgd, source, desired)
	createLPXTestObjects(t, ctx, reconciler.Client, objects...)
	publishSelectedLPXForTest(t, ctx, reconciler, dgd, desired)
	pcs := findLPXTestPodCliqueSet(t, objects)
	pcsUID := pcs.UID
	removed := getLPXRequest(t, ctx, reconciler.Client, dgd.Namespace, desired.requests[1].requestName)
	removedUID := removed.UID
	removed.Finalizers = []string{"scheduler.example/cleanup"}
	require.NoError(t, reconciler.Update(ctx, removed))

	t.Log("Scale down and begin direct request deletion")
	scaledDown := withLPXTestReplicas(t, desired, 1)
	_, classification, err := reconciler.reconcileLPXPublicationFence(
		ctx, dgd, scaledDown, observedLPXTestPodCliqueSet(t, ctx, reconciler, dgd, scaledDown),
	)
	require.NoError(t, err)
	require.IsType(t, &lpxRetiring{}, classification)
	terminating := getLPXRequest(t, ctx, reconciler.Client, dgd.Namespace, removed.Name)
	require.False(t, terminating.DeletionTimestamp.IsZero())

	t.Log("Scale back up while the same request name is still terminating")
	_, classification, err = reconciler.reconcileLPXPublicationFence(
		ctx, dgd, desired, observedLPXTestPodCliqueSet(t, ctx, reconciler, dgd, desired),
	)
	require.NoError(t, err)
	require.IsType(t, &lpxRetiring{}, classification)
	storedPCS := &grovev1alpha1.PodCliqueSet{}
	require.NoError(t, reconciler.Get(ctx, client.ObjectKeyFromObject(pcs), storedPCS))
	require.Equal(t, pcsUID, storedPCS.UID)

	t.Log("Recreate only the request after scheduler cleanup completes")
	terminating.Finalizers = nil
	require.NoError(t, reconciler.Update(ctx, terminating))
	requireLPXRequestNotFound(t, ctx, reconciler.Client, dgd.Namespace, removed.Name)
	classification = publishSelectedLPXForTest(t, ctx, reconciler, dgd, desired)
	require.IsType(t, &lpxOpen{}, classification)
	recreated := getLPXRequest(t, ctx, reconciler.Client, dgd.Namespace, removed.Name)
	require.NotEqual(t, removedUID, recreated.UID)
	require.Equal(t, pcsUID, metav1.GetControllerOf(recreated).UID)
	require.NoError(t, reconciler.Get(ctx, client.ObjectKeyFromObject(pcs), storedPCS))
	require.Equal(t, pcsUID, storedPCS.UID)
}

func TestLPXElapsedDeadlineDoesNotFenceBoundGeneration(t *testing.T) {
	t.Log("Publish a request that reaches Bound before its persisted deadline is observed again")
	ctx := t.Context()
	dgd, source, registry := newLPXTestDGD(t, lpx.PipelineSingle)
	source.Spec.Scheduling = deadlineTestScheduling()
	reconciler, desired := newPreparedLPXTestReconciler(t, registry, ctx, dgd, source)
	createLPXTestObjects(t, ctx, reconciler.Client, lpxMaterializedObjects(t, reconciler, dgd, source, desired)...)
	publishSelectedLPXForTest(t, ctx, reconciler, dgd, desired)
	request := getLPXRequest(t, ctx, reconciler.Client, dgd.Namespace, desired.requests[0].requestName)
	request.Status = deadlineTestRequest(dgd, deadlineTestPCS(dgd, metav1.GetControllerOf(request).UID), request.Name, time.Now(), lpxv1alpha1.RequestPhaseBound).Status
	require.NoError(t, reconciler.Update(ctx, request))

	t.Log("Terminal scheduler disposition wins the deadline race and permits ordinary observation")
	currents, classification, err := reconciler.reconcileLPXPublicationFence(
		ctx, dgd, desired, observedLPXTestPodCliqueSet(t, ctx, reconciler, dgd, desired),
	)
	require.NoError(t, err)
	require.Len(t, currents, 1)
	require.Nil(t, classification)
	classification, wake, err := reconciler.reconcileLPXRequestDeadlines(
		ctx, dgd, source, observedLPXTestPodCliqueSet(t, ctx, reconciler, dgd, desired), desired.requests,
	)
	require.NoError(t, err)
	require.Nil(t, classification)
	require.True(t, wake.IsZero())
}

func TestLPXRecordedDeadlineFailureFencesLateBoundRequest(t *testing.T) {
	t.Log("Record expiry before the scheduler reports a terminal phase")
	ctx := t.Context()
	dgd, source, registry := newLPXTestDGD(t, lpx.PipelineSingle)
	source.Spec.Scheduling = deadlineTestScheduling()
	reconciler, desired := newPreparedLPXTestReconciler(t, registry, ctx, dgd, source)
	createLPXTestObjects(t, ctx, reconciler.Client, lpxMaterializedObjects(t, reconciler, dgd, source, desired)...)
	publishSelectedLPXForTest(t, ctx, reconciler, dgd, desired)
	request := getLPXRequest(t, ctx, reconciler.Client, dgd.Namespace, desired.requests[0].requestName)
	request.Status = deadlineTestRequest(dgd, deadlineTestPCS(dgd, metav1.GetControllerOf(request).UID), request.Name, time.Now().Add(-time.Minute), lpxv1alpha1.RequestPhaseBound).Status
	require.NoError(t, reconciler.Update(ctx, request))
	dgd.Status.Conditions = []metav1.Condition{{
		Type: lpxSchedulingFailedCondition, Status: metav1.ConditionTrue,
		Reason: lpxSchedulingDeadlineExceededReason, ObservedGeneration: dgd.Generation,
	}}

	t.Log("Keep the recorded failure authoritative for the current generation")
	currents, classification, err := reconciler.reconcileLPXPublicationFence(
		ctx, dgd, desired, observedLPXTestPodCliqueSet(t, ctx, reconciler, dgd, desired),
	)
	require.NoError(t, err)
	require.Len(t, currents, 1)
	require.IsType(t, &lpxDeadlineExceeded{}, classification)
}

func TestLPXPreflightPreservesRequestsDuringNativeGroveSync(t *testing.T) {
	ctx := t.Context()
	dgd, source, registry := newLPXTestDGD(t, lpx.PipelineSingle)
	reconciler, desired := newPreparedLPXTestReconciler(t, registry, ctx, dgd, source)

	createLPXTestObjects(t, ctx, reconciler.Client, lpxMaterializedObjects(t, reconciler, dgd, source, desired)...)
	rendered := renderLPXTestPodCliqueSet(t, ctx, reconciler, dgd, source, desired)
	publishSelectedLPXForTest(t, ctx, reconciler, dgd, desired)
	published := getLPXRequest(t, ctx, reconciler.Client, dgd.Namespace, desired.requests[0].requestName)

	pcsKey := types.NamespacedName{Namespace: dgd.Namespace, Name: desired.plan.PodCliqueSetName}
	drifted := &grovev1alpha1.PodCliqueSet{}
	require.NoError(t, reconciler.Get(ctx, pcsKey, drifted))
	require.NotEmpty(t, drifted.Spec.Template.Cliques)
	drifted.Spec.Template.Cliques[0].Spec.PodSpec.PriorityClassName = "externally-mutated"
	require.NoError(t, reconciler.Update(ctx, drifted))
	require.NoError(t, reconciler.Get(ctx, pcsKey, drifted))

	_, _, err := commoncontroller.SyncObservedResource(
		ctx,
		reconciler,
		dgd,
		drifted,
		rendered,
		commoncontroller.WithPreservedListOrder(),
	)
	require.NoError(t, err)
	require.Equal(t, published, getLPXRequest(t, ctx, reconciler.Client, dgd.Namespace, published.Name))

	t.Log("Correct the spec in place without stopping the existing PCS")
	require.NoError(t, reconciler.Get(ctx, pcsKey, drifted))
	require.Equal(t, int32(1), drifted.Spec.Replicas)
	require.Equal(t, types.UID("pcs-uid"), drifted.UID)
	require.NotEqual(t, "externally-mutated", drifted.Spec.Template.Cliques[0].Spec.PodSpec.PriorityClassName)

	foreignPCS := drifted.DeepCopy()
	foreignPCS.ResourceVersion = ""
	require.NotEmpty(t, foreignPCS.OwnerReferences)
	foreignPCS.OwnerReferences[0].UID = "foreign-owner"
	foreignReconciler := newLPXTestReconciler(t, registry, dgd, source, foreignPCS)
	_, err = foreignReconciler.reconcileSelectedLPXFromCurrentRequests(ctx, dgd, desired, nil)
	require.EqualError(t, err, fmt.Sprintf("refusing to inspect PodCliqueSet %q without the exact LPXGraphDeployment controller owner", foreignPCS.Name))

	t.Log("An explicitly stopped PCS still cannot publish another request")
	drifted.Spec.Replicas = 0
	require.NoError(t, reconciler.Update(ctx, drifted))
	reader, ok := reconciler.apiReader.(client.WithWatch)
	require.True(t, ok)
	pcsGets := 0
	reconciler.apiReader = interceptor.NewClient(reader, interceptor.Funcs{
		Get: func(ctx context.Context, delegated client.WithWatch, key client.ObjectKey, object client.Object, opts ...client.GetOption) error {
			if _, ok := object.(*grovev1alpha1.PodCliqueSet); ok {
				pcsGets++
			} else {
				return errors.New("retired Grove observation must not read dependencies")
			}
			return delegated.Get(ctx, key, object, opts...)
		},
		List: func(context.Context, client.WithWatch, client.ObjectList, ...client.ListOption) error {
			return errors.New("retired Grove observation must not read dependencies")
		},
	})
	classification, err := reconciler.reconcileSelectedLPXFromCurrentRequests(ctx, dgd, desired, nil)
	require.NoError(t, err)
	closed := requireLPXClosed(t, classification)
	require.Equal(t, "The selected Grove PodCliqueSet has 0 replicas, want 1", closed.incomplete)
	require.Equal(t, 1, pcsGets)
	reconciler.apiReader = reader

	t.Log("Restore the explicit hold in place without rotating the existing request")
	_, _, err = commoncontroller.SyncObservedResource(
		ctx,
		reconciler,
		dgd,
		drifted,
		rendered,
		commoncontroller.WithPreservedListOrder(),
	)
	require.NoError(t, err)

	recreated := &grovev1alpha1.PodCliqueSet{}
	require.NoError(t, reconciler.Get(ctx, pcsKey, recreated))
	require.Equal(t, drifted.UID, recreated.UID)
	for _, clique := range recreated.Spec.Template.Cliques {
		require.NotEqual(t, "externally-mutated", clique.Spec.PodSpec.PriorityClassName)
	}
	require.Equal(t, published, getLPXRequest(t, ctx, reconciler.Client, dgd.Namespace, published.Name))
}

func TestGroveSpecSyncPreservesWorkloadAfterTopologyRejection(t *testing.T) {
	t.Log("Keep an existing two-partition engine while preparing a three-partition successor")
	ctx := t.Context()
	oldDGD, oldSource, oldRegistry := newLPXTestDGD(t, lpx.PipelineSingle)
	oldPrepareReconciler := newLPXTestReconciler(t, oldRegistry, oldDGD, oldSource)
	oldDesired, rejected := requirePreparedLPX(t, oldPrepareReconciler, ctx, oldDGD, oldSource)
	require.Nil(t, rejected)

	oldPCS := findLPXTestPodCliqueSet(t, lpxMaterializedObjects(t, oldPrepareReconciler, oldDGD, oldSource, oldDesired))
	published := deadlineTestRequest(oldDGD, oldPCS, oldDesired.requests[0].requestName, time.Now(), lpxv1alpha1.RequestPhaseBound)

	const successorBuildID = "build-v2-successor"
	successorRegistry := newLPXTestRegistryWithPartitionsAndMode(
		t,
		successorBuildID,
		[]int{7, 8, 9},
		manifestcapnpv2.CompilationMode_lpuOnly,
	)
	successorDGD := oldDGD.DeepCopy()
	successorSource := oldSource.DeepCopy()
	successorDGD.Generation++
	successorSource.Spec.Components[0].LPX.BuildID = successorBuildID
	reconciler := newLPXTestReconciler(t, successorRegistry, successorDGD, successorSource, oldPCS, published)
	successorDesired, rejected := requirePreparedLPX(t, reconciler, ctx, successorDGD, successorSource)
	require.Nil(t, rejected)

	t.Log("The successor changes Agent minAvailable, which pinned Grove rejects as immutable")
	pcs := &grovev1alpha1.PodCliqueSet{}
	pcsKey := types.NamespacedName{
		Namespace: successorDGD.Namespace,
		Name:      successorDesired.plan.PodCliqueSetName,
	}
	require.NoError(t, reconciler.Get(ctx, pcsKey, pcs))
	published = getLPXRequest(t, ctx, reconciler.Client, successorDGD.Namespace, published.Name)
	successorRendered := renderLPXTestPodCliqueSet(t, ctx, reconciler, successorDGD, successorSource, successorDesired)
	for index, clique := range successorRendered.Spec.Template.Cliques {
		if clique.Annotations[lpxv1alpha1.PodRoleAnnotation] == lpxv1alpha1.PodRoleAgent {
			require.Equal(t, ptr.To(int32(4)), pcs.Spec.Template.Cliques[index].Spec.MinAvailable)
			require.Equal(t, ptr.To(int32(6)), clique.Spec.MinAvailable)
		}
	}

	t.Log("Return the admission error without replacing or scaling down the existing workload")
	admissionErr := apierrors.NewForbidden(grovev1alpha1.SchemeGroupVersion.WithResource("podcliquesets").GroupResource(), pcs.Name,
		errors.New("spec.template.cliques.spec.minAvailable: field is immutable"))
	reconciler.Client = interceptor.NewClient(reconciler.Client.(client.WithWatch), interceptor.Funcs{
		Update: func(ctx context.Context, delegated client.WithWatch, object client.Object, opts ...client.UpdateOption) error {
			if _, ok := object.(*grovev1alpha1.PodCliqueSet); ok {
				return admissionErr
			}
			return delegated.Update(ctx, object, opts...)
		},
	})
	changed, synced, err := commoncontroller.SyncObservedResource(
		ctx,
		reconciler,
		successorDGD,
		pcs,
		successorRendered,
		commoncontroller.WithPreservedListOrder(),
	)
	require.ErrorIs(t, err, admissionErr)
	require.Nil(t, synced)
	require.False(t, changed)

	t.Log("The complete stored PCS and published request remain unchanged")
	stored := &grovev1alpha1.PodCliqueSet{}
	require.NoError(t, reconciler.Get(ctx, pcsKey, stored))
	require.Equal(t, pcs, stored)
	require.Equal(t, published, getLPXRequest(t, ctx, reconciler.Client, successorDGD.Namespace, published.Name))
}

func TestLPXRetirementOverridesRuntimeReadiness(t *testing.T) {
	t.Log("Retirement must replace both successful and pending runtime results")
	retiring := &lpxRetiring{retirementReason: "test retirement"}
	_, isError := any(retiring).(error)
	require.False(t, isError, "expected retirement must stay outside the operational error channel")
	for _, base := range []reconcileOutcome{{State: nvidiacomv1beta1.DGDStateSuccessful}, {State: nvidiacomv1beta1.DGDStatePending}} {
		require.Equal(t, lpxResult(retiring), overlayLPXResult(base, retiring))
	}
}

func TestLPXRestartPreservesBoundProofAndFinalization(t *testing.T) {
	t.Log("Prepare the LPU-only runtime without adding scheduler configuration to its source")
	ctx := context.Background()
	dgd, source, registry := newLPXTestDGD(t, lpx.PipelineSingle)
	delete(source.Annotations, consts.KubeAnnotationLPXSchedulerBackend)
	reconciler, desired := newPreparedLPXTestReconciler(t, registry, ctx, dgd, source)
	require.Nil(t, source.Spec.Scheduling)
	require.NotContains(t, source.Annotations, consts.KubeAnnotationLPXSchedulerBackend)

	t.Log("Materialize the selected runtime")
	objects := lpxMaterializedObjects(t, reconciler, dgd, source, desired)
	createLPXTestObjects(t, ctx, reconciler.Client, objects...)

	t.Log("Keep an otherwise-ready runtime gated before the exact request is Bound")
	classification := publishSelectedLPXForTest(t, ctx, reconciler, dgd, desired)
	require.IsType(t, &lpxOpen{}, classification)
	runtimeReady := reconcileOutcome{
		State:   nvidiacomv1beta1.DGDStateSuccessful,
		Reason:  "all_resources_are_ready",
		Message: "All resources are ready",
	}
	require.Equal(t, "LPXPublished", overlayLPXResult(runtimeReady, classification).Reason)

	t.Log("Recreate the controller and derive the same open attempt from API objects")
	reconciler = &graphReconciler{
		Client:                reconciler.Client,
		recorder:              reconciler.recorder,
		apiReader:             reconciler.apiReader,
		runtimeConfig:         reconciler.runtimeConfig,
		modelRegistry:         registry,
		Config:                reconciler.Config,
		DockerSecretRetriever: reconciler.DockerSecretRetriever,
	}
	rederived, rejected := requirePreparedLPX(t, reconciler, ctx, dgd, source)
	require.Nil(t, rejected)
	desired = rederived
	classification, err := reconcileSelectedLPXForTest(t, ctx, reconciler, dgd, desired)
	require.NoError(t, err)
	require.IsType(t, &lpxOpen{}, classification)

	t.Log("Observe the exact current scheduler receipt after the controller restart")
	request := getLPXRequest(t, ctx, reconciler.Client, dgd.Namespace, desired.requests[0].requestName)
	request.Generation = 7
	request.Status = newLPXNodeLocalBoundStatus(7, 3, "sha256:bound-plan")
	require.NoError(t, reconciler.Update(ctx, request))

	classification, err = reconcileSelectedLPXForTest(t, ctx, reconciler, dgd, desired)
	require.NoError(t, err)
	bound, ok := classification.(*lpxBound)
	require.True(t, ok)

	t.Log("Return an exact node-local Bound receipt to the base Grove runtime result")
	require.Equal(t, runtimeReady, overlayLPXResult(runtimeReady, bound))
	runtimePending := reconcileOutcome{
		State:   nvidiacomv1beta1.DGDStatePending,
		Reason:  "some_resources_are_not_ready",
		Message: "Some resources are not ready",
	}
	require.Equal(t, runtimePending, overlayLPXResult(runtimePending, bound))

	t.Log("Keep runtime readiness gated when the Bound receipt belongs to an older generation")
	request = getLPXRequest(t, ctx, reconciler.Client, dgd.Namespace, desired.requests[0].requestName)
	request.Status.ObservedGeneration = ptr.To(int64(6))
	require.NoError(t, reconciler.Update(ctx, request))
	classification, err = reconcileSelectedLPXForTest(t, ctx, reconciler, dgd, desired)
	require.NoError(t, err)
	require.IsType(t, &lpxOpen{}, classification)
	require.Equal(t, "LPXPublished", overlayLPXResult(runtimeReady, classification).Reason)

	t.Log("Stale failure diagnostics are not attributed to the current request generation")
	request = getLPXRequest(t, ctx, reconciler.Client, dgd.Namespace, desired.requests[0].requestName)
	request.Status.Phase = lpxv1alpha1.RequestPhaseUnsupported
	request.Status.Diagnostics = []lpxv1alpha1.StatusDiagnostic{{
		Code: "StaleUnsupported", Subject: "request/old", Detail: "belongs to an older generation",
	}}
	require.NoError(t, reconciler.Update(ctx, request))
	classification, err = reconcileSelectedLPXForTest(t, ctx, reconciler, dgd, desired)
	require.NoError(t, err)
	require.IsType(t, &lpxOpen{}, classification)
	require.NotContains(t, lpxResult(classification).Message, "StaleUnsupported")

	t.Log("The owner chain is sufficient for deletion without Dynamo finalizers")
	request = getLPXRequest(t, ctx, reconciler.Client, dgd.Namespace, desired.requests[0].requestName)
	pcs := &grovev1alpha1.PodCliqueSet{}
	require.NoError(t, reconciler.Get(ctx, client.ObjectKey{Namespace: dgd.Namespace, Name: desired.plan.PodCliqueSetName}, pcs))
	require.True(t, metav1.IsControlledBy(pcs, dgd))
	require.True(t, metav1.IsControlledBy(request, pcs))
	require.Empty(t, request.Finalizers)
	require.Empty(t, dgd.Finalizers)
}

func TestSelectedNodeLocalLPXReadinessUsesOneFixedScalingGroup(t *testing.T) {
	for _, pipeline := range []lpx.Pipeline{lpx.PipelineSingle, lpx.PipelineLPX} {
		t.Run(string(pipeline), func(t *testing.T) {
			ctx := t.Context()
			deployment, source, registry := newLPXTestDGD(t, pipeline)
			if pipeline == lpx.PipelineLPX {
				source.Spec.Components[0].Replicas = ptr.To(int32(1))
				source.Spec.Components[0].ComponentRole(nvidiacomv1beta1.ComponentRoleLPXConductor).Replicas = ptr.To(int32(2))
			}
			reconciler, desired := newPreparedLPXTestReconciler(t, registry, ctx, deployment, source)
			objects := lpxMaterializedObjects(t, reconciler, deployment, source, desired)
			for _, object := range objects {
				switch live := object.(type) {
				case *grovev1alpha1.PodCliqueScalingGroup:
					live.Status.Replicas, live.Status.UpdatedReplicas = 1, 1
					live.Status.AvailableReplicas, live.Status.ScheduledReplicas = 1, 1
				case *grovev1alpha1.PodClique:
					live.Status.Replicas, live.Status.UpdatedReplicas = live.Spec.Replicas, live.Spec.Replicas
					live.Status.ReadyReplicas, live.Status.ScheduledReplicas = live.Spec.Replicas, live.Spec.Replicas
					require.Equal(t, desired.workload.LPXComponentName(), live.Labels[consts.KubeLabelDynamoComponent])
				}
			}
			createLPXTestObjects(t, ctx, reconciler.Client, objects...)
			pcs := findLPXTestPodCliqueSet(t, objects)
			rootReads := 0
			base, ok := reconciler.Client.(client.WithWatch)
			require.True(t, ok)
			reader := interceptor.NewClient(base, interceptor.Funcs{Get: func(ctx context.Context, reader client.WithWatch, key client.ObjectKey, object client.Object, opts ...client.GetOption) error {
				if _, ok := object.(*grovev1alpha1.PodCliqueSet); ok {
					rootReads++
				}
				return reader.Get(ctx, key, object, opts...)
			}})
			observe := func() dynamo.GroveReadiness {
				ready, err := dynamo.EvaluateLPXGroveReadiness(ctx, reader, source, deployment, pcs)
				require.NoError(t, err)
				require.Len(t, ready.ComponentStatuses, 1)
				require.Zero(t, rootReads, "reuse the synchronized PCS observation")
				return ready
			}
			t.Log("Report one complete engine, never separate GPU or Agent component capacity")
			ready := observe()
			require.True(t, ready.Ready)
			status := ready.ComponentStatuses[desired.workload.LPXComponentName()]
			require.Equal(t, nvidiacomv1beta1.ComponentKindPodCliqueScalingGroup, status.ComponentKind)
			require.Equal(t, []string{desired.plan.LPXScalingGroup}, status.ComponentNames)
			require.Equal(t, int32(1), status.Replicas)
			require.Equal(t, int32(1), status.UpdatedReplicas)
			require.Equal(t, ptr.To(int32(1)), status.AvailableReplicas)
			require.Equal(t, ptr.To(int32(1)), status.ScheduledReplicas)
			if desired.plan.CyborgClique != "" {
				t.Log("A stale GPU-role scale cannot satisfy complete-engine readiness")
				gpu := findLPXTestClique(t, objects, desired.plan.CyborgClique).DeepCopy()
				require.NoError(t, reconciler.Get(ctx, client.ObjectKeyFromObject(gpu), gpu))
				gpu.Spec.Replicas, gpu.Status.Replicas, gpu.Status.UpdatedReplicas = 1, 1, 1
				gpu.Status.ReadyReplicas, gpu.Status.ScheduledReplicas = 1, 1
				require.NoError(t, reconciler.Update(ctx, gpu))
				ready = observe()
				require.False(t, ready.Ready)
				require.Equal(t, nvidiacomv1beta1.DGDReadyReasonUpdating, ready.Classification)
				t.Log("Partial GPU readiness contributes zero complete engine replicas")
				gpu.Spec.Replicas, gpu.Status.Replicas, gpu.Status.UpdatedReplicas = 2, 2, 2
				gpu.Status.ScheduledReplicas = 2
				require.NoError(t, reconciler.Update(ctx, gpu))
				ready = observe()
				require.False(t, ready.Ready)
				require.Equal(t, nvidiacomv1beta1.DGDReadyReasonPodsNotReady, ready.Classification)
				require.Equal(t, ptr.To(int32(0)), ready.ComponentStatuses[desired.workload.LPXComponentName()].AvailableReplicas)
				gpu.Status.ReadyReplicas = 2
				require.NoError(t, reconciler.Update(ctx, gpu))
				require.True(t, observe().Ready)
			}
			t.Log("Reject stale complete-engine capacity even when every existing role is ready")
			group := &grovev1alpha1.PodCliqueScalingGroup{}
			require.NoError(t, reconciler.Get(ctx, client.ObjectKey{Namespace: deployment.Namespace, Name: desired.plan.LPXScalingGroup}, group))
			group.Spec.Replicas = 2
			require.NoError(t, reconciler.Update(ctx, group))
			ready = observe()
			require.False(t, ready.Ready)
			require.Equal(t, nvidiacomv1beta1.DGDReadyReasonUpdating, ready.Classification)
		})
	}
}

func TestLPXClassifiesCurrentSchedulerReceipts(t *testing.T) {
	tests := []struct {
		name        string
		phase       lpxv1alpha1.RequestPhase
		wantState   nvidiacomv1beta1.DGDState
		wantReason  string
		wantMessage string
	}{
		{name: "pending", phase: lpxv1alpha1.RequestPhasePending, wantState: nvidiacomv1beta1.DGDStatePending, wantReason: "LPXSchedulerPending"},
		{name: "no fit", phase: lpxv1alpha1.RequestPhaseNoFit, wantState: nvidiacomv1beta1.DGDStatePending, wantReason: "LPXNoFit"},
		{name: "unsupported", phase: lpxv1alpha1.RequestPhaseUnsupported, wantState: nvidiacomv1beta1.DGDStateFailed, wantReason: "LPXUnsupported"},
		{name: "planned", phase: lpxv1alpha1.RequestPhasePlanned, wantState: nvidiacomv1beta1.DGDStatePending, wantReason: "LPXPlanned"},
		{name: "reserving", phase: lpxv1alpha1.RequestPhaseReserving, wantState: nvidiacomv1beta1.DGDStatePending, wantReason: "LPXReserving"},
		{name: "binding", phase: lpxv1alpha1.RequestPhaseBinding, wantState: nvidiacomv1beta1.DGDStatePending, wantReason: "LPXBinding"},
		{name: "degraded", phase: lpxv1alpha1.RequestPhaseDegraded, wantState: nvidiacomv1beta1.DGDStatePending, wantReason: "LPXDegraded"},
		{
			name: "unknown phase", phase: lpxv1alpha1.RequestPhase("future"),
			wantState: nvidiacomv1beta1.DGDStateFailed, wantReason: "LPXSchedulerStatusInvalid", wantMessage: `unknown phase "future"`,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Log("Record a scheduler status observation for the current generation")
			request := &lpxv1alpha1.LPUPipelineRequest{
				ObjectMeta: metav1.ObjectMeta{Generation: 7},
				Spec:       lpxv1alpha1.LPUPipelineRequestSpec{ExecutionBackend: lpxv1alpha1.ExecutionBackendNodeLocal},
				Status: &lpxv1alpha1.LPUPipelineRequestStatus{
					Phase:              test.phase,
					ObservedGeneration: ptr.To(int64(7)),
					Diagnostics: []lpxv1alpha1.StatusDiagnostic{{
						Code: "ExactDiagnostic", Subject: "requests/current", Detail: "the exact scheduler detail",
					}},
				},
			}

			t.Log("Classify the current receipt and preserve its phase, journal and diagnostic")
			classification := classifyPublishedLPX(request)
			observed, ok := classification.(*lpxSchedulerObserved)
			require.True(t, ok)
			require.Equal(t, test.phase, observed.Phase)
			result := lpxResult(observed)
			require.Equal(t, test.wantState, result.State)
			require.Equal(t, test.wantReason, result.Reason)
			require.Contains(t, result.Message, "ExactDiagnostic")
			require.Contains(t, result.Message, "requests/current")
			require.Contains(t, result.Message, "the exact scheduler detail")
			require.Equal(t, 1, strings.Count(result.Message, "LPX diagnostics:"))
			require.Equal(t, 1, strings.Count(result.Message, "the exact scheduler detail"))
			if test.wantMessage != "" {
				require.Contains(t, result.Message, test.wantMessage)
			}

			t.Log("The first failure outranks open requests; otherwise the first non-Bound receipt wins")
			before := request.DeepCopy()
			later := request.DeepCopy()
			later.Status.Diagnostics[0].Detail = "a later scheduler receipt"
			requests := []lpxModelMaterializing{{requestName: "open"}, {requestName: "observed"}, {requestName: "later"}}
			currents := map[string]*lpxv1alpha1.LPUPipelineRequest{"open": {}, "observed": request, "later": later}
			var want lpxClassification = &lpxOpen{}
			if test.wantState == nvidiacomv1beta1.DGDStateFailed {
				want = observed
			}
			require.Equal(t, want, classifyLPXCurrentRequests(requests, currents))
			slices.Reverse(requests)
			require.Equal(t, (*lpxSchedulerObserved)(later.Status), classifyLPXCurrentRequests(requests, currents))
			require.Equal(t, before, request)
		})
	}
}

func TestLPXSchedulerDiagnosticMessageRespectsConditionLimit(t *testing.T) {
	diagnostics := []lpxv1alpha1.StatusDiagnostic{{
		Code:    strings.Repeat("c", 253),
		Subject: strings.Repeat("s", 253),
		Detail:  strings.Repeat("界", 4096/len("界")),
	}}
	for range 31 {
		diagnostics = append(diagnostics, diagnostics[0])
	}

	message := lpxMessageWithDiagnostics("LPX scheduler diagnostic summary", diagnostics)

	require.LessOrEqual(t, len(message), maxDGDConditionMessageSize)
	require.True(t, utf8.ValidString(message))
	require.True(t, strings.HasSuffix(message, "…"))
}

func TestLPXRequestIdentityAndRetirementFence(t *testing.T) {
	type digestInput struct {
		namespace string
		name      string
		uid       types.UID
		model     string
	}
	digest := func(input digestInput, replicaIndex int32) string {
		return digestAttemptKey(
			input.namespace,
			input.name,
			input.uid,
			input.model,
			replicaIndex,
		)
	}
	base := digestInput{
		namespace: "ns", name: "dgd", uid: "uid-a", model: "default",
	}
	baseDigest := digest(base, 0)
	require.NotEmpty(t, baseDigest)
	for name, mutate := range map[string]func(*digestInput){
		"namespace": func(key *digestInput) { key.namespace = "other-ns" },
		"name":      func(key *digestInput) { key.name = "other-dgd" },
		"uid":       func(key *digestInput) { key.uid = "uid-b" },
		"model":     func(key *digestInput) { key.model = "model-b" },
	} {
		t.Run(name, func(t *testing.T) {
			changed := base
			mutate(&changed)
			require.NotEqual(t, baseDigest, digest(changed, 0))
			mutate(&changed)
			require.NotEmpty(t, digest(changed, 0))
		})
	}
	require.Equal(t, baseDigest, digest(base, 0), "change-then-revert must return to the same complete tuple digest")
	require.NotEqual(t, baseDigest, digest(base, 1))
	require.LessOrEqual(t, len(lpxRequestName("a-very-long-but-readable-dynamo-graph-deployment-name", baseDigest)), 63)
	ctx := context.Background()
	dgd, source, registry := newLPXTestDGD(t, lpx.PipelineSingle)
	reconciler, desired := newPreparedLPXTestReconciler(t, registry, ctx, dgd, source)

	t.Log("Fence the exact PCS owner and immutable deployment annotations")
	projection := &desired.requests[0]
	identity := &lpxGroveIdentity{pcs: deadlineTestPCS(dgd, "pcs-uid"), plan: desired.plan}
	current := &lpxv1alpha1.LPUPipelineRequest{
		ObjectMeta: metav1.ObjectMeta{UID: "request-uid", Annotations: lpxRequestAnnotations(dgd, projection), OwnerReferences: []metav1.OwnerReference{*metav1.NewControllerRef(identity.pcs, grovev1alpha1.SchemeGroupVersion.WithKind("PodCliqueSet"))}},
		Spec:       projection.modelProjection.RequestSpec(identity.plan, projection.agentCliqueName),
	}
	require.NoError(t, validateCurrentLPXRequest(dgd, projection, identity, current))
	changed := *identity
	changed.pcs = identity.pcs.DeepCopy()
	changed.pcs.UID = "replacement-pcs-uid"
	require.ErrorContains(t, validateCurrentLPXRequest(dgd, projection, &changed, current), "exact PCS owner")

	createLPXTestObjects(t, ctx, reconciler.Client, lpxMaterializedObjects(t, reconciler, dgd, source, desired)...)
	publishSelectedLPXForTest(t, ctx, reconciler, dgd, desired)

	dgd.Generation++
	pcs := observedLPXTestPodCliqueSet(t, ctx, reconciler, dgd, desired)
	classification, _, err := reconciler.reconcileLPXKnownIntentFence(ctx, dgd, source, pcs, nil)
	require.NoError(t, err)
	require.Nil(t, classification)
	require.NoError(t, reconciler.Get(ctx, client.ObjectKey{Namespace: dgd.Namespace, Name: desired.requests[0].requestName}, &lpxv1alpha1.LPUPipelineRequest{}))

	classification, _, err = reconciler.reconcileLPXKnownIntentFence(ctx, dgd, source, pcs, nil)
	require.NoError(t, err)
	require.Nil(t, classification)
}

func TestLPXDisabledPreservesPublishedWorkloadUntilDeletion(t *testing.T) {
	for _, scenario := range []struct{ name, message string }{
		{"LPX", "LPX integration is disabled"},
		{"Grove", "Grove is disabled"},
	} {
		t.Run(scenario.name, func(t *testing.T) {
			t.Log("Publish a complete workload, discovery service and scheduling request")
			ctx := t.Context()
			child, source, registry := newLPXTestDGD(t, lpx.PipelineSingle)
			source.Annotations[consts.KubeAnnotationDynamoDiscoveryBackend] = string(configv1alpha1.DiscoveryBackendKubernetes)
			source.Spec.Scheduling = deadlineTestScheduling()
			r, selected := newPreparedLPXTestReconciler(t, registry, ctx, child, source)
			createLPXTestObjects(t, ctx, r.Client, lpxMaterializedObjects(t, r, child, source, selected)...)
			key := client.ObjectKeyFromObject(child)
			for range 4 {
				_, err := r.Reconcile(ctx, ctrl.Request{NamespacedName: key})
				require.NoError(t, err)
			}
			require.NoError(t, r.Get(ctx, key, child))
			before := []client.ObjectList{
				&grovev1alpha1.PodCliqueSetList{}, &corev1.ConfigMapList{},
				&corev1.ServiceList{}, &lpxv1alpha1.LPUPipelineRequestList{},
			}
			for _, list := range before {
				require.NoError(t, r.List(ctx, list))
				require.Positive(t, meta.LenList(list))
			}

			t.Log("Disable the provider before an input edit and an expired deadline can retire the workload")
			if scenario.name == "LPX" {
				r.runtimeConfig.Gate.LPX = false
			} else {
				r.runtimeConfig.Gate.Grove = false
			}
			r.modelRegistry = nil
			require.NoError(t, r.Get(ctx, client.ObjectKeyFromObject(source), source))
			originalSpec := source.Spec.DeepCopy()
			source.Spec.Components[0].LPX.BuildID = "edited-while-disabled"
			require.NoError(t, r.Update(ctx, source))
			for range 2 {
				result, err := r.Reconcile(ctx, ctrl.Request{NamespacedName: key})
				require.NoError(t, err)
				require.Zero(t, result)
			}
			require.NoError(t, r.Get(ctx, key, child))
			failed := meta.FindStatusCondition(child.Status.Conditions, "Failed")
			require.NotNil(t, failed)
			require.Equal(t, metav1.ConditionTrue, failed.Status)
			require.Equal(t, "LPXUnavailable", failed.Reason)
			require.Equal(t, scenario.message, failed.Message)
			require.Equal(t, child.Generation, failed.ObservedGeneration)
			require.Equal(t, metav1.ConditionFalse, meta.FindStatusCondition(child.Status.Conditions, "Ready").Status)
			for _, list := range before {
				current := list.DeepCopyObject().(client.ObjectList)
				require.NoError(t, r.List(ctx, current))
				require.Equal(t, list, current)
			}

			t.Log("Re-enable unchanged intent and resume the original PCS and request identities")
			source.Spec = *originalSpec
			require.NoError(t, r.Update(ctx, source))
			r.runtimeConfig.Gate.LPX, r.runtimeConfig.Gate.Grove = true, true
			r.modelRegistry = registry
			_, err := r.Reconcile(ctx, ctrl.Request{NamespacedName: key})
			require.NoError(t, err)
			require.NoError(t, r.Get(ctx, key, child))
			require.NotEqual(t, "LPXUnavailable", meta.FindStatusCondition(child.Status.Conditions, "Ready").Reason)
			for _, list := range before {
				current := list.DeepCopyObject().(client.ObjectList)
				require.NoError(t, r.List(ctx, current))
				require.Equal(t, list, current)
			}

			t.Log("Deletion remains available while disabled and needs no controller finalizer")
			if scenario.name == "LPX" {
				r.runtimeConfig.Gate.LPX = false
			} else {
				r.runtimeConfig.Gate.Grove = false
			}
			r.modelRegistry = nil
			require.Empty(t, child.Finalizers)
			require.NoError(t, r.Delete(ctx, child))
			_, err = r.Reconcile(ctx, ctrl.Request{NamespacedName: key})
			require.NoError(t, err)
			require.True(t, apierrors.IsNotFound(r.Get(ctx, key, &nvidiacomv1alpha1.LPXGraphDeployment{})))
		})
	}
}

func TestLPXFinalizationAfterRequestAPIRemoval(t *testing.T) {
	resource := lpxv1alpha1.GroupVersion.WithResource("lpupipelinerequests").GroupResource()
	for _, scenario := range []struct {
		name    string
		listErr error
	}{
		{"removed CRD", apierrors.NewNotFound(resource, "")},
		{"forbidden", apierrors.NewForbidden(resource, "", errors.New("denied"))},
		{"transport error", errors.New("connection reset")},
	} {
		t.Run(scenario.name, func(t *testing.T) {
			t.Log("Keep a legacy controller finalizer while the request API cannot be read")
			ctx := t.Context()
			source := newLPXTestSource(lpx.PipelineSingle, "build-v2")
			child := newLPXTestDeployment(t, source)
			child.Finalizers = []string{legacyLPXGraphDeploymentFinalizer}
			r := newLPXTestReconciler(t, nil, child, source)
			r.apiReader = interceptor.NewClient(r.Client.(client.WithWatch), interceptor.Funcs{
				List: func(context.Context, client.WithWatch, client.ObjectList, ...client.ListOption) error {
					return scenario.listErr
				},
			})

			t.Log("Ordinary reconciliation still fails when requests cannot be listed")
			_, err := r.listOwnedLPXRequests(ctx, child, nil)
			require.ErrorIs(t, err, scenario.listErr)

			t.Log("Release the legacy finalizer without reading requests; owner garbage collection owns cleanup")
			key := client.ObjectKeyFromObject(child)
			require.NoError(t, r.Delete(ctx, child))
			_, err = r.Reconcile(ctx, ctrl.Request{NamespacedName: key})
			require.NoError(t, err)
			require.True(t, apierrors.IsNotFound(r.Get(ctx, key, child)))
		})
	}
}

func TestLPXControllerPreservesPublishedWorkWhenProviderEditIsInvalid(t *testing.T) {
	t.Log("Reject an incompatible provider edit without deleting the existing engine")
	dgd, source, registry := newLPXTestDGD(t, lpx.PipelineSingle)
	reconciler, desired := newPreparedLPXTestReconciler(t, registry, t.Context(), dgd, source)
	createLPXTestObjects(t, t.Context(), reconciler.Client, lpxMaterializedObjects(t, reconciler, dgd, source, desired)...)
	publishSelectedLPXForTest(t, t.Context(), reconciler, dgd, desired)

	require.NoError(t, reconciler.Get(t.Context(), client.ObjectKeyFromObject(source), source))
	source.Annotations[consts.KubeAnnotationWorkloadProvider] = consts.WorkloadProviderComponent
	require.NoError(t, reconciler.Update(t.Context(), source))
	_, err := reconciler.Reconcile(t.Context(), ctrl.Request{NamespacedName: client.ObjectKeyFromObject(dgd)})
	require.ErrorContains(t, err, "LPX requires the Grove workload provider")
	request := getLPXRequest(t, t.Context(), reconciler.Client, dgd.Namespace, desired.requests[0].requestName)
	require.Nil(t, request.DeletionTimestamp)
}

func TestLPXPublicationFenceRejectsForeignExactNameCollisionFromAuthoritativeList(t *testing.T) {
	t.Log("Give foreign requests the desired names in the opposite lexical order")
	ctx := t.Context()
	source := newLPXTestSource(lpx.PipelineSingle, "build-v2")
	dgd := newLPXTestDeployment(t, source)
	foreign := &lpxv1alpha1.LPUPipelineRequest{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "z-foreign-request",
			Namespace: dgd.Namespace,
			UID:       "foreign-collision-uid",
			Labels:    map[string]string{lpxOwnerUIDLabel: string(dgd.UID)},
			OwnerReferences: []metav1.OwnerReference{{
				APIVersion: grovev1alpha1.SchemeGroupVersion.String(),
				Kind:       lpxPodCliqueSetKind,
				Name:       lpxTestOtherName,
				UID:        "other-uid",
				Controller: ptr.To(true),
			}},
		},
	}
	second := foreign.DeepCopy()
	second.Name, second.UID = "a-foreign-request", "second-foreign-collision-uid"
	second.OwnerReferences[0].APIVersion = nvidiacomv1beta1.GroupVersion.String()
	second.OwnerReferences[0].Kind = nvidiacomv1beta1.DynamoGraphDeploymentGVK.Kind
	desired := &lpxMaterializing{requests: []lpxModelMaterializing{
		{requestName: foreign.Name}, {requestName: second.Name},
	}}
	reconciler := newLPXTestReconciler(t, nil, dgd, source)

	t.Log("Complete an empty preflight fence before foreign requests occupy the desired names")
	currents, retiring, err := reconciler.reconcileLPXPublicationFence(ctx, dgd, desired, nil)
	require.NoError(t, err)
	require.Empty(t, currents)
	require.Nil(t, retiring)
	for _, request := range []*lpxv1alpha1.LPUPipelineRequest{foreign, second} {
		require.NoError(t, reconciler.Create(ctx, request))
	}

	t.Log("Exclude foreign requests from owned publications and the known-intent retirement fence")
	requests, err := reconciler.listOwnedLPXRequests(ctx, dgd, nil)
	require.NoError(t, err)
	require.Empty(t, requests)
	classification, _, err := reconciler.reconcileLPXKnownIntentFence(ctx, dgd, source, nil, nil)
	require.NoError(t, err)
	require.Nil(t, classification)

	t.Log("Observe labeled exact-name collisions with one scoped authoritative list and no per-request Gets")
	reader, ok := reconciler.apiReader.(client.WithWatch)
	require.True(t, ok)
	requestGets := 0
	requestListCalls := make([]client.ListOptions, 0)
	reconciler.apiReader = interceptor.NewClient(reader, interceptor.Funcs{
		Get: func(
			ctx context.Context,
			delegated client.WithWatch,
			key client.ObjectKey,
			object client.Object,
			opts ...client.GetOption,
		) error {
			if _, ok := object.(*lpxv1alpha1.LPUPipelineRequest); ok {
				requestGets++
			}
			return delegated.Get(ctx, key, object, opts...)
		},
		List: func(
			ctx context.Context,
			delegated client.WithWatch,
			list client.ObjectList,
			opts ...client.ListOption,
		) error {
			if _, ok := list.(*lpxv1alpha1.LPUPipelineRequestList); ok {
				requestListCalls = append(requestListCalls, *(&client.ListOptions{}).ApplyOptions(opts))
			}
			return delegated.List(ctx, list, opts...)
		},
	})

	currents, retiring, err = reconciler.reconcileLPXPublicationFence(ctx, dgd, desired, nil)
	require.EqualError(t, err, fmt.Sprintf("LPX request name %q is occupied by a foreign owner", foreign.Name))
	require.Nil(t, currents)
	require.Nil(t, retiring)
	require.Zero(t, requestGets, "exact-name collision detection must not issue one request Get per desired projection")
	require.Len(t, requestListCalls, 1)
	require.Zero(t, requestListCalls[0].Limit)
	require.Empty(t, requestListCalls[0].Continue)
	require.Equal(t, dgd.Namespace, requestListCalls[0].Namespace)
	require.Equal(t, lpxOwnerUIDLabel+"="+string(dgd.UID), requestListCalls[0].LabelSelector.String())

	t.Log("Refresh publication state and refuse the foreign requests created after the empty preflight")
	classification, err = reconcileSelectedLPXForTest(t, ctx, reconciler, dgd, desired)
	require.ErrorContains(t, err, "occupied by a foreign owner")
	require.Nil(t, classification)
	require.Len(t, requestListCalls, 2, "final publication must refresh the authoritative request snapshot")
	for _, request := range []*lpxv1alpha1.LPUPipelineRequest{foreign, second} {
		stored := &lpxv1alpha1.LPUPipelineRequest{}
		require.NoError(t, reconciler.Get(ctx, client.ObjectKeyFromObject(request), stored))
		require.Equal(t, request.UID, stored.UID)
		require.Equal(t, request.OwnerReferences, stored.OwnerReferences)
		require.True(t, stored.DeletionTimestamp.IsZero())
	}
	allRequests := &lpxv1alpha1.LPUPipelineRequestList{}
	require.NoError(t, reconciler.List(ctx, allRequests, client.InNamespace(dgd.Namespace)))
	require.Len(t, allRequests.Items, 2, "the fresh final snapshot must not publish alongside the collisions")
}

func requireLPXClosed(t *testing.T, classification lpxClassification) *lpxClosed {
	t.Helper()
	closed, ok := classification.(*lpxClosed)
	require.True(t, ok, "classification=%T", classification)
	return closed
}

func newLPXNodeLocalBoundStatus(generation, _ int64, _ lpxv1alpha1.PlanDigest) *lpxv1alpha1.LPUPipelineRequestStatus {
	return &lpxv1alpha1.LPUPipelineRequestStatus{Phase: lpxv1alpha1.RequestPhaseBound, ObservedGeneration: &generation}
}

func findLPXTestClique(t *testing.T, objects []client.Object, name string) *grovev1alpha1.PodClique {
	t.Helper()
	for _, object := range objects {
		if clique, ok := object.(*grovev1alpha1.PodClique); ok && clique.Name == name {
			return clique
		}
	}
	t.Fatalf("PodClique %q not found", name)
	return nil
}

func findLPXTestScalingGroup(t *testing.T, objects []client.Object, name string) *grovev1alpha1.PodCliqueScalingGroup {
	t.Helper()
	for _, object := range objects {
		if group, ok := object.(*grovev1alpha1.PodCliqueScalingGroup); ok && group.Name == name {
			return group
		}
	}
	t.Fatalf("PodCliqueScalingGroup %q not found", name)
	return nil
}

func findLPXTestPodCliqueSet(t *testing.T, objects []client.Object) *grovev1alpha1.PodCliqueSet {
	t.Helper()
	for _, object := range objects {
		if pcs, ok := object.(*grovev1alpha1.PodCliqueSet); ok {
			return pcs
		}
	}
	t.Fatal("PodCliqueSet not found")
	return nil
}

func renderLPXTestPodCliqueSet(
	t *testing.T, ctx context.Context, reconciler *graphReconciler,
	dgd *nvidiacomv1alpha1.LPXGraphDeployment, source *nvidiacomv1beta1.DynamoGraphDeployment, desired *lpxMaterializing,
) *grovev1alpha1.PodCliqueSet {
	t.Helper()
	rendered, _, err := renderPodCliqueSet(ctx, source, reconciler.Config, reconciler.runtimeConfig,
		reconciler.DockerSecretRetriever, desired.workload, desired.plan, dgd)
	require.NoError(t, err)
	return rendered
}

func newLPXTestDGD(t *testing.T, pipeline lpx.Pipeline) (*nvidiacomv1alpha1.LPXGraphDeployment, *nvidiacomv1beta1.DynamoGraphDeployment, lpx.ModelRegistry) {
	t.Helper()
	compilationMode := manifestcapnpv2.CompilationMode_lpuOnly
	if pipeline == lpx.PipelineLPX {
		compilationMode = manifestcapnpv2.CompilationMode_lpx
	}
	const buildID = "build-v2"
	registry := newLPXTestRegistryWithPartitionsAndMode(t, buildID, []int{7, 8}, compilationMode)
	source := newLPXTestSource(pipeline, buildID)
	return newLPXTestDeployment(t, source), source, registry
}

func newLPXTestDeployment(t *testing.T, source *nvidiacomv1beta1.DynamoGraphDeployment) *nvidiacomv1alpha1.LPXGraphDeployment {
	t.Helper()
	revision, err := dynamo.LPXInputRevision(source, "")
	require.NoError(t, err)
	return &nvidiacomv1alpha1.LPXGraphDeployment{
		TypeMeta: metav1.TypeMeta{APIVersion: nvidiacomv1alpha1.GroupVersion.String(), Kind: "LPXGraphDeployment"},
		ObjectMeta: metav1.ObjectMeta{Name: source.Name, Namespace: source.Namespace, UID: types.UID("lpx-" + string(source.UID)), Generation: 1,
			Annotations:     map[string]string{lpx.DGDGenerationAnnotation: strconv.FormatInt(source.Generation, 10)},
			OwnerReferences: []metav1.OwnerReference{*metav1.NewControllerRef(source, nvidiacomv1beta1.DynamoGraphDeploymentGVK)},
		},
		Spec: nvidiacomv1alpha1.LPXGraphDeploymentSpec{InputRevision: revision},
	}
}

func newLPXTestSource(pipeline lpx.Pipeline, buildID string) *nvidiacomv1beta1.DynamoGraphDeployment {
	one := int32(1)
	source := &nvidiacomv1beta1.DynamoGraphDeployment{
		TypeMeta: metav1.TypeMeta{APIVersion: nvidiacomv1beta1.GroupVersion.String(), Kind: "DynamoGraphDeployment"},
		ObjectMeta: metav1.ObjectMeta{
			Name: "selected-source", Namespace: "test", UID: "source-uid", Generation: 3,
			Annotations: map[string]string{consts.KubeAnnotationLPXSchedulerBackend: consts.LPXSchedulerBackend},
		},
		Spec: nvidiacomv1beta1.DynamoGraphDeploymentSpec{
			Components: []nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec{{
				ComponentName: "lpx", ComponentType: nvidiacomv1beta1.ComponentTypeLPX, Replicas: &one,
				LPX: &nvidiacomv1beta1.LPXConfig{BuildID: buildID},
				Roles: []nvidiacomv1beta1.ComponentRoleSpec{{Name: nvidiacomv1beta1.ComponentRoleLPXConductor}, {
					Name: nvidiacomv1beta1.ComponentRoleLPXAgent,
					PodTemplate: &corev1.PodTemplateSpec{Spec: corev1.PodSpec{
						Containers: []corev1.Container{{
							Name: consts.MainContainerName, Image: "lpu-runtime",
							VolumeMounts: []corev1.VolumeMount{
								{Name: consts.ModelStorageVolumeName, MountPath: "/models"},
								{Name: "config", MountPath: "/configs"},
								{Name: "host-dev", MountPath: "/dev"},
								{Name: "host-sys", MountPath: "/sys"},
								{Name: "ssh-secret", MountPath: "/ssh-pk", ReadOnly: true},
								{Name: "hugepages", MountPath: "/dev/hugepages"},
								{Name: "single-v2-ssh-key", MountPath: "/tmp/dynamo-lpu-ssh"},
							},
						}},
						Volumes: []corev1.Volume{
							{Name: consts.ModelStorageVolumeName, VolumeSource: corev1.VolumeSource{EmptyDir: &corev1.EmptyDirVolumeSource{}}},
							{Name: "host-dev", VolumeSource: corev1.VolumeSource{HostPath: &corev1.HostPathVolumeSource{Path: "/dev", Type: ptr.To(corev1.HostPathDirectory)}}},
							{Name: "host-sys", VolumeSource: corev1.VolumeSource{HostPath: &corev1.HostPathVolumeSource{Path: "/sys", Type: ptr.To(corev1.HostPathDirectory)}}},
							{Name: "hugepages", VolumeSource: corev1.VolumeSource{EmptyDir: &corev1.EmptyDirVolumeSource{Medium: corev1.StorageMediumHugePages}}},
							{Name: "ssh-secret", VolumeSource: corev1.VolumeSource{Secret: &corev1.SecretVolumeSource{SecretName: "ssh-secret", DefaultMode: ptr.To[int32](0644)}}},
							{Name: "single-v2-ssh-key", VolumeSource: corev1.VolumeSource{EmptyDir: &corev1.EmptyDirVolumeSource{}}},
						},
					}},
				}},
			}},
		},
	}
	// Each role authors its own startup; the conductor cannot inherit the Agent command.
	agentTemplate := source.Spec.Components[0].ComponentRole(nvidiacomv1beta1.ComponentRoleLPXAgent).PodTemplate
	agentTemplate.Spec.Containers[0].Command = []string{"/bin/quasar-entrypoint"}
	conductorTemplate := agentTemplate.DeepCopy()
	conductorTemplate.Spec.Containers[0].Command = []string{"/bin/nova"}
	source.Spec.Components[0].ComponentRole(nvidiacomv1beta1.ComponentRoleLPXConductor).PodTemplate = conductorTemplate
	if pipeline == lpx.PipelineLPX {
		*source.Spec.Components[0].ComponentRole(nvidiacomv1beta1.ComponentRoleLPXConductor) = nvidiacomv1beta1.ComponentRoleSpec{
			Name:     nvidiacomv1beta1.ComponentRoleLPXConductor,
			Replicas: &one,
			PodTemplate: &corev1.PodTemplateSpec{Spec: corev1.PodSpec{
				ResourceClaims: []corev1.PodResourceClaim{{Name: "gpu", ResourceClaimTemplateName: ptr.To("gpu")}},
				Containers: []corev1.Container{{Name: consts.MainContainerName, Image: "cyborg-runtime",
					Resources: corev1.ResourceRequirements{Claims: []corev1.ResourceClaim{{Name: "gpu"}}},
					VolumeMounts: []corev1.VolumeMount{
						{Name: "config", MountPath: "/configs"},
						{Name: consts.ModelStorageVolumeName, MountPath: "/models"},
					},
				}},
				Volumes: []corev1.Volume{
					{Name: consts.ModelStorageVolumeName, VolumeSource: corev1.VolumeSource{EmptyDir: &corev1.EmptyDirVolumeSource{}}},
				},
			}},
		}
	}
	return source
}

func newLPXSpecDecodeTestDGD(t *testing.T) (*nvidiacomv1alpha1.LPXGraphDeployment, *nvidiacomv1beta1.DynamoGraphDeployment, lpx.ModelRegistry) {
	t.Helper()
	source := newLPXSpecDecodeTestSource()
	root := t.TempDir()
	writeLPXTestBuild(t, root, "draft-build", []int{7, 8}, manifestcapnpv2.CompilationMode_lpuOnly)
	writeLPXTestBuild(t, root, "target-build", []int{9, 10}, manifestcapnpv2.CompilationMode_lpuOnly)
	registryURL := (&url.URL{Scheme: lpx.BuildSchemeFile, Path: root}).String()
	registry, err := lpx.NewModelRegistry(registryURL, nil)
	require.NoError(t, err)
	return newLPXTestDeployment(t, source), source, registry
}

func newLPXSpecDecodeTestSource() *nvidiacomv1beta1.DynamoGraphDeployment {
	source := newLPXTestSource(lpx.PipelineSingle, "target-build")
	target := &source.Spec.Components[0]
	draft := target.DeepCopy()
	draft.ComponentName, draft.LPX.BuildID, draft.Replicas = "draft", "draft-build", ptr.To(int32(2))
	draft.Roles = []nvidiacomv1beta1.ComponentRoleSpec{*draft.ComponentRole(nvidiacomv1beta1.ComponentRoleLPXAgent)}
	source.Spec.Components = append(source.Spec.Components, *draft)
	return source
}

func newLPXTestRegistryWithPartitionsAndMode(
	t *testing.T,
	buildID string,
	partitionIDs []int,
	compilationMode manifestcapnpv2.CompilationMode,
) lpx.ModelRegistry {
	t.Helper()
	root := t.TempDir()
	writeLPXTestBuild(t, root, buildID, partitionIDs, compilationMode)
	registryURL := (&url.URL{Scheme: lpx.BuildSchemeFile, Path: root}).String()
	registry, err := lpx.NewModelRegistry(registryURL, nil)
	require.NoError(t, err)
	return registry
}

func writeLPXTestBuild(
	t *testing.T,
	root string,
	buildID string,
	partitionIDs []int,
	compilationMode manifestcapnpv2.CompilationMode,
) {
	t.Helper()
	require.NotEmpty(t, partitionIDs)
	buildDir := filepath.Join(root, buildID)
	require.NoError(t, os.Mkdir(buildDir, 0o700))

	t.Log("Encode the V2 manifest header and the tokenizer consumed by rendered runtimes")
	message, manifest := newTestGraphManifest(t)

	t.Log("Retain the opaque compiler provenance without the removed V1 publication contract")
	populateTestGraphBuild(t, manifest, buildID)

	t.Log("Describe the same one-batch runtime with explicit V2 runtime I/O and prop-sync evidence")
	deployment, program := newTestGraphProgram(t, manifest, compilationMode, uint32(len(partitionIDs)*2), 8192)
	program.SetNumKvCaches(1)
	program.SetNumBatchSplitDivisions(1)
	chains, err := deployment.NewSelectedPropSyncChains(1)
	require.NoError(t, err)
	selectedIDs, err := chains.At(0).NewPartitionIds(int32(len(partitionIDs)))
	require.NoError(t, err)
	for index, partitionID := range partitionIDs {
		selectedIDs.Set(index, uint32(partitionID))
	}

	t.Log("Package the LPU partitions and the hybrid CUDA marker in the flat V2 artifact inventory")
	artifacts, err := manifest.NewArtifacts()
	require.NoError(t, err)
	partitionCount := len(partitionIDs)
	if compilationMode == manifestcapnpv2.CompilationMode_lpx {
		partitionCount++
	}
	partitions, err := artifacts.NewPartitions(int32(partitionCount))
	require.NoError(t, err)
	for index, partitionID := range partitionIDs {
		partition := partitions.At(index)
		ref, err := partition.NewPartition()
		require.NoError(t, err)
		ref.SetDeviceType(manifestcapnpv2.DeviceType_lpu)
		ref.SetPartitionId(uint32(partitionID))
		detail, err := partition.Detail().NewLpu()
		require.NoError(t, err)
		require.NoError(t, detail.SetPath(fmt.Sprintf("part-%d", partitionID)))
		require.NoError(t, detail.SetTopology("URSA_V2__Q8__16C__G_96_25__KP_FEC__GHZ_1_0__NO_FPGA"))
		detail.SetNumChips(16)
		detail.SetDevicesPerNode(8)
	}
	if compilationMode == manifestcapnpv2.CompilationMode_lpx {
		partition := partitions.At(len(partitionIDs))
		ref, err := partition.NewPartition()
		require.NoError(t, err)
		ref.SetDeviceType(manifestcapnpv2.DeviceType_cuda)
		ref.SetPartitionId(uint32(len(partitionIDs)))
		_, err = partition.Detail().NewCuda()
		require.NoError(t, err)
	}

	t.Log("Publish the required V2 compiler manifest")
	payload, err := message.Marshal()
	require.NoError(t, err)
	require.NoError(t, os.WriteFile(filepath.Join(buildDir, "manifest.v2.capnp.bin"), payload, 0o600))
}

func newLPXTestReconciler(
	t *testing.T,
	registry lpx.ModelRegistry,
	dgd *nvidiacomv1alpha1.LPXGraphDeployment,
	source *nvidiacomv1beta1.DynamoGraphDeployment,
	objects ...client.Object,
) *graphReconciler {
	t.Helper()
	revision, err := dynamo.LPXInputRevision(source, "")
	require.NoError(t, err)
	dgd.Spec.InputRevision = revision
	if dgd.ResourceVersion == "" {
		dgd.ResourceVersion = "1"
	}
	scheme := newLPXTestScheme(t)
	seed := append([]client.Object{dgd.DeepCopy(), source.DeepCopy()}, objects...)
	base := fake.NewClientBuilder().
		WithScheme(scheme).
		WithIndex(&nvidiacomv1alpha1.LPXGraphDeployment{}, lpxSourceOwnerIndex, lpxSourceOwnerReferences).
		WithStatusSubresource(&nvidiacomv1beta1.DynamoGraphDeployment{}, &nvidiacomv1alpha1.LPXGraphDeployment{}).
		WithObjects(seed...).
		Build()
	nextLPRUID := 0
	wrapped := interceptor.NewClient(base, interceptor.Funcs{
		// The fake client does not implement scale for Grove custom resources.
		SubResourceUpdate: func(ctx context.Context, delegated client.Client, subresource string, object client.Object, opts ...client.SubResourceUpdateOption) error {
			if group, ok := object.(*grovev1alpha1.PodCliqueScalingGroup); ok && subresource == "scale" {
				options := (&client.SubResourceUpdateOptions{}).ApplyOptions(opts)
				scale := options.SubResourceBody.(*autoscalingv1.Scale)
				group.ResourceVersion = scale.ResourceVersion
				group.Spec.Replicas = scale.Spec.Replicas
				return delegated.Update(ctx, group)
			}
			return delegated.SubResource(subresource).Update(ctx, object, opts...)
		},
		Create: func(ctx context.Context, delegated client.WithWatch, object client.Object, opts ...client.CreateOption) error {
			if request, ok := object.(*lpxv1alpha1.LPUPipelineRequest); ok {
				if request.UID == "" {
					nextLPRUID++
					request.UID = types.UID(fmt.Sprintf("lpr-%s-%d", request.Name, nextLPRUID))
				}
				if request.Generation == 0 {
					request.Generation = 1
				}
				if request.CreationTimestamp.IsZero() {
					request.CreationTimestamp = metav1.NewTime(time.Now().UTC().Truncate(time.Second))
				}
			}
			return delegated.Create(ctx, object, opts...)
		},
	})
	recorder := events.NewFakeRecorder(100)
	config := &configv1alpha1.OperatorConfiguration{
		LPX: configv1alpha1.LPXConfiguration{Enabled: true},
	}
	runtimeConfig := &commoncontroller.RuntimeConfig{Gate: features.Gates{Grove: true, DRA: true, LPX: true}}
	return &graphReconciler{
		Client:        wrapped,
		recorder:      recorder,
		apiReader:     wrapped,
		runtimeConfig: runtimeConfig,
		modelRegistry: registry,
		Config:        config,
	}
}

func lpxMaterializedObjects(
	t *testing.T,
	reconciler *graphReconciler,
	dgd *nvidiacomv1alpha1.LPXGraphDeployment,
	source *nvidiacomv1beta1.DynamoGraphDeployment,
	desired *lpxMaterializing,
) []client.Object {
	t.Helper()
	t.Log("Render LPX intent once and emulate the API-server and Grove observations")
	pcs := renderLPXTestPodCliqueSet(t, t.Context(), reconciler, dgd, source, desired)
	pcs.TypeMeta = metav1.TypeMeta{APIVersion: grovev1alpha1.SchemeGroupVersion.String(), Kind: "PodCliqueSet"}
	pcs.UID, pcs.Generation = "pcs-uid", 1
	pcs.OwnerReferences = []metav1.OwnerReference{*metav1.NewControllerRef(dgd, nvidiacomv1alpha1.LPXGraphDeploymentGVK)}
	const generationHash = "pcs-generation-hash"
	pcs.Status = grovev1alpha1.PodCliqueSetStatus{
		ObservedGeneration: ptr.To(pcs.Generation), CurrentGenerationHash: ptr.To(generationHash),
	}

	// Grove materializes the rendered scaling-group configuration once.
	groupTemplate := pcs.Spec.Template.PodCliqueScalingGroupConfigs[0].DeepCopy()
	group := &grovev1alpha1.PodCliqueScalingGroup{
		TypeMeta: metav1.TypeMeta{APIVersion: grovev1alpha1.SchemeGroupVersion.String(), Kind: "PodCliqueScalingGroup"},
		ObjectMeta: metav1.ObjectMeta{
			Name: desired.plan.LPXScalingGroup, Namespace: pcs.Namespace, UID: "lpu-group-uid",
			Generation: 1, Labels: grovecommon.GetDefaultLabelsForPodCliqueSetManagedResources(pcs.Name), Annotations: groupTemplate.Annotations,
			OwnerReferences: []metav1.OwnerReference{*metav1.NewControllerRef(pcs, grovev1alpha1.SchemeGroupVersion.WithKind("PodCliqueSet"))},
		},
		Spec: grovev1alpha1.PodCliqueScalingGroupSpec{
			Replicas: ptr.Deref(groupTemplate.Replicas, 1), MinAvailable: groupTemplate.MinAvailable,
			CliqueNames: groupTemplate.CliqueNames,
		},
		Status: grovev1alpha1.PodCliqueScalingGroupStatus{
			ObservedGeneration: ptr.To(int64(1)), CurrentPodCliqueSetGenerationHash: ptr.To(generationHash),
		},
	}
	group.Labels[grovecommon.LabelPartOfKey] = pcs.Name
	group.Labels[grovecommon.LabelPodCliqueSetReplicaIndex] = "0"
	objects := []client.Object{pcs, group}

	// Each Grove replica owns independent cliques.
	for replicaIndex := int32(0); replicaIndex < group.Spec.Replicas; replicaIndex++ {
		parent := grovecommon.ResourceNameReplica{Name: group.Name, Replica: int(replicaIndex)}
		for _, template := range pcs.Spec.Template.Cliques {
			rendered := template.DeepCopy()
			name := grovecommon.GeneratePodCliqueName(parent, rendered.Name)
			clique := &grovev1alpha1.PodClique{
				TypeMeta: metav1.TypeMeta{APIVersion: grovev1alpha1.SchemeGroupVersion.String(), Kind: "PodClique"},
				ObjectMeta: metav1.ObjectMeta{
					Name: name, Namespace: pcs.Namespace, UID: types.UID(name + "-uid"), Generation: 1,
					Labels: rendered.Labels, Annotations: rendered.Annotations,
					OwnerReferences: []metav1.OwnerReference{*metav1.NewControllerRef(group, grovev1alpha1.SchemeGroupVersion.WithKind("PodCliqueScalingGroup"))},
				},
				Spec: rendered.Spec,
				Status: grovev1alpha1.PodCliqueStatus{
					ObservedGeneration: ptr.To(int64(1)), CurrentPodCliqueSetGenerationHash: ptr.To(generationHash),
					CurrentPodTemplateHash: ptr.To(rendered.Name + "-pod-template-hash"),
				},
			}
			clique.Labels[grovecommon.LabelPartOfKey] = pcs.Name
			clique.Labels[grovecommon.LabelPodCliqueScalingGroup] = group.Name
			clique.Labels[grovecommon.LabelPodCliqueScalingGroupReplicaIndex] = strconv.FormatInt(int64(replicaIndex), 10)
			for index, dependency := range clique.Spec.StartsAfter {
				clique.Spec.StartsAfter[index] = grovecommon.GeneratePodCliqueName(parent, dependency)
			}
			objects = append(objects, clique)
		}
	}
	return objects
}

func withLPXTestReplicas(t *testing.T, desired *lpxMaterializing, replicas int32) *lpxMaterializing {
	t.Helper()
	require.NotNil(t, desired)
	require.NotNil(t, desired.plan)
	require.Positive(t, desired.plan.Replicas)
	require.GreaterOrEqual(t, replicas, int32(0))
	require.LessOrEqual(t, replicas, desired.plan.Replicas)
	require.Zero(t, len(desired.requests)%int(desired.plan.Replicas))
	requestsPerReplica := len(desired.requests) / int(desired.plan.Replicas)
	next := *desired
	next.requests = slices.Clone(desired.requests[:int(replicas)*requestsPerReplica])
	plan := *desired.plan
	plan.Replicas = replicas
	next.plan = &plan
	return &next
}

func createLPXTestObjects(t *testing.T, ctx context.Context, kubeClient client.Client, objects ...client.Object) {
	t.Helper()
	for _, object := range objects {
		require.NoError(t, kubeClient.Create(ctx, object.DeepCopyObject().(client.Object)))
	}
}

func observedLPXTestPodCliqueSet(
	t *testing.T,
	ctx context.Context,
	reconciler *graphReconciler,
	deployment *nvidiacomv1alpha1.LPXGraphDeployment,
	desired *lpxMaterializing,
) *grovev1alpha1.PodCliqueSet {
	t.Helper()
	if desired.plan == nil {
		return nil
	}
	pcs := &grovev1alpha1.PodCliqueSet{}
	err := reconciler.Get(ctx, client.ObjectKey{Namespace: deployment.Namespace, Name: desired.plan.PodCliqueSetName}, pcs)
	if apierrors.IsNotFound(err) {
		return nil
	}
	require.NoError(t, err)
	return pcs
}

func reconcileSelectedLPXForTest(
	t *testing.T,
	ctx context.Context,
	reconciler *graphReconciler,
	deployment *nvidiacomv1alpha1.LPXGraphDeployment,
	desired *lpxMaterializing,
) (lpxClassification, error) {
	t.Helper()
	pcs := observedLPXTestPodCliqueSet(t, ctx, reconciler, deployment, desired)
	currents, classification, err := reconciler.reconcileLPXPublicationFence(ctx, deployment, desired, pcs)
	if classification != nil || err != nil {
		return classification, err
	}
	return reconciler.reconcileSelectedLPXFromCurrentRequests(ctx, deployment, desired, currents)
}

func publishSelectedLPXForTest(
	t *testing.T,
	ctx context.Context,
	reconciler *graphReconciler,
	deployment *nvidiacomv1alpha1.LPXGraphDeployment,
	desired *lpxMaterializing,
) lpxClassification {
	t.Helper()
	classification, err := reconcileSelectedLPXForTest(t, ctx, reconciler, deployment, desired)
	require.NoError(t, err)
	return classification
}

func getLPXRequest(t *testing.T, ctx context.Context, kubeClient client.Reader, namespace, name string) *lpxv1alpha1.LPUPipelineRequest {
	t.Helper()
	request := &lpxv1alpha1.LPUPipelineRequest{}
	require.NoError(t, kubeClient.Get(ctx, types.NamespacedName{Namespace: namespace, Name: name}, request))
	return request
}

func requireLPXRequestNotFound(
	t *testing.T,
	ctx context.Context,
	kubeClient client.Reader,
	namespace string,
	name string,
) {
	t.Helper()
	request := &lpxv1alpha1.LPUPipelineRequest{}
	err := kubeClient.Get(ctx, types.NamespacedName{Namespace: namespace, Name: name}, request)
	require.True(t, apierrors.IsNotFound(err), "expected LPX request %s/%s to be absent, got %v", namespace, name, err)
}
