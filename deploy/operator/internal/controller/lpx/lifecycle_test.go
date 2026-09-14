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
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/features"
	manifestcapnpv2 "github.com/ai-dynamo/dynamo/deploy/operator/internal/thirdparty/capnp/gbuild_manifest/v2"
	lpxv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/internal/thirdparty/lpxscheduler/v1alpha1"
	grovecommon "github.com/ai-dynamo/grove/operator/api/common"
	groveconstants "github.com/ai-dynamo/grove/operator/api/common/constants"
	grovev1alpha1 "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	groveschedulerv1alpha1 "github.com/ai-dynamo/grove/scheduler/api/core/v1alpha1"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	resourcev1 "k8s.io/api/resource/v1"
	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/api/resource"
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
		grovev1alpha1.AddToScheme, groveschedulerv1alpha1.AddToScheme, lpxv1alpha1.AddToScheme,
	} {
		require.NoError(t, add(scheme))
	}
	return scheme
}

type snapshotFailureRegistry struct {
	*lpx.ModelRegistry
	err error
}

type downloadOrderedLPXRegistry struct {
	*lpx.ModelRegistry
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
	r.calls = append(r.calls, "download")
	if len(r.calls) == 1 {
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
	desired, rejected, err := reconciler.prepareLPXMaterializing(ctx, dgd, source)
	require.NoError(t, err)
	return desired, rejected
}

func newPreparedLPXTestReconciler(
	t *testing.T,
	registry *lpx.ModelRegistry,
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
	ordinaryPodGang := findLPXTestPodGang(t, objects, corev1.DefaultSchedulerName)
	require.Len(t, ordinaryPodGang.Spec.PodGroups, 1)
	require.Equal(t, desired.plan.CyborgClique, ordinaryPodGang.Spec.PodGroups[0].Name)
	createLPXTestObjects(t, ctx, reconciler.Client, objects...)

	classification, err := reconciler.reconcileSelectedLPX(ctx, dgd, desired)
	require.NoError(t, err)
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

func TestNodeLocalSpecDecodePublishesOneRequestAndAgentCliquePerModelProjection(t *testing.T) {
	ctx := t.Context()
	dgd, source, registry := newLPXSpecDecodeTestDGD(t)

	reconciler, desired := newPreparedLPXTestReconciler(t, registry, ctx, dgd, source)
	require.Len(t, desired.requests, 3)
	require.Len(t, desired.plan.Agents, 3)
	objects := lpxMaterializedObjects(t, reconciler, dgd, source, desired)
	createLPXTestObjects(t, ctx, reconciler.Client, objects...)

	classification, err := reconciler.reconcileSelectedLPX(ctx, dgd, desired)
	require.NoError(t, err)
	require.IsType(t, &lpxOpen{}, classification)
	requests, err := reconciler.listOwnedLPXRequests(ctx, dgd)
	require.NoError(t, err)
	require.Len(t, requests, 3)

	requestByModel := make(map[string]lpxv1alpha1.LPUPipelineRequest, len(requests))
	for _, request := range requests {
		model := request.Annotations[lpxModelAnnotation]
		requestByModel[model] = request
		require.Equal(t, model, request.Spec.NodeLocal.Model)
		require.Equal(t, "selected-dgd-0-lpx-scheduler", request.Spec.PodGangRef.Name)
	}
	for _, projection := range desired.requests {
		request, found := requestByModel[projection.modelProjection.Model()]
		require.True(t, found)
		require.Equal(t, projection.modelProjection.Digest().String(), request.Annotations[lpx.WorkloadDigestAnnotation])
	}

	for index, expected := range desired.plan.Agents {
		projection := &desired.requests[index]
		clique := findLPXTestClique(t, objects, expected.CliqueName)
		require.Equal(t, int32(expected.Replicas), clique.Spec.Replicas)
		require.Equal(t, ptr.To(int32(expected.Replicas)), clique.Spec.MinAvailable)
		require.Equal(t, projection.modelProjection.Digest().String(), clique.Annotations[lpx.WorkloadDigestAnnotation])
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
	classification, err = reconciler.reconcileSelectedLPX(ctx, dgd, desired)
	require.NoError(t, err)
	require.Equal(t, nvidiacomv1beta1.DGDStateFailed, lpxResult(classification).State)
	require.Equal(t, "LPXUnsupported", lpxResult(classification).Reason)
	require.Contains(t, lpxResult(classification).Message, "TargetUnsupported")
}

func TestLPXPublicationWitnessReadsOnlyNamedChildrenAndNoPods(t *testing.T) {
	t.Log("Materialize two runtime replicas and preserve their distinct publication identities")
	ctx := t.Context()
	dgd, source, registry := newLPXTestDGD(t, lpx.PipelineLPX)
	source.Spec.Components[0].Replicas = ptr.To(int32(2))
	source.Spec.Components[0].ComponentRole(nvidiacomv1beta1.ComponentRoleLPXConductor).Replicas = ptr.To(int32(2))
	reconciler, desired := newPreparedLPXTestReconciler(t, registry, ctx, dgd, source)
	require.Equal(t, int32(2), desired.plan.Replicas)
	require.Len(t, desired.requests, 2)
	require.NotEqual(t, desired.requests[0].requestName, desired.requests[1].requestName)
	for replicaIndex := int32(0); replicaIndex < 2; replicaIndex++ {
		require.Equal(t, replicaIndex, desired.requests[replicaIndex].replicaIndex)
	}

	objects := lpxMaterializedObjects(t, reconciler, dgd, source, desired)
	scalingGroup := findLPXTestScalingGroup(t, objects, desired.plan.LPXScalingGroup)
	require.Equal(t, int32(2), scalingGroup.Spec.Replicas)

	t.Log("Leave Agent status unobserved and drift generated-child replica counts")
	replicaOneAgent := findLPXTestClique(t, objects, desired.plan.ForReplica(1).Agents[0].CliqueName)
	replicaOneAgent.Status.ObservedGeneration = nil
	replicaOneAgent.Status.Replicas = replicaOneAgent.Spec.Replicas
	replicaOneAgent.Status.UpdatedReplicas = replicaOneAgent.Spec.Replicas
	replicaOneAgent.Status.ScheduleGatedReplicas = replicaOneAgent.Spec.Replicas
	replicaOneAgent.Status.ScheduledReplicas = 0
	replicaOneAgent.Status.Conditions = nil
	scalingGroup.Spec.Replicas = 9
	agentClique := findLPXTestClique(t, objects, desired.plan.Agents[0].CliqueName)
	agentClique.Spec.Replicas = 9
	createLPXTestObjects(t, ctx, reconciler.Client, objects...)

	t.Log("Allow only deterministic engine-clique and parent-PCSG reads; reject Pods and broad clique lists")
	reader, ok := reconciler.apiReader.(client.WithWatch)
	require.True(t, ok)
	cachedClient, ok := reconciler.Client.(client.WithWatch)
	require.True(t, ok)
	forbiddenReads := make([]string, 0)
	podGangLists := 0
	lifecycleRequestListCalls, dependencyListCalls := 0, 0
	scalingGroupGets := 0
	conductorGets := 0
	cyborgGets := 0
	agentGets := 0
	forbidden := interceptor.Funcs{
		Get: func(
			ctx context.Context,
			delegated client.WithWatch,
			key client.ObjectKey,
			object client.Object,
			opts ...client.GetOption,
		) error {
			switch object.(type) {
			case *corev1.Pod:
				forbiddenReads = append(forbiddenReads, fmt.Sprintf("get %T %s", object, key))
			case *grovev1alpha1.PodCliqueScalingGroup:
				if key.Name != desired.plan.LPXScalingGroup {
					forbiddenReads = append(forbiddenReads, fmt.Sprintf("get %T %s", object, key))
				} else {
					scalingGroupGets++
				}
			case *grovev1alpha1.PodClique:
				known := false
				for _, request := range desired.requests {
					plan := desired.plan.ForReplica(request.replicaIndex)
					for _, agent := range plan.Agents {
						if key.Name == agent.CliqueName {
							agentGets++
							known = true
						}
					}
					if key.Name == plan.ConductorClique {
						conductorGets++
						known = true
						break
					}
					if key.Name == plan.CyborgClique {
						cyborgGets++
						known = true
						break
					}
				}
				if !known {
					forbiddenReads = append(forbiddenReads, fmt.Sprintf("get %T %s", object, key))
				}
			}
			return delegated.Get(ctx, key, object, opts...)
		},
		List: func(
			ctx context.Context,
			delegated client.WithWatch,
			list client.ObjectList,
			opts ...client.ListOption,
		) error {
			if _, request := list.(*lpxv1alpha1.LPUPipelineRequestList); request {
				lifecycleRequestListCalls++
			} else {
				dependencyListCalls++
			}
			switch list.(type) {
			case *groveschedulerv1alpha1.PodGangList:
				podGangLists++
			case *corev1.PodList, *grovev1alpha1.PodCliqueList,
				*grovev1alpha1.PodCliqueScalingGroupList:
				forbiddenReads = append(forbiddenReads, fmt.Sprintf("list %T", list))
			}
			return delegated.List(ctx, list, opts...)
		},
	}
	reconciler.apiReader = interceptor.NewClient(reader, forbidden)
	reconciler.Client = interceptor.NewClient(cachedClient, forbidden)

	t.Log("Publish directly from the PCS and PodGang witness")
	classification, err := reconciler.reconcileSelectedLPX(ctx, dgd, desired)
	require.NoError(t, err)
	require.IsType(t, &lpxOpen{}, classification)
	require.Empty(t, forbiddenReads)
	require.Equal(t, 1, podGangLists)
	require.Equal(t, 1, scalingGroupGets)
	require.Zero(t, conductorGets)
	require.Equal(t, 2, cyborgGets)
	require.Equal(t, 2, agentGets)
	requests, err := reconciler.listOwnedLPXRequests(ctx, dgd)
	require.NoError(t, err)
	require.Len(t, requests, 2)
	require.NotEqual(t, requests[0].Spec.PodGangRef.Name, requests[1].Spec.PodGangRef.Name)
	require.NotNil(t, requests[0].Spec.CyborgPodCliqueRef)
	require.NotNil(t, requests[1].Spec.CyborgPodCliqueRef)
	require.NotEqual(t, requests[0].Spec.CyborgPodCliqueRef.Name, requests[1].Spec.CyborgPodCliqueRef.Name)
	published := getLPXRequest(t, ctx, reconciler.Client, dgd.Namespace, desired.requests[0].requestName)

	t.Log("Repeat publication using the same witnesses and preserve the request")
	classification, err = reconciler.reconcileSelectedLPX(ctx, dgd, desired)
	require.NoError(t, err)
	require.IsType(t, &lpxOpen{}, classification)
	require.Equal(t, published.UID, getLPXRequest(t, ctx, reconciler.Client, dgd.Namespace, desired.requests[0].requestName).UID)
	require.Empty(t, forbiddenReads)
	require.Equal(t, 2, scalingGroupGets)
	require.Zero(t, conductorGets)
	require.Equal(t, 4, cyborgGets)
	require.Equal(t, 4, agentGets)

	t.Log("Leave unsupported generated-child replica mutations untouched")
	require.NoError(t, cachedClient.Get(ctx, client.ObjectKeyFromObject(scalingGroup), scalingGroup))
	require.Equal(t, int32(9), scalingGroup.Spec.Replicas)
	require.NoError(t, cachedClient.Get(ctx, client.ObjectKeyFromObject(agentClique), agentClique))
	require.Equal(t, int32(9), agentClique.Spec.Replicas)

	t.Log("Retire the complete runtime while scheduler finalizers hold both requests")
	for index := range requests {
		requests[index].Finalizers = []string{"scheduling.lpu.nvidia.com/test-cleanup"}
		require.NoError(t, reconciler.Update(ctx, &requests[index]))
	}
	_, err = reconciler.retireLPXRequest(ctx, dgd, desired.plan.PodCliqueSetName, &requests[0], "test group retirement")
	require.NoError(t, err)
	requests, err = reconciler.listOwnedLPXRequests(ctx, dgd)
	require.NoError(t, err)
	require.Len(t, requests, 2)
	for index := range requests {
		require.False(t, requests[index].DeletionTimestamp.IsZero())
	}
	pcs := &grovev1alpha1.PodCliqueSet{}
	require.NoError(t, reconciler.Get(ctx, types.NamespacedName{
		Namespace: dgd.Namespace,
		Name:      desired.plan.PodCliqueSetName,
	}, pcs))
	require.Zero(t, pcs.Spec.Replicas)

	t.Log("Restore nonzero Grove scale while scheduler finalizers still hold both requests in deletion")
	pcs.Spec.Replicas = 1
	require.NoError(t, reconciler.Update(ctx, pcs))
	lifecycleRequestListCalls, dependencyListCalls = 0, 0

	t.Log("Re-establish scale-to-zero using only the authoritative attempt-wide request observation")
	classification, err = reconciler.retireLPXRequest(ctx, dgd, desired.plan.PodCliqueSetName, &requests[0], "test group retirement")
	require.NoError(t, err)
	require.IsType(t, &lpxRetiring{}, classification)
	require.Positive(t, lifecycleRequestListCalls)
	require.Zero(t, dependencyListCalls)
	require.NoError(t, reconciler.Get(ctx, client.ObjectKeyFromObject(pcs), pcs))
	require.Zero(t, pcs.Spec.Replicas)
}

func TestSelectedLPXObservesLimitsOnlyCyborgGPUIntent(t *testing.T) {
	t.Log("Materialize a hybrid Cyborg whose classic GPU intent exists only in limits")
	dgd, source, registry := newLPXTestDGD(t, lpx.PipelineLPX)
	cyborgPodSpec := &source.Spec.Components[0].ComponentRole(nvidiacomv1beta1.ComponentRoleLPXConductor).PodTemplate.Spec
	cyborgPodSpec.ResourceClaims = nil
	cyborgPodSpec.Containers[0].Resources = corev1.ResourceRequirements{
		Limits: corev1.ResourceList{
			corev1.ResourceName(consts.KubeResourceGPUNvidia): resource.MustParse("1"),
		},
	}
	reconciler, desired := newPreparedLPXTestReconciler(t, registry, t.Context(), dgd, source)
	objects := lpxMaterializedObjects(t, reconciler, dgd, source, desired)
	cyborg := findLPXTestClique(t, objects, desired.plan.CyborgClique)
	cyborg.Spec.PodSpec = *cyborgPodSpec.DeepCopy()
	cyborg.Spec.PodSpec.SchedulerName = corev1.DefaultSchedulerName
	pcs := findLPXTestPodCliqueSet(t, objects)
	for _, template := range pcs.Spec.Template.Cliques {
		if template.Name == desired.plan.CyborgTemplate {
			template.Spec.PodSpec = *cyborg.Spec.PodSpec.DeepCopy()
		}
	}
	container := cyborg.Spec.PodSpec.Containers[0]
	require.Empty(t, container.Resources.Requests)
	require.NotEmpty(t, container.Resources.Limits)

	t.Log("Observe the raw Grove PodSpec without relying on Kubernetes Pod request defaulting")
	createLPXTestObjects(t, t.Context(), reconciler.Client, objects...)
	classification, err := reconciler.reconcileSelectedLPX(t.Context(), dgd, desired)
	require.NoError(t, err)
	require.IsType(t, &lpxOpen{}, classification)

	t.Log("Retire the published request if the live classic GPU intent disappears")
	storedCyborg := &grovev1alpha1.PodClique{}
	require.NoError(t, reconciler.Get(t.Context(), client.ObjectKeyFromObject(cyborg), storedCyborg))
	storedCyborg.Spec.PodSpec.ResourceClaims = nil
	for index := range storedCyborg.Spec.PodSpec.Containers {
		if storedCyborg.Spec.PodSpec.Containers[index].Name == consts.MainContainerName ||
			storedCyborg.Spec.PodSpec.Containers[index].Name == "cyborg" {
			storedCyborg.Spec.PodSpec.Containers[index].Resources = corev1.ResourceRequirements{}
		}
	}
	require.NoError(t, reconciler.Update(t.Context(), storedCyborg))
	classification, err = reconciler.reconcileSelectedLPX(t.Context(), dgd, desired)
	require.NoError(t, err)
	require.IsType(t, &lpxClosed{}, classification)
	requireLPXRequestNotFound(t, t.Context(), reconciler.Client, dgd.Namespace, desired.requests[0].requestName)
}

func TestLPXPublishedWorkloadFailurePreservesExistingEngines(t *testing.T) {
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
			classification, err := reconciler.reconcileSelectedLPX(ctx, dgd, desired)
			require.NoError(t, err)
			require.IsType(t, &lpxOpen{}, classification)
			requestBefore := getLPXRequest(t, ctx, reconciler.Client, dgd.Namespace, desired.requests[0].requestName)
			key := client.ObjectKeyFromObject(dgd)
			pcsKey := client.ObjectKey{Namespace: dgd.Namespace, Name: desired.plan.PodCliqueSetName}
			pcsBefore := &grovev1alpha1.PodCliqueSet{}
			require.NoError(t, reconciler.Get(ctx, pcsKey, pcsBefore))

			t.Log("Fail the actual snapshot dependency or remove required runtime SSH configuration")
			message := "node-local LPU runtime requires an MPI SSH secret name"
			if test.snapshotError != nil {
				reconciler.modelRegistry = &snapshotFailureRegistry{ModelRegistry: registry, err: test.snapshotError}
				message = test.snapshotError.Error()
			} else {
				reconciler.Config.MPI.SSHSecretName = ""
			}
			_, err = reconciler.Reconcile(ctx, ctrl.Request{NamespacedName: key})
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

func TestLPXAuthoritativeLifecycleListConsumesContinuePages(t *testing.T) {
	t.Log("Create lifecycle objects spanning multiple authoritative pages")
	ctx := t.Context()
	source := newLPXTestSource(lpx.PipelineSingle, "build-v2")
	dgd := newLPXTestDeployment(t, source)
	reconciler := newLPXTestReconciler(t, nil, dgd, source)
	reader, ok := reconciler.apiReader.(client.WithWatch)
	require.True(t, ok)

	expected := make([]lpxv1alpha1.LPUPipelineRequest, lpxLifecycleListPageSize+1)
	for index := range expected {
		expected[index] = lpxv1alpha1.LPUPipelineRequest{ObjectMeta: metav1.ObjectMeta{
			Name:      fmt.Sprintf("request-%02d", index),
			Namespace: dgd.Namespace,
		}}
	}
	const nextPageToken = "lpx-request-page-2"

	t.Log("Install a paginated reader that repopulates each supplied page in place")
	requestGets := 0
	requestListCalls := make([]client.ListOptions, 0, 2)
	reconciler.apiReader = interceptor.NewClient(reader, interceptor.Funcs{
		Get: func(
			ctx context.Context,
			delegated client.WithWatch,
			key client.ObjectKey,
			object client.Object,
			opts ...client.GetOption,
		) error {
			if _, request := object.(*lpxv1alpha1.LPUPipelineRequest); request {
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
			requestList, request := list.(*lpxv1alpha1.LPUPipelineRequestList)
			if !request {
				return delegated.List(ctx, list, opts...)
			}
			options := *(&client.ListOptions{}).ApplyOptions(opts)
			requestListCalls = append(requestListCalls, options)
			switch options.Continue {
			case "":
				requestList.Items = append(requestList.Items[:0], expected[:lpxLifecycleListPageSize]...)
				requestList.Continue = nextPageToken
			case nextPageToken:
				requestList.Items = append(requestList.Items[:0], expected[lpxLifecycleListPageSize:]...)
				requestList.Continue = ""
			default:
				return fmt.Errorf("unexpected LPX request continuation token %q", options.Continue)
			}
			return nil
		},
	})

	t.Log("Retain raw callback pointers until every continuation page is consumed")
	retained := make([]*lpxv1alpha1.LPUPipelineRequest, 0, len(expected))
	require.NoError(t, visitLifecycleObjectPages(
		ctx,
		reconciler.apiReader,
		&lpxv1alpha1.LPUPipelineRequestList{},
		func(object k8sruntime.Object) error {
			request := object.(*lpxv1alpha1.LPUPipelineRequest)
			retained = append(retained, request)
			return nil
		},
		client.InNamespace(dgd.Namespace),
	))

	t.Log("Verify retained first-page objects and exact pagination options after traversal")
	require.Len(t, retained, len(expected))
	for index := range expected {
		require.Equal(t, expected[index], *retained[index])
	}
	require.Zero(t, requestGets, "pagination must not fall back to per-object request Gets")
	require.Len(t, requestListCalls, 2, "the authoritative list must make exactly one call per returned page")
	require.Equal(t, "", requestListCalls[0].Continue)
	require.Equal(t, nextPageToken, requestListCalls[1].Continue)
	for _, call := range requestListCalls {
		require.Equal(t, int64(lpxLifecycleListPageSize), call.Limit)
		require.Equal(t, dgd.Namespace, call.Namespace)
		require.Nil(t, call.LabelSelector)
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
	synced, modified, err := reconciler.reconcileGrovePodCliqueSetForLPX(ctx, dgd, cached, desired)
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
	synced, modified, err = reconciler.reconcileGrovePodCliqueSetForLPX(ctx, dgd, synced, desired)
	require.NoError(t, err)
	require.False(t, modified)
	require.Zero(t, updates)
	require.Equal(t, want, synced)

	t.Log("Template metadata still updates through Spec without changing root metadata or native controls")
	desired.Spec.Template.Cliques[0].Annotations = map[string]string{"test.example/template": "next"}
	synced, modified, err = reconciler.reconcileGrovePodCliqueSetForLPX(ctx, dgd, synced, desired)
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
	synced, modified, err = reconciler.reconcileGrovePodCliqueSetForLPX(ctx, dgd, cached, desired)
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
			observed, modified, err := reconciler.reconcileGrovePodCliqueSetForLPX(ctx, dgd, nil, desired.DeepCopy())
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
			synced, modified, err := reconciler.reconcileGrovePodCliqueSetForLPX(ctx, dgd, observed, desired)
			require.NoError(t, err)
			require.Equal(t, reorder, modified)
			require.Equal(t, desired.Spec.Template.Cliques[0].Spec.PodSpec.InitContainers, synced.Spec.Template.Cliques[0].Spec.PodSpec.InitContainers)
			require.Equal(t, before, observed, "synchronization must not mutate its observation")
			if !reorder {
				require.Equal(t, before, synced, "API defaults must not cause a spec rewrite")
			}

			t.Log("The next reconciliation is a no-op")
			_, modified, err = reconciler.reconcileGrovePodCliqueSetForLPX(ctx, dgd, synced, desired)
			require.NoError(t, err)
			require.False(t, modified)
		})
	}
}

func TestLPXPodCliqueSetPublicationFence(t *testing.T) {
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
	synced, modified, err := reconciler.reconcileGrovePodCliqueSetForLPX(ctx, dgd, nil, desired.DeepCopy())
	require.ErrorIs(t, err, fenceErr)
	require.False(t, modified)
	require.Nil(t, synced)
	require.Zero(t, writes)
	require.Equal(t, 1, lists)
	require.True(t, apierrors.IsNotFound(base.Get(ctx, client.ObjectKeyFromObject(desired), &grovev1alpha1.PodCliqueSet{})))

	t.Log("Recover the observation and create the exact child-owned PCS")
	failFence = false
	writes, lists = 0, 0
	synced, modified, err = reconciler.reconcileGrovePodCliqueSetForLPX(ctx, dgd, nil, desired.DeepCopy())
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
	_, modified, err = reconciler.reconcileGrovePodCliqueSetForLPX(ctx, dgd, nil, desired.DeepCopy())
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
	synced, modified, err = reconciler.reconcileGrovePodCliqueSetForLPX(ctx, dgd, live, desired)
	require.NoError(t, err)
	require.True(t, modified)
	require.Equal(t, live.UID, synced.UID)
	require.Equal(t, 1, writes)
	require.Zero(t, lists)
	require.NoError(t, base.Get(ctx, client.ObjectKeyFromObject(live), stored))
	require.True(t, apiequality.Semantic.DeepEqual(desired.Spec, stored.Spec))
	require.Equal(t, live.Annotations[lpx.WorkloadDigestAnnotation], stored.Annotations[lpx.WorkloadDigestAnnotation])
}

func TestLPXPreflightPreservesRequestsDuringNativeGroveSync(t *testing.T) {
	ctx := t.Context()
	dgd, source, registry := newLPXTestDGD(t, lpx.PipelineSingle)
	reconciler, desired := newPreparedLPXTestReconciler(t, registry, ctx, dgd, source)

	createLPXTestObjects(t, ctx, reconciler.Client, lpxMaterializedObjects(t, reconciler, dgd, source, desired)...)
	rendered := renderLPXTestPodCliqueSet(t, ctx, reconciler, dgd, source, desired)
	_, err := reconciler.reconcileSelectedLPX(ctx, dgd, desired)
	require.NoError(t, err)
	_, err = reconciler.reconcileSelectedLPX(ctx, dgd, desired)
	require.NoError(t, err)
	published := getLPXRequest(t, ctx, reconciler.Client, dgd.Namespace, desired.requests[0].requestName)

	pcsKey := types.NamespacedName{Namespace: dgd.Namespace, Name: desired.plan.PodCliqueSetName}
	drifted := &grovev1alpha1.PodCliqueSet{}
	require.NoError(t, reconciler.Get(ctx, pcsKey, drifted))
	require.NotEmpty(t, drifted.Spec.Template.Cliques)
	drifted.Spec.Template.Cliques[0].Spec.PodSpec.PriorityClassName = "externally-mutated"
	require.NoError(t, reconciler.Update(ctx, drifted))
	require.NoError(t, reconciler.Get(ctx, pcsKey, drifted))

	_, _, err = reconciler.reconcileGrovePodCliqueSetForLPX(
		ctx,
		dgd,
		drifted,
		rendered,
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
	_, err = foreignReconciler.reconcileSelectedLPXFromCurrentRequests(ctx, dgd, desired, nil, false)
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
	classification, err := reconciler.reconcileSelectedLPXFromCurrentRequests(ctx, dgd, desired, nil, false)
	require.NoError(t, err)
	closed := requireLPXClosed(t, classification)
	require.Equal(t, "The retired Grove attempt is held at zero; successor publication waits for post-retirement Grove synchronization", closed.incomplete)
	require.Equal(t, 1, pcsGets)
	reconciler.apiReader = reader

	t.Log("Restore the explicit hold in place without rotating the existing request")
	_, _, err = reconciler.reconcileGrovePodCliqueSetForLPX(
		ctx,
		dgd,
		drifted,
		rendered,
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
	published := deadlineTestRequest(oldDGD, oldDesired.requests[0].requestName, time.Now(), lpxv1alpha1.RequestPhaseBound)

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
	synced, changed, err := reconciler.reconcileGrovePodCliqueSetForLPX(
		ctx,
		successorDGD,
		pcs,
		successorRendered,
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

	t.Log("Materialize the selected runtime alongside an unrelated ordinary frontend group")
	objects := lpxMaterializedObjects(t, reconciler, dgd, source, desired)
	ordinary := findLPXTestPodGang(t, objects, corev1.DefaultSchedulerName)
	ordinary.Spec.PodGroups = append(ordinary.Spec.PodGroups, groveschedulerv1alpha1.PodGroup{
		Name: "frontend", MinReplicas: 1,
		PodReferences: []groveschedulerv1alpha1.NamespacedName{{Namespace: dgd.Namespace, Name: "frontend-0"}},
	})
	createLPXTestObjects(t, ctx, reconciler.Client, objects...)

	t.Log("Keep an otherwise-ready runtime gated before the exact request is Bound")
	classification, err := reconciler.reconcileSelectedLPX(ctx, dgd, desired)
	require.NoError(t, err)
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
	wantDigest := desired.requests[0].attemptDigest
	desired = rederived
	classification, err = reconciler.reconcileSelectedLPX(ctx, dgd, desired)
	require.NoError(t, err)
	require.IsType(t, &lpxOpen{}, classification)
	require.Equal(t, wantDigest, desired.requests[0].attemptDigest)

	t.Log("Observe the exact current scheduler receipt after the controller restart")
	request := getLPXRequest(t, ctx, reconciler.Client, dgd.Namespace, desired.requests[0].requestName)
	request.Generation = 7
	request.Status = newLPXNodeLocalBoundStatus(7, 3, "sha256:bound-plan")
	require.NoError(t, reconciler.Update(ctx, request))

	classification, err = reconciler.reconcileSelectedLPX(ctx, dgd, desired)
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

	t.Run("unknown backend arm is not bound", func(t *testing.T) {
		t.Log("Reject an execution backend outside the scheduler API contract")
		unknownBackend := request.DeepCopy()
		unknownBackend.Spec.ExecutionBackend = "unknown"
		unknownBackend.Status.Committed.Plan.Placement = lpxv1alpha1.PlanPlacement{ExecutionBackend: "unknown"}
		require.IsType(t, &lpxSchedulerObserved{}, classifyPublishedLPX(unknownBackend))
	})

	t.Log("Keep runtime readiness gated when the Bound receipt belongs to an older generation")
	request = getLPXRequest(t, ctx, reconciler.Client, dgd.Namespace, desired.requests[0].requestName)
	request.Status.ObservedGeneration = ptr.To(int64(6))
	require.NoError(t, reconciler.Update(ctx, request))
	classification, err = reconciler.reconcileSelectedLPX(ctx, dgd, desired)
	require.NoError(t, err)
	require.IsType(t, &lpxOpen{}, classification)
	require.Equal(t, "LPXPublished", overlayLPXResult(runtimeReady, classification).Reason)

	t.Log("Stale failure diagnostics are not attributed to the current request generation")
	request = getLPXRequest(t, ctx, reconciler.Client, dgd.Namespace, desired.requests[0].requestName)
	request.Status.Phase = lpxv1alpha1.RequestPhaseUnsupported
	request.Status.Committed = nil
	request.Status.Diagnostics = []lpxv1alpha1.StatusDiagnostic{{
		Code: "StaleUnsupported", Subject: "request/old", Detail: "belongs to an older generation",
	}}
	require.NoError(t, reconciler.Update(ctx, request))
	classification, err = reconciler.reconcileSelectedLPX(ctx, dgd, desired)
	require.NoError(t, err)
	require.IsType(t, &lpxOpen{}, classification)
	require.NotContains(t, lpxResult(classification).Message, "StaleUnsupported")

	t.Log("Delete the stored child while scheduler cleanup holds its published request")
	request = getLPXRequest(t, ctx, reconciler.Client, dgd.Namespace, desired.requests[0].requestName)
	request.Finalizers = []string{lpxAttemptRecordingFinalizer, "scheduling.lpu.nvidia.com/lpx-cleanup"}
	require.NoError(t, reconciler.Update(ctx, request))
	key := client.ObjectKeyFromObject(dgd)
	require.NoError(t, reconciler.Get(ctx, key, dgd))
	require.NoError(t, reconciler.Delete(ctx, dgd))
	_, err = reconciler.Reconcile(ctx, ctrl.Request{NamespacedName: key})
	waiting := fmt.Sprintf("waiting for LPX request %q to finish fail-closed retirement", request.Name)
	require.EqualError(t, err, waiting)
	request = getLPXRequest(t, ctx, reconciler.Client, dgd.Namespace, request.Name)
	require.False(t, request.DeletionTimestamp.IsZero())
	require.Equal(t, []string{lpxAttemptRecordingFinalizer, "scheduling.lpu.nvidia.com/lpx-cleanup"}, request.Finalizers)
	pcs := &grovev1alpha1.PodCliqueSet{}
	require.NoError(t, reconciler.Get(ctx, client.ObjectKey{Namespace: dgd.Namespace, Name: desired.plan.PodCliqueSetName}, pcs))
	require.Zero(t, pcs.Spec.Replicas, "child finalization must stop the workload before scheduler cleanup finishes")

	t.Log("Keep the child finalizer until the scheduler releases the exact request")
	_, err = reconciler.Reconcile(ctx, ctrl.Request{NamespacedName: key})
	require.EqualError(t, err, waiting)
	require.NoError(t, reconciler.Get(ctx, key, dgd))
	require.Contains(t, dgd.Finalizers, lpxGraphDeploymentFinalizer)
	request.Finalizers = []string{lpxAttemptRecordingFinalizer}
	require.NoError(t, reconciler.Update(ctx, request))
	_, err = reconciler.Reconcile(ctx, ctrl.Request{NamespacedName: key})
	require.EqualError(t, err, waiting)
	requireLPXRequestNotFound(t, ctx, reconciler.Client, dgd.Namespace, request.Name)
	_, err = reconciler.Reconcile(ctx, ctrl.Request{NamespacedName: key})
	require.NoError(t, err)
	require.True(t, apierrors.IsNotFound(reconciler.Get(ctx, key, &nvidiacomv1alpha1.LPXGraphDeployment{})))
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
		name          string
		phase         lpxv1alpha1.RequestPhase
		releaseReason *lpxv1alpha1.ReleaseReason
		wantState     nvidiacomv1beta1.DGDState
		wantReason    string
		wantMessage   string
	}{
		{name: "pending", phase: lpxv1alpha1.RequestPhasePending, wantState: nvidiacomv1beta1.DGDStatePending, wantReason: "LPXSchedulerPending"},
		{name: "no fit", phase: lpxv1alpha1.RequestPhaseNoFit, wantState: nvidiacomv1beta1.DGDStatePending, wantReason: "LPXNoFit"},
		{name: "unsupported", phase: lpxv1alpha1.RequestPhaseUnsupported, wantState: nvidiacomv1beta1.DGDStateFailed, wantReason: "LPXUnsupported"},
		{name: "planned", phase: lpxv1alpha1.RequestPhasePlanned, wantState: nvidiacomv1beta1.DGDStatePending, wantReason: "LPXPlanned"},
		{name: "reserving", phase: lpxv1alpha1.RequestPhaseReserving, wantState: nvidiacomv1beta1.DGDStatePending, wantReason: "LPXReserving"},
		{name: "binding", phase: lpxv1alpha1.RequestPhaseBinding, wantState: nvidiacomv1beta1.DGDStatePending, wantReason: "LPXBinding"},
		{name: "degraded", phase: lpxv1alpha1.RequestPhaseDegraded, wantState: nvidiacomv1beta1.DGDStateFailed, wantReason: "LPXDegraded"},
		{
			name: "failure release in progress", phase: lpxv1alpha1.RequestPhaseReleasing,
			releaseReason: ptr.To(lpxv1alpha1.ReleaseReasonBindingFailed),
			wantState:     nvidiacomv1beta1.DGDStateFailed, wantReason: "LPXReleasing", wantMessage: "bindingFailed",
		},
		{
			name: "request change release in progress", phase: lpxv1alpha1.RequestPhaseReleasing,
			releaseReason: ptr.To(lpxv1alpha1.ReleaseReasonRequestChanged),
			wantState:     nvidiacomv1beta1.DGDStatePending, wantReason: "LPXReleasing", wantMessage: "requestChanged",
		},
		{
			name: "failure release complete", phase: lpxv1alpha1.RequestPhaseReleased,
			releaseReason: ptr.To(lpxv1alpha1.ReleaseReasonDependencyChanged),
			wantState:     nvidiacomv1beta1.DGDStateFailed, wantReason: "LPXReleased", wantMessage: "dependencyChanged",
		},
		{
			name: "deletion release complete", phase: lpxv1alpha1.RequestPhaseReleased,
			releaseReason: ptr.To(lpxv1alpha1.ReleaseReasonRequestDeleted),
			wantState:     nvidiacomv1beta1.DGDStatePending, wantReason: "LPXReleased", wantMessage: "requestDeleted",
		},
		{
			name: "release without journal", phase: lpxv1alpha1.RequestPhaseReleasing,
			wantState: nvidiacomv1beta1.DGDStateFailed, wantReason: "LPXSchedulerStatusInvalid",
			wantMessage: "without the required release journal",
		},
		{
			name: "release with unknown reason", phase: lpxv1alpha1.RequestPhaseReleased,
			releaseReason: ptr.To(lpxv1alpha1.ReleaseReason("futureReason")),
			wantState:     nvidiacomv1beta1.DGDStateFailed, wantReason: "LPXSchedulerStatusInvalid",
			wantMessage: "unknown release reason",
		},
		{
			name: "bound without exact proof", phase: lpxv1alpha1.RequestPhaseBound,
			wantState: nvidiacomv1beta1.DGDStateFailed, wantReason: "LPXSchedulerStatusInvalid", wantMessage: "without the exact current-generation plan proof",
		},
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
			if test.releaseReason != nil {
				request.Status.Committed = &lpxv1alpha1.Committed{
					Execution: lpxv1alpha1.CommittedExecution{
						Release: &lpxv1alpha1.ReleaseJournal{Reason: *test.releaseReason, RequestedAtGeneration: 7},
					},
				}
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
			if test.wantMessage != "" {
				require.Contains(t, result.Message, test.wantMessage)
			}
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

func TestLPXNativeRolloutPreservesOtherEngines(t *testing.T) {
	for _, pipeline := range []lpx.Pipeline{lpx.PipelineSingle, lpx.PipelineLPX} {
		t.Run(string(pipeline), func(t *testing.T) {
			t.Log("Publish two engines and stage a third under the same PCS")
			ctx := t.Context()
			_, source, registry := newLPXTestDGD(t, pipeline)
			source.Spec.Components[0].Replicas = ptr.To[int32](3)
			deployment := newLPXTestDeployment(t, source)
			reconciler, desired := newPreparedLPXTestReconciler(t, registry, ctx, deployment, source)
			objects := lpxMaterializedObjects(t, reconciler, deployment, source, desired)
			createLPXTestObjects(t, ctx, reconciler.Client, objects...)
			allRequests := desired.requests
			desired.requests = allRequests[:2]
			_, err := reconciler.reconcileSelectedLPX(ctx, deployment, desired)
			require.NoError(t, err)
			desired.requests = allRequests
			old := &lpxv1alpha1.LPUPipelineRequest{}
			kept := &lpxv1alpha1.LPUPipelineRequest{}
			require.NoError(t, reconciler.Get(ctx, client.ObjectKey{Namespace: deployment.Namespace, Name: allRequests[0].requestName}, old))
			require.NoError(t, reconciler.Get(ctx, client.ObjectKey{Namespace: deployment.Namespace, Name: allRequests[1].requestName}, kept))
			old.Finalizers = []string{"test.example/scheduler"}
			require.NoError(t, reconciler.Update(ctx, old))
			require.Nil(t, old.Status, "native identity changes must not require a scheduler receipt")

			t.Log("Grove replaces engine zero; engine one remains old, while engine two is ready to publish")
			pcs := findLPXTestPodCliqueSet(t, objects)
			require.NoError(t, reconciler.Get(ctx, client.ObjectKeyFromObject(pcs), pcs))
			pcs.Status.CurrentGenerationHash = ptr.To("next-hash")
			pcs.Status.ObservedGeneration = ptr.To[int64](0)
			require.NoError(t, reconciler.Update(ctx, pcs))
			for _, object := range objects {
				clique, ok := object.(*grovev1alpha1.PodClique)
				if !ok || clique.Labels[grovecommon.LabelPodCliqueScalingGroupReplicaIndex] == "1" {
					continue
				}
				require.NoError(t, reconciler.Get(ctx, client.ObjectKeyFromObject(clique), clique))
				require.NoError(t, reconciler.Delete(ctx, clique))
				clique.UID += "-replacement"
				clique.ResourceVersion = ""
				clique.Status.CurrentPodCliqueSetGenerationHash = ptr.To("next-hash")
				require.NoError(t, reconciler.Create(ctx, clique))
			}
			_, err = reconciler.reconcileSelectedLPX(ctx, deployment, desired)
			require.NoError(t, err)
			live := &lpxv1alpha1.LPUPipelineRequest{}
			require.NoError(t, reconciler.Get(ctx, client.ObjectKeyFromObject(old), live))
			require.NotNil(t, live.DeletionTimestamp)
			require.NoError(t, reconciler.Get(ctx, client.ObjectKeyFromObject(kept), kept))
			require.Nil(t, kept.DeletionTimestamp)
			require.NoError(t, reconciler.Get(ctx, client.ObjectKey{Namespace: deployment.Namespace, Name: allRequests[2].requestName}, &lpxv1alpha1.LPUPipelineRequest{}))

			t.Log("Publish the replacement under its stable name after its own finalizer finishes")
			live.Finalizers = nil
			require.NoError(t, reconciler.Update(ctx, live))
			_, err = reconciler.reconcileSelectedLPX(ctx, deployment, desired)
			require.NoError(t, err)
			require.NoError(t, reconciler.Get(ctx, client.ObjectKeyFromObject(old), live))
			require.NotEqual(t, old.UID, live.UID)
			require.NoError(t, reconciler.Get(ctx, client.ObjectKeyFromObject(pcs), pcs))
			require.Equal(t, types.UID("pcs-uid"), pcs.UID)

			t.Log("An owned deleting Agent retires its request even while its PodGang is absent")
			live.Finalizers = []string{"test.example/scheduler"}
			require.NoError(t, reconciler.Update(ctx, live))
			agent := findLPXTestClique(t, objects, desired.plan.Agents[0].CliqueName)
			require.NoError(t, reconciler.Get(ctx, client.ObjectKeyFromObject(agent), agent))
			agent.Finalizers = []string{"test.example/grove"}
			require.NoError(t, reconciler.Update(ctx, agent))
			require.NoError(t, reconciler.Delete(ctx, agent))
			require.NoError(t, reconciler.Delete(ctx, &groveschedulerv1alpha1.PodGang{ObjectMeta: metav1.ObjectMeta{
				Name: live.Spec.PodGangRef.Name, Namespace: deployment.Namespace,
			}}))
			_, err = reconciler.reconcileSelectedLPX(ctx, deployment, desired)
			require.NoError(t, err)
			require.False(t, getLPXRequest(t, ctx, reconciler.Client, deployment.Namespace, live.Name).DeletionTimestamp.IsZero())
			require.Equal(t, kept.UID, getLPXRequest(t, ctx, reconciler.Client, deployment.Namespace, kept.Name).UID)
		})
	}
}

func TestLPXAttemptIdentityAndRetirementFence(t *testing.T) {
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

	t.Log("Fence exact PCS and PodGang UID annotations")
	projection := &desired.requests[0]
	identity := &lpxGroveIdentity{pcsUID: "pcs-uid", podGangName: "podgang", podGangUID: "podgang-uid"}
	current := &lpxv1alpha1.LPUPipelineRequest{
		ObjectMeta: metav1.ObjectMeta{UID: "request-uid", Annotations: lpxRequestAnnotations(dgd, projection, identity)},
		Spec:       projection.modelProjection.RequestSpec(dgd.Namespace, identity.podGangName, identity.cyborgClique),
	}
	require.NoError(t, validateCurrentLPXRequest(dgd, projection, identity, current))
	changed := *identity
	changed.pcsUID = "replacement-pcs-uid"
	require.ErrorContains(t, validateCurrentLPXRequest(dgd, projection, &changed, current), "immutable annotation")
	changed = *identity
	changed.podGangUID = "replacement-podgang-uid"
	require.ErrorContains(t, validateCurrentLPXRequest(dgd, projection, &changed, current), "immutable annotation")

	createLPXTestObjects(t, ctx, reconciler.Client, lpxMaterializedObjects(t, reconciler, dgd, source, desired)...)
	_, err := reconciler.reconcileSelectedLPX(ctx, dgd, desired)
	require.NoError(t, err)
	_, err = reconciler.reconcileSelectedLPX(ctx, dgd, desired)
	require.NoError(t, err)

	dgd.Generation++
	classification, _, err := reconciler.reconcileLPXKnownIntentFence(ctx, dgd, source, nil)
	require.NoError(t, err)
	require.Nil(t, classification)
	require.NoError(t, reconciler.Get(ctx, client.ObjectKey{Namespace: dgd.Namespace, Name: desired.requests[0].requestName}, &lpxv1alpha1.LPUPipelineRequest{}))

	classification, _, err = reconciler.reconcileLPXKnownIntentFence(ctx, dgd, source, nil)
	require.NoError(t, err)
	require.Nil(t, classification)
}

func TestLPXDisabledPreservesPublishedWorkloadUntilDeletion(t *testing.T) {
	for _, scenario := range []struct{ name, message string }{
		{"LPX", "LPX integration is disabled"},
		{"Grove", "Grove is disabled"},
	} {
		t.Run(scenario.name, func(t *testing.T) {
			t.Log("Publish a complete workload, discovery service and scheduling attempt")
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
			require.NotNil(t, child.Status.Placement)
			require.NotNil(t, child.Status.Placement.LPXAttempt)
			require.NotNil(t, child.Status.Placement.LPXAttempt.DeadlineAt)
			placement := child.Status.Placement.DeepCopy()
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
			child.Status.Placement.LPXAttempt.DeadlineAt = ptr.To(metav1.NewTime(time.Now().Add(-time.Minute)))
			require.NoError(t, r.Status().Update(ctx, child))
			disabledPlacement := child.Status.Placement.DeepCopy()
			for range 2 {
				result, err := r.Reconcile(ctx, ctrl.Request{NamespacedName: key})
				require.NoError(t, err)
				require.Zero(t, result)
			}
			require.NoError(t, r.Get(ctx, key, child))
			require.Equal(t, disabledPlacement, child.Status.Placement)
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
			child.Status.Placement = placement
			require.NoError(t, r.Status().Update(ctx, child))
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

			t.Log("Explicit deletion still retires requests and releases the child finalizer while disabled")
			if scenario.name == "LPX" {
				r.runtimeConfig.Gate.LPX = false
			} else {
				r.runtimeConfig.Gate.Grove = false
			}
			r.modelRegistry = nil
			require.NoError(t, r.Delete(ctx, child))
			_, err = r.Reconcile(ctx, ctrl.Request{NamespacedName: key})
			require.ErrorContains(t, err, "waiting for LPX request")
			request := getLPXRequest(t, ctx, r.Client, child.Namespace, selected.requests[0].requestName)
			require.False(t, request.DeletionTimestamp.IsZero())
			pcs := &grovev1alpha1.PodCliqueSet{}
			require.NoError(t, r.Get(ctx, client.ObjectKey{Namespace: child.Namespace, Name: dynamo.PCSNameForLPX(child)}, pcs))
			require.Zero(t, pcs.Spec.Replicas)
			_, err = r.Reconcile(ctx, ctrl.Request{NamespacedName: key})
			require.ErrorContains(t, err, "waiting for LPX request")
			requireLPXRequestNotFound(t, ctx, r.Client, child.Namespace, selected.requests[0].requestName)
			_, err = r.Reconcile(ctx, ctrl.Request{NamespacedName: key})
			require.NoError(t, err)
			require.True(t, apierrors.IsNotFound(r.Get(ctx, key, &nvidiacomv1alpha1.LPXGraphDeployment{})))
		})
	}
}

func TestLPXControllerPreservesPublishedWorkWhenProviderEditIsInvalid(t *testing.T) {
	t.Log("Reject an incompatible provider edit without deleting the existing engine")
	dgd, source, registry := newLPXTestDGD(t, lpx.PipelineSingle)
	reconciler, desired := newPreparedLPXTestReconciler(t, registry, t.Context(), dgd, source)
	createLPXTestObjects(t, t.Context(), reconciler.Client, lpxMaterializedObjects(t, reconciler, dgd, source, desired)...)
	_, err := reconciler.reconcileSelectedLPX(t.Context(), dgd, desired)
	require.NoError(t, err)

	require.NoError(t, reconciler.Get(t.Context(), client.ObjectKeyFromObject(source), source))
	source.Annotations[consts.KubeAnnotationWorkloadProvider] = consts.WorkloadProviderComponent
	require.NoError(t, reconciler.Update(t.Context(), source))
	_, err = reconciler.Reconcile(t.Context(), ctrl.Request{NamespacedName: client.ObjectKeyFromObject(dgd)})
	require.ErrorContains(t, err, "LPX requires the Grove workload provider")
	request := getLPXRequest(t, t.Context(), reconciler.Client, dgd.Namespace, desired.requests[0].requestName)
	require.Nil(t, request.DeletionTimestamp)
}

func TestLPUOnlyWaitsForCurrentConductorWitness(t *testing.T) {
	tests := []struct{ name, defect, want string }{
		{name: "stale owner", defect: "owner", want: "ordinary-scheduler identity"},
		{name: "wrong PodGang", defect: "podgang", want: "ordinary-scheduler identity"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			dgd, source, registry := newLPXTestDGD(t, lpx.PipelineSingle)
			reconciler, desired := newPreparedLPXTestReconciler(t, registry, t.Context(), dgd, source)
			objects := lpxMaterializedObjects(t, reconciler, dgd, source, desired)
			conductor := findLPXTestClique(t, objects, desired.plan.ConductorClique)
			switch test.defect {
			case "owner":
				conductor.OwnerReferences[0].UID = "previous-scaling-group-uid"
			case "podgang":
				conductor.Labels[grovecommon.LabelPodGang] = "other-podgang"
			}
			createLPXTestObjects(t, t.Context(), reconciler.Client, objects...)
			classification, err := reconciler.reconcileSelectedLPX(t.Context(), dgd, desired)
			require.NoError(t, err)
			closed := requireLPXClosed(t, classification)
			require.Contains(t, closed.incomplete, test.want)
			requireLPXRequestNotFound(t, t.Context(), reconciler.Client, dgd.Namespace, desired.requests[0].requestName)
		})
	}
}

func TestNodeLocalHybridPublishesAndFencesCurrentCyborgMaterialization(t *testing.T) {
	tests := []struct {
		name               string
		generation         int64
		observedGeneration *int64
	}{
		{
			name:               "absent generation then claim drift",
			generation:         1,
			observedGeneration: nil,
		},
		{
			name:               "lagging generation then UID replacement",
			generation:         2,
			observedGeneration: ptr.To(int64(1)),
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Log("Arrange current Cyborg materialization evidence without an observed generation")
			ctx := t.Context()
			dgd, source, registry := newLPXTestDGD(t, lpx.PipelineLPX)
			reconciler, desired := newPreparedLPXTestReconciler(t, registry, ctx, dgd, source)
			objects := lpxMaterializedObjects(t, reconciler, dgd, source, desired)
			cyborg := findLPXTestClique(t, objects, desired.plan.CyborgClique)
			cyborg.Generation = test.generation
			cyborg.Status.ObservedGeneration = test.observedGeneration
			createLPXTestObjects(t, ctx, reconciler.Client, objects...)

			t.Log("Publish the NodeLocal request with the exact Cyborg name and UID")
			classification, err := reconciler.reconcileSelectedLPX(ctx, dgd, desired)
			require.NoError(t, err)
			require.IsType(t, &lpxOpen{}, classification)
			request := getLPXRequest(
				t,
				ctx,
				reconciler.Client,
				dgd.Namespace,
				desired.requests[0].requestName,
			)
			require.NotNil(t, request.Spec.CyborgPodCliqueRef)
			require.Equal(t, cyborg.Name, request.Spec.CyborgPodCliqueRef.Name)
			require.Equal(t, string(cyborg.UID), request.Spec.CyborgPodCliqueRef.UID)
			publishedUID := request.UID

			storedCyborg := &grovev1alpha1.PodClique{}
			require.NoError(t, reconciler.Get(ctx, client.ObjectKeyFromObject(cyborg), storedCyborg))
			if test.observedGeneration == nil {
				t.Log("Drift a live Claim reference without an observed generation")
				require.NotEmpty(t, storedCyborg.Spec.PodSpec.ResourceClaims)
				storedCyborg.Generation++
				storedCyborg.Spec.PodSpec.ResourceClaims[0].ResourceClaimTemplateName = ptr.To("stale-gpu")
				require.NoError(t, reconciler.Update(ctx, storedCyborg))

				t.Log("An unobserved PCS edit permits temporary old-child references without retiring the engine")
				pcs := findLPXTestPodCliqueSet(t, objects)
				require.NoError(t, reconciler.Get(ctx, client.ObjectKeyFromObject(pcs), pcs))
				pcs.Generation++
				require.NoError(t, reconciler.Update(ctx, pcs))
				_, err = reconciler.reconcileSelectedLPX(ctx, dgd, desired)
				require.NoError(t, err)
				require.Equal(t, publishedUID, getLPXRequest(t, ctx, reconciler.Client, dgd.Namespace, request.Name).UID)
				pcs.Status.ObservedGeneration = ptr.To(pcs.Generation)
				require.NoError(t, reconciler.Update(ctx, pcs))
			} else {
				t.Log("Advance only the global generation without rotating the published request")
				storedCyborg.Status.ObservedGeneration = ptr.To(test.generation)
				require.NoError(t, reconciler.Update(ctx, storedCyborg))
				classification, err = reconciler.reconcileSelectedLPX(ctx, dgd, desired)
				require.NoError(t, err)
				require.IsType(t, &lpxOpen{}, classification)
				request = getLPXRequest(t, ctx, reconciler.Client, dgd.Namespace, desired.requests[0].requestName)
				require.Equal(t, publishedUID, request.UID)

				t.Log("Replace the Cyborg clique at the same name with a different UID")
				require.NoError(t, reconciler.Delete(ctx, storedCyborg))
				replacement := storedCyborg.DeepCopy()
				replacement.ResourceVersion = ""
				replacement.UID = "replacement-cyborg-uid"
				replacement.DeletionTimestamp = nil
				replacement.CreationTimestamp = metav1.Time{}
				require.NoError(t, reconciler.Create(ctx, replacement))
			}

			t.Log("Retire the request when its exact published materialization no longer holds")
			classification, err = reconciler.reconcileSelectedLPX(ctx, dgd, desired)
			require.NoError(t, err)
			require.IsType(t, &lpxClosed{}, classification)
			requireLPXRequestNotFound(t, ctx, reconciler.Client, dgd.Namespace, desired.requests[0].requestName)
		})
	}
}

func TestNodeLocalHybridWaitsForCurrentCyborgWitness(t *testing.T) {
	tests := []struct {
		name        string
		defect      string
		wantMessage string
	}{
		{
			name:        "missing current scaling group",
			defect:      "missing-scaling-group",
			wantMessage: "current LPU PodCliqueScalingGroup",
		},
		{
			name:        "stale scaling-group owner",
			defect:      "stale-scaling-group-owner",
			wantMessage: "current PodCliqueSet owner identity",
		},
		{
			name:        "stale Cyborg owner",
			defect:      "stale-cyborg-owner",
			wantMessage: "current LPU scaling-group owner identity",
		},
		{
			name:        "missing PCS generation hash",
			defect:      "missing-pcs-hash",
			wantMessage: "current PodCliqueSet generation",
		},
		{
			name:        "mismatched PCS generation hash",
			defect:      "mismatched-pcs-hash",
			wantMessage: "current PodCliqueSet generation",
		},
		{
			name:        "missing Pod template hash",
			defect:      "missing-template-hash",
			wantMessage: "current PodCliqueSet generation",
		},
		{
			name:        "empty Pod template hash",
			defect:      "empty-template-hash",
			wantMessage: "current PodCliqueSet generation",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Log("Arrange one incomplete hybrid Cyborg witness")
			ctx := t.Context()
			dgd, source, registry := newLPXTestDGD(t, lpx.PipelineLPX)
			reconciler, desired := newPreparedLPXTestReconciler(t, registry, ctx, dgd, source)
			objects := lpxMaterializedObjects(t, reconciler, dgd, source, desired)
			cyborg := findLPXTestClique(t, objects, desired.plan.CyborgClique)
			switch test.defect {
			case "missing-scaling-group":
				scalingGroup := findLPXTestScalingGroup(t, objects, desired.plan.LPXScalingGroup)
				objects = slices.DeleteFunc(objects, func(object client.Object) bool { return object == scalingGroup })
			case "stale-scaling-group-owner":
				scalingGroup := findLPXTestScalingGroup(t, objects, desired.plan.LPXScalingGroup)
				require.NotEmpty(t, scalingGroup.OwnerReferences)
				scalingGroup.OwnerReferences[0].UID = "previous-pcs-uid"
			case "stale-cyborg-owner":
				require.NotEmpty(t, cyborg.OwnerReferences)
				cyborg.OwnerReferences[0].UID = "previous-scaling-group-uid"
			case "missing-pcs-hash":
				cyborg.Status.CurrentPodCliqueSetGenerationHash = nil
			case "mismatched-pcs-hash":
				cyborg.Status.CurrentPodCliqueSetGenerationHash = ptr.To("previous-pcs-generation")
			case "missing-template-hash":
				cyborg.Status.CurrentPodTemplateHash = nil
			case "empty-template-hash":
				cyborg.Status.CurrentPodTemplateHash = ptr.To("")
			default:
				require.FailNow(t, "unknown materialization defect", "defect: %s", test.defect)
			}
			createLPXTestObjects(t, ctx, reconciler.Client, objects...)

			t.Log("Keep publication pending until a dependency watch supplies current evidence")
			classification, err := reconciler.reconcileSelectedLPX(ctx, dgd, desired)
			require.NoError(t, err)
			closed := requireLPXClosed(t, classification)
			require.Contains(t, closed.incomplete, test.wantMessage)
			require.Equal(t, nvidiacomv1beta1.DGDStatePending, lpxResult(closed).State)
			requireLPXRequestNotFound(
				t,
				ctx,
				reconciler.Client,
				dgd.Namespace,
				desired.requests[0].requestName,
			)
		})
	}
}

func TestLPXSchedulerScopedGangAndStartupWitnessesAreDisjoint(t *testing.T) {
	const (
		crossReplicaAgent = "other replica Agent enters ordinary gang"
		missingLPXGang    = "Waiting for one replica-0 LPX PodGang; found 0 candidates"
		missingOrdinary   = "Waiting for one replica-0 ordinary-scheduler PodGang; found 0 candidates"
	)
	tests := []struct {
		name, wantMessage string
	}{
		{"LPX gang unavailable", missingLPXGang},
		{"LPX gang is stale", "Waiting for Grove to publish current-generation PodGangs"},
		{"LPX gang has incomplete references", missingLPXGang},
		{"LPX gang has an extra group", missingLPXGang},
		{"LPX gang is ambiguous", "Waiting for one replica-0 LPX PodGang; found 2 candidates"},
		{"Agent enters ordinary gang", missingOrdinary},
		{crossReplicaAgent, missingOrdinary},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Log("Materialize the complete hybrid runtime and its scheduler-scoped gangs")
			ctx := context.Background()
			dgd, source, registry := newLPXTestDGD(t, lpx.PipelineLPX)
			if test.name == crossReplicaAgent {
				source.Spec.Components[0].Replicas = ptr.To(int32(2))
				source.Spec.Components[0].ComponentRole(nvidiacomv1beta1.ComponentRoleLPXConductor).Replicas = ptr.To(int32(1))
			}
			reconciler, desired := newPreparedLPXTestReconciler(t, registry, ctx, dgd, source)
			objects := lpxMaterializedObjects(t, reconciler, dgd, source, desired)

			t.Log("Introduce only the selected gang or startup-witness defect")
			gang := findLPXTestPodGang(t, objects, lpx.SchedulerName)
			switch test.name {
			case "LPX gang unavailable":
				gang.Labels[grovecommon.LabelPartOfKey] = lpxTestOtherName
			case "LPX gang is stale":
				gang.Annotations[lpxDeploymentUIDAnnotation] = "different-lgd"
			case "LPX gang has incomplete references":
				gang.Spec.PodGroups[0].PodReferences = nil
			case "LPX gang has an extra group":
				gang.Spec.PodGroups = append(gang.Spec.PodGroups, groveschedulerv1alpha1.PodGroup{Name: "unexpected"})
			case "LPX gang is ambiguous":
				duplicate := gang.DeepCopy()
				duplicate.Name, duplicate.UID, duplicate.ResourceVersion = duplicate.Name+"-duplicate", "duplicate-uid", ""
				objects = append(objects, duplicate)
			case "Agent enters ordinary gang", crossReplicaAgent:
				ordinary := findLPXTestPodGang(t, objects, corev1.DefaultSchedulerName)
				agent := desired.plan.Agents[0]
				if test.name == crossReplicaAgent {
					agent = desired.plan.ForReplica(1).Agents[0]
				}
				ordinary.Spec.PodGroups = append(ordinary.Spec.PodGroups, groveschedulerv1alpha1.PodGroup{
					Name: agent.CliqueName, MinReplicas: 1,
					PodReferences: []groveschedulerv1alpha1.NamespacedName{{Namespace: ordinary.Namespace, Name: agent.CliqueName + "-0"}},
				})
			}
			createLPXTestObjects(t, ctx, reconciler.Client, objects...)

			t.Log("Keep the invalid engine unpublished while allowing an independent sibling")
			classification, err := reconciler.reconcileSelectedLPX(ctx, dgd, desired)
			require.NoError(t, err)
			requests, err := reconciler.listOwnedLPXRequests(ctx, dgd)
			require.NoError(t, err)
			requireLPXRequestNotFound(t, ctx, reconciler.Client, dgd.Namespace, desired.requests[0].requestName)
			if test.name == crossReplicaAgent {
				require.IsType(t, &lpxOpen{}, classification)
				require.Len(t, requests, 1)
				require.Equal(t, desired.requests[1].requestName, requests[0].Name)
			} else {
				require.IsType(t, &lpxClosed{}, classification)
				require.Equal(t, test.wantMessage, classification.(*lpxClosed).incomplete)
				require.Empty(t, requests)
			}
		})
	}
}

func TestLPXLiveRequestSizeGrowthRetiresPublication(t *testing.T) {
	ctx := context.Background()
	dgd, source, registry := newLPXTestDGD(t, lpx.PipelineSingle)
	reconciler, desired := newPreparedLPXTestReconciler(t, registry, ctx, dgd, source)
	require.NotNil(t, desired)
	createLPXTestObjects(t, ctx, reconciler.Client, lpxMaterializedObjects(t, reconciler, dgd, source, desired)...)

	t.Log("Publish the request, then model API-server metadata growth")
	classification, err := reconciler.reconcileSelectedLPX(ctx, dgd, desired)
	require.NoError(t, err)
	require.IsType(t, &lpxOpen{}, classification)
	storedClient := reconciler.Client
	storedWithWatch, ok := storedClient.(client.WithWatch)
	require.True(t, ok)
	reconciler.apiReader = interceptor.NewClient(storedWithWatch, interceptor.Funcs{
		List: func(
			ctx context.Context,
			delegated client.WithWatch,
			list client.ObjectList,
			opts ...client.ListOption,
		) error {
			if err := delegated.List(ctx, list, opts...); err != nil {
				return err
			}
			requests, ok := list.(*lpxv1alpha1.LPUPipelineRequestList)
			if !ok {
				return nil
			}
			for index := range requests.Items {
				request := &requests.Items[index]
				if request.Name != desired.requests[0].requestName {
					continue
				}
				request.ManagedFields = []metav1.ManagedFieldsEntry{{
					Manager:    "metadata-growth-test",
					FieldsType: "FieldsV1",
					FieldsV1: &metav1.FieldsV1{
						Raw: []byte(`{"padding":"` + strings.Repeat("x", 850*1024) + `"}`),
					},
				}}
			}
			return nil
		},
	})

	t.Log("Retire the now-oversized publication")
	classification, err = reconciler.reconcileSelectedLPX(ctx, dgd, desired)
	require.NoError(t, err)
	require.IsType(t, &lpxClosed{}, classification)
	requireLPXRequestNotFound(t, ctx, storedClient, dgd.Namespace, desired.requests[0].requestName)
}

func TestLPXAttemptFenceRejectsForeignExactNameCollisionFromAuthoritativeList(t *testing.T) {
	t.Log("Give foreign requests the desired names in the opposite lexical order")
	ctx := t.Context()
	source := newLPXTestSource(lpx.PipelineSingle, "build-v2")
	dgd := newLPXTestDeployment(t, source)
	foreign := &lpxv1alpha1.LPUPipelineRequest{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "z-foreign-request",
			Namespace: dgd.Namespace,
			UID:       "foreign-collision-uid",
			OwnerReferences: []metav1.OwnerReference{{
				APIVersion: nvidiacomv1beta1.GroupVersion.String(),
				Kind:       nvidiacomv1beta1.DynamoGraphDeploymentGVK.Kind,
				Name:       lpxTestOtherName,
				UID:        "other-uid",
				Controller: ptr.To(true),
			}},
		},
	}
	second := foreign.DeepCopy()
	second.Name, second.UID = "a-foreign-request", "second-foreign-collision-uid"
	desired := &lpxMaterializing{requests: []lpxModelMaterializing{
		{requestName: foreign.Name}, {requestName: second.Name},
	}}
	reconciler := newLPXTestReconciler(t, nil, dgd, source)

	t.Log("Complete an empty preflight fence before foreign requests occupy the desired names")
	currents, retiring, err := reconciler.reconcileLPXAttemptFence(ctx, dgd, desired)
	require.NoError(t, err)
	require.Empty(t, currents)
	require.Nil(t, retiring)
	for _, request := range []*lpxv1alpha1.LPUPipelineRequest{foreign, second} {
		require.NoError(t, reconciler.Create(ctx, request))
	}

	t.Log("Exclude foreign requests from owned publications and the known-intent retirement fence")
	requests, err := reconciler.listOwnedLPXRequests(ctx, dgd)
	require.NoError(t, err)
	require.Empty(t, requests)
	classification, _, err := reconciler.reconcileLPXKnownIntentFence(ctx, dgd, source, nil)
	require.NoError(t, err)
	require.Nil(t, classification)

	t.Log("Observe exact-name collisions with one bounded authoritative list and no per-request Gets")
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

	currents, retiring, err = reconciler.reconcileLPXAttemptFence(ctx, dgd, desired)
	require.EqualError(t, err, fmt.Sprintf("LPX request name %q is occupied by a foreign owner", foreign.Name))
	require.Nil(t, currents)
	require.Nil(t, retiring)
	require.Zero(t, requestGets, "exact-name collision detection must not issue one request Get per desired projection")
	require.Len(t, requestListCalls, 1)
	require.Equal(t, int64(lpxLifecycleListPageSize), requestListCalls[0].Limit)
	require.Equal(t, dgd.Namespace, requestListCalls[0].Namespace)
	require.True(t, requestListCalls[0].LabelSelector == nil || requestListCalls[0].LabelSelector.Empty())

	t.Log("Refresh publication state and refuse the foreign requests created after the empty preflight")
	classification, err = reconciler.reconcileSelectedLPX(ctx, dgd, desired)
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

func newLPXNodeLocalBoundStatus(generation, revision int64, planDigest lpxv1alpha1.PlanDigest) *lpxv1alpha1.LPUPipelineRequestStatus {
	// Construct a fresh exact-generation NodeLocal scheduler receipt.
	return &lpxv1alpha1.LPUPipelineRequestStatus{
		Phase:              lpxv1alpha1.RequestPhaseBound,
		ObservedGeneration: ptr.To(generation),
		LastPlanRevision:   revision,
		Committed: &lpxv1alpha1.Committed{
			Execution: lpxv1alpha1.CommittedExecution{
				AcceptedGeneration: ptr.To(generation),
				NodeLocal:          &lpxv1alpha1.NodeLocalExecution{},
			},
			Plan: lpxv1alpha1.CommittedPlan{
				PlannedFromGeneration: generation,
				Revision:              revision,
				PlanDigest:            planDigest,
				Placement: lpxv1alpha1.PlanPlacement{
					ExecutionBackend: lpxv1alpha1.ExecutionBackendNodeLocal,
					NodeLocal:        &lpxv1alpha1.NodeLocalPlacement{},
				},
			},
		},
	}
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

func findLPXTestPodGang(t *testing.T, objects []client.Object, schedulerName string) *groveschedulerv1alpha1.PodGang {
	t.Helper()
	for _, object := range objects {
		if gang, ok := object.(*groveschedulerv1alpha1.PodGang); ok && gang.Labels[grovecommon.LabelSchedulerName] == schedulerName {
			return gang
		}
	}
	t.Fatalf("PodGang for scheduler %q not found", schedulerName)
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
		reconciler.Client, reconciler.DockerSecretRetriever, desired.workload, desired.plan, dgd)
	require.NoError(t, err)
	return rendered
}

func newLPXTestDGD(t *testing.T, pipeline lpx.Pipeline) (*nvidiacomv1alpha1.LPXGraphDeployment, *nvidiacomv1beta1.DynamoGraphDeployment, *lpx.ModelRegistry) {
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
			Finalizers:      []string{lpxGraphDeploymentFinalizer},
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
				LPX: &nvidiacomv1beta1.LPXConfig{
					BuildID:  buildID,
					Settings: &apiextensionsv1.JSON{Raw: []byte(`{"prop_sync":true}`)},
				},
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
	if pipeline == lpx.PipelineLPX {
		*source.Spec.Components[0].ComponentRole(nvidiacomv1beta1.ComponentRoleLPXConductor) = nvidiacomv1beta1.ComponentRoleSpec{
			Name:     nvidiacomv1beta1.ComponentRoleLPXConductor,
			Replicas: &one,
			PodTemplate: &corev1.PodTemplateSpec{Spec: corev1.PodSpec{
				ResourceClaims: []corev1.PodResourceClaim{{Name: "gpu", ResourceClaimTemplateName: ptr.To("gpu")}},
				Containers: []corev1.Container{{Name: consts.MainContainerName, Image: "cyborg-runtime",
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

func newLPXSpecDecodeTestDGD(t *testing.T) (*nvidiacomv1alpha1.LPXGraphDeployment, *nvidiacomv1beta1.DynamoGraphDeployment, *lpx.ModelRegistry) {
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
) *lpx.ModelRegistry {
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
	registry lpxModelRegistry,
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
		WithIndex(&nvidiacomv1beta1.DynamoGraphDeployment{}, lpxTopologyBindingRefIndex, lpxTopologyBindingReferences).
		WithIndex(&nvidiacomv1beta1.DynamoGraphDeployment{}, lpxResourceClaimRefIndex, lpxDRAClaimReferences(false)).
		WithIndex(&nvidiacomv1beta1.DynamoGraphDeployment{}, lpxResourceClaimTemplateRefIndex, lpxDRAClaimReferences(true)).
		WithIndex(&resourcev1.ResourceClaim{}, lpxDeviceClassRefIndex, lpxDeviceClassReferences).
		WithIndex(&resourcev1.ResourceClaimTemplate{}, lpxDeviceClassRefIndex, lpxDeviceClassReferences).
		WithStatusSubresource(&nvidiacomv1beta1.DynamoGraphDeployment{}, &nvidiacomv1alpha1.LPXGraphDeployment{}).
		WithObjects(seed...).
		Build()
	nextLPRUID := 0
	wrapped := interceptor.NewClient(base, interceptor.Funcs{
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
		MPI: configv1alpha1.MPIConfiguration{SSHSecretName: "ssh-secret"},
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

	// Each Grove replica owns independent cliques and one gang per scheduler.
	for replicaIndex := int32(0); replicaIndex < group.Spec.Replicas; replicaIndex++ {
		gangs := make(map[string]*groveschedulerv1alpha1.PodGang, 2)
		for _, scheduler := range []string{corev1.DefaultSchedulerName, lpx.SchedulerName} {
			name := fmt.Sprintf("selected-dgd-%d-%s", replicaIndex, scheduler)
			gangs[scheduler] = &groveschedulerv1alpha1.PodGang{
				TypeMeta: metav1.TypeMeta{APIVersion: groveschedulerv1alpha1.SchemeGroupVersion.String(), Kind: "PodGang"},
				ObjectMeta: metav1.ObjectMeta{
					Name: name, Namespace: pcs.Namespace, UID: types.UID(name + "-uid"),
					Labels: maps.Clone(pcs.Labels), Annotations: maps.Clone(pcs.Annotations),
					OwnerReferences: []metav1.OwnerReference{*metav1.NewControllerRef(pcs, grovev1alpha1.SchemeGroupVersion.WithKind("PodCliqueSet"))},
				},
			}
			gangs[scheduler].Labels[grovecommon.LabelPartOfKey] = pcs.Name
			gangs[scheduler].Labels[grovecommon.LabelSchedulerName] = scheduler
		}
		parent := grovecommon.ResourceNameReplica{Name: group.Name, Replica: int(replicaIndex)}
		for _, template := range pcs.Spec.Template.Cliques {
			rendered := template.DeepCopy()
			name := grovecommon.GeneratePodCliqueName(parent, rendered.Name)
			gang := gangs[rendered.Spec.PodSpec.SchedulerName]
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
			clique.Labels[grovecommon.LabelPodGang] = gang.Name
			clique.Labels[grovecommon.LabelPodCliqueScalingGroup] = group.Name
			clique.Labels[grovecommon.LabelPodCliqueScalingGroupReplicaIndex] = strconv.FormatInt(int64(replicaIndex), 10)
			for index, dependency := range clique.Spec.StartsAfter {
				clique.Spec.StartsAfter[index] = grovecommon.GeneratePodCliqueName(parent, dependency)
			}
			objects = append(objects, clique)

			// Gang references, not fake Pods, witness each clique's rendered cardinality.
			podGroup := groveschedulerv1alpha1.PodGroup{Name: name, MinReplicas: ptr.Deref(clique.Spec.MinAvailable, clique.Spec.Replicas)}
			for podIndex := int32(0); podIndex < clique.Spec.Replicas; podIndex++ {
				podGroup.PodReferences = append(podGroup.PodReferences, groveschedulerv1alpha1.NamespacedName{
					Namespace: pcs.Namespace, Name: fmt.Sprintf("%s-%d", name, podIndex),
				})
			}
			gang.Spec.PodGroups = append(gang.Spec.PodGroups, podGroup)
		}
		objects = append(objects, gangs[corev1.DefaultSchedulerName], gangs[lpx.SchedulerName])
	}
	return objects
}

func createLPXTestObjects(t *testing.T, ctx context.Context, kubeClient client.Client, objects ...client.Object) {
	t.Helper()
	for _, object := range objects {
		require.NoError(t, kubeClient.Create(ctx, object.DeepCopyObject().(client.Object)))
	}
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
