/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package controller

import (
	"context"
	"maps"
	"strings"
	"testing"

	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	commoncontroller "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/features"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/tools/record"
	"k8s.io/utils/ptr"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
	"sigs.k8s.io/controller-runtime/pkg/event"
	disaggregatedsetv1 "sigs.k8s.io/lws/api/disaggregatedset/v1"
	leaderworkersetv1 "sigs.k8s.io/lws/api/leaderworkerset/v1"
	disaggregatedsetutils "sigs.k8s.io/lws/pkg/utils/disaggregatedset"
)

func TestShouldUseDisaggregatedSetSeparatesAPIAvailabilityFromSelection(t *testing.T) {
	dgd := newEnvtestDSHappyPathDGD("selection-capabilities")
	tests := []struct {
		name          string
		runtimeConfig *commoncontroller.RuntimeConfig
		wantUse       bool
		wantReason    string
	}{
		{
			name:          "API unavailable",
			runtimeConfig: &commoncontroller.RuntimeConfig{},
			wantReason:    "API is not available",
		},
		{
			name: "API available but pathway disabled",
			runtimeConfig: &commoncontroller.RuntimeConfig{
				Capabilities: features.Capabilities{DisaggregatedSetAPI: true},
			},
			wantReason: "pathway is disabled",
		},
		{
			name: "API and pathway available",
			runtimeConfig: &commoncontroller.RuntimeConfig{
				Gate:         features.Gates{LWS: true, DisaggregatedSet: true},
				Capabilities: features.Capabilities{DisaggregatedSetAPI: true},
			},
			wantUse: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			use, reason := shouldUseDisaggregatedSet(dgd, tt.runtimeConfig)
			require.Equal(t, tt.wantUse, use)
			if tt.wantReason == "" {
				require.Empty(t, reason)
			} else {
				require.Contains(t, reason, tt.wantReason)
			}
		})
	}
}

func TestSyncDisaggregatedSetPreservesUnmanagedMetadata(t *testing.T) {
	scheme := runtime.NewScheme()
	require.NoError(t, nvidiacomv1beta1.AddToScheme(scheme))
	scheme.AddKnownTypeWithName(disaggregatedSetGVK, &unstructured.Unstructured{})
	scheme.AddKnownTypeWithName(disaggregatedSetGVK.GroupVersion().WithKind("DisaggregatedSetList"), &unstructured.UnstructuredList{})

	dgd := &nvidiacomv1beta1.DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "demo", Namespace: "default", UID: "demo-uid"},
	}
	current := newDisaggregatedSetObject()
	current.SetName(disaggregatedSetName(dgd))
	current.SetNamespace(dgd.Namespace)
	current.SetLabels(map[string]string{"example.com/keep": "label"})
	current.SetAnnotations(map[string]string{"example.com/keep": "annotation"})
	current.SetOwnerReferences([]metav1.OwnerReference{
		*dgdControllerOwnerReference(dgd),
		{APIVersion: "v1", Kind: "ConfigMap", Name: "keep", UID: "keep-uid"},
	})
	current.Object["spec"] = map[string]any{"roles": []any{}}
	desired := current.DeepCopy()
	desired.SetLabels(map[string]string{consts.KubeLabelDynamoGraphDeploymentName: dgd.Name})
	desired.SetAnnotations(map[string]string{"example.com/desired": "annotation"})
	desired.SetOwnerReferences([]metav1.OwnerReference{*dgdControllerOwnerReference(dgd)})
	desired.Object["spec"] = map[string]any{"roles": []any{
		map[string]any{"name": "prefill"},
		map[string]any{"name": "decode"},
	}}
	k8sClient := fake.NewClientBuilder().WithScheme(scheme).WithObjects(dgd, current).Build()
	recorder := record.NewFakeRecorder(10)
	workloads := newDisaggregatedSetWorkloadsReconciler(k8sClient, recorder, nil, nil, nil, newDGDWorkerRolloutReconciler(k8sClient, recorder))

	modified, synced, err := workloads.syncDisaggregatedSet(t.Context(), dgd, desired)
	require.NoError(t, err)
	require.True(t, modified)
	require.Equal(t, "label", synced.GetLabels()["example.com/keep"])
	require.Equal(t, "annotation", synced.GetAnnotations()["example.com/keep"])
	require.Equal(t, "annotation", synced.GetAnnotations()["example.com/desired"])
	require.Len(t, synced.GetOwnerReferences(), 2)
	persisted := newDisaggregatedSetObject()
	require.NoError(t, workloads.Get(t.Context(), client.ObjectKeyFromObject(current), persisted))
	require.Equal(t, "label", persisted.GetLabels()["example.com/keep"])
	require.Equal(t, "annotation", persisted.GetAnnotations()["example.com/desired"])
}

func TestDisaggregatedSetServiceSelectorIsRevisionScoped(t *testing.T) {
	service := &corev1.Service{}
	setDisaggregatedSetServiceSelector(service, "demo-ds", "prefill", "abc12345")

	require.True(t, isDisaggregatedSetServiceSelector(service))
	require.Equal(t, map[string]string{
		disaggregatedsetv1.SetNameLabelKey:  "demo-ds",
		disaggregatedsetv1.RoleLabelKey:     "prefill",
		disaggregatedsetv1.RevisionLabelKey: "abc12345",
	}, service.Spec.Selector)
}

func TestDisaggregatedSetServiceSelectorCutover(t *testing.T) {
	existingDCDSelector := map[string]string{consts.KubeLabelDynamoSelector: "demo-prefill"}
	existingDSSelector := map[string]string{
		disaggregatedsetv1.SetNameLabelKey:  "demo-ds",
		disaggregatedsetv1.RoleLabelKey:     "prefill",
		disaggregatedsetv1.RevisionLabelKey: "old12345",
	}
	tests := []struct {
		name        string
		hasExisting bool
		targetReady bool
		existing    map[string]string
		want        map[string]string
	}{
		{
			name:        "DCD selector remains active while the first DS revision is pending",
			hasExisting: true,
			existing:    existingDCDSelector,
			want:        existingDCDSelector,
		},
		{
			name:        "old DS revision remains active while the target revision is pending",
			hasExisting: true,
			existing:    existingDSSelector,
			want:        existingDSSelector,
		},
		{
			name: "a new service selects the target revision immediately",
			want: map[string]string{
				disaggregatedsetv1.SetNameLabelKey:  "demo-ds",
				disaggregatedsetv1.RoleLabelKey:     "prefill",
				disaggregatedsetv1.RevisionLabelKey: "new12345",
			},
		},
		{
			name:        "a ready target replaces the active selector",
			hasExisting: true,
			targetReady: true,
			existing:    existingDCDSelector,
			want: map[string]string{
				disaggregatedsetv1.SetNameLabelKey:  "demo-ds",
				disaggregatedsetv1.RoleLabelKey:     "prefill",
				disaggregatedsetv1.RevisionLabelKey: "new12345",
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			service := &corev1.Service{}
			existing := &corev1.Service{Spec: corev1.ServiceSpec{Selector: maps.Clone(tt.existing)}}

			setDesiredDisaggregatedSetServiceSelector(
				service,
				existing,
				tt.hasExisting,
				"demo-ds",
				"prefill",
				"new12345",
				tt.targetReady,
			)

			require.Equal(t, tt.want, service.Spec.Selector)
		})
	}
}

func TestDisaggregatedSetCleanupUsesAPICapabilityWhenSelectionIsDisabled(t *testing.T) {
	scheme := runtime.NewScheme()
	require.NoError(t, nvidiacomv1beta1.AddToScheme(scheme))
	require.NoError(t, corev1.AddToScheme(scheme))
	scheme.AddKnownTypeWithName(disaggregatedSetGVK, &unstructured.Unstructured{})
	scheme.AddKnownTypeWithName(disaggregatedSetGVK.GroupVersion().WithKind("DisaggregatedSetList"), &unstructured.UnstructuredList{})

	dgd := &nvidiacomv1beta1.DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "demo", Namespace: "default", UID: "dgd-uid"},
		Spec: nvidiacomv1beta1.DynamoGraphDeploymentSpec{
			Components: []nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec{{ComponentName: "prefill"}},
		},
	}
	dcd := &nvidiacomv1beta1.DynamoComponentDeployment{ObjectMeta: metav1.ObjectMeta{
		Name:            "demo-prefill",
		Namespace:       dgd.Namespace,
		UID:             "dcd-uid",
		Labels:          map[string]string{consts.KubeLabelDynamoGraphDeploymentName: dgd.Name},
		OwnerReferences: []metav1.OwnerReference{*dgdControllerOwnerReference(dgd)},
	}}
	service := &corev1.Service{ObjectMeta: metav1.ObjectMeta{
		Name:            dynamo.NormalizeKubeResourceName(dcd.Name),
		Namespace:       dgd.Namespace,
		OwnerReferences: []metav1.OwnerReference{*dgdControllerOwnerReference(dgd)},
	}}
	ds := newDisaggregatedSetObject()
	ds.SetName(disaggregatedSetName(dgd))
	ds.SetNamespace(dgd.Namespace)
	ds.SetOwnerReferences([]metav1.OwnerReference{*dgdControllerOwnerReference(dgd)})
	k8sClient := fake.NewClientBuilder().WithScheme(scheme).WithObjects(dgd, dcd, service, ds).Build()
	reconciler := &DynamoGraphDeploymentReconciler{
		Client:    &staleDisaggregatedSetClient{Client: k8sClient},
		APIReader: k8sClient,
		Recorder:  record.NewFakeRecorder(10),
		RuntimeConfig: &commoncontroller.RuntimeConfig{
			Capabilities: features.Capabilities{DisaggregatedSetAPI: true},
		},
	}

	require.NoError(t, reconciler.newDisaggregatedSetCompatibilityCleanup(true).Reconcile(t.Context(), dgd))
	persistedService := &corev1.Service{}
	require.NoError(t, k8sClient.Get(t.Context(), client.ObjectKeyFromObject(service), persistedService))
	require.True(t, metav1.IsControlledBy(persistedService, dcd))
	err := k8sClient.Get(t.Context(), client.ObjectKeyFromObject(ds), newDisaggregatedSetObject())
	require.True(t, apierrors.IsNotFound(err))
}

type staleDisaggregatedSetClient struct {
	client.Client
}

func (c *staleDisaggregatedSetClient) Get(
	ctx context.Context,
	key client.ObjectKey,
	obj client.Object,
	opts ...client.GetOption,
) error {
	if obj.GetObjectKind().GroupVersionKind() == disaggregatedSetGVK {
		return apierrors.NewNotFound(schema.GroupResource{
			Group:    disaggregatedSetGVK.Group,
			Resource: "disaggregatedsets",
		}, key.Name)
	}
	return c.Client.Get(ctx, key, obj, opts...)
}

func TestSelectDisaggregatedSetComponents(t *testing.T) {
	t.Run("selects multinode worker roles", func(t *testing.T) {
		dgd := &nvidiacomv1beta1.DynamoGraphDeployment{
			Spec: nvidiacomv1beta1.DynamoGraphDeploymentSpec{
				Components: []nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec{
					{
						ComponentName: "prefill",
						ComponentType: nvidiacomv1beta1.ComponentTypePrefill,
						Multinode:     &nvidiacomv1beta1.MultinodeSpec{NodeCount: 2},
						Replicas:      ptr.To(int32(2)),
					},
					{
						ComponentName: "decode",
						ComponentType: nvidiacomv1beta1.ComponentTypeDecode,
						Multinode:     &nvidiacomv1beta1.MultinodeSpec{NodeCount: 2},
						Replicas:      ptr.To(int32(2)),
					},
					{
						ComponentName: "frontend",
						ComponentType: nvidiacomv1beta1.ComponentTypeFrontend,
					},
				},
			},
		}

		selection, reason := selectDisaggregatedSetComponents(dgd)
		require.Empty(t, reason)
		require.Equal(t, "prefill", selection.componentToRole["prefill"])
		require.Equal(t, "decode", selection.componentToRole["decode"])
		require.Len(t, selection.componentToRole, 2)
	})

	t.Run("rejects scaling adapter", func(t *testing.T) {
		dgd := &nvidiacomv1beta1.DynamoGraphDeployment{
			Spec: nvidiacomv1beta1.DynamoGraphDeploymentSpec{
				Components: []nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec{
					{
						ComponentName:  "prefill",
						ComponentType:  nvidiacomv1beta1.ComponentTypePrefill,
						Multinode:      &nvidiacomv1beta1.MultinodeSpec{NodeCount: 2},
						ScalingAdapter: &nvidiacomv1beta1.ScalingAdapter{},
						Replicas:       ptr.To(int32(2)),
					},
					{
						ComponentName: "decode",
						ComponentType: nvidiacomv1beta1.ComponentTypeDecode,
						Multinode:     &nvidiacomv1beta1.MultinodeSpec{NodeCount: 2},
						Replicas:      ptr.To(int32(2)),
					},
				},
			},
		}

		_, reason := selectDisaggregatedSetComponents(dgd)
		require.Contains(t, reason, "scalingAdapter")
	})
}

func TestDisaggregatedSetChildNamesFitDNSLabelLimit(t *testing.T) {
	dgd := &nvidiacomv1beta1.DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: strings.Repeat("d", 63)},
		Spec: nvidiacomv1beta1.DynamoGraphDeploymentSpec{
			Components: []nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec{
				{
					ComponentName: strings.Repeat("p", 63),
					ComponentType: nvidiacomv1beta1.ComponentTypeWorker,
					Multinode:     &nvidiacomv1beta1.MultinodeSpec{NodeCount: 2},
				},
				{
					ComponentName: strings.Repeat("q", 63),
					ComponentType: nvidiacomv1beta1.ComponentTypeWorker,
					Multinode:     &nvidiacomv1beta1.MultinodeSpec{NodeCount: 2},
				},
			},
		},
	}

	selection, reason := selectDisaggregatedSetComponents(dgd)
	require.Empty(t, reason)
	require.Len(t, selection.componentToRole, 2)
	setName := disaggregatedSetName(dgd)
	require.LessOrEqual(t, len(setName), maxDisaggregatedSetNameLength)
	for _, roleName := range selection.componentToRole {
		require.LessOrEqual(t, len(roleName), maxDisaggregatedSetRoleNameLength)
		childName := disaggregatedsetutils.GenerateName(setName, roleName, strings.Repeat("a", disaggregatedSetRevisionLength))
		require.LessOrEqual(t, len(childName), 63)
	}
}

func TestCheckDisaggregatedSetReadiness(t *testing.T) {
	ds := newDisaggregatedSetObject()
	ds.SetName("demo")
	ds.SetGeneration(3)
	ds.Object["status"] = map[string]any{
		"observedGeneration": int64(2),
		"roleStatuses": []any{
			map[string]any{"name": "prefill", "replicas": int64(2), "updatedReplicas": int64(2), "readyReplicas": int64(2)},
			map[string]any{"name": "decode", "replicas": int64(2), "updatedReplicas": int64(2), "readyReplicas": int64(1)},
		},
	}
	selection := disaggregatedSetSelection{
		componentToRole: map[string]string{"prefill": "prefill", "decode": "decode"},
		desiredReplicas: map[string]int32{"prefill": 2, "decode": 2},
	}

	t.Log("stale observedGeneration keeps the DisaggregatedSet unready")
	ready, reason, statuses := checkDisaggregatedSetReadiness(ds, selection)
	require.False(t, ready)
	require.Contains(t, reason, "observed generation")
	require.Equal(t, int32(2), ptr.Deref(statuses["prefill"].ReadyReplicas, 0))

	t.Log("lagging decode role readiness keeps the DisaggregatedSet unready")
	ds.Object["status"].(map[string]any)["observedGeneration"] = int64(3)
	ready, reason, statuses = checkDisaggregatedSetReadiness(ds, selection)
	require.False(t, ready)
	require.Contains(t, reason, "decode")
	require.Equal(t, int32(1), ptr.Deref(statuses["decode"].ReadyReplicas, 0))

	t.Log("all roles at desired ready replicas report ready")
	ds.Object["status"].(map[string]any)["roleStatuses"] = []any{
		map[string]any{"name": "prefill", "replicas": int64(2), "updatedReplicas": int64(2), "readyReplicas": int64(2)},
		map[string]any{"name": "decode", "replicas": int64(2), "updatedReplicas": int64(2), "readyReplicas": int64(2)},
	}
	ready, _, _ = checkDisaggregatedSetReadiness(ds, selection)
	require.True(t, ready)
}

func TestCheckDisaggregatedSetChildLWSReadinessWaitsForRemovedRoles(t *testing.T) {
	selection := disaggregatedSetSelection{
		componentToRole: map[string]string{"prefill": "prefill", "decode": "decode"},
		desiredReplicas: map[string]int32{"prefill": 1, "decode": 1},
	}
	readyChild := func(name string) *leaderworkersetv1.LeaderWorkerSet {
		return &leaderworkersetv1.LeaderWorkerSet{
			ObjectMeta: metav1.ObjectMeta{Name: name, Generation: 1},
			Spec:       leaderworkersetv1.LeaderWorkerSetSpec{Replicas: ptr.To[int32](1)},
			Status: leaderworkersetv1.LeaderWorkerSetStatus{
				ObservedGeneration: 1,
				Replicas:           1,
				UpdatedReplicas:    1,
				ReadyReplicas:      1,
			},
		}
	}
	prefill := readyChild("demo-prefill-target")
	decode := readyChild("demo-decode-target")
	removed := readyChild("demo-legacy-worker-old")
	targetByRole := map[string]*leaderworkersetv1.LeaderWorkerSet{
		"prefill": prefill,
		"decode":  decode,
	}
	childrenByRole := map[string][]*leaderworkersetv1.LeaderWorkerSet{
		"prefill":       {prefill},
		"decode":        {decode},
		"legacy-worker": {removed},
	}

	t.Log("a removed role with live replicas keeps the DisaggregatedSet unready")
	ready, reason, _ := checkDisaggregatedSetChildLWSReadiness(selection, targetByRole, childrenByRole)
	require.False(t, ready)
	require.Contains(t, reason, removed.Name)

	t.Log("the target becomes ready after the removed role is fully drained")
	removed.Spec.Replicas = ptr.To[int32](0)
	removed.Status.Replicas = 0
	removed.Status.UpdatedReplicas = 0
	removed.Status.ReadyReplicas = 0
	ready, _, _ = checkDisaggregatedSetChildLWSReadiness(selection, targetByRole, childrenByRole)
	require.True(t, ready)
}

func TestDisaggregatedSetStatusReadinessWaitsForRemovedRoleChildren(t *testing.T) {
	ds := newDisaggregatedSetObject()
	ds.SetName("demo")
	ds.SetNamespace(testNamespace)
	ds.SetUID("demo-uid")
	ds.SetGeneration(2)
	ds.Object["status"] = map[string]any{
		"observedGeneration": int64(2),
		"roleStatuses": []any{
			map[string]any{"name": "prefill", "replicas": int64(1), "updatedReplicas": int64(1), "readyReplicas": int64(1)},
			map[string]any{"name": "decode", "replicas": int64(1), "updatedReplicas": int64(1), "readyReplicas": int64(1)},
		},
	}
	removed := &leaderworkersetv1.LeaderWorkerSet{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "demo-old-legacy-worker",
			Namespace: testNamespace,
			Labels: map[string]string{
				disaggregatedsetv1.SetNameLabelKey: "demo",
				disaggregatedsetv1.RoleLabelKey:    "legacy-worker",
			},
			OwnerReferences: []metav1.OwnerReference{{
				APIVersion: disaggregatedsetv1.GroupVersion.String(),
				Kind:       "DisaggregatedSet",
				Name:       ds.GetName(),
				UID:        ds.GetUID(),
				Controller: ptr.To(true),
			}},
		},
		Spec:   leaderworkersetv1.LeaderWorkerSetSpec{Replicas: ptr.To[int32](1)},
		Status: leaderworkersetv1.LeaderWorkerSetStatus{Replicas: 1, ReadyReplicas: 1},
	}
	scheme := runtime.NewScheme()
	require.NoError(t, leaderworkersetv1.AddToScheme(scheme))
	workloads := &disaggregatedSetWorkloadsReconciler{Client: fake.NewClientBuilder().WithScheme(scheme).WithObjects(removed).Build()}
	selection := disaggregatedSetSelection{
		componentToRole: map[string]string{"prefill": "prefill", "decode": "decode"},
		desiredReplicas: map[string]int32{"prefill": 1, "decode": 1},
	}

	ready, reason, _, err := workloads.checkDisaggregatedSetReadiness(t.Context(), ds, selection)
	require.NoError(t, err)
	require.False(t, ready)
	require.Contains(t, reason, removed.Name)
}

func TestDisaggregatedSetPredicatesObserveRoutingMetadata(t *testing.T) {
	baseDS := newDisaggregatedSetObject()
	baseDS.SetLabels(map[string]string{consts.KubeLabelDynamoGraphDeploymentName: "demo"})
	relabeledDS := baseDS.DeepCopy()
	relabeledDS.SetLabels(map[string]string{consts.KubeLabelDynamoGraphDeploymentName: "other"})
	reownedDS := baseDS.DeepCopy()
	reownedDS.SetOwnerReferences([]metav1.OwnerReference{{
		APIVersion: nvidiacomv1beta1.GroupVersion.String(),
		Kind:       dynamoGraphDeploymentKind,
		Name:       "demo",
		UID:        "demo-uid",
		Controller: ptr.To(true),
	}})
	statusUpdatedDS := baseDS.DeepCopy()
	statusUpdatedDS.Object["status"] = map[string]any{"observedGeneration": int64(1)}

	require.False(t, disaggregatedSetStatusChanged(baseDS, baseDS.DeepCopy()))
	require.True(t, disaggregatedSetStatusChanged(baseDS, relabeledDS))
	require.True(t, disaggregatedSetStatusChanged(baseDS, reownedDS))
	require.True(t, disaggregatedSetStatusChanged(baseDS, statusUpdatedDS))

	baseLWS := &leaderworkersetv1.LeaderWorkerSet{ObjectMeta: metav1.ObjectMeta{
		Labels: map[string]string{
			consts.KubeLabelDynamoGraphDeploymentName: "demo",
		},
	}}
	relabeledLWS := baseLWS.DeepCopy()
	relabeledLWS.SetLabels(map[string]string{
		consts.KubeLabelDynamoGraphDeploymentName: "other",
	})
	statusUpdatedLWS := baseLWS.DeepCopy()
	statusUpdatedLWS.Status.Conditions = []metav1.Condition{{Type: "Available", Status: metav1.ConditionTrue}}

	require.False(t, leaderWorkerSetStatusChanged(baseLWS, baseLWS.DeepCopy()))
	require.True(t, leaderWorkerSetStatusChanged(baseLWS, relabeledLWS))
	require.True(t, leaderWorkerSetStatusChanged(baseLWS, statusUpdatedLWS))
}

func TestWorkloadRoutingAnnotationsChanged(t *testing.T) {
	t.Log("no change in routing annotations does not trigger update predicate")
	oldDGD := &nvidiacomv1beta1.DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{
			Annotations: map[string]string{
				consts.KubeAnnotationEnableGrove:            consts.KubeLabelValueFalse,
				consts.KubeAnnotationEnableDisaggregatedSet: consts.KubeLabelValueFalse,
			},
		},
	}
	newDGD := oldDGD.DeepCopy()
	require.False(t, workloadRoutingAnnotationsChanged(updateEvent(oldDGD, newDGD)))

	t.Log("enabling DisaggregatedSet triggers update predicate")
	newDGD = oldDGD.DeepCopy()
	newDGD.Annotations[consts.KubeAnnotationEnableDisaggregatedSet] = consts.KubeLabelValueTrue
	require.True(t, workloadRoutingAnnotationsChanged(updateEvent(oldDGD, newDGD)))

	t.Log("disabling DisaggregatedSet triggers update predicate")
	oldDGD = newDGD.DeepCopy()
	newDGD = oldDGD.DeepCopy()
	newDGD.Annotations[consts.KubeAnnotationEnableDisaggregatedSet] = consts.KubeLabelValueFalse
	require.True(t, workloadRoutingAnnotationsChanged(updateEvent(oldDGD, newDGD)))

	t.Log("removing DisaggregatedSet triggers update predicate")
	newDGD = oldDGD.DeepCopy()
	delete(newDGD.Annotations, consts.KubeAnnotationEnableDisaggregatedSet)
	require.True(t, workloadRoutingAnnotationsChanged(updateEvent(oldDGD, newDGD)))
}

func updateEvent(oldObj, newObj *nvidiacomv1beta1.DynamoGraphDeployment) event.UpdateEvent {
	return event.UpdateEvent{ObjectOld: oldObj, ObjectNew: newObj}
}
