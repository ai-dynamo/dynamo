/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package controller

import (
	"maps"
	"strings"
	"testing"

	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/checkpoint"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	commoncontroller "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/features"
	"github.com/stretchr/testify/require"
	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/tools/events"
	"k8s.io/utils/ptr"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
	"sigs.k8s.io/controller-runtime/pkg/event"
	disaggregatedsetv1 "sigs.k8s.io/lws/api/disaggregatedset/v1"
	leaderworkersetv1 "sigs.k8s.io/lws/api/leaderworkerset/v1"
	disaggregatedsetutils "sigs.k8s.io/lws/pkg/utils/disaggregatedset"
)

const disaggregatedSetUnitTestNamespace = "default"

func TestDisaggregatedSetEligibilityDoesNotSelectAProvider(t *testing.T) {
	dgd := newEnvtestDSHappyPathDGD("selection-eligibility")
	tests := []struct {
		name       string
		gate       features.Gate
		wantReason string
	}{
		{
			name:       "LWS gate supports all eligible roles",
			gate:       features.Gates{LWS: true},
			wantReason: "",
		},
		{
			name:       "selection validation rejects scaling adapter",
			gate:       features.Gates{LWS: true},
			wantReason: "scalingAdapter",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.wantReason != "" {
				dgd.Spec.Components[0].ScalingAdapter = &nvidiacomv1beta1.ScalingAdapter{}
			}
			require.Contains(t, disaggregatedSetEligibilityReason(dgd, tt.gate), tt.wantReason)
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
	resources := newDisaggregatedSetResourceReconciler(k8sClient)

	synced, modified, err := resources.Reconcile(t.Context(), dgd, desired)
	require.NoError(t, err)
	require.True(t, modified)
	require.Equal(t, "label", synced.GetLabels()["example.com/keep"])
	require.Equal(t, "annotation", synced.GetAnnotations()["example.com/keep"])
	require.Equal(t, "annotation", synced.GetAnnotations()["example.com/desired"])
	require.Len(t, synced.GetOwnerReferences(), 2)
	persisted := newDisaggregatedSetObject()
	require.NoError(t, k8sClient.Get(t.Context(), client.ObjectKeyFromObject(current), persisted))
	require.Equal(t, "label", persisted.GetLabels()["example.com/keep"])
	require.Equal(t, "annotation", persisted.GetAnnotations()["example.com/desired"])
}

func TestSyncDGDStableServicePrunesManagedMetadataAndPreservesExternalMetadata(t *testing.T) {
	scheme := runtime.NewScheme()
	require.NoError(t, nvidiacomv1beta1.AddToScheme(scheme))
	require.NoError(t, corev1.AddToScheme(scheme))

	dgd := &nvidiacomv1beta1.DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "demo", Namespace: "default", UID: "demo-uid"},
	}
	desired := &corev1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name:        "demo-prefill",
			Namespace:   dgd.Namespace,
			Labels:      map[string]string{"example.com/managed-label": "old"},
			Annotations: map[string]string{"example.com/managed-annotation": "old"},
		},
		Spec: corev1.ServiceSpec{Ports: []corev1.ServicePort{{Port: 8080}}},
	}
	kubeClient := fake.NewClientBuilder().WithScheme(scheme).WithObjects(dgd).Build()
	stableResources := newDisaggregatedSetStableResourcesReconciler(kubeClient, nil)
	require.NoError(t, stableResources.syncDGDStableService(t.Context(), dgd, desired))

	stored := &corev1.Service{}
	require.NoError(t, kubeClient.Get(t.Context(), client.ObjectKeyFromObject(desired), stored))
	stored.Labels["external.example.com/keep"] = "label"
	stored.Annotations["external.example.com/keep"] = "annotation"
	require.NoError(t, kubeClient.Update(t.Context(), stored))

	updatedDesired := desired.DeepCopy()
	updatedDesired.Labels = map[string]string{"example.com/current-label": "new"}
	updatedDesired.Annotations = map[string]string{"example.com/current-annotation": "new"}
	require.NoError(t, stableResources.syncDGDStableService(t.Context(), dgd, updatedDesired))

	require.NoError(t, kubeClient.Get(t.Context(), client.ObjectKeyFromObject(desired), stored))
	require.NotContains(t, stored.Labels, "example.com/managed-label")
	require.NotContains(t, stored.Annotations, "example.com/managed-annotation")
	require.Equal(t, "new", stored.Labels["example.com/current-label"])
	require.Equal(t, "new", stored.Annotations["example.com/current-annotation"])
	require.Equal(t, "label", stored.Labels["external.example.com/keep"])
	require.Equal(t, "annotation", stored.Annotations["external.example.com/keep"])
	require.NotEmpty(t, stored.Annotations[managedServiceMetadataAnnotation])
}

func TestDisaggregatedSetServiceSelectorIsRevisionScoped(t *testing.T) {
	service := &corev1.Service{}
	setDisaggregatedSetServiceSelector(service, "demo-ds", "prefill", "abc12345")

	require.Equal(t, map[string]string{
		disaggregatedsetv1.SetNameLabelKey:  "demo-ds",
		disaggregatedsetv1.RoleLabelKey:     "prefill",
		disaggregatedsetv1.RevisionLabelKey: "abc12345",
	}, service.Spec.Selector, "the stable component Service must aggregate every slice")
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

func TestDeleteStaleDisaggregatedSetServicesRemovesUndesiredModelService(t *testing.T) {
	scheme := runtime.NewScheme()
	require.NoError(t, nvidiacomv1beta1.AddToScheme(scheme))
	require.NoError(t, corev1.AddToScheme(scheme))

	dgd := &nvidiacomv1beta1.DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "demo", Namespace: "default", UID: "demo-uid"},
	}
	controlledService := func(name string, labels map[string]string) *corev1.Service {
		return &corev1.Service{ObjectMeta: metav1.ObjectMeta{
			Name:            name,
			Namespace:       dgd.Namespace,
			Labels:          labels,
			OwnerReferences: []metav1.OwnerReference{*dgdControllerOwnerReference(dgd)},
		}}
	}
	desiredComponent := controlledService("demo-prefill", map[string]string{
		consts.KubeLabelDynamoGraphDeploymentName: dgd.Name,
		consts.KubeLabelDynamoComponent:           "prefill",
	})
	staleComponent := controlledService("demo-removed", map[string]string{
		consts.KubeLabelDynamoGraphDeploymentName: dgd.Name,
		consts.KubeLabelDynamoComponent:           "removed",
	})
	staleModel := controlledService(dynamo.GenerateServiceName("removed-model"), map[string]string{
		consts.KubeLabelDynamoBaseModelHash: dynamo.HashModelName("removed-model"),
	})
	foreignModel := controlledService(dynamo.GenerateServiceName("foreign-model"), map[string]string{
		consts.KubeLabelDynamoBaseModelHash: dynamo.HashModelName("foreign-model"),
	})
	foreignModel.OwnerReferences = nil
	staleUnlabeled := controlledService("demo-unlabeled", nil)
	foreignUnlabeled := controlledService("foreign-unlabeled", nil)
	foreignUnlabeled.OwnerReferences = nil

	k8sClient := fake.NewClientBuilder().WithScheme(scheme).
		WithObjects(dgd, desiredComponent, staleComponent, staleModel, foreignModel, staleUnlabeled, foreignUnlabeled).
		Build()
	stableResources := newDisaggregatedSetStableResourcesReconciler(k8sClient, nil)

	t.Log("stale component and model services are removed while desired and foreign services remain")
	require.NoError(t, stableResources.DeleteStale(t.Context(), dgd, map[string]struct{}{
		desiredComponent.Name: {},
	}))
	for _, name := range []string{staleComponent.Name, staleModel.Name, staleUnlabeled.Name} {
		err := k8sClient.Get(t.Context(), client.ObjectKey{Name: name, Namespace: dgd.Namespace}, &corev1.Service{})
		require.True(t, apierrors.IsNotFound(err))
	}
	for _, name := range []string{desiredComponent.Name, foreignModel.Name, foreignUnlabeled.Name} {
		require.NoError(t, k8sClient.Get(t.Context(), client.ObjectKey{Name: name, Namespace: dgd.Namespace}, &corev1.Service{}))
	}
}

func TestDeleteOwnedSelectedDCDsUsesOwnerReference(t *testing.T) {
	scheme := runtime.NewScheme()
	require.NoError(t, nvidiacomv1beta1.AddToScheme(scheme))

	dgd := &nvidiacomv1beta1.DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "demo", Namespace: "default", UID: "demo-uid"},
	}
	owned := func(name string, labels map[string]string, owner *metav1.OwnerReference) *nvidiacomv1beta1.DynamoComponentDeployment {
		var ownerReferences []metav1.OwnerReference
		if owner != nil {
			ownerReferences = []metav1.OwnerReference{*owner}
		}
		return &nvidiacomv1beta1.DynamoComponentDeployment{
			ObjectMeta: metav1.ObjectMeta{
				Name:            name,
				Namespace:       dgd.Namespace,
				Labels:          labels,
				OwnerReferences: ownerReferences,
			},
			Spec: nvidiacomv1beta1.DynamoComponentDeploymentSpec{
				DynamoComponentDeploymentSharedSpec: nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec{
					ComponentName: "prefill",
				},
			},
		}
	}
	ownedByDGD := dgdControllerOwnerReference(dgd)
	foreignOwner := ownedByDGD.DeepCopy()
	foreignOwner.UID = "other-dgd-uid"
	selected := disaggregatedSetSelection{componentToRole: map[string]string{"prefill": "prefill"}}
	ownedWithoutGraphLabel := owned("owned-without-graph-label", nil, ownedByDGD)
	ownedWithStaleGraphLabel := owned("owned-with-stale-graph-label", map[string]string{
		consts.KubeLabelDynamoGraphDeploymentName: "other-graph",
	}, ownedByDGD)
	foreignLabeled := owned("foreign-labeled", map[string]string{
		consts.KubeLabelDynamoGraphDeploymentName: dgd.Name,
	}, foreignOwner)
	foreignUnlabeled := owned("foreign-unlabeled", nil, foreignOwner)

	kubeClient := fake.NewClientBuilder().WithScheme(scheme).
		WithObjects(dgd, ownedWithoutGraphLabel, ownedWithStaleGraphLabel, foreignLabeled, foreignUnlabeled).
		Build()
	auxiliaryDCDs := newDisaggregatedSetAuxiliaryDCDReconciler(kubeClient, events.NewFakeRecorder(10))

	require.NoError(t, auxiliaryDCDs.DeleteSelected(t.Context(), dgd, selected))
	for _, name := range []string{ownedWithoutGraphLabel.Name, ownedWithStaleGraphLabel.Name} {
		err := kubeClient.Get(t.Context(), client.ObjectKey{Name: name, Namespace: dgd.Namespace}, &nvidiacomv1beta1.DynamoComponentDeployment{})
		require.True(t, apierrors.IsNotFound(err), "owned selected DCD %s should be deleted", name)
	}
	for _, name := range []string{foreignLabeled.Name, foreignUnlabeled.Name} {
		require.NoError(t, kubeClient.Get(t.Context(), client.ObjectKey{Name: name, Namespace: dgd.Namespace}, &nvidiacomv1beta1.DynamoComponentDeployment{}))
	}
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

func TestDCDAndDirectDisaggregatedSetRenderingParity(t *testing.T) {
	dgd := newEnvtestDSHappyPathDGD("render-parity")
	dgd.Spec.Labels = map[string]string{"example.com/graph-label": "value"}
	dgd.Spec.Annotations = map[string]string{"example.com/graph-annotation": "value"}
	dgd.Spec.Experimental = &nvidiacomv1beta1.DynamoGraphDeploymentExperimentalSpec{
		KvTransferPolicy: &nvidiacomv1beta1.KvTransferPolicy{
			LabelKey: "topology.kubernetes.io/zone",
			Domain:   nvidiacomv1beta1.TopologyDomain("zone"),
		},
	}
	checkpointInfo := &checkpoint.CheckpointInfo{
		Enabled:                   true,
		AutomaticCapture:          true,
		StartupPolicy:             nvidiacomv1alpha1.CheckpointStartupPolicyImmediate,
		SnapshotCompatibilityHash: "compatibility-v1",
		AutomaticSnapshotJob: &checkpoint.SnapshotJobReference{
			Name: "render-parity-checkpoint",
			UID:  types.UID("render-parity-checkpoint-uid"),
		},
	}
	dgd.Spec.Components[0].Experimental = &nvidiacomv1beta1.ExperimentalSpec{
		Checkpoint: &nvidiacomv1beta1.ComponentCheckpointConfig{Enabled: true},
	}
	dgd.Spec.Components[0].PodTemplate.Annotations = map[string]string{}
	require.NoError(t, checkpoint.ApplyRestoreCandidateMetadata(
		dgd.Spec.Components[0].PodTemplate.Annotations,
		checkpointInfo,
	))
	rollingUpdateCtx := dynamo.RollingUpdateContext{NewWorkerHash: "worker123"}
	normalized, err := dynamo.NormalizeDynamoGraphDeploymentComponents(dgd, nil, nil, rollingUpdateCtx)
	require.NoError(t, err)
	component := normalized["prefill"]
	require.NotNil(t, component)
	dcds, err := dynamo.GenerateDynamoComponentsDeploymentsFromNormalized(
		dgd,
		map[string]*nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec{"prefill": component},
		rollingUpdateCtx,
	)
	require.NoError(t, err)
	dcd := dcds["prefill"]
	require.NotNil(t, dcd)
	dcd.SetOwnerReferences([]metav1.OwnerReference{*dgdControllerOwnerReference(dgd)})

	renderScheme := runtime.NewScheme()
	require.NoError(t, appsv1.AddToScheme(renderScheme))
	require.NoError(t, corev1.AddToScheme(renderScheme))
	require.NoError(t, leaderworkersetv1.AddToScheme(renderScheme))
	kubeClient := fake.NewClientBuilder().WithScheme(renderScheme).Build()
	config := &configv1alpha1.OperatorConfiguration{
		Discovery: configv1alpha1.DiscoveryConfiguration{Backend: configv1alpha1.DiscoveryBackendKubernetes},
	}
	renderer := newDCDWorkloadRenderer(
		kubeClient,
		config,
		&commoncontroller.RuntimeConfig{Gate: features.Gates{Checkpoint: true}},
		&mockDockerSecretRetriever{GetSecretsFunc: func(string, string) ([]string, error) { return nil, nil }},
	)
	dcdLeader, dcdWorker, err := renderer.renderMultinodePodTemplateSpecs(t.Context(), dcd)
	require.NoError(t, err)
	backendFramework, err := dynamo.BackendFrameworkForComponent(component, dgd)
	require.NoError(t, err)
	directLeader, directWorker, err := renderer.renderMultinodePodTemplateSpecsForDGDComponent(
		t.Context(),
		dgd,
		component,
		"prefill",
		dcd.Name,
		dynamo.GetDynamoNamespace(dgd, component),
		backendFramework,
		checkpointInfo,
	)
	require.NoError(t, err)

	require.Equal(t, dcdLeader, directLeader)
	require.Equal(t, dcdWorker, directWorker)
	for _, template := range []*corev1.PodTemplateSpec{directLeader, directWorker} {
		require.Equal(t, "topology.kubernetes.io/zone", template.Annotations[consts.KubeAnnotationTopologyLabelKey])
		require.Equal(t, "worker123", template.Labels[consts.KubeLabelDynamoWorkerHash])
		require.Equal(t, "kubernetes", template.Labels[consts.KubeLabelDynamoDiscoveryBackend])
		require.Equal(t, "value", template.Labels["example.com/graph-label"])
		require.Equal(t, "value", template.Annotations["example.com/graph-annotation"])
		require.Equal(t, "render-parity-checkpoint", template.Annotations[consts.CheckpointNameAnnotation])
	}
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
		childName := disaggregatedsetutils.GenerateName(setName, 99, strings.Repeat("a", disaggregatedSetRevisionLength), roleName)
		require.LessOrEqual(t, len(childName), 63)
		serviceName := childName + "-prv"
		require.LessOrEqual(t, len(serviceName), 63)
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
	targetByIdentity := map[disaggregatedSetChildIdentity][]*leaderworkersetv1.LeaderWorkerSet{
		{slice: 0, role: "prefill"}: {prefill},
		{slice: 0, role: "decode"}:  {decode},
	}
	childrenByRole := map[string][]*leaderworkersetv1.LeaderWorkerSet{
		"prefill":       {prefill},
		"decode":        {decode},
		"legacy-worker": {removed},
	}

	t.Log("a removed role with live replicas keeps the DisaggregatedSet unready")
	ready, reason, _ := checkDisaggregatedSetChildLWSReadiness(selection, 1, targetByIdentity, childrenByRole)
	require.False(t, ready)
	require.Contains(t, reason, removed.Name)

	t.Log("the target becomes ready after the removed role is fully drained")
	removed.Spec.Replicas = ptr.To[int32](0)
	removed.Status.Replicas = 0
	removed.Status.UpdatedReplicas = 0
	removed.Status.ReadyReplicas = 0
	ready, _, _ = checkDisaggregatedSetChildLWSReadiness(selection, 1, targetByIdentity, childrenByRole)
	require.True(t, ready)
}

func TestCheckDisaggregatedSetReadinessTracksEverySlice(t *testing.T) {
	typedDS := &disaggregatedsetv1.DisaggregatedSet{
		ObjectMeta: metav1.ObjectMeta{Name: "demo", Namespace: disaggregatedSetUnitTestNamespace, UID: "demo-uid"},
		Spec: disaggregatedsetv1.DisaggregatedSetSpec{
			Slices: ptr.To[int32](2),
			Roles: []disaggregatedsetv1.DisaggregatedRoleSpec{
				{Name: "prefill"},
				{Name: "decode"},
			},
		},
	}
	dsObject, err := runtime.DefaultUnstructuredConverter.ToUnstructured(typedDS)
	require.NoError(t, err)
	ds := newDisaggregatedSetObject()
	ds.Object = dsObject
	ds.SetGroupVersionKind(disaggregatedSetGVK)
	revision := disaggregatedsetutils.ComputeRevision(typedDS.Spec.Roles)

	readyChild := func(slice int, role string) *leaderworkersetv1.LeaderWorkerSet {
		return &leaderworkersetv1.LeaderWorkerSet{
			ObjectMeta: metav1.ObjectMeta{
				Name:       disaggregatedsetutils.GenerateName(ds.GetName(), slice, revision, role),
				Namespace:  ds.GetNamespace(),
				Generation: 1,
				Labels:     disaggregatedsetutils.GenerateLabels(ds.GetName(), slice, revision, role),
				OwnerReferences: []metav1.OwnerReference{{
					APIVersion: disaggregatedsetv1.GroupVersion.String(),
					Kind:       "DisaggregatedSet",
					Name:       ds.GetName(),
					UID:        ds.GetUID(),
					Controller: ptr.To(true),
				}},
			},
			Spec: leaderworkersetv1.LeaderWorkerSetSpec{Replicas: ptr.To[int32](1)},
			Status: leaderworkersetv1.LeaderWorkerSetStatus{
				ObservedGeneration: 1,
				Replicas:           1,
				UpdatedReplicas:    1,
				ReadyReplicas:      1,
			},
		}
	}
	children := []client.Object{
		readyChild(0, "prefill"),
		readyChild(0, "decode"),
		readyChild(1, "prefill"),
		readyChild(1, "decode"),
	}
	// A label-less pre-v0.10 child is slice 0 during an in-place upgrade.
	delete(children[0].GetLabels(), disaggregatedsetv1.SliceLabelKey)

	scheme := runtime.NewScheme()
	require.NoError(t, leaderworkersetv1.AddToScheme(scheme))
	k8sClient := fake.NewClientBuilder().WithScheme(scheme).WithObjects(children...).Build()
	resolver := newDisaggregatedSetReadinessResolver(k8sClient)
	selection := disaggregatedSetSelection{
		componentToRole: map[string]string{"prefill": "prefill", "decode": "decode"},
		desiredReplicas: map[string]int32{"prefill": 1, "decode": 1},
	}

	readiness, err := resolver.Resolve(t.Context(), ds, selection)
	require.NoError(t, err)
	require.True(t, readiness.Ready, readiness.Reason)
	require.Equal(t, int32(2), readiness.ComponentStatuses["prefill"].Replicas)
	require.Equal(t, int32(2), readiness.ComponentStatuses["prefill"].UpdatedReplicas)
	require.Equal(t, int32(2), ptr.Deref(readiness.ComponentStatuses["prefill"].ReadyReplicas, 0))

	t.Log("a child whose labels claim a target identity but whose name does not match is stale")
	duplicate := readyChild(0, "prefill")
	duplicate.Name += "-duplicate"
	require.NoError(t, k8sClient.Create(t.Context(), duplicate))
	readiness, err = resolver.Resolve(t.Context(), ds, selection)
	require.NoError(t, err)
	require.False(t, readiness.Ready)
	require.Contains(t, readiness.Reason, "stale role \"prefill\" child LeaderWorkerSet")
	require.Contains(t, readiness.Reason, duplicate.Name)
	require.NoError(t, k8sClient.Delete(t.Context(), duplicate))

	t.Log("one missing slice role cannot be hidden by aggregate replica counts")
	missing := children[3].(*leaderworkersetv1.LeaderWorkerSet)
	require.NoError(t, k8sClient.Delete(t.Context(), missing))
	readiness, err = resolver.Resolve(t.Context(), ds, selection)
	require.NoError(t, err)
	require.False(t, readiness.Ready)
	require.Contains(t, readiness.Reason, "slice 1")
}

func TestDisaggregatedSetPathwayObservesWorkerHash(t *testing.T) {
	const targetHash = "target12"
	dgd := &nvidiacomv1beta1.DynamoGraphDeployment{
		Spec: nvidiacomv1beta1.DynamoGraphDeploymentSpec{
			Components: []nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec{
				{ComponentName: "prefill", ComponentType: consts.ComponentTypePrefill},
				{ComponentName: "decode", ComponentType: consts.ComponentTypeDecode},
				{ComponentName: "aux-worker", ComponentType: consts.ComponentTypeWorker},
			},
		},
	}
	selection := disaggregatedSetSelection{componentToRole: map[string]string{
		"prefill": "prefill",
		"decode":  "decode",
	}}
	role := func(name string) disaggregatedsetv1.DisaggregatedRoleSpec {
		labels := map[string]string{consts.KubeLabelDynamoWorkerHash: targetHash}
		return disaggregatedsetv1.DisaggregatedRoleSpec{
			Name: name,
			LeaderWorkerSetTemplateSpec: leaderworkersetv1.LeaderWorkerSetTemplateSpec{
				Spec: leaderworkersetv1.LeaderWorkerSetSpec{
					LeaderWorkerTemplate: leaderworkersetv1.LeaderWorkerTemplate{
						LeaderTemplate: &corev1.PodTemplateSpec{ObjectMeta: metav1.ObjectMeta{Labels: maps.Clone(labels)}},
						WorkerTemplate: corev1.PodTemplateSpec{ObjectMeta: metav1.ObjectMeta{Labels: maps.Clone(labels)}},
					},
				},
			},
		}
	}
	typedDS := &disaggregatedsetv1.DisaggregatedSet{Spec: disaggregatedsetv1.DisaggregatedSetSpec{
		Roles: []disaggregatedsetv1.DisaggregatedRoleSpec{role("prefill"), role("decode")},
	}}
	dsObject, err := runtime.DefaultUnstructuredConverter.ToUnstructured(typedDS)
	require.NoError(t, err)
	ds := newDisaggregatedSetObject()
	ds.Object = dsObject
	dcds := map[string]*nvidiacomv1beta1.DynamoComponentDeployment{
		"aux-worker": {ObjectMeta: metav1.ObjectMeta{Labels: map[string]string{consts.KubeLabelDynamoWorkerHash: targetHash}}},
	}

	observed, err := disaggregatedSetPathwayObservesWorkerHash(dgd, ds, selection, dcds, targetHash)
	require.NoError(t, err)
	require.True(t, observed)

	typedDS.Spec.Roles[0].Spec.LeaderWorkerTemplate.WorkerTemplate.Labels[consts.KubeLabelDynamoWorkerHash] = "stale"
	ds.Object, err = runtime.DefaultUnstructuredConverter.ToUnstructured(typedDS)
	require.NoError(t, err)
	observed, err = disaggregatedSetPathwayObservesWorkerHash(dgd, ds, selection, dcds, targetHash)
	require.NoError(t, err)
	require.False(t, observed)
}

func TestDisaggregatedSetReadinessDiscoversOwnedChildrenByOwnerReference(t *testing.T) {
	typedDS := &disaggregatedsetv1.DisaggregatedSet{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "demo-ds",
			Namespace: disaggregatedSetUnitTestNamespace,
			UID:       "demo-ds-uid",
		},
		Spec: disaggregatedsetv1.DisaggregatedSetSpec{
			Roles: []disaggregatedsetv1.DisaggregatedRoleSpec{{Name: "prefill"}, {Name: "decode"}},
		},
	}
	dsObject, err := runtime.DefaultUnstructuredConverter.ToUnstructured(typedDS)
	require.NoError(t, err)
	ds := newDisaggregatedSetObject()
	ds.Object = dsObject
	ds.SetGroupVersionKind(disaggregatedSetGVK)
	revision := disaggregatedsetutils.ComputeRevision(typedDS.Spec.Roles)
	controllerOwner := metav1.OwnerReference{
		APIVersion: disaggregatedsetv1.GroupVersion.String(),
		Kind:       disaggregatedSetGVK.Kind,
		Name:       ds.GetName(),
		UID:        ds.GetUID(),
		Controller: ptr.To(true),
	}
	readyChild := func(role string) *leaderworkersetv1.LeaderWorkerSet {
		return &leaderworkersetv1.LeaderWorkerSet{
			ObjectMeta: metav1.ObjectMeta{
				Name:            disaggregatedsetutils.GenerateName(ds.GetName(), 0, revision, role),
				Namespace:       ds.GetNamespace(),
				Generation:      1,
				Labels:          disaggregatedsetutils.GenerateLabels(ds.GetName(), 0, revision, role),
				OwnerReferences: []metav1.OwnerReference{controllerOwner},
			},
			Spec: leaderworkersetv1.LeaderWorkerSetSpec{Replicas: ptr.To[int32](1)},
			Status: leaderworkersetv1.LeaderWorkerSetStatus{
				ObservedGeneration: 1,
				Replicas:           1,
				UpdatedReplicas:    1,
				ReadyReplicas:      1,
			},
		}
	}
	ownedMissingLabel := readyChild("removed-missing-label")
	ownedMissingLabel.Labels[disaggregatedsetv1.RevisionLabelKey] = "old-revision"
	delete(ownedMissingLabel.Labels, disaggregatedsetv1.SetNameLabelKey)
	ownedStaleLabel := readyChild("removed-stale-label")
	ownedStaleLabel.Labels[disaggregatedsetv1.RevisionLabelKey] = "old-revision"
	ownedStaleLabel.Labels[disaggregatedsetv1.SetNameLabelKey] = "old-ds-name"
	foreignLabeled := readyChild("foreign-labeled")
	foreignLabeled.OwnerReferences = nil
	foreignUnlabeled := readyChild("foreign-unlabeled")
	foreignUnlabeled.OwnerReferences = nil
	delete(foreignUnlabeled.Labels, disaggregatedsetv1.SetNameLabelKey)

	scheme := runtime.NewScheme()
	require.NoError(t, leaderworkersetv1.AddToScheme(scheme))
	objects := []client.Object{
		readyChild("prefill"),
		readyChild("decode"),
		ownedMissingLabel,
		ownedStaleLabel,
		foreignLabeled,
		foreignUnlabeled,
	}
	resolver := newDisaggregatedSetReadinessResolver(fake.NewClientBuilder().WithScheme(scheme).WithObjects(objects...).Build())
	selection := disaggregatedSetSelection{
		componentToRole: map[string]string{"prefill": "prefill", "decode": "decode"},
		desiredReplicas: map[string]int32{"prefill": 1, "decode": 1},
	}

	readiness, err := resolver.Resolve(t.Context(), ds, selection)
	require.NoError(t, err)
	require.False(t, readiness.Ready)
	require.Contains(t, readiness.Reason, ownedMissingLabel.Name)
	require.Contains(t, readiness.Reason, ownedStaleLabel.Name)
	require.NotContains(t, readiness.Reason, foreignLabeled.Name)
	require.NotContains(t, readiness.Reason, foreignUnlabeled.Name)
}

func TestDisaggregatedSetReadinessDoesNotTrustChildIdentityLabels(t *testing.T) {
	typedDS := &disaggregatedsetv1.DisaggregatedSet{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "demo-ds",
			Namespace: disaggregatedSetUnitTestNamespace,
			UID:       "demo-ds-uid",
		},
		Spec: disaggregatedsetv1.DisaggregatedSetSpec{
			Roles: []disaggregatedsetv1.DisaggregatedRoleSpec{{Name: "prefill"}, {Name: "decode"}},
		},
	}
	dsObject, err := runtime.DefaultUnstructuredConverter.ToUnstructured(typedDS)
	require.NoError(t, err)
	ds := newDisaggregatedSetObject()
	ds.Object = dsObject
	ds.SetGroupVersionKind(disaggregatedSetGVK)
	revision := disaggregatedsetutils.ComputeRevision(typedDS.Spec.Roles)
	controllerOwner := metav1.OwnerReference{
		APIVersion: disaggregatedsetv1.GroupVersion.String(),
		Kind:       disaggregatedSetGVK.Kind,
		Name:       ds.GetName(),
		UID:        ds.GetUID(),
		Controller: ptr.To(true),
	}
	readyChild := func(name, role string) *leaderworkersetv1.LeaderWorkerSet {
		return &leaderworkersetv1.LeaderWorkerSet{
			ObjectMeta: metav1.ObjectMeta{
				Name:            name,
				Namespace:       ds.GetNamespace(),
				Generation:      1,
				Labels:          disaggregatedsetutils.GenerateLabels(ds.GetName(), 0, revision, role),
				OwnerReferences: []metav1.OwnerReference{controllerOwner},
			},
			Spec: leaderworkersetv1.LeaderWorkerSetSpec{Replicas: ptr.To[int32](1)},
			Status: leaderworkersetv1.LeaderWorkerSetStatus{
				ObservedGeneration: 1,
				Replicas:           1,
				UpdatedReplicas:    1,
				ReadyReplicas:      1,
			},
		}
	}
	decodeName := disaggregatedsetutils.GenerateName(ds.GetName(), 0, revision, "decode")
	decode := readyChild(decodeName, "decode")
	decode.Labels[disaggregatedsetv1.RoleLabelKey] = "stale-role-label"
	forgedPrefill := readyChild("old-prefill-workload", "prefill")

	scheme := runtime.NewScheme()
	require.NoError(t, leaderworkersetv1.AddToScheme(scheme))
	resolver := newDisaggregatedSetReadinessResolver(fake.NewClientBuilder().WithScheme(scheme).WithObjects(
		decode,
		forgedPrefill,
	).Build())
	selection := disaggregatedSetSelection{
		componentToRole: map[string]string{"prefill": "prefill", "decode": "decode"},
		desiredReplicas: map[string]int32{"prefill": 1, "decode": 1},
	}

	readiness, err := resolver.Resolve(t.Context(), ds, selection)
	require.NoError(t, err)
	require.False(t, readiness.Ready)
	require.Contains(t, readiness.Reason, "prefill role \"prefill\" slice 0 has 0 target LeaderWorkerSets")
	require.Contains(t, readiness.Reason, forgedPrefill.Name)
}

func TestDisaggregatedSetWatchMapperMapsNonzeroSlice(t *testing.T) {
	ds := newDisaggregatedSetObject()
	ds.SetName("demo-ds")
	ds.SetNamespace(disaggregatedSetUnitTestNamespace)
	ds.SetUID("demo-ds-uid")
	ds.SetOwnerReferences([]metav1.OwnerReference{{
		APIVersion: nvidiacomv1beta1.GroupVersion.String(),
		Kind:       dynamoGraphDeploymentKind,
		Name:       "demo",
		Controller: ptr.To(true),
	}})
	scheme := runtime.NewScheme()
	scheme.AddKnownTypeWithName(disaggregatedSetGVK, &unstructured.Unstructured{})
	scheme.AddKnownTypeWithName(disaggregatedSetGVK.GroupVersion().WithKind("DisaggregatedSetList"), &unstructured.UnstructuredList{})
	watches := newDisaggregatedSetWatchSetup(fake.NewClientBuilder().WithScheme(scheme).WithObjects(ds).Build())
	child := &leaderworkersetv1.LeaderWorkerSet{ObjectMeta: metav1.ObjectMeta{
		Name:      "demo-ds-1-abc12345-prefill",
		Namespace: disaggregatedSetUnitTestNamespace,
		Labels: map[string]string{
			disaggregatedsetv1.SetNameLabelKey: "stale-ds-name",
			disaggregatedsetv1.SliceLabelKey:   "1",
		},
		OwnerReferences: []metav1.OwnerReference{{
			APIVersion: disaggregatedsetv1.GroupVersion.String(),
			Kind:       disaggregatedSetGVK.Kind,
			Name:       ds.GetName(),
			UID:        ds.GetUID(),
			Controller: ptr.To(true),
		}},
	}}

	require.Equal(t, []ctrl.Request{{NamespacedName: types.NamespacedName{
		Name: "demo", Namespace: disaggregatedSetUnitTestNamespace,
	}}}, watches.mapChildLWSToDGD(t.Context(), child))

	child.Labels = nil
	require.Len(t, watches.mapChildLWSToDGD(t.Context(), child), 1, "owner reference must remain authoritative when the label is missing")
	child.OwnerReferences[0].UID = "foreign-ds-uid"
	require.Empty(t, watches.mapChildLWSToDGD(t.Context(), child), "a name-reused foreign owner must not enqueue the DGD")
}

func TestDisaggregatedSetStatusReadinessWaitsForRemovedRoleChildren(t *testing.T) {
	ds := newDisaggregatedSetObject()
	ds.SetName("demo")
	ds.SetNamespace(disaggregatedSetUnitTestNamespace)
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
			Namespace: disaggregatedSetUnitTestNamespace,
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
	resolver := newDisaggregatedSetReadinessResolver(fake.NewClientBuilder().WithScheme(scheme).WithObjects(removed).Build())
	selection := disaggregatedSetSelection{
		componentToRole: map[string]string{"prefill": "prefill", "decode": "decode"},
		desiredReplicas: map[string]int32{"prefill": 1, "decode": 1},
	}

	readiness, err := resolver.Resolve(t.Context(), ds, selection)
	require.NoError(t, err)
	require.False(t, readiness.Ready)
	require.Contains(t, readiness.Reason, removed.Name)
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
