//go:build !clustertest

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package enginegroup

import (
	"encoding/json"
	"fmt"
	"net/http"
	"testing"

	api "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
	domain "github.com/ai-dynamo/dynamo/deploy/operator/internal/enginegroup"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/podcache"
	grovecommon "github.com/ai-dynamo/grove/operator/api/common"
	grove "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
)

func TestVLLMRuntimeResolutionBeforeAPIReadiness(t *testing.T) {
	t.Log("construct the DGD, PCS, PCSG and exact member clique from one native declaration")
	kube, group := newVLLMRuntimeTestWorld(t)
	capacity, err := ResolveVLLMGroveCapacity(t.Context(), kube, group)
	require.NoError(t, err)
	transport := &vllmNoNativeRequests{}
	httpClient := &http.Client{Transport: transport}
	_, err = ResolveVLLMMembership(t.Context(), capacity, httpClient)
	require.NoError(t, err)
	assert.Empty(t, transport.hosts)

	t.Log("persist launch identities before publication and use the watched Group observation to resume")
	require.NoError(t, capacity.ReconcileAllocations(t.Context()))
	require.NoError(t, kube.Get(t.Context(), client.ObjectKeyFromObject(group), group))
	capacity, err = ResolveVLLMGroveCapacity(t.Context(), kube, group)
	require.NoError(t, err)
	require.NoError(t, capacity.ReconcileAllocations(t.Context()))

	t.Log("resolve direct native clients while every Pod is NotReady and every engine API is unreachable")
	membership, err := ResolveVLLMMembership(t.Context(), capacity, httpClient)
	require.NoError(t, err)
	assert.Equal(t, int32(2), membership.MinDataParallelSize)
	assert.Equal(t, int32(4), membership.MaxDataParallelSize)
	processes := membership.Processes
	require.Len(t, processes, 2)
	for rank, process := range processes {
		assert.Equal(t, int32(rank), process.Rank)
		assert.Equal(t, domain.ReplicaID(fmt.Sprintf("replica-%d", rank)), process.Membership.ReplicaID)
		assert.Equal(t, domain.NativeMemberID(fmt.Sprintf("dp-%d", rank)), process.Membership.Members[0].ID)
		assert.Equal(t, domain.RuntimeIncarnationID(fmt.Sprintf("pod-%d-uid", rank)), process.Membership.Members[0].RuntimeIncarnation)
	}
	assert.Empty(t, transport.hosts, "runtime resolution must not wait for native API readiness")
	_, err = processes[1].Client.Observe(t.Context())
	require.Error(t, err)
	assert.Equal(t, []string{"10.0.0.11:18000"}, transport.hosts, "control is sent to the exact Pod, never a load-balanced Service")

	t.Log("changing the live target does not reinterpret the immutable launch declaration")
	group.Spec.Replicas = 4
	require.NoError(t, kube.Update(t.Context(), group))
	capacity, err = ResolveVLLMGroveCapacity(t.Context(), kube, group)
	require.NoError(t, err)
	assert.Equal(t, int32(2), capacity.Source.InitialReplicas)
	assert.Equal(t, int32(4), group.Spec.Replicas)
}

func TestVLLMNativeProcessFencesOnlyTheEngineContainer(t *testing.T) {
	t.Log("project a physical allocation with an independently restarted telemetry container")
	pod := podcache.Project(&corev1.Pod{UID: "pod-uid",
		Status: corev1.PodStatus{ContainerStatuses: []corev1.ContainerStatus{{Name: consts.MainContainerName}, {Name: "telemetry", RestartCount: 2}}}})
	members, err := resolveVLLMLaunchMembers(pod, "slot-0", 1)
	require.NoError(t, err)
	assert.Equal(t, domain.RuntimeIncarnationID("pod-uid"), members[0].RuntimeIncarnation)

	t.Log("the engine container itself cannot restart under the same Pod UID")
	pod.Status.ContainerStatuses = []corev1.ContainerStatus{{Name: consts.MainContainerName, RestartCount: 1}}
	_, err = resolveVLLMLaunchMembers(pod, "slot-0", 1)
	require.ErrorIs(t, err, domain.ErrRecoveryRequired)
}

func TestVLLMRuntimeResolutionRefusesChangedAuthority(t *testing.T) {
	for _, scenario := range []string{"parent-uid", "profile", "clique-uid", "pod-ip", "container-restart", "binding-uid", "catalog-loss"} {
		t.Run(scenario, func(t *testing.T) {
			t.Log("start from exact, persisted launch bindings in the production Pod-cache representation")
			kube, group := newVLLMRuntimeTestWorld(t)
			capacity, err := ResolveVLLMGroveCapacity(t.Context(), kube, group)
			require.NoError(t, err)
			require.NoError(t, capacity.ReconcileAllocations(t.Context()))
			require.NoError(t, kube.Get(t.Context(), client.ObjectKeyFromObject(group), group))
			capacity, err = ResolveVLLMGroveCapacity(t.Context(), kube, group)
			require.NoError(t, err)
			require.NoError(t, capacity.ReconcileAllocations(t.Context()))
			pod := &corev1.Pod{}
			require.NoError(t, kube.Get(t.Context(), client.ObjectKey{Namespace: group.Namespace, Name: "allocation-1"}, pod))

			t.Log("change one authority and refuse to manufacture a new process lifetime")
			switch scenario {
			case "parent-uid":
				group.OwnerReferences[0].UID = "replacement-parent"
			case "profile":
				group.Annotations[consts.KubeAnnotationDynamoEngineGroupProfile] = "other-profile"
			case "clique-uid":
				group.Annotations[consts.KubeAnnotationDynamoEngineGroupPodCliqueUID] = "replacement-clique"
			case "catalog-loss":
				delete(group.Annotations, consts.KubeAnnotationDynamoVLLMCapacityBindings)
			case "pod-ip":
				pod.Status.PodIP = ""
			case "container-restart":
				pod.Status.ContainerStatuses = []corev1.ContainerStatus{{Name: "main", RestartCount: 1}}
			case "binding-uid":
				bindings, err := capacity.capacityBindings()
				require.NoError(t, err)
				bindings.Slots["slot-1"] = "another-pod"
				encoded, err := json.Marshal(bindings)
				require.NoError(t, err)
				group.Annotations[consts.KubeAnnotationDynamoVLLMCapacityBindings] = string(encoded)
			}
			require.NoError(t, kube.Update(t.Context(), group))
			status := pod.Status.DeepCopy()
			require.NoError(t, kube.Update(t.Context(), pod))
			pod.Status = *status
			require.NoError(t, kube.Status().Update(t.Context(), pod))
			capacity, err = ResolveVLLMGroveCapacity(t.Context(), kube, group)
			if err == nil {
				_, err = ResolveVLLMMembership(t.Context(), capacity, &http.Client{Transport: &vllmNoNativeRequests{}})
			}
			require.Error(t, err)
			if scenario == "pod-ip" {
				assert.ErrorIs(t, err, ErrRuntimeUnavailable)
			}
		})
	}
}

func TestVLLMRendezvousRefusesReplacementRoot(t *testing.T) {
	t.Log("persist initial slot authority before any engine process has become Ready")
	kube, group := newVLLMRuntimeTestWorld(t)
	capacity, err := ResolveVLLMGroveCapacity(t.Context(), kube, group)
	require.NoError(t, err)
	require.NoError(t, capacity.ReconcileAllocations(t.Context()))
	require.NoError(t, kube.Get(t.Context(), client.ObjectKeyFromObject(group), group))

	t.Log("simulate lifecycle-owner-confirmed teardown followed by a new Pod in the same root slot")
	root := &corev1.Pod{}
	require.NoError(t, kube.Get(t.Context(), client.ObjectKey{Namespace: group.Namespace, Name: "allocation-0"}, root))
	root.Finalizers = nil
	require.NoError(t, kube.Update(t.Context(), root))
	require.NoError(t, kube.Delete(t.Context(), root))
	root.UID, root.ResourceVersion = "replacement-root", ""
	root.Annotations = nil
	root.Status.PodIP = "10.0.0.30"
	require.NoError(t, kube.Create(t.Context(), root))

	t.Log("never redirect new launches to the replacement, even before native API observation")
	_, err = ResolveVLLMGroveCapacity(t.Context(), kube, group)
	require.ErrorIs(t, err, domain.ErrRecoveryRequired)
	assert.Empty(t, root.Annotations)
}

func TestEngineGroupParentWatchUsesExactOwners(t *testing.T) {
	t.Log("use the same parent chain consumed by native runtime resolution")
	kube, group := newVLLMRuntimeTestWorld(t)
	reconciler := &Reconciler{Client: kube}
	dgd := &api.DynamoGraphDeployment{}
	require.NoError(t, kube.Get(t.Context(), client.ObjectKey{Namespace: group.Namespace, Name: "deployment"}, dgd))
	pcs := &grove.PodCliqueSet{}
	require.NoError(t, kube.Get(t.Context(), client.ObjectKey{Namespace: group.Namespace, Name: "deployment"}, pcs))
	world := &grove.PodCliqueScalingGroup{}
	require.NoError(t, kube.Get(t.Context(), client.ObjectKey{Namespace: group.Namespace, Name: "world"}, world))

	t.Log("each asynchronous dependency can wake the logical world without inspecting the event edge")
	for _, parent := range []client.Object{dgd, pcs, world} {
		requests := reconciler.engineGroupsForParent(t.Context(), parent)
		require.Len(t, requests, 1)
		assert.Equal(t, client.ObjectKeyFromObject(group), requests[0].NamespacedName)
	}

	t.Log("same-name foreign parent incarnations cannot enqueue the old world's authority")
	dgd.UID = "another-dgd"
	assert.Empty(t, reconciler.engineGroupsForParent(t.Context(), dgd))
	world.OwnerReferences[0].UID = "another-pcs"
	assert.Empty(t, reconciler.engineGroupsForParent(t.Context(), world))
}

func newVLLMRuntimeTestWorld(t *testing.T) (client.Client, *api.DynamoGraphDeploymentEngineGroup) {
	t.Helper()
	return newVLLMRuntimeTestWorldForTransport(t, false)
}

func newVLLMRuntimeTestWorldForTransport(t *testing.T, worker bool) (client.Client, *api.DynamoGraphDeploymentEngineGroup) {
	t.Helper()
	scheme := runtime.NewScheme()
	require.NoError(t, api.AddToScheme(scheme))
	require.NoError(t, grove.AddToScheme(scheme))
	require.NoError(t, corev1.AddToScheme(scheme))
	component := api.DynamoComponentDeploymentSharedSpec{
		ComponentName: "worker", ComponentType: api.ComponentTypeWorker, Replicas: new(int32(1)),
		EngineGroup: &api.ComponentEngineGroupSpec{InitialSize: 2},
		PodTemplate: &corev1.PodTemplateSpec{Spec: corev1.PodSpec{RestartPolicy: corev1.RestartPolicyNever,
			ReadinessGates: []corev1.PodReadinessGate{{ConditionType: corev1.PodConditionType(consts.EngineGroupRetainedReadinessGate)}}, Containers: []corev1.Container{{
				Name: consts.MainContainerName, Image: "vllm:native", Command: []string{"vllm", "serve", "model"},
				Args: []string{
					"--elastic-ep-auto-bootstrap", "--elastic-ep-initial-dp-size=2", "--data-parallel-size-local=1", "--data-parallel-external-lb",
					"--data-parallel-rank=$(DYNAMO_RANK)", "--data-parallel-address=$(DYNAMO_LEADER_ADDRESS)",
					"--tensor-parallel-size=1", "--distributed-executor-backend=mp", "--elastic-ep-max-dp-size=4",
					"--enable-expert-parallel", "--enable-eplb", "--enable-elastic-ep", "--port=18000", "--data-parallel-rpc-port=29550",
				},
				Env:       []corev1.EnvVar{{Name: "VLLM_ENABLE_V1_MULTIPROCESSING", Value: "1"}, {Name: "VLLM_USE_RUST_FRONTEND", Value: "0"}, {Name: "VLLM_ELASTIC_EP_DRAIN_REQUESTS", Value: "1"}},
				Resources: corev1.ResourceRequirements{Limits: corev1.ResourceList{consts.KubeResourceGPUNvidia: resource.MustParse("1")}},
			}}}},
	}
	if worker {
		main := &component.PodTemplate.Spec.Containers[0]
		main.Command = []string{"python3", "-m", "dynamo.vllm", "--model", "model"}
		main.Args = append(main.Args[:len(main.Args)-2], "--data-parallel-rpc-port=29550")
		main.Env = append(main.Env, corev1.EnvVar{Name: "DYN_SYSTEM_PORT", Value: "18000"}, corev1.EnvVar{Name: "DYNAMO_ENGINE_GROUP", Value: "1"},
			corev1.EnvVar{Name: "DYNAMO_POD_UID", ValueFrom: &corev1.EnvVarSource{FieldRef: &corev1.ObjectFieldSelector{FieldPath: "metadata.uid"}}})
	}
	dgd := &api.DynamoGraphDeployment{Name: "deployment", Namespace: "test", UID: "dgd-uid",
		Spec: api.DynamoGraphDeploymentSpec{BackendFramework: string(dynamo.BackendFrameworkVLLM), Components: []api.DynamoComponentDeploymentSharedSpec{component}}}
	_, profile, err := dynamo.ResolveVLLMComponentEngineGroup(&component, dgd.Spec.Env)
	require.NoError(t, err)
	group := dynamo.RenderVLLMEngineGroup(dgd, &component, 0, profile)
	group.UID, group.Finalizers = "group-uid", []string{engineGroupFinalizer}
	pcs := &grove.PodCliqueSet{
		Name: dynamo.PCSNameForDGD(dgd, (*api.DynamoComponentDeploymentSharedSpec).ManagedByExternalController), Namespace: group.Namespace, UID: "pcs-uid",
		OwnerReferences: []metav1.OwnerReference{*metav1.NewControllerRef(dgd, api.GroupVersion.WithKind("DynamoGraphDeployment"))}}
	world := &grove.PodCliqueScalingGroup{Name: "world", Namespace: group.Namespace, UID: "world-uid",
		OwnerReferences: []metav1.OwnerReference{*metav1.NewControllerRef(pcs, grove.SchemeGroupVersion.WithKind("PodCliqueSet"))}}
	template := component.PodTemplate.DeepCopy()
	clique := &grove.PodClique{Name: "members", Namespace: group.Namespace, UID: "clique-uid",
		Labels:          map[string]string{consts.KubeLabelDynamoEngineGroup: group.Name, grovecommon.LabelPodCliqueScalingGroupReplicaIndex: "0"},
		OwnerReferences: []metav1.OwnerReference{*metav1.NewControllerRef(world, grove.SchemeGroupVersion.WithKind("PodCliqueScalingGroup"))},
		Spec:            grove.PodCliqueSpec{Replicas: 2, MinAvailable: new(int32(2)), PodSpec: template.Spec}}
	group.Annotations[consts.KubeAnnotationDynamoEngineGroupPodClique] = clique.Name
	group.Annotations[consts.KubeAnnotationDynamoEngineGroupPodCliqueUID] = string(clique.UID)
	objects := []client.Object{dgd, group, pcs, world, clique}
	for rank := range int32(2) {
		pod := newVLLMGroveLaunchPod(clique, rank)
		pod.UID = types.UID(fmt.Sprintf("pod-%d-uid", rank))
		pod.Status.PodIP = fmt.Sprintf("10.0.0.%d", 10+rank)
		objects = append(objects, podcache.Project(pod))
	}
	kube := fake.NewClientBuilder().WithScheme(scheme).WithObjects(objects...).WithStatusSubresource(&corev1.Pod{}).Build()
	return kube, group
}

type vllmNoNativeRequests struct{ hosts []string }

func (r *vllmNoNativeRequests) RoundTrip(request *http.Request) (*http.Response, error) {
	r.hosts = append(r.hosts, request.URL.Host)
	return nil, fmt.Errorf("native API is not ready")
}
