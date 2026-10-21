/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package controller

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strconv"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	rbacv1 "k8s.io/api/rbac/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/utils/ptr"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"

	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	commoncontroller "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/features"
	disaggregatedsetv1 "sigs.k8s.io/lws/api/disaggregatedset/v1"
	leaderworkersetv1 "sigs.k8s.io/lws/api/leaderworkerset/v1"
)

// rolloutProbeServer implements the released frontend's worker-attribution response shape.
type rolloutProbeServer struct {
	rank        uint32
	mode        atomic.Int32
	chat        atomic.Int32
	completions atomic.Int32
}

func (s *rolloutProbeServer) ServeHTTP(w http.ResponseWriter, request *http.Request) {
	var body struct {
		Model     string `json:"model"`
		MaxTokens int    `json:"max_tokens"`
		Stream    bool   `json:"stream"`
		Nvext     struct {
			InstanceID    uint64   `json:"backend_instance_id"`
			PrefillID     uint64   `json:"prefill_worker_id"`
			DPRank        *uint32  `json:"dp_rank"`
			PrefillDPRank *uint32  `json:"prefill_dp_rank"`
			Fields        []string `json:"extra_fields"`
		} `json:"nvext"`
	}
	if err := json.NewDecoder(request.Body).Decode(&body); err != nil || request.Method != http.MethodPost ||
		body.Model != "test-model" || body.MaxTokens != 1 || body.Stream || (body.Nvext.InstanceID != 42 && body.Nvext.PrefillID != 42) ||
		len(body.Nvext.Fields) != 1 || body.Nvext.Fields[0] != "worker_id" {
		http.Error(w, "invalid inference probe", http.StatusBadRequest)
		return
	}

	// Exact worker targeting must include the rank for only the phase being probed.
	rank, other := body.Nvext.DPRank, body.Nvext.PrefillDPRank
	if body.Nvext.PrefillID == 42 {
		rank, other = body.Nvext.PrefillDPRank, body.Nvext.DPRank
	}
	if rank == nil || *rank != s.rank || other != nil {
		http.Error(w, "invalid probe DP rank", http.StatusBadRequest)
		return
	}

	if request.URL.Path == "/v1/chat/completions" {
		s.chat.Add(1)
	} else if request.URL.Path == "/v1/completions" {
		s.completions.Add(1)
	} else {
		http.NotFound(w, request)
		return
	}

	// Simulate admission delay, absent attribution, and an ignored targeting hint.
	switch s.mode.Load() {
	case 1:
		w.WriteHeader(http.StatusNotFound)
	case 2:
		_, _ = fmt.Fprint(w, `{"choices":[{"finish_reason":"length"}],"usage":{"completion_tokens":1}}`)
	case 3:
		_, _ = fmt.Fprint(w, `{"choices":[{"finish_reason":"length"}],"usage":{"completion_tokens":1},"nvext":{"worker_id":{"decode_worker_id":41}}}`)
	default:
		phase := "decode_worker_id"
		if body.Nvext.PrefillID == 42 {
			phase = "prefill_worker_id"
		}
		_, _ = fmt.Fprintf(w, `{"choices":[{"finish_reason":"length"}],"usage":{"completion_tokens":1},"nvext":{"worker_id":{"%s":42}}}`, phase)
	}
}

func TestBuildRollingUpdateContext_ReplacementInference(t *testing.T) {
	t.Log("Build a constrained two-replica rollout with one old and one Ready replacement")
	dgd := createTestDGD("inference-gate", map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
		"worker":   {ComponentType: consts.ComponentTypeWorker, Replicas: ptr.To(int32(2))},
		"frontend": {ComponentType: consts.ComponentTypeFrontend},
		"standby":  {ComponentType: consts.ComponentTypeFrontend, Replicas: ptr.To(int32(0))},
	})
	dgd.UID = "graph-uid"
	dgd.Annotations = map[string]string{consts.KubeAnnotationDynamoDiscoveryBackend: "kubernetes"}
	dgd.Spec.Annotations = map[string]string{
		KubeAnnotationDeploymentRollingUpdateMaxSurge:       "0",
		KubeAnnotationDeploymentRollingUpdateMaxUnavailable: "1",
	}
	hashes, err := desiredWorkerHashes(dgd)
	require.NoError(t, err)
	newName := dynamo.GetDCDResourceName(dgd, "worker", hashes.v2)
	oldDCD := &nvidiacomv1beta1.DynamoComponentDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "inference-gate-worker-old", Namespace: dgd.Namespace,
			Labels: map[string]string{consts.KubeLabelDynamoGraphDeploymentName: dgd.Name, consts.KubeLabelDynamoWorkerHash: "old"}},
		Spec: nvidiacomv1beta1.DynamoComponentDeploymentSpec{DynamoComponentDeploymentSharedSpec: nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec{
			ComponentName: "worker", ComponentType: consts.ComponentTypeWorker, Replicas: ptr.To(int32(1)),
		}},
		Status: nvidiacomv1beta1.DynamoComponentDeploymentStatus{Component: &nvidiacomv1beta1.ComponentReplicaStatus{
			Replicas: 1, ReadyReplicas: ptr.To(int32(1)), AvailableReplicas: ptr.To(int32(1)),
		}},
	}
	require.NoError(t, ctrl.SetControllerReference(dgd, oldDCD, testScheme))
	newDCD := oldDCD.DeepCopy()
	newDCD.Name = newName
	newDCD.Labels[consts.KubeLabelDynamoWorkerHash] = hashes.v2
	worker := &corev1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "new-worker", Namespace: dgd.Namespace, UID: "new-pod-uid",
		Labels: map[string]string{consts.KubeLabelDynamoGraphDeploymentName: dgd.Name, consts.KubeLabelDynamoComponent: "worker",
			consts.KubeLabelDynamoSelector: newName}},
		Status: corev1.PodStatus{Conditions: []corev1.PodCondition{{Type: corev1.PodReady, Status: corev1.ConditionTrue}}},
	}
	metadata := &unstructured.Unstructured{Object: map[string]any{
		"apiVersion": "nvidia.com/v1alpha1", "kind": "DynamoWorkerMetadata",
		"metadata": map[string]any{"name": "worker-container", "namespace": dgd.Namespace},
		"spec": map[string]any{"data": map[string]any{"model_cards": map[string]any{
			"new/backend/generate/42": map[string]any{"instance_id": int64(42), "card_json": map[string]any{
				"display_name": "test-model", "model_type": "Chat | Completions", "worker_type": "aggregated",
				"runtime_config": map[string]any{"data_parallel_start_rank": int64(4), "data_parallel_size": int64(2)},
			}},
		}}},
	}}
	metadata.SetOwnerReferences([]metav1.OwnerReference{{APIVersion: "v1", Kind: "Pod", Name: worker.Name, UID: worker.UID, Controller: ptr.To(false)}})

	t.Log("Serve two independent frontend endpoints so a single successful replica cannot release the gate")
	first, second := &rolloutProbeServer{rank: 4}, &rolloutProbeServer{rank: 4}
	server1, server2 := httptest.NewServer(first), httptest.NewServer(second)
	defer server1.Close()
	defer server2.Close()
	frontend1 := rolloutTestFrontendPod(t, dgd, "frontend-1", server1.URL)
	frontend2 := rolloutTestFrontendPod(t, dgd, "frontend-2", server2.URL)
	r := createTestReconcilerWithStatus(dgd, withObjects(oldDCD, newDCD, worker, metadata, frontend1, frontend2))

	t.Log("Withhold final old-worker deletion for 404, missing attribution, and old-worker responses")
	for _, mode := range []int32{1, 2, 3} {
		second.mode.Store(mode)
		rollout, pending, err := r.buildRollingUpdateContext(context.Background(), dgd)
		require.NoError(t, err)
		require.True(t, pending)
		require.Equal(t, int32(1), rollout.OldWorkerReplicaTargetsByDCD[oldDCD.Name], "mode %d", mode)
		require.Equal(t, int32(1), rollout.NewWorkerReplicaTargetsByComponent["worker"], "must respect maxSurge=0")
	}

	t.Log("Release the last old replica only when both frontends execute on the replacement")
	second.mode.Store(0)
	rollout, pending, err := r.buildRollingUpdateContext(context.Background(), dgd)
	require.NoError(t, err)
	require.False(t, pending)
	require.Equal(t, int32(0), rollout.OldWorkerReplicaTargetsByDCD[oldDCD.Name])
	require.Positive(t, first.chat.Load())
	require.Positive(t, first.completions.Load())
	require.Positive(t, second.chat.Load())
	require.Positive(t, second.completions.Load())

	t.Log("A zero-replica frontend with a remaining Ready Pod must still prove replacement inference")
	spare := &rolloutProbeServer{rank: 4}
	spare.mode.Store(1)
	spareServer := httptest.NewServer(spare)
	defer spareServer.Close()
	sparePod := rolloutTestFrontendPod(t, dgd, "standby-1", spareServer.URL)
	sparePod.Labels[consts.KubeLabelDynamoComponent] = "standby"
	require.NoError(t, r.Create(t.Context(), sparePod))
	rollout, pending, err = r.buildRollingUpdateContext(t.Context(), dgd)
	require.NoError(t, err)
	require.True(t, pending)
	require.Equal(t, int32(1), rollout.OldWorkerReplicaTargetsByDCD[oldDCD.Name])
	require.Positive(t, spare.chat.Load())
	require.NoError(t, r.Delete(t.Context(), sparePod))

	t.Log("Absent default-count and positive-count frontends still block retirement")
	for _, replicas := range []*int32{nil, ptr.To(int32(1))} {
		requiredDGD := dgd.DeepCopy()
		for i := range requiredDGD.Spec.Components {
			if requiredDGD.Spec.Components[i].ComponentName == "standby" {
				requiredDGD.Spec.Components[i].Replicas = replicas
			}
		}
		rollout, pending, err = r.buildRollingUpdateContext(t.Context(), requiredDGD)
		require.NoError(t, err)
		require.True(t, pending)
		require.Equal(t, int32(1), rollout.OldWorkerReplicaTargetsByDCD[oldDCD.Name])
	}

	t.Log("A non-Kubernetes rollout never requests inference retries even when a frontend would fail")
	legacyDGD := dgd.DeepCopy()
	legacyDGD.Annotations[consts.KubeAnnotationDynamoDiscoveryBackend] = "etcd"
	second.mode.Store(1)
	rollout, pending, err = r.buildRollingUpdateContext(t.Context(), legacyDGD)
	require.NoError(t, err)
	require.False(t, pending)

	t.Log("TLS, disabled nvext, and opaque environments keep legacy retirement without probes")
	for _, container := range []corev1.Container{
		{Name: consts.MainContainerName, Env: []corev1.EnvVar{{Name: "DYN_TLS_CERT_PATH", Value: "/tls/server.crt"}}},
		{Name: consts.MainContainerName, Env: []corev1.EnvVar{{Name: "DYN_TLS_CLIENT_CA_CERT_PATH", ValueFrom: &corev1.EnvVarSource{SecretKeyRef: &corev1.SecretKeySelector{LocalObjectReference: corev1.LocalObjectReference{Name: "tls"}, Key: "ca"}}}}},
		{Name: consts.RuntimeContainerName, Args: []string{"--tls-cert-path=/tls/server.crt", "--tls-key-path=/tls/server.key"}},
		{Name: consts.MainContainerName, EnvFrom: []corev1.EnvFromSource{{ConfigMapRef: &corev1.ConfigMapEnvSource{LocalObjectReference: corev1.LocalObjectReference{Name: "frontend-env"}}}}},
		{Name: consts.RuntimeContainerName, Env: []corev1.EnvVar{{Name: "DYN_DISABLE_FRONTEND_NVEXT", Value: " YES "}}},
		{Name: consts.MainContainerName, Env: []corev1.EnvVar{{Name: "DYN_DISABLE_FRONTEND_NVEXT", ValueFrom: &corev1.EnvVarSource{ConfigMapKeyRef: &corev1.ConfigMapKeySelector{LocalObjectReference: corev1.LocalObjectReference{Name: "frontend-env"}, Key: "disable-nvext"}}}}},
	} {
		configured := frontend2.DeepCopy()
		require.NoError(t, r.Get(t.Context(), client.ObjectKeyFromObject(frontend2), configured))
		configured.Spec = frontend2.DeepCopy().Spec
		if container.Name == consts.RuntimeContainerName {
			configured.Spec.InitContainers = []corev1.Container{container}
		} else {
			configured.Spec.Containers[0].Env = container.Env
			configured.Spec.Containers[0].EnvFrom = container.EnvFrom
		}
		require.NoError(t, r.Update(t.Context(), configured))
		before := first.chat.Load() + second.chat.Load()
		rollout, pending, err = r.buildRollingUpdateContext(t.Context(), dgd)
		require.NoError(t, err)
		require.False(t, pending)
		require.Zero(t, rollout.OldWorkerReplicaTargetsByDCD[oldDCD.Name])
		require.Equal(t, before, first.chat.Load()+second.chat.Load())
	}

	t.Log("Values that do not disable nvext keep the gate closed on missing attribution")
	second.mode.Store(2)
	for _, value := range []string{"false", "unknown"} {
		configured := frontend2.DeepCopy()
		require.NoError(t, r.Get(t.Context(), client.ObjectKeyFromObject(frontend2), configured))
		configured.Spec = frontend2.DeepCopy().Spec
		configured.Spec.Containers[0].Env = []corev1.EnvVar{{Name: "DYN_DISABLE_FRONTEND_NVEXT", Value: value}}
		require.NoError(t, r.Update(t.Context(), configured))
		rollout, pending, err = r.buildRollingUpdateContext(t.Context(), dgd)
		require.NoError(t, err)
		require.True(t, pending, "value %q", value)
		require.Equal(t, int32(1), rollout.OldWorkerReplicaTargetsByDCD[oldDCD.Name])
	}
	restored := frontend2.DeepCopy()
	require.NoError(t, r.Get(t.Context(), client.ObjectKeyFromObject(frontend2), restored))
	restored.Spec = frontend2.DeepCopy().Spec
	require.NoError(t, r.Update(t.Context(), restored))

	t.Log("Drive the production program: retry a blocked gate after ten seconds, then stop polling after success")
	programDGD := dgd.DeepCopy()
	programDGD.Annotations[consts.AnnotationCurrentWorkerHashV2] = "old"
	programDGD.Status.RollingUpdate = &nvidiacomv1beta1.RollingUpdateStatus{Phase: nvidiacomv1beta1.RollingUpdatePhaseInProgress}
	programOwner := createTestDGDReconcilerWithStatus(programDGD, withObjects(oldDCD, newDCD, worker, metadata, frontend1, frontend2))
	programOwner.Config.Namespace.Restricted = dgd.Namespace
	require.NoError(t, rbacv1.AddToScheme(programOwner.Scheme()))
	program := programOwner.newComponentProgram()
	programResult, err := program.Reconcile(t.Context(), workloadProgramRequest{DGD: programDGD})
	require.NoError(t, err)
	require.Equal(t, 10*time.Second, programResult.RequeueAfter)
	require.True(t, isRollingUpdateInProgress(&programResult.Status))
	second.mode.Store(0)
	programResult, err = program.Reconcile(t.Context(), workloadProgramRequest{DGD: programDGD})
	require.NoError(t, err)
	require.Zero(t, programResult.RequeueAfter)
	require.True(t, isRollingUpdateInProgress(&programResult.Status), "rollout phase alone must not schedule polling")

	t.Log("DisaggregatedSet retries a blocked gate even without a subsequent resource event")
	dsDGD := programDGD.DeepCopy()
	roles := newEnvtestDSHappyPathDGD("roles").Spec.Components
	for i := range dsDGD.Spec.Components {
		if dsDGD.Spec.Components[i].ComponentName == "worker" {
			dsDGD.Spec.Components[i] = roles[1]
			dsDGD.Spec.Components[i].ComponentName = "worker"
			dsDGD.Spec.Components[i].Replicas = ptr.To(int32(2))
		}
	}
	dsDGD.Spec.Components = append(dsDGD.Spec.Components, roles[0])
	dsHashes, err := desiredWorkerHashes(dsDGD)
	require.NoError(t, err)
	dsReplacement := newDCD.DeepCopy()
	dsReplacement.Name = dynamo.GetDCDResourceName(dsDGD, "worker", dsHashes.v2)
	dsReplacement.Labels[consts.KubeLabelDynamoWorkerHash] = dsHashes.v2
	dsWorker := worker.DeepCopy()
	dsWorker.Labels[consts.KubeLabelDynamoSelector] = dsReplacement.Name
	dsOwner := createTestDGDReconcilerWithStatus(dsDGD, withObjects(oldDCD, dsReplacement, dsWorker, metadata, frontend1, frontend2))
	dsOwner.Config.Namespace.Restricted = dsDGD.Namespace
	dsOwner.RuntimeConfig.Gate = features.Gates{LWS: true, DisaggregatedSet: true}
	require.NoError(t, rbacv1.AddToScheme(dsOwner.Scheme()))
	require.NoError(t, disaggregatedsetv1.AddToScheme(dsOwner.Scheme()))
	require.NoError(t, appsv1.AddToScheme(dsOwner.Scheme()))
	require.NoError(t, leaderworkersetv1.AddToScheme(dsOwner.Scheme()))
	dsProgram := dsOwner.newDisaggregatedSetProgram()
	second.mode.Store(1)
	dsResult, err := dsProgram.Reconcile(t.Context(), workloadProgramRequest{DGD: dsDGD})
	require.NoError(t, err)
	require.Equal(t, 10*time.Second, dsResult.RequeueAfter)
	second.mode.Store(0)
	dsResult, err = dsProgram.Reconcile(t.Context(), workloadProgramRequest{DGD: dsDGD})
	require.NoError(t, err)
	require.Zero(t, dsResult.RequeueAfter)

	t.Log("A recreated worker cannot reuse metadata belonging to the previous Pod UID")
	worker.UID = "recreated-pod-uid"
	require.NoError(t, r.Update(context.Background(), worker))
	rollout, pending, err = r.buildRollingUpdateContext(context.Background(), dgd)
	require.NoError(t, err)
	require.True(t, pending)
	require.Equal(t, int32(1), rollout.OldWorkerReplicaTargetsByDCD[oldDCD.Name])

	t.Log("Route metadata changes back to the owning graph, rejecting stale Pod ownership")
	owner := createTestDGDReconcilerWithStatus(dgd, withObjects(worker))
	require.Empty(t, owner.mapRolloutMetadataToDGDRequests(context.Background(), metadata))
	metadata.SetOwnerReferences([]metav1.OwnerReference{{APIVersion: "v1", Kind: "Pod", Name: worker.Name, UID: worker.UID, Controller: ptr.To(false)}})
	worker.Labels[consts.KubeLabelDynamoComponentType] = consts.ComponentTypeWorker
	require.NoError(t, owner.Update(context.Background(), worker))
	require.Equal(t, []ctrl.Request{{NamespacedName: client.ObjectKeyFromObject(dgd)}}, owner.mapRolloutMetadataToDGDRequests(context.Background(), metadata))
	require.Equal(t, []ctrl.Request{{NamespacedName: client.ObjectKeyFromObject(dgd)}}, mapDGDServingPodToRequests(context.Background(), frontend1))
}

func TestRolloutInferenceClient_BypassesProxy(t *testing.T) {
	t.Log("Configure a proxy on the default transport and serve a replacement frontend directly")
	var proxyRequests atomic.Int32
	proxy := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		proxyRequests.Add(1)
		http.Error(w, "Pod IP is unreachable through proxy", http.StatusBadGateway)
	}))
	defer proxy.Close()
	proxyURL, err := url.Parse(proxy.URL)
	require.NoError(t, err)
	defaultTransport := http.DefaultTransport
	defer func() { http.DefaultTransport = defaultTransport }()
	transport := defaultTransport.(*http.Transport).Clone()
	transport.Proxy = http.ProxyURL(proxyURL)
	defer transport.CloseIdleConnections()
	http.DefaultTransport = transport
	frontend := httptest.NewServer(&rolloutProbeServer{})
	defer frontend.Close()

	t.Log("Verify the control request reaches the proxy even for the local target")
	control := &http.Client{Transport: transport, Timeout: time.Second}
	response, err := control.Get(frontend.URL)
	require.NoError(t, err)
	require.NoError(t, response.Body.Close())
	require.Equal(t, http.StatusBadGateway, response.StatusCode)
	require.Equal(t, int32(1), proxyRequests.Load())

	t.Log("The production rollout client reaches the frontend without changing the default transport")
	owner := &DynamoGraphDeploymentReconciler{
		Config:          &configv1alpha1.OperatorConfiguration{},
		RuntimeConfig:   &commoncontroller.RuntimeConfig{},
		inferenceClient: newRolloutInferenceClient(),
	}
	defer owner.inferenceClient.CloseIdleConnections()
	r := owner.newComponentProgram().rollout
	err = r.probeReplacementInference(t.Context(), frontend.URL+"/v1/chat/completions", "/v1/chat/completions",
		rolloutInferenceTarget{Model: "test-model", InstanceID: 42})
	require.NoError(t, err)
	require.Equal(t, int32(1), proxyRequests.Load())
	require.NotNil(t, transport.Proxy)
}

func rolloutTestFrontendPod(t *testing.T, dgd *nvidiacomv1beta1.DynamoGraphDeployment, name, address string) *corev1.Pod {
	t.Helper()
	parsed, err := url.Parse(address)
	require.NoError(t, err)
	port, err := strconv.ParseInt(parsed.Port(), 10, 32)
	require.NoError(t, err)
	return &corev1.Pod{ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: dgd.Namespace, UID: types.UID(name),
		Labels: map[string]string{consts.KubeLabelDynamoGraphDeploymentName: dgd.Name, consts.KubeLabelDynamoComponent: "frontend",
			consts.KubeLabelDynamoComponentType: consts.ComponentTypeFrontend}},
		Spec: corev1.PodSpec{Containers: []corev1.Container{{Name: consts.MainContainerName,
			Ports: []corev1.ContainerPort{{Name: consts.DynamoContainerPortName, ContainerPort: int32(port)}},
		}}},
		Status: corev1.PodStatus{PodIP: parsed.Hostname(), Conditions: []corev1.PodCondition{{Type: corev1.PodReady, Status: corev1.ConditionTrue}}},
	}
}

func TestReplacementInferenceTargets_ModelRoles(t *testing.T) {
	tests := []struct {
		name          string
		modelType     string
		workerType    string
		surface       string
		prefill       bool
		wantTargets   int
		wantError     bool
		runtimeConfig map[string]any
		wantRank      uint32
	}{
		{name: "legacy prefill uses chat only", modelType: "Prefill", surface: "Chat", prefill: true, wantTargets: 1},
		{name: "DP prefill uses completions only", workerType: "prefill", surface: "Completions", prefill: true, wantTargets: 1,
			runtimeConfig: map[string]any{"data_parallel_start_rank": int64(7), "data_parallel_size": int64(2)}, wantRank: 7},
		{name: "prefill waits for paired serving metadata", workerType: "prefill", prefill: true, wantError: true},
		{name: "surface-less encoder retains legacy rollout", workerType: "encode"},
		{name: "embedding retains legacy rollout", modelType: "Embedding"},
		{name: "incomplete model card cannot bypass gate", wantError: true},
		{name: "empty DP rank range cannot certify replacement", modelType: "Chat", runtimeConfig: map[string]any{"data_parallel_size": int64(0)}, wantError: true},
		{name: "overflowing DP rank range cannot certify replacement", modelType: "Chat", runtimeConfig: map[string]any{"data_parallel_start_rank": int64(1<<32 - 1), "data_parallel_size": int64(1)}, wantError: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Log("Read replacement metadata with an exact Pod owner")
			metadata := unstructured.Unstructured{Object: map[string]any{
				"spec": map[string]any{"data": map[string]any{"model_cards": map[string]any{
					"new/backend/generate/42": map[string]any{"instance_id": int64(42), "card_json": map[string]any{
						"display_name": "test-model", "model_type": tt.modelType, "worker_type": tt.workerType,
					}},
				}}},
			}}
			if tt.runtimeConfig != nil {
				require.NoError(t, unstructured.SetNestedMap(metadata.Object, tt.runtimeConfig, "spec", "data", "model_cards", "new/backend/generate/42", "card_json", "runtime_config"))
			}
			metadata.SetOwnerReferences([]metav1.OwnerReference{{APIVersion: "v1", Kind: "Pod", Name: "new-worker", UID: "new", Controller: ptr.To(true)}})
			cards := []unstructured.Unstructured{metadata}
			if tt.surface != "" {
				paired := metadata.DeepCopy()
				paired.SetOwnerReferences([]metav1.OwnerReference{{APIVersion: "v1", Kind: "Pod", Name: "decode-worker", UID: "decode", Controller: ptr.To(false)}})
				require.NoError(t, unstructured.SetNestedField(paired.Object, tt.surface, "spec", "data", "model_cards", "new/backend/generate/42", "card_json", "model_type"))
				require.NoError(t, unstructured.SetNestedField(paired.Object, "decode", "spec", "data", "model_cards", "new/backend/generate/42", "card_json", "worker_type"))
				cards = append(cards, *paired)
			}
			targets, found, err := replacementInferenceTargets(cards, map[types.UID]bool{"new": true}, map[types.UID]bool{"decode": true})
			if tt.wantError {
				require.Error(t, err)
				return
			}
			require.NoError(t, err)
			require.True(t, found)
			require.Len(t, targets, tt.wantTargets)
			if tt.wantTargets == 0 {
				return
			}
			require.Equal(t, tt.prefill, targets[0].Prefill)
			require.Equal(t, tt.wantRank, targets[0].DPRank)
			require.Len(t, targets[0].Paths, 1)
			path := "/v1/chat/completions"
			if tt.surface == "Completions" {
				path = "/v1/completions"
			}
			require.Equal(t, path, targets[0].Paths[0])

			if tt.surface == "Chat" {
				t.Log("An unrelated serving Pod cannot supply the prefill surface")
				_, _, err = replacementInferenceTargets(cards, map[types.UID]bool{"new": true}, nil)
				require.ErrorContains(t, err, "no paired serving surface")
			}

			t.Log("Verify that a real HTTP probe sends and checks prefill attribution")
			server := httptest.NewServer(&rolloutProbeServer{rank: tt.wantRank})
			defer server.Close()
			r := createTestReconcilerWithStatus(createTestDGD("probe", nil))
			require.NoError(t, r.probeReplacementInference(context.Background(), server.URL+targets[0].Paths[0], targets[0].Paths[0], targets[0]))
		})
	}
}
