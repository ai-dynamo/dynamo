/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package controller

import (
	"context"
	"encoding/json"
	"reflect"
	"strings"
	"testing"
	"time"

	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	nvidiacomv1beta2 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta2"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dgdrrunpublisher"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/gpu"
	batchv1 "k8s.io/api/batch/v1"
	corev1 "k8s.io/api/core/v1"
	rbacv1 "k8s.io/api/rbac/v1"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	clientgoscheme "k8s.io/client-go/kubernetes/scheme"
	"k8s.io/utils/ptr"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
	"sigs.k8s.io/yaml"
)

type fixedRunHardware struct {
	info  gpu.GPUInfo
	err   error
	calls []nvidiacomv1beta1.GPUSKUType
}

func (f *fixedRunHardware) Discover(_ context.Context, sku nvidiacomv1beta1.GPUSKUType) (*gpu.GPUInfo, error) {
	f.calls = append(f.calls, sku)
	if f.err != nil {
		return nil, f.err
	}
	return &f.info, nil
}

type dgdrRunFixture struct {
	t          *testing.T
	client     client.Client
	reconciler *DynamoGraphDeploymentRunV1beta2Reconciler
}

func newDGDRRunFixture(t *testing.T, hardware dgdrRunHardwareDiscoverer, objects ...client.Object) *dgdrRunFixture {
	t.Helper()
	scheme := runtime.NewScheme()
	if err := clientgoscheme.AddToScheme(scheme); err != nil {
		t.Fatalf("add Kubernetes types: %v", err)
	}
	if err := nvidiacomv1beta2.AddToScheme(scheme); err != nil {
		t.Fatalf("add v1beta2 types: %v", err)
	}
	c := fake.NewClientBuilder().
		WithScheme(scheme).
		WithObjects(objects...).
		WithIndex(&batchv1.Job{}, dgdrRunJobControllerUIDIndex, dgdrRunJobControllerUIDs).
		WithStatusSubresource(&nvidiacomv1beta2.DynamoGraphDeploymentRun{}, &batchv1.Job{}, &corev1.Pod{}).
		Build()
	return &dgdrRunFixture{
		t:      t,
		client: c,
		reconciler: &DynamoGraphDeploymentRunV1beta2Reconciler{
			Client:                  c,
			Hardware:                hardware,
			PlannerImage:            "example.com/dynamo-planner:1.2.3",
			OperatorImage:           "example.com/dynamo-operator:1.2.3",
			OperatorImagePullPolicy: corev1.PullAlways,
		},
	}
}

func (f *dgdrRunFixture) reconcile() {
	f.t.Helper()
	if _, err := f.reconciler.Reconcile(f.t.Context(), ctrl.Request{NamespacedName: types.NamespacedName{
		Namespace: "default", Name: "search-1",
	}}); err != nil {
		f.t.Fatalf("reconcile: %v", err)
	}
}

func (f *dgdrRunFixture) run() *nvidiacomv1beta2.DynamoGraphDeploymentRun {
	f.t.Helper()
	run := &nvidiacomv1beta2.DynamoGraphDeploymentRun{}
	if err := f.client.Get(f.t.Context(), types.NamespacedName{Namespace: "default", Name: "search-1"}, run); err != nil {
		f.t.Fatalf("get Run: %v", err)
	}
	return run
}

func (f *dgdrRunFixture) job() *batchv1.Job {
	f.t.Helper()
	jobs := &batchv1.JobList{}
	if err := f.client.List(f.t.Context(), jobs, client.InNamespace("default")); err != nil {
		f.t.Fatalf("list Jobs: %v", err)
	}
	if len(jobs.Items) != 1 {
		f.t.Fatalf("Jobs = %d, want 1", len(jobs.Items))
	}
	return &jobs.Items[0]
}

func dgdrRunForTest(t *testing.T, inputOverrides map[string]any) *nvidiacomv1beta2.DynamoGraphDeploymentRun {
	t.Helper()
	raw, err := json.Marshal(inputOverrides)
	if err != nil {
		t.Fatalf("marshal parameters: %v", err)
	}
	return &nvidiacomv1beta2.DynamoGraphDeploymentRun{
		ObjectMeta: metav1.ObjectMeta{Name: "search-1", Namespace: "default", UID: "run-uid", Generation: 1},
		Spec: nvidiacomv1beta2.DynamoGraphDeploymentRunSpec{NativeSearchInput: nvidiacomv1beta2.NativeSearchInput{
			Model: &nvidiacomv1beta2.ModelSpec{Name: "Qwen/Qwen3-0.6B"},
			Hardware: &nvidiacomv1beta2.HardwareSpec{GPU: &nvidiacomv1beta2.GPUHardwareSpec{
				SKU: ptr.To("h200_sxm"), Budget: ptr.To[int32](16),
			}},
			AISimulate:     &nvidiacomv1beta2.AISimulateSpec{InputOverrides: &runtime.RawExtension{Raw: raw}},
			Recommendation: &nvidiacomv1beta2.RecommendationSpec{MaxCandidates: 2},
		}},
	}
}

func TestV1beta2RunControllerCreatesOneIsolatedAISimulateJob(t *testing.T) {
	hardware := &fixedRunHardware{info: gpu.GPUInfo{GPUsPerNode: 8}}
	run := dgdrRunForTest(t, map[string]any{
		"engine":       map[string]any{"hardware": "h100_sxm"},
		"optimization": map[string]any{"constraints": map[string]any{"max_candidate_gpus": 12}},
		"dgdr_run": map[string]any{
			"max_candidates": 7,
			"runtime_image":  "example.com/custom-vllm-runtime:1.2.3",
			"name":           "native-output-name",
			"snapshot_dir":   "/work/results",
		},
	})
	f := newDGDRRunFixture(t, hardware, run)

	f.reconcile()
	f.reconcile()

	job := f.job()
	assertDGDRRunJobIsolation(t, job, run, f.reconciler.PlannerImage, f.reconciler.OperatorImage)
	assertDGDRRunInput(t, f.client, job, hardware)
	assertDGDRRunSupportResources(t, f.client, job, run)
	stored := f.run()
	assertDGDRRunPendingStatus(t, stored, job.Name)
	assertDGDRRunStatusInputMatchesConfigMap(t, f.client, stored, job)
}

func assertDGDRRunJobIsolation(
	t *testing.T,
	job *batchv1.Job,
	run *nvidiacomv1beta2.DynamoGraphDeploymentRun,
	plannerImage string,
	operatorImage string,
) {
	t.Helper()
	if !metav1.IsControlledBy(job, run) {
		t.Fatal("Job is not controlled by its Run")
	}
	if got := job.Annotations[dgdrRunGPUsPerNodeAnnotation]; got != "8" {
		t.Fatalf("GPUs-per-node annotation = %q, want 8", got)
	}
	if job.Spec.BackoffLimit == nil || *job.Spec.BackoffLimit != 0 {
		t.Fatalf("backoffLimit = %v, want 0", job.Spec.BackoffLimit)
	}
	if job.Spec.Template.Spec.AutomountServiceAccountToken == nil || *job.Spec.Template.Spec.AutomountServiceAccountToken {
		t.Fatal("the pod must disable automatic service-account token mounting")
	}
	sweeper := containerNamed(t, job.Spec.Template.Spec.Containers, dgdrRunSweeperContainer)
	publisher := containerNamed(t, job.Spec.Template.Spec.Containers, dgdrRunPublisherContainer)
	if sweeper.Image != plannerImage {
		t.Fatalf("sweeper image = %q, want operator-configured planner image %q", sweeper.Image, plannerImage)
	}
	if want := []string{"aisimulate"}; !reflect.DeepEqual(sweeper.Command, want) {
		t.Fatalf("sweeper command = %q, want %q", sweeper.Command, want)
	}
	if want := []string{
		"recommend", "--stack", "dynamo",
		"--config", dgdrRunInputPath,
		"--output", "dgdr_run",
		"--output-dir", "/work/results",
		"--overwrite",
	}; !reflect.DeepEqual(sweeper.Args, want) {
		t.Fatalf("sweeper args = %q, want %q", sweeper.Args, want)
	}
	if publisher.Image != operatorImage || publisher.ImagePullPolicy != corev1.PullAlways {
		t.Fatalf("publisher image = %q/%s, want configured operator image/Always", publisher.Image, publisher.ImagePullPolicy)
	}
	if hasMount(sweeper.VolumeMounts, dgdrRunServiceAccountVolume) {
		t.Fatal("sweeper unexpectedly receives the Kubernetes service-account token")
	}
	if !hasMount(publisher.VolumeMounts, dgdrRunServiceAccountVolume) {
		t.Fatal("publisher does not receive its projected Kubernetes credentials")
	}
	if !hasMountAt(sweeper.VolumeMounts, dgdrRunSnapshotVolume, "/work/results") ||
		!hasMountAt(publisher.VolumeMounts, dgdrRunSnapshotVolume, "/work/results") {
		t.Fatal("sweeper and publisher do not share the parameter-overridden snapshot directory")
	}
}

func assertDGDRRunStatusInputMatchesConfigMap(
	t *testing.T,
	c client.Client,
	run *nvidiacomv1beta2.DynamoGraphDeploymentRun,
	job *batchv1.Job,
) {
	t.Helper()
	if run.Status.AISimulate == nil || run.Status.AISimulate.Input == nil {
		t.Fatal("status.aisimulate.input is empty")
	}
	var statusInput map[string]any
	if err := json.Unmarshal(run.Status.AISimulate.Input.Raw, &statusInput); err != nil {
		t.Fatalf("decode status.aisimulate.input: %v", err)
	}
	configMap := &corev1.ConfigMap{}
	if err := c.Get(t.Context(), client.ObjectKeyFromObject(job), configMap); err != nil {
		t.Fatalf("get input ConfigMap: %v", err)
	}
	var configInput map[string]any
	if err := yaml.Unmarshal([]byte(configMap.Data[dgdrRunInputKey]), &configInput); err != nil {
		t.Fatalf("decode input ConfigMap: %v", err)
	}
	if !reflect.DeepEqual(statusInput, configInput) {
		t.Fatalf("status input = %#v, want exact ConfigMap input %#v", statusInput, configInput)
	}
}

func assertDGDRRunInput(t *testing.T, c client.Client, job *batchv1.Job, hardware *fixedRunHardware) {
	t.Helper()
	configMap := &corev1.ConfigMap{}
	if err := c.Get(t.Context(), client.ObjectKeyFromObject(job), configMap); err != nil {
		t.Fatalf("get input ConfigMap: %v", err)
	}
	if configMap.Immutable == nil || !*configMap.Immutable {
		t.Fatal("input ConfigMap is not immutable")
	}
	var input map[string]any
	if err := yaml.Unmarshal([]byte(configMap.Data[dgdrRunInputKey]), &input); err != nil {
		t.Fatalf("decode input ConfigMap: %v", err)
	}
	output := input["dgdr_run"].(map[string]any)
	if output["name"] != "native-output-name" || output["snapshot_dir"] != "/work/results" {
		t.Fatalf("native parameter overrides were not preserved: %v", output)
	}
	if output["max_candidates"] != float64(7) || output["runtime_image"] != "example.com/custom-vllm-runtime:1.2.3" {
		t.Fatalf("native parameter overrides were not preserved: %v", output)
	}
	if output["derive_runtime_images"] != true {
		t.Fatalf("derive_runtime_images = %v, want true for a backend search", output["derive_runtime_images"])
	}
	if got := input["engine"].(map[string]any)["hardware"]; got != "h100_sxm" {
		t.Fatalf("engine.hardware = %v, want authoritative input override h100_sxm", got)
	}
	constraints := input["optimization"].(map[string]any)["constraints"].(map[string]any)
	if got := constraints["max_candidate_gpus"]; got != float64(12) {
		t.Fatalf("optimization.constraints.max_candidate_gpus = %v, want authoritative input override 12", got)
	}
	if !reflect.DeepEqual(hardware.calls, []nvidiacomv1beta1.GPUSKUType{"h200_sxm"}) {
		t.Fatalf("hardware discovery calls = %v, want the admitted typed SKU h200_sxm", hardware.calls)
	}
}

func assertDGDRRunSupportResources(
	t *testing.T,
	c client.Client,
	job *batchv1.Job,
	run *nvidiacomv1beta2.DynamoGraphDeploymentRun,
) {
	t.Helper()
	for _, object := range []client.Object{
		&corev1.ServiceAccount{}, &rbacv1.Role{}, &rbacv1.RoleBinding{},
	} {
		if err := c.Get(t.Context(), client.ObjectKeyFromObject(job), object); err != nil {
			t.Fatalf("get %T: %v", object, err)
		}
		if !metav1.IsControlledBy(object, run) {
			t.Fatalf("%T is not controlled by the Run", object)
		}
	}
}

func assertDGDRRunPendingStatus(t *testing.T, stored *nvidiacomv1beta2.DynamoGraphDeploymentRun, jobName string) {
	t.Helper()
	if stored.Status.Phase != nvidiacomv1beta2.DGDRRunPhasePending || stored.Status.JobRef == nil || stored.Status.JobRef.Name != jobName {
		t.Fatalf("Run status = %+v, want Pending and JobRef %q", stored.Status, jobName)
	}
	condition := meta.FindStatusCondition(stored.Status.Conditions, nvidiacomv1beta2.DGDRRunConditionCompleted)
	if condition == nil || condition.Status != metav1.ConditionFalse || condition.Reason != dgdrRunReasonPending {
		t.Fatalf("Completed condition = %+v, want False/%s", condition, dgdrRunReasonPending)
	}
}

func TestV1beta2RunControllerOwnsOnlyLifecycleStatus(t *testing.T) {
	hardware := &fixedRunHardware{info: gpu.GPUInfo{GPUsPerNode: 8}}
	f := newDGDRRunFixture(t, hardware, dgdrRunForTest(t, map[string]any{}))
	f.reconcile()
	job := f.job()

	run := f.run()
	progressTime := metav1.NewTime(time.Date(2026, 10, 23, 10, 0, 0, 0, time.UTC))
	run.Status.Progress = &nvidiacomv1beta2.DGDRRunProgress{Candidates: &nvidiacomv1beta2.CandidateProgress{Published: 1}}
	run.Status.CandidateRefs = []corev1.LocalObjectReference{{Name: "candidate-a"}}
	run.Status.LastProgressTime = &progressTime
	if err := f.client.Status().Update(t.Context(), run); err != nil {
		t.Fatalf("publish progress status: %v", err)
	}

	start := metav1.NewTime(progressTime.Add(time.Minute))
	job.Status.StartTime = &start
	job.Status.Active = 1
	if err := f.client.Status().Update(t.Context(), job); err != nil {
		t.Fatalf("mark Job running: %v", err)
	}
	f.reconcile()
	run = f.run()
	if run.Status.Phase != nvidiacomv1beta2.DGDRRunPhaseRunning || run.Status.StartTime == nil || !run.Status.StartTime.Equal(&start) {
		t.Fatalf("running status = %+v", run.Status)
	}
	if run.Status.Progress == nil || len(run.Status.CandidateRefs) != 1 || run.Status.LastProgressTime == nil {
		t.Fatalf("publisher-owned status was lost: %+v", run.Status)
	}

	completed := metav1.NewTime(start.Add(5 * time.Minute))
	job = f.job()
	job.Status.Active = 0
	job.Status.CompletionTime = &completed
	job.Status.Conditions = []batchv1.JobCondition{{
		Type: batchv1.JobComplete, Status: corev1.ConditionTrue, LastTransitionTime: completed,
	}}
	if err := f.client.Status().Update(t.Context(), job); err != nil {
		t.Fatalf("complete Job: %v", err)
	}
	f.reconcile()
	run = f.run()
	condition := meta.FindStatusCondition(run.Status.Conditions, nvidiacomv1beta2.DGDRRunConditionCompleted)
	if run.Status.Phase != nvidiacomv1beta2.DGDRRunPhaseSucceeded || condition == nil || condition.Status != metav1.ConditionTrue {
		t.Fatalf("completed status = %+v", run.Status)
	}
	if run.Status.CompletionTime == nil || !run.Status.CompletionTime.Equal(&completed) {
		t.Fatalf("completionTime = %v, want %v", run.Status.CompletionTime, completed)
	}
}

func TestV1beta2RunControllerClassifiesPublisherProtocolFailure(t *testing.T) {
	f := newDGDRRunFixture(t, &fixedRunHardware{info: gpu.GPUInfo{GPUsPerNode: 8}}, dgdrRunForTest(t, map[string]any{}))
	f.reconcile()
	job := f.job()
	pod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      job.Name + "-pod",
			Namespace: job.Namespace,
			Labels:    map[string]string{batchv1.JobNameLabel: job.Name},
			OwnerReferences: []metav1.OwnerReference{{
				APIVersion: batchv1.SchemeGroupVersion.String(), Kind: "Job", Name: job.Name, UID: job.UID,
				Controller: ptr.To(true),
			}},
		},
	}
	if err := f.client.Create(t.Context(), pod); err != nil {
		t.Fatalf("create Job pod: %v", err)
	}
	pod.Status.ContainerStatuses = []corev1.ContainerStatus{{
		Name: dgdrRunPublisherContainer,
		State: corev1.ContainerState{Terminated: &corev1.ContainerStateTerminated{
			ExitCode: int32(dgdrrunpublisher.ExitProtocolViolation), Message: "terminal snapshot missing",
		}},
	}}
	if err := f.client.Status().Update(t.Context(), pod); err != nil {
		t.Fatalf("update Job pod status: %v", err)
	}

	failed := metav1.Now()
	job.Status.Conditions = []batchv1.JobCondition{{
		Type: batchv1.JobFailed, Status: corev1.ConditionTrue, LastTransitionTime: failed,
	}}
	if err := f.client.Status().Update(t.Context(), job); err != nil {
		t.Fatalf("fail Job: %v", err)
	}
	f.reconcile()

	run := f.run()
	condition := meta.FindStatusCondition(run.Status.Conditions, nvidiacomv1beta2.DGDRRunConditionCompleted)
	if run.Status.Phase != nvidiacomv1beta2.DGDRRunPhaseFailed || condition == nil || condition.Reason != dgdrRunReasonProtocolViolation {
		t.Fatalf("failed status = %+v", run.Status)
	}
	if !strings.Contains(condition.Message, "terminal snapshot missing") {
		t.Fatalf("failure message = %q, want publisher termination message", condition.Message)
	}
}

func TestDGDRRunResourceNameIsStableAndBounded(t *testing.T) {
	short := dgdrRunResourceName("search-1")
	if short != "search-1-aisimulate" {
		t.Fatalf("short name = %q", short)
	}
	long := strings.Repeat("a", 200)
	one, two := dgdrRunResourceName(long), dgdrRunResourceName(long)
	if one != two || len(one) > dgdrRunMaxJobNameLength || !strings.HasSuffix(one, "-aisimulate") {
		t.Fatalf("bounded name = %q", one)
	}
}

func containerNamed(t *testing.T, containers []corev1.Container, name string) corev1.Container {
	t.Helper()
	for _, container := range containers {
		if container.Name == name {
			return container
		}
	}
	t.Fatalf("container %q not found", name)
	return corev1.Container{}
}

func hasMount(mounts []corev1.VolumeMount, name string) bool {
	for _, mount := range mounts {
		if mount.Name == name {
			return true
		}
	}
	return false
}

func hasMountAt(mounts []corev1.VolumeMount, name, path string) bool {
	for _, mount := range mounts {
		if mount.Name == name && mount.MountPath == path {
			return true
		}
	}
	return false
}
