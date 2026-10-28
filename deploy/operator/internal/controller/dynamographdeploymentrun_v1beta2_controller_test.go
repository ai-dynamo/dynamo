/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package controller

import (
	"context"
	"encoding/json"
	"errors"
	"reflect"
	"strings"
	"testing"
	"time"

	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	nvidiacomv1beta2 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta2"
	commoncontroller "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dgdrrunpublisher"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/gpu"
	batchv1 "k8s.io/api/batch/v1"
	corev1 "k8s.io/api/core/v1"
	rbacv1 "k8s.io/api/rbac/v1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/validation/field"
	clientgoscheme "k8s.io/client-go/kubernetes/scheme"
	"k8s.io/utils/ptr"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
	"sigs.k8s.io/controller-runtime/pkg/client/interceptor"
	"sigs.k8s.io/controller-runtime/pkg/reconcile"
	"sigs.k8s.io/yaml"
)

type fixedRunHardware struct {
	info  gpu.GPUInfo
	err   error
	calls []nvidiacomv1beta1.GPUSKUType
}

type fixedRunSecrets struct {
	byImage map[string][]string
	err     error
}

func (f *fixedRunSecrets) GetSecrets(_ string, image string) ([]string, error) {
	if f.err != nil {
		return nil, f.err
	}
	return append([]string(nil), f.byImage[image]...), nil
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
	return newDGDRRunFixtureWithInterceptor(t, hardware, interceptor.Funcs{}, objects...)
}

func newDGDRRunFixtureWithInterceptor(
	t *testing.T,
	hardware dgdrRunHardwareDiscoverer,
	funcs interceptor.Funcs,
	objects ...client.Object,
) *dgdrRunFixture {
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
		WithInterceptorFuncs(funcs).
		Build()
	return &dgdrRunFixture{
		t:      t,
		client: c,
		reconciler: &DynamoGraphDeploymentRunV1beta2Reconciler{
			Client:                  c,
			APIReader:               c,
			Hardware:                hardware,
			PlannerImage:            "example.com/dynamo-planner:1.2.3",
			OperatorImage:           "example.com/dynamo-operator:1.2.3",
			OperatorImagePullPolicy: corev1.PullAlways,
		},
	}
}

func (f *dgdrRunFixture) reconcile() {
	f.t.Helper()
	if _, err := f.reconciler.Reconcile(f.t.Context(), ctrl.Request{
		Namespace: "default", Name: "search-1"}); err != nil {
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
	jobs := f.jobs()
	if len(jobs) != 1 {
		f.t.Fatalf("Jobs = %d, want 1", len(jobs))
	}
	return &jobs[0]
}

func (f *dgdrRunFixture) jobs() []batchv1.Job {
	f.t.Helper()
	jobs := &batchv1.JobList{}
	if err := f.client.List(f.t.Context(), jobs, client.InNamespace("default")); err != nil {
		f.t.Fatalf("list Jobs: %v", err)
	}
	return jobs.Items
}

func dgdrRunForTest(t *testing.T, inputOverrides map[string]any) *nvidiacomv1beta2.DynamoGraphDeploymentRun {
	t.Helper()
	raw, err := json.Marshal(inputOverrides)
	if err != nil {
		t.Fatalf("marshal parameters: %v", err)
	}
	metric := nvidiacomv1beta2.ObjectiveMetric("ThroughputPerGPU")
	return &nvidiacomv1beta2.DynamoGraphDeploymentRun{
		Name: "search-1", Namespace: "default", UID: "run-uid", Generation: 1,
		Annotations: map[string]string{dgdrV1beta2GenerationAnnotation: "1"},
		Spec: nvidiacomv1beta2.DynamoGraphDeploymentRunSpec{NativeSearchInput: nvidiacomv1beta2.NativeSearchInput{
			Model:    &nvidiacomv1beta2.ModelSpec{Name: "Qwen/Qwen3-0.6B"},
			Backends: []nvidiacomv1beta2.Backend{nvidiacomv1beta2.BackendVLLM},
			Hardware: &nvidiacomv1beta2.HardwareSpec{GPU: &nvidiacomv1beta2.GPUHardwareSpec{
				SKU: new("h200_sxm"), Budget: ptr.To[int32](16),
			}},
			Workload: &nvidiacomv1beta2.WorkloadSpec{Static: &nvidiacomv1beta2.StaticWorkloadSpec{}},
			Objective: &nvidiacomv1beta2.ObjectiveSpec{
				Mode: nvidiacomv1beta2.ObjectiveModeOptimize, Metric: &metric, SLA: &nvidiacomv1beta2.ObjectiveSLASpec{},
			},
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
			"namespace":      "another-namespace",
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

func TestV1beta2RunControllerRetainsAndTerminalizesASupersededRun(t *testing.T) {
	hardware := &fixedRunHardware{info: gpu.GPUInfo{GPUsPerNode: 8}}
	f := newDGDRRunFixture(t, hardware, dgdrRunForTest(t, map[string]any{}))
	f.reconcile()
	f.reconcile()

	run := f.run()
	run.Status.Progress = &nvidiacomv1beta2.DGDRRunProgress{
		Candidates: &nvidiacomv1beta2.CandidateProgress{Published: 1},
	}
	run.Status.CandidateRefs = []corev1.LocalObjectReference{{Name: "candidate-a"}}
	if err := f.client.Status().Update(t.Context(), run); err != nil {
		t.Fatalf("publish candidate history: %v", err)
	}
	run = f.run()
	if run.Annotations == nil {
		run.Annotations = map[string]string{}
	}
	run.Annotations[dgdrV1beta2CancelRequestedByGenerationAnnotation] = "3"
	if err := f.client.Update(t.Context(), run); err != nil {
		t.Fatalf("request cancellation: %v", err)
	}

	t.Log("The first cancellation pass stops the Job but retains the Run and its history")
	f.reconcile()
	if jobs := f.jobs(); len(jobs) != 0 {
		t.Fatalf("Jobs = %d, want the execution stopped", len(jobs))
	}
	run = f.run()
	if meta.IsStatusConditionTrue(run.Status.Conditions, nvidiacomv1beta2.DGDRRunConditionCompleted) {
		t.Fatal("Run completed before Job deletion was observed")
	}
	if run.Status.Progress == nil || len(run.Status.CandidateRefs) != 1 {
		t.Fatalf("candidate history was lost while stopping the Job: %+v", run.Status)
	}

	t.Log("A coalesced Request generation becomes the terminal cancellation cause")
	run.Annotations[dgdrV1beta2CancelRequestedByGenerationAnnotation] = "4"
	if err := f.client.Update(t.Context(), run); err != nil {
		t.Fatalf("coalesce cancellation generation: %v", err)
	}
	f.reconcile()
	run = f.run()
	condition := meta.FindStatusCondition(run.Status.Conditions, nvidiacomv1beta2.DGDRRunConditionCompleted)
	if run.Status.Phase != nvidiacomv1beta2.DGDRRunPhaseFailed || condition == nil ||
		condition.Status != metav1.ConditionTrue || condition.Reason != dgdrRunReasonSuperseded {
		t.Fatalf("superseded status = %+v", run.Status)
	}
	if !strings.Contains(condition.Message, "generation 4") {
		t.Fatalf("superseded message = %q, want latest generation 4", condition.Message)
	}
	if run.Status.CompletionTime == nil || run.Status.Progress == nil || len(run.Status.CandidateRefs) != 1 {
		t.Fatalf("terminal cancellation lost completion or candidate history: %+v", run.Status)
	}

	t.Log("Retries are idempotent and never recreate the stopped Job")
	completionTime := run.Status.CompletionTime.DeepCopy()
	f.reconcile()
	if jobs := f.jobs(); len(jobs) != 0 {
		t.Fatalf("Jobs = %d after retry, want none", len(jobs))
	}
	if got := f.run().Status.CompletionTime; got == nil || !got.Equal(completionTime) {
		t.Fatalf("completionTime = %v, want stable %v", got, completionTime)
	}
}

func TestV1beta2RunControllerCancelsBeforeCreatingExecutionResources(t *testing.T) {
	hardware := &fixedRunHardware{info: gpu.GPUInfo{GPUsPerNode: 8}}
	run := dgdrRunForTest(t, map[string]any{})
	run.Annotations[dgdrV1beta2CancelRequestedByGenerationAnnotation] = "2"
	f := newDGDRRunFixture(t, hardware, run)

	f.reconcile()
	stored := f.run()
	condition := meta.FindStatusCondition(stored.Status.Conditions, nvidiacomv1beta2.DGDRRunConditionCompleted)
	if condition == nil || condition.Status != metav1.ConditionTrue || condition.Reason != dgdrRunReasonSuperseded {
		t.Fatalf("Completed condition = %+v, want True/%s", condition, dgdrRunReasonSuperseded)
	}
	if jobs := f.jobs(); len(jobs) != 0 {
		t.Fatalf("Jobs = %d, want none", len(jobs))
	}
	if len(hardware.calls) != 0 {
		t.Fatalf("hardware discovery calls = %v, want none for a canceled Run", hardware.calls)
	}
}

func TestV1beta2RunControllerRequiresANewerCancellationGeneration(t *testing.T) {
	tests := []struct {
		name             string
		runGeneration    string
		cancelGeneration string
		wantCanceled     bool
	}{
		{name: "malformed Run generation", runGeneration: "malformed", cancelGeneration: "2"},
		{name: "zero Run generation", runGeneration: "0", cancelGeneration: "2"},
		{name: "malformed cancellation generation", runGeneration: "1", cancelGeneration: "malformed"},
		{name: "zero cancellation generation", runGeneration: "1", cancelGeneration: "0"},
		{name: "equal cancellation generation", runGeneration: "1", cancelGeneration: "1"},
		{name: "older cancellation generation", runGeneration: "2", cancelGeneration: "1"},
		{name: "newer cancellation generation", runGeneration: "1", cancelGeneration: "2", wantCanceled: true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			hardware := &fixedRunHardware{info: gpu.GPUInfo{GPUsPerNode: 8}}
			run := dgdrRunForTest(t, map[string]any{})
			run.Annotations[dgdrV1beta2GenerationAnnotation] = tt.runGeneration
			run.Annotations[dgdrV1beta2CancelRequestedByGenerationAnnotation] = tt.cancelGeneration
			f := newDGDRRunFixture(t, hardware, run)

			t.Log("Reconcile the Run with the cancellation marker")
			f.reconcile()

			completed := meta.IsStatusConditionTrue(f.run().Status.Conditions, nvidiacomv1beta2.DGDRRunConditionCompleted)
			if completed != tt.wantCanceled {
				t.Fatalf("Completed = %t, want %t", completed, tt.wantCanceled)
			}
			if tt.wantCanceled {
				if jobs := f.jobs(); len(jobs) != 0 {
					t.Fatalf("Jobs = %d, want no execution for a superseded Run", len(jobs))
				}
				if len(hardware.calls) != 0 {
					t.Fatalf("hardware discovery calls = %v, want none", hardware.calls)
				}
				return
			}
			if jobs := f.jobs(); len(jobs) != 1 {
				t.Fatalf("Jobs = %d, want normal execution", len(jobs))
			}
			if len(hardware.calls) != 1 {
				t.Fatalf("hardware discovery calls = %v, want normal execution", hardware.calls)
			}
		})
	}
}

func TestV1beta2RunControllerRejectsSupportResourceOwnershipCollisionsWithoutMutation(t *testing.T) {
	resourceKinds := []struct {
		name string
		new  func(string) client.Object
	}{
		{name: "ServiceAccount", new: func(name string) client.Object {
			return &corev1.ServiceAccount{
				Name:      name,
				Namespace: "default",
				Labels:    map[string]string{"existing": "service-account"},
			}
		}},
		{name: "Role", new: func(name string) client.Object {
			return &rbacv1.Role{
				Name:      name,
				Namespace: "default",
				Rules:     []rbacv1.PolicyRule{{APIGroups: []string{"existing.example"}, Resources: []string{"kept"}, Verbs: []string{"get"}}},
			}
		}},
		{name: "RoleBinding", new: func(name string) client.Object {
			return &rbacv1.RoleBinding{
				Name:      name,
				Namespace: "default",
				Labels:    map[string]string{"existing": "role-binding"},
			}
		}},
		{name: "ConfigMap", new: func(name string) client.Object {
			return &corev1.ConfigMap{
				Name:      name,
				Namespace: "default",
				Data:      map[string]string{"existing": "config-map"},
			}
		}},
	}
	owners := []struct {
		name string
		ref  *metav1.OwnerReference
	}{
		{name: "ownerless"},
		{name: "foreign", ref: &metav1.OwnerReference{
			APIVersion: "v1", Kind: "ConfigMap", Name: "foreign", UID: "foreign-uid", Controller: new(true),
		}},
		{name: "stale Run UID", ref: &metav1.OwnerReference{
			APIVersion: nvidiacomv1beta2.GroupVersion.String(), Kind: "DynamoGraphDeploymentRun",
			Name: "search-1", UID: "stale-run-uid", Controller: new(true),
		}},
	}

	for _, resourceKind := range resourceKinds {
		for _, owner := range owners {
			t.Run(resourceKind.name+"/"+owner.name, func(t *testing.T) {
				run := dgdrRunForTest(t, map[string]any{})
				collision := resourceKind.new(dgdrRunResourceName(run.Name))
				if owner.ref != nil {
					collision.SetOwnerReferences([]metav1.OwnerReference{*owner.ref})
				}
				f := newDGDRRunFixture(t, &fixedRunHardware{info: gpu.GPUInfo{GPUsPerNode: 8}}, run, collision)

				before := collision.DeepCopyObject().(client.Object)
				if err := f.client.Get(t.Context(), client.ObjectKeyFromObject(collision), before); err != nil {
					t.Fatalf("get collision before reconcile: %v", err)
				}
				_, err := f.reconciler.Reconcile(t.Context(), ctrl.Request{NamespacedName: client.ObjectKeyFromObject(run)})
				if !errors.Is(err, reconcile.TerminalError(nil)) {
					t.Fatalf("reconcile error = %v, want terminal ownership conflict", err)
				}
				if _, ok := errors.AsType[*commoncontroller.OwnershipConflictError](err); !ok {
					t.Fatalf("reconcile error = %T %v, want OwnershipConflictError", err, err)
				}

				after := collision.DeepCopyObject().(client.Object)
				if err := f.client.Get(t.Context(), client.ObjectKeyFromObject(collision), after); err != nil {
					t.Fatalf("get collision after reconcile: %v", err)
				}
				if !apiequality.Semantic.DeepEqual(before, after) {
					t.Fatalf("colliding resource was mutated\nbefore: %#v\nafter:  %#v", before, after)
				}
				condition := meta.FindStatusCondition(f.run().Status.Conditions, nvidiacomv1beta2.DGDRRunConditionCompleted)
				if condition == nil || condition.Status != metav1.ConditionTrue ||
					condition.Reason != commoncontroller.EventReasonOwnershipConflict {
					t.Fatalf("Completed condition = %+v, want terminal ownership conflict", condition)
				}
			})
		}
	}
}

func TestV1beta2RunControllerClassifiesPermanentAndRetryableFailures(t *testing.T) {
	tests := []struct {
		name           string
		newFixture     func(*testing.T) *dgdrRunFixture
		wantTerminal   bool
		wantReason     string
		wantErrMessage string
	}{
		{
			name: "immutable lowering error is terminal",
			newFixture: func(t *testing.T) *dgdrRunFixture {
				run := dgdrRunForTest(t, map[string]any{})
				f := newDGDRRunFixture(t, &fixedRunHardware{info: gpu.GPUInfo{GPUsPerNode: 8}}, run)
				f.reconciler.PlannerImage = "example.com/dynamo-planner@sha256:0123"
				f.reconciler.DynamoVersion = "1.7.0"
				return f
			},
			wantTerminal: true, wantReason: dgdrRunReasonInvalidExecution,
			wantErrMessage: "digest-only planner image",
		},
		{
			name: "invalid Kubernetes resource is terminal",
			newFixture: func(t *testing.T) *dgdrRunFixture {
				run := dgdrRunForTest(t, map[string]any{})
				return newDGDRRunFixtureWithInterceptor(t, &fixedRunHardware{info: gpu.GPUInfo{GPUsPerNode: 8}}, interceptor.Funcs{
					Create: func(ctx context.Context, c client.WithWatch, object client.Object, opts ...client.CreateOption) error {
						if _, isConfigMap := object.(*corev1.ConfigMap); isConfigMap {
							return apierrors.NewInvalid(schema.GroupKind{Kind: "ConfigMap"}, object.GetName(),
								field.ErrorList{field.Invalid(field.NewPath("data"), nil, "invalid execution")})
						}
						return c.Create(ctx, object, opts...)
					},
				}, run)
			},
			wantTerminal: true, wantReason: dgdrRunReasonInvalidResource,
			wantErrMessage: "invalid execution",
		},
		{
			name: "hardware discovery error is retryable",
			newFixture: func(t *testing.T) *dgdrRunFixture {
				return newDGDRRunFixture(t, &fixedRunHardware{err: errors.New("node inventory unavailable")},
					dgdrRunForTest(t, map[string]any{}))
			},
			wantErrMessage: "node inventory unavailable",
		},
		{
			name: "registry secret error is retryable",
			newFixture: func(t *testing.T) *dgdrRunFixture {
				f := newDGDRRunFixture(t, &fixedRunHardware{info: gpu.GPUInfo{GPUsPerNode: 8}},
					dgdrRunForTest(t, map[string]any{}))
				f.reconciler.DockerSecretRetriever = &fixedRunSecrets{err: errors.New("registry index unavailable")}
				return f
			},
			wantErrMessage: "registry index unavailable",
		},
		{
			name: "terminal status write failure remains retryable",
			newFixture: func(t *testing.T) *dgdrRunFixture {
				run := dgdrRunForTest(t, map[string]any{})
				f := newDGDRRunFixtureWithInterceptor(t, &fixedRunHardware{info: gpu.GPUInfo{GPUsPerNode: 8}}, interceptor.Funcs{
					SubResourcePatch: func(
						_ context.Context, _ client.Client, _ string, _ client.Object,
						_ client.Patch, _ ...client.SubResourcePatchOption,
					) error {
						return errors.New("status writer unavailable")
					},
				}, run)
				f.reconciler.PlannerImage = "example.com/dynamo-planner@sha256:0123"
				f.reconciler.DynamoVersion = "1.7.0"
				return f
			},
			wantErrMessage: "status writer unavailable",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			f := tt.newFixture(t)
			_, err := f.reconciler.Reconcile(t.Context(), ctrl.Request{NamespacedName: client.ObjectKeyFromObject(f.run())})
			if err == nil || !strings.Contains(err.Error(), tt.wantErrMessage) {
				t.Fatalf("reconcile error = %v, want %q", err, tt.wantErrMessage)
			}
			if got := errors.Is(err, reconcile.TerminalError(nil)); got != tt.wantTerminal {
				t.Fatalf("terminal error = %t, want %t: %v", got, tt.wantTerminal, err)
			}
			condition := meta.FindStatusCondition(f.run().Status.Conditions, nvidiacomv1beta2.DGDRRunConditionCompleted)
			if tt.wantTerminal {
				if condition == nil || condition.Status != metav1.ConditionTrue || condition.Reason != tt.wantReason {
					t.Fatalf("Completed condition = %+v, want True/%s", condition, tt.wantReason)
				}
			} else if condition != nil && condition.Status == metav1.ConditionTrue {
				t.Fatalf("retryable failure terminalized the Run: %+v", condition)
			}
		})
	}
}

func TestDGDRRunPermanentErrorClassification(t *testing.T) {
	invalid := apierrors.NewInvalid(schema.GroupKind{Kind: "Job"}, "search-1", field.ErrorList{
		field.Invalid(field.NewPath("spec"), nil, "invalid"),
	})
	tests := []struct {
		name string
		err  error
		want bool
	}{
		{name: "explicit immutable execution error", err: &dgdrRunPermanentExecutionError{cause: errors.New("invalid input")}, want: true},
		{name: "ownership conflict", err: &commoncontroller.OwnershipConflictError{Cause: errors.New("foreign owner")}, want: true},
		{name: "invalid API object", err: invalid, want: true},
		{name: "forbidden", err: apierrors.NewForbidden(schema.GroupResource{Group: "batch", Resource: "jobs"}, "search-1", errors.New("denied"))},
		{name: "conflict", err: apierrors.NewConflict(schema.GroupResource{Group: "batch", Resource: "jobs"}, "search-1", errors.New("changed"))},
		{name: "timeout", err: apierrors.NewTimeoutError("timed out", 1)},
		{name: "retryable invalid discovery error", err: &dgdrRunRetryableError{cause: invalid}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := dgdrRunIsPermanentError(tt.err); got != tt.want {
				t.Fatalf("permanent = %t, want %t for %T %v", got, tt.want, tt.err, tt.err)
			}
		})
	}
}

func TestV1beta2RunControllerUsesUnfilteredTopologyForAnExternalHardwareSKU(t *testing.T) {
	hardware := &fixedRunHardware{info: gpu.GPUInfo{Model: "Future Accelerator", GPUsPerNode: 8}}
	run := dgdrRunForTest(t, map[string]any{})
	*run.Spec.Hardware.GPU.SKU = "future_accelerator"
	f := newDGDRRunFixture(t, hardware, run)

	f.reconcile()

	if len(hardware.calls) != 1 || hardware.calls[0] != "" {
		t.Fatalf("hardware discovery filters = %v, want one unfiltered topology lookup", hardware.calls)
	}
	if got := f.job().Spec.Template.Spec.Containers[0].Image; got != f.reconciler.PlannerImage {
		t.Fatalf("sweeper image = %q, want %q", got, f.reconciler.PlannerImage)
	}
}

func TestV1beta2RunControllerRejectsAmbiguousTopologyForAnExternalHardwareSKU(t *testing.T) {
	hardware := &fixedRunHardware{info: gpu.GPUInfo{
		Model: "Future Accelerator", GPUsPerNode: 8, OtherModels: []string{"Another Accelerator"},
	}}
	run := dgdrRunForTest(t, map[string]any{})
	*run.Spec.Hardware.GPU.SKU = "future_accelerator"
	f := newDGDRRunFixture(t, hardware, run)

	_, err := f.reconciler.Reconcile(t.Context(), ctrl.Request{NamespacedName: client.ObjectKeyFromObject(run)})
	if err == nil || !strings.Contains(err.Error(), "among multiple GPU models") {
		t.Fatalf("reconcile error = %v, want ambiguous external-SKU topology failure", err)
	}
}

func TestV1beta2RunControllerFreezesExecutionBeforeCreatingTheJob(t *testing.T) {
	hardware := &fixedRunHardware{info: gpu.GPUInfo{GPUsPerNode: 8}}
	run := dgdrRunForTest(t, map[string]any{})
	f := newDGDRRunFixture(t, hardware, run)
	f.reconciler.DockerSecretRetriever = &fixedRunSecrets{byImage: map[string][]string{
		"example.com/dynamo-planner:1.2.3":  {"planner-secret", "shared-secret"},
		"example.com/dynamo-operator:1.2.3": {"shared-secret", "publisher-secret"},
	}}

	t.Log("Persist the immutable execution artifact before any Job exists")
	execution, err := f.reconciler.reconcileInput(t.Context(), run, nil, dgdrRunResourceName(run.Name))
	if err != nil {
		t.Fatalf("reconcile input: %v", err)
	}
	if execution.PlannerImage != "example.com/dynamo-planner:1.2.3" || execution.PublisherImage != "example.com/dynamo-operator:1.2.3" {
		t.Fatalf("frozen images = %q/%q", execution.PlannerImage, execution.PublisherImage)
	}

	t.Log("Simulate an operator upgrade before the next reconciliation")
	f.reconciler.PlannerImage = "example.com/dynamo-planner:9.9.9"
	f.reconciler.OperatorImage = "example.com/dynamo-operator:9.9.9"
	f.reconciler.OperatorImagePullPolicy = corev1.PullNever
	f.reconciler.DockerSecretRetriever = &fixedRunSecrets{byImage: map[string][]string{
		"example.com/dynamo-planner:9.9.9": {"new-secret"},
	}}
	f.reconcile()

	t.Log("Verify the Job uses only the previously frozen identity and credentials")
	job := f.job()
	planner := containerNamed(t, job.Spec.Template.Spec.Containers, dgdrRunSweeperContainer)
	publisher := containerNamed(t, job.Spec.Template.Spec.Containers, dgdrRunPublisherContainer)
	if planner.Image != "example.com/dynamo-planner:1.2.3" || publisher.Image != "example.com/dynamo-operator:1.2.3" {
		t.Fatalf("Job images = %q/%q, want frozen images", planner.Image, publisher.Image)
	}
	wantSecrets := []corev1.LocalObjectReference{{Name: "planner-secret"}, {Name: "shared-secret"}, {Name: "publisher-secret"}}
	if !reflect.DeepEqual(job.Spec.Template.Spec.ImagePullSecrets, wantSecrets) {
		t.Fatalf("imagePullSecrets = %v, want %v", job.Spec.Template.Spec.ImagePullSecrets, wantSecrets)
	}
}

func TestV1beta2RunControllerFailsClosedWithoutExecutionConfig(t *testing.T) {
	f := newDGDRRunFixture(t, &fixedRunHardware{info: gpu.GPUInfo{GPUsPerNode: 8}}, dgdrRunForTest(t, map[string]any{}))
	f.reconcile()

	t.Log("Delete the immutable execution artifact while retaining the Job")
	job := f.job()
	configMap := &corev1.ConfigMap{}
	if err := f.client.Get(t.Context(), client.ObjectKeyFromObject(job), configMap); err != nil {
		t.Fatalf("get ConfigMap: %v", err)
	}
	if err := f.client.Delete(t.Context(), configMap); err != nil {
		t.Fatalf("delete ConfigMap: %v", err)
	}

	t.Log("Verify a real deletion fails closed instead of reconstructing or hot-looping")
	_, err := f.reconciler.Reconcile(t.Context(), ctrl.Request{NamespacedName: client.ObjectKeyFromObject(f.run())})
	if err == nil || !strings.Contains(err.Error(), "immutable execution ConfigMap") {
		t.Fatalf("reconcile error = %v, want missing immutable execution ConfigMap", err)
	}
}

func TestV1beta2RunControllerWaitsForExecutionConfigCacheLag(t *testing.T) {
	f := newDGDRRunFixture(t, &fixedRunHardware{info: gpu.GPUInfo{GPUsPerNode: 8}}, dgdrRunForTest(t, map[string]any{}))
	f.reconcile()

	job := f.job()
	configMap := &corev1.ConfigMap{}
	if err := f.client.Get(t.Context(), client.ObjectKeyFromObject(job), configMap); err != nil {
		t.Fatalf("get ConfigMap: %v", err)
	}
	directReader := fake.NewClientBuilder().WithScheme(f.client.Scheme()).WithObjects(configMap.DeepCopy()).Build()
	if err := f.client.Delete(t.Context(), configMap); err != nil {
		t.Fatalf("remove ConfigMap from simulated cache: %v", err)
	}
	f.reconciler.APIReader = directReader

	t.Log("Wait only when the uncached API confirms that the immutable artifact still exists")
	result, err := f.reconciler.Reconcile(t.Context(), ctrl.Request{NamespacedName: client.ObjectKeyFromObject(f.run())})
	if err != nil || result.RequeueAfter != dgdrRunExecutionCacheRetry {
		t.Fatalf("reconcile result/error = %+v/%v, want bounded cache-lag retry", result, err)
	}
}

func TestV1beta2RunControllerRejectsJobExecutionSkew(t *testing.T) {
	f := newDGDRRunFixture(t, &fixedRunHardware{info: gpu.GPUInfo{GPUsPerNode: 8}}, dgdrRunForTest(t, map[string]any{}))
	f.reconcile()

	t.Log("Mutate the observed Job away from the immutable execution identity")
	job := f.job()
	container := &job.Spec.Template.Spec.Containers[0]
	container.Image = "example.com/dynamo-planner:other"
	if err := f.client.Update(t.Context(), job); err != nil {
		t.Fatalf("update Job: %v", err)
	}

	t.Log("Verify reconciliation detects the inconsistency instead of accepting skew")
	_, err := f.reconciler.Reconcile(t.Context(), ctrl.Request{NamespacedName: client.ObjectKeyFromObject(f.run())})
	if err == nil || !strings.Contains(err.Error(), "planner image identities differ") {
		t.Fatalf("reconcile error = %v, want planner identity mismatch", err)
	}
}

func TestV1beta2RunControllerReportsImagePullSecretErrors(t *testing.T) {
	f := newDGDRRunFixture(t, &fixedRunHardware{info: gpu.GPUInfo{GPUsPerNode: 8}}, dgdrRunForTest(t, map[string]any{}))
	f.reconciler.DockerSecretRetriever = &fixedRunSecrets{err: errors.New("registry index unavailable")}

	t.Log("Verify execution is not frozen without resolving its image credentials")
	_, err := f.reconciler.Reconcile(t.Context(), ctrl.Request{NamespacedName: client.ObjectKeyFromObject(f.run())})
	if err == nil || !strings.Contains(err.Error(), "registry index unavailable") {
		t.Fatalf("reconcile error = %v, want image pull secret error", err)
	}
}

func TestV1beta2RunControllerRejectsDigestOnlyPlannerImage(t *testing.T) {
	f := newDGDRRunFixture(t, &fixedRunHardware{info: gpu.GPUInfo{GPUsPerNode: 8}}, dgdrRunForTest(t, map[string]any{}))
	f.reconciler.PlannerImage = "example.com/dynamo-planner@sha256:0123"
	f.reconciler.DynamoVersion = "1.7.0"

	t.Log("Reject the Planner before freezing an unrelated runtime release tag")
	_, err := f.reconciler.Reconcile(t.Context(), ctrl.Request{NamespacedName: client.ObjectKeyFromObject(f.run())})
	if err == nil || !strings.Contains(err.Error(), "digest-only planner image cannot identify sibling runtime images") {
		t.Fatalf("reconcile error = %v, want digest-only Planner rejection", err)
	}
	jobs := &batchv1.JobList{}
	if err := f.client.List(t.Context(), jobs, client.InNamespace("default")); err != nil {
		t.Fatalf("list Jobs: %v", err)
	}
	if len(jobs.Items) != 0 {
		t.Fatalf("Jobs = %d, want none for an ambiguous runtime source", len(jobs.Items))
	}
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
	podSecurity := job.Spec.Template.Spec.SecurityContext
	if podSecurity == nil || podSecurity.RunAsNonRoot == nil || !*podSecurity.RunAsNonRoot ||
		podSecurity.SeccompProfile == nil || podSecurity.SeccompProfile.Type != corev1.SeccompProfileTypeRuntimeDefault {
		t.Fatalf("pod security context = %+v, want runAsNonRoot and RuntimeDefault seccomp", podSecurity)
	}
	sweeper := containerNamed(t, job.Spec.Template.Spec.Containers, dgdrRunSweeperContainer)
	publisher := containerNamed(t, job.Spec.Template.Spec.Containers, dgdrRunPublisherContainer)
	for _, container := range []corev1.Container{sweeper, publisher} {
		security := container.SecurityContext
		if security == nil || security.AllowPrivilegeEscalation == nil || *security.AllowPrivilegeEscalation ||
			security.Capabilities == nil || !reflect.DeepEqual(security.Capabilities.Drop, []corev1.Capability{"ALL"}) {
			t.Fatalf("container %q security context = %+v, want restricted defaults", container.Name, security)
		}
	}
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
		"--output-dir", dgdrRunSnapshotPath,
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
	if !hasMountAt(sweeper.VolumeMounts, dgdrRunSnapshotVolume, dgdrRunSnapshotPath) ||
		!hasMountAt(publisher.VolumeMounts, dgdrRunSnapshotVolume, dgdrRunSnapshotPath) {
		t.Fatal("sweeper and publisher do not share the frozen snapshot directory")
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

func TestDGDRRunStatusInputPreservesLargeIntegers(t *testing.T) {
	const seed = "9007199254740993"

	t.Log("Project the frozen YAML input into status without decoding numbers through float64")
	input, err := dgdrRunStatusInput("optimizer:\n  seed: " + seed + "\n")
	if err != nil {
		t.Fatalf("project status input: %v", err)
	}
	if got := string(input.Raw); !strings.Contains(got, `"seed":`+seed) {
		t.Fatalf("status input lost integer precision: %s", got)
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
	for key, want := range map[string]string{
		dgdrRunPlannerImageKey:        "example.com/dynamo-planner:1.2.3",
		dgdrRunPlannerPullPolicyKey:   string(corev1.PullIfNotPresent),
		dgdrRunRuntimeImageSourceKey:  "example.com/dynamo-planner:1.2.3",
		dgdrRunPublisherImageKey:      "example.com/dynamo-operator:1.2.3",
		dgdrRunPublisherPullPolicyKey: string(corev1.PullAlways),
		dgdrRunDynamoVersionKey:       "1.2.3",
		dgdrRunGPUsPerNodeKey:         "8",
		dgdrRunSnapshotPathKey:        dgdrRunSnapshotPath,
	} {
		if got := configMap.Data[key]; got != want {
			t.Fatalf("ConfigMap data[%q] = %q, want %q", key, got, want)
		}
	}
	var input map[string]any
	if err := yaml.Unmarshal([]byte(configMap.Data[dgdrRunInputKey]), &input); err != nil {
		t.Fatalf("decode input ConfigMap: %v", err)
	}
	output := input["dgdr_run"].(map[string]any)
	if output["name"] != "search-1" || output["max_candidates"] != float64(2) {
		t.Fatalf("operator-owned identity and candidate cap were not restored: %v", output)
	}
	if output["snapshot_dir"] != dgdrRunSnapshotPath || output["namespace"] != "default" {
		t.Fatalf("operator-owned transport overrides were not restored: %v", output)
	}
	if output["runtime_image"] != "example.com/dynamo-planner:1.2.3" || output["runtime_version_override"] != "1.2.3" {
		t.Fatalf("operator-owned runtime identity was not restored: %v", output)
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

func TestDGDRRunInputResolvesRuntimeVersion(t *testing.T) {
	tests := []struct {
		name          string
		plannerImage  string
		dynamoVersion string
		wantOverride  string
		wantRuntime   string
		wantErr       bool
	}{
		{name: "canonical planner tag", plannerImage: "example.com/dynamo-planner:1.7.0", wantOverride: "1.7.0", wantRuntime: "example.com/dynamo-planner:1.7.0"},
		{name: "v-prefixed planner tag", plannerImage: "example.com/dynamo-planner:v1.7.0", wantOverride: "1.7.0", wantRuntime: "example.com/dynamo-planner:v1.7.0"},
		{name: "prerelease planner tag", plannerImage: "example.com/dynamo-planner:1.7.0-rc.1", wantOverride: "1.7.0", wantRuntime: "example.com/dynamo-planner:1.7.0-rc.1"},
		{name: "build metadata planner tag", plannerImage: "example.com/dynamo-planner:1.7.0+build.5", wantOverride: "1.7.0", wantRuntime: "example.com/dynamo-planner:1.7.0+build.5"},
		{name: "tag and digest", plannerImage: "example.com/dynamo-planner:v1.7.0-rc.1@sha256:0123", wantOverride: "1.7.0", wantRuntime: "example.com/dynamo-planner:v1.7.0-rc.1@sha256:0123"},
		{name: "digest only", plannerImage: "example.com/dynamo-planner@sha256:0123", dynamoVersion: "1.7.0", wantErr: true},
		{name: "opaque CI tag", plannerImage: "example.com/dynamo-planner:git-sha", dynamoVersion: "1.7.0", wantOverride: "1.7.0", wantRuntime: "example.com/dynamo-planner:git-sha"},
		{name: "untagged planner image", plannerImage: "example.com/dynamo-planner", dynamoVersion: "1.7.0", wantErr: true},
		{name: "opaque tag without canonical version", plannerImage: "example.com/dynamo-planner:git-sha", dynamoVersion: "1.7", wantErr: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			version, err := dgdrRunDynamoVersion(tt.plannerImage, tt.dynamoVersion)
			var runtimeImage string
			if err == nil {
				runtimeImage, err = dgdrRunRuntimeImageSource(tt.plannerImage)
			}
			if (err != nil) != tt.wantErr {
				t.Fatalf("resolve execution identity error = %v, wantErr %t", err, tt.wantErr)
			}
			if tt.wantErr {
				return
			}
			input, err := dgdrRunInput(dgdrRunForTest(t, map[string]any{}), runtimeImage, version, 8)
			if err != nil {
				t.Fatalf("lower input: %v", err)
			}
			output := input["dgdr_run"].(map[string]any)
			if got := output["runtime_version_override"]; got != tt.wantOverride {
				t.Fatalf("dgdr_run.runtime_version_override = %v, want %s", got, tt.wantOverride)
			}
			if got := output["runtime_image"]; got != tt.wantRuntime {
				t.Fatalf("dgdr_run.runtime_image = %v, want %s", got, tt.wantRuntime)
			}
		})
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
		Name:      job.Name + "-pod",
		Namespace: job.Namespace,
		Labels:    map[string]string{batchv1.JobNameLabel: job.Name},
		OwnerReferences: []metav1.OwnerReference{{
			APIVersion: batchv1.SchemeGroupVersion.String(), Kind: "Job", Name: job.Name, UID: job.UID,
			Controller: new(true),
		}},
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

func TestV1beta2RunControllerPrefersPublisherOutcomeWhenBothContainersFail(t *testing.T) {
	f := newDGDRRunFixture(t, &fixedRunHardware{info: gpu.GPUInfo{GPUsPerNode: 8}}, dgdrRunForTest(t, map[string]any{}))
	f.reconcile()
	job := f.job()
	outcome, err := json.Marshal(dgdrrunpublisher.TerminalOutcome{
		Phase: "Failed", Message: "terminal snapshot is invalid", Error: "candidate rank is duplicated",
	})
	if err != nil {
		t.Fatalf("encode terminal outcome: %v", err)
	}

	t.Log("Report both a generic sweeper failure and the publisher's structured protocol outcome")
	pod := &corev1.Pod{
		Name:      job.Name + "-pod",
		Namespace: job.Namespace,
		Labels:    map[string]string{batchv1.JobNameLabel: job.Name},
		OwnerReferences: []metav1.OwnerReference{{
			APIVersion: batchv1.SchemeGroupVersion.String(), Kind: "Job", Name: job.Name, UID: job.UID,
			Controller: new(true),
		}},
	}
	if err := f.client.Create(t.Context(), pod); err != nil {
		t.Fatalf("create Job pod: %v", err)
	}
	pod.Status.ContainerStatuses = []corev1.ContainerStatus{
		{
			Name: dgdrRunSweeperContainer,
			State: corev1.ContainerState{Terminated: &corev1.ContainerStateTerminated{
				ExitCode: 1, Message: "planner process exited",
			}},
		},
		{
			Name: dgdrRunPublisherContainer,
			State: corev1.ContainerState{Terminated: &corev1.ContainerStateTerminated{
				ExitCode: int32(dgdrrunpublisher.ExitProtocolViolation), Message: string(outcome),
			}},
		},
	}
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

	t.Log("Prefer and decode the publisher outcome instead of reporting the sweeper exit")
	f.reconcile()
	run := f.run()
	condition := meta.FindStatusCondition(run.Status.Conditions, nvidiacomv1beta2.DGDRRunConditionCompleted)
	if condition == nil || condition.Reason != dgdrRunReasonProtocolViolation {
		t.Fatalf("Completed condition = %+v, want %s", condition, dgdrRunReasonProtocolViolation)
	}
	if condition.Message != "terminal snapshot is invalid: candidate rank is duplicated" {
		t.Fatalf("failure message = %q, want decoded publisher diagnostics", condition.Message)
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

func TestRequireDGDRV1beta2RESTMappingsIncludesCandidates(t *testing.T) {
	groupVersion := nvidiacomv1beta2.GroupVersion
	mapper := meta.NewDefaultRESTMapper([]schema.GroupVersion{groupVersion})
	mapper.Add(groupVersion.WithKind("DynamoGraphDeploymentRequest"), meta.RESTScopeNamespace)
	mapper.Add(groupVersion.WithKind("DynamoGraphDeploymentRun"), meta.RESTScopeNamespace)

	t.Log("Verify startup fails when Candidate publication cannot succeed")
	err := requireDGDRV1beta2RESTMappings(mapper)
	if err == nil || !strings.Contains(err.Error(), "DynamoGraphDeploymentCandidate") {
		t.Fatalf("preflight error = %v, want missing Candidate API", err)
	}

	t.Log("Register the complete execution API and verify preflight succeeds")
	mapper.Add(groupVersion.WithKind("DynamoGraphDeploymentCandidate"), meta.RESTScopeNamespace)
	if err := requireDGDRV1beta2RESTMappings(mapper); err != nil {
		t.Fatalf("complete API preflight: %v", err)
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
