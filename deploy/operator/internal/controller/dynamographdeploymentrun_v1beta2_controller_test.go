/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package controller

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"reflect"
	"strings"
	"testing"
	"time"

	nvidiacomv1beta2 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta2"
	commoncontroller "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dgdrrunpublisher"
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

type dgdrRunFixture struct {
	t          *testing.T
	client     client.Client
	reconciler *DynamoGraphDeploymentRunV1beta2Reconciler
}

type staleJobListOnceClient struct {
	client.Client
	hide bool
}

func (c *staleJobListOnceClient) List(
	ctx context.Context,
	list client.ObjectList,
	opts ...client.ListOption,
) error {
	if _, ok := list.(*batchv1.JobList); ok && c.hide {
		c.hide = false
		return nil
	}
	return c.Client.List(ctx, list, opts...)
}

func newDGDRRunFixture(t *testing.T, objects ...client.Object) *dgdrRunFixture {
	t.Helper()
	return newDGDRRunFixtureWithInterceptor(t, interceptor.Funcs{}, objects...)
}

func newDGDRRunFixtureWithInterceptor(
	t *testing.T,
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
		WithIndex(&nvidiacomv1beta2.DynamoGraphDeploymentRun{}, dgdrRunTraceClaimIndex, dgdrRunTraceClaimNames).
		WithStatusSubresource(&nvidiacomv1beta2.DynamoGraphDeploymentRun{}, &batchv1.Job{}, &corev1.Pod{}).
		WithInterceptorFuncs(funcs).
		Build()
	return &dgdrRunFixture{
		t:      t,
		client: c,
		reconciler: &DynamoGraphDeploymentRunV1beta2Reconciler{
			Client:                  c,
			APIReader:               c,
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
			Hardware: &nvidiacomv1beta2.HardwareSpec{
				GPU: &nvidiacomv1beta2.GPUHardwareSpec{
					SKU: new("h200_sxm"), Budget: ptr.To[int32](16),
				},
				NumGPUsPerNode: ptr.To[int32](8),
			},
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
	f := newDGDRRunFixture(t, run)

	f.reconcile()
	f.reconcile()

	job := f.job()
	assertDGDRRunJobIsolation(t, job, run, f.reconciler.PlannerImage, f.reconciler.OperatorImage)
	assertDGDRRunInput(t, f.client, job)
	assertDGDRRunSupportResources(t, f.client, job, run)
	stored := f.run()
	assertDGDRRunPendingStatus(t, stored, job.Name)
	assertDGDRRunStatusInputMatchesConfigMap(t, f.client, stored, job)
}

func TestV1beta2RunControllerRetainsAndTerminalizesASupersededRun(t *testing.T) {
	f := newDGDRRunFixture(t, dgdrRunForTest(t, map[string]any{}))
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
	run := dgdrRunForTest(t, map[string]any{})
	run.Annotations[dgdrV1beta2CancelRequestedByGenerationAnnotation] = "2"
	f := newDGDRRunFixture(t, run)

	f.reconcile()
	stored := f.run()
	condition := meta.FindStatusCondition(stored.Status.Conditions, nvidiacomv1beta2.DGDRRunConditionCompleted)
	if condition == nil || condition.Status != metav1.ConditionTrue || condition.Reason != dgdrRunReasonSuperseded {
		t.Fatalf("Completed condition = %+v, want True/%s", condition, dgdrRunReasonSuperseded)
	}
	if jobs := f.jobs(); len(jobs) != 0 {
		t.Fatalf("Jobs = %d, want none", len(jobs))
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
			run := dgdrRunForTest(t, map[string]any{})
			run.Annotations[dgdrV1beta2GenerationAnnotation] = tt.runGeneration
			run.Annotations[dgdrV1beta2CancelRequestedByGenerationAnnotation] = tt.cancelGeneration
			f := newDGDRRunFixture(t, run)

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
				return
			}
			if jobs := f.jobs(); len(jobs) != 1 {
				t.Fatalf("Jobs = %d, want normal execution", len(jobs))
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
				f := newDGDRRunFixture(t, run, collision)

				before := collision.DeepCopyObject().(client.Object)
				if err := f.client.Get(t.Context(), client.ObjectKeyFromObject(collision), before); err != nil {
					t.Fatalf("get collision before reconcile: %v", err)
				}
				result, err := f.reconciler.Reconcile(t.Context(), ctrl.Request{NamespacedName: client.ObjectKeyFromObject(run)})
				if err != nil || result.RequeueAfter != dgdrRunExecutionCacheRetry {
					t.Fatalf("marker result/error = %+v/%v, want durable ownership-failure marker", result, err)
				}
				_, err = f.reconciler.Reconcile(t.Context(), ctrl.Request{NamespacedName: client.ObjectKeyFromObject(run)})
				if !errors.Is(err, reconcile.TerminalError(nil)) {
					t.Fatalf("reconcile error = %v, want terminal ownership conflict", err)
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
				f := newDGDRRunFixture(t, run)
				f.reconciler.PlannerImage = "example.com/dynamo-planner"
				f.reconciler.DynamoVersion = "1.7.0"
				return f
			},
			wantTerminal: true, wantReason: dgdrRunReasonInvalidExecution,
			wantErrMessage: "shared release tag is required",
		},
		{
			name: "invalid Kubernetes resource is terminal",
			newFixture: func(t *testing.T) *dgdrRunFixture {
				run := dgdrRunForTest(t, map[string]any{})
				return newDGDRRunFixtureWithInterceptor(t, interceptor.Funcs{
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
			name: "registry secret error is retryable",
			newFixture: func(t *testing.T) *dgdrRunFixture {
				f := newDGDRRunFixture(t, dgdrRunForTest(t, map[string]any{}))
				f.reconciler.DockerSecretRetriever = &fixedRunSecrets{err: errors.New("registry index unavailable")}
				return f
			},
			wantErrMessage: "registry index unavailable",
		},
		{
			name: "terminal status write failure remains retryable",
			newFixture: func(t *testing.T) *dgdrRunFixture {
				run := dgdrRunForTest(t, map[string]any{})
				f := newDGDRRunFixtureWithInterceptor(t, interceptor.Funcs{
					SubResourcePatch: func(
						_ context.Context, _ client.Client, _ string, _ client.Object,
						_ client.Patch, _ ...client.SubResourcePatchOption,
					) error {
						return errors.New("status writer unavailable")
					},
				}, run)
				f.reconciler.PlannerImage = "example.com/dynamo-planner"
				f.reconciler.DynamoVersion = "1.7.0"
				return f
			},
			wantErrMessage: "status writer unavailable",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			f := tt.newFixture(t)
			result, err := f.reconciler.Reconcile(t.Context(), ctrl.Request{NamespacedName: client.ObjectKeyFromObject(f.run())})
			if tt.wantTerminal {
				if err != nil || result.RequeueAfter != dgdrRunExecutionCacheRetry {
					t.Fatalf("marker result/error = %+v/%v, want durable state retry", result, err)
				}
				condition := meta.FindStatusCondition(f.run().Status.Conditions, nvidiacomv1beta2.DGDRRunConditionCompleted)
				if condition == nil || condition.Status != metav1.ConditionFalse || condition.Reason != tt.wantReason {
					t.Fatalf("marker condition = %+v, want False/%s", condition, tt.wantReason)
				}
				_, err = f.reconciler.Reconcile(t.Context(), ctrl.Request{NamespacedName: client.ObjectKeyFromObject(f.run())})
			}
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
		{name: "explicitly retryable invalid error", err: &dgdrRunRetryableError{cause: invalid}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := dgdrRunIsPermanentError(tt.err); got != tt.want {
				t.Fatalf("permanent = %t, want %t for %T %v", got, tt.want, tt.err, tt.err)
			}
		})
	}
}

func TestV1beta2RunControllerOmitsUnspecifiedGPUsPerNode(t *testing.T) {
	run := dgdrRunForTest(t, map[string]any{})
	run.Spec.Hardware.NumGPUsPerNode = nil
	f := newDGDRRunFixture(t, run)

	f.reconcile()

	input := dgdrRunInputFromConfigMap(t, f.client, f.job())
	output := input["dgdr_run"].(map[string]any)
	if _, found := output["num_gpus_per_node"]; found {
		t.Fatalf("dgdr_run.num_gpus_per_node = %v, want adapter-derived node width", output["num_gpus_per_node"])
	}
}

func TestV1beta2RunControllerLowersHardwareOptOutFromInputOverrides(t *testing.T) {
	run := dgdrRunForTest(t, map[string]any{
		"engine": map[string]any{"hardware": "future_accelerator"},
		"optimization": map[string]any{
			"constraints": map[string]any{"max_candidate_gpus": int64(24)},
		},
	})
	run.Spec.Hardware = &nvidiacomv1beta2.HardwareSpec{}
	f := newDGDRRunFixture(t, run)

	f.reconcile()

	input := dgdrRunInputFromConfigMap(t, f.client, f.job())
	output := input["dgdr_run"].(map[string]any)
	if _, found := output["num_gpus_per_node"]; found {
		t.Fatalf("dgdr_run.num_gpus_per_node = %v, want AISimulate system-spec resolution", output["num_gpus_per_node"])
	}
	if got := input["engine"].(map[string]any)["hardware"]; got != "future_accelerator" {
		t.Fatalf("engine.hardware = %v, want authoritative input override", got)
	}
}

func TestV1beta2RunControllerFreezesExecutionBeforeCreatingTheJob(t *testing.T) {
	run := dgdrRunForTest(t, map[string]any{})
	f := newDGDRRunFixture(t, run)
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
	f := newDGDRRunFixture(t, dgdrRunForTest(t, map[string]any{}))
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

	t.Log("Persist the permanent failure marker before taking cleanup action")
	result, err := f.reconciler.Reconcile(t.Context(), ctrl.Request{NamespacedName: client.ObjectKeyFromObject(f.run())})
	if err != nil || result.RequeueAfter != dgdrRunExecutionCacheRetry {
		t.Fatalf("marker result/error = %+v/%v, want durable state retry", result, err)
	}
	if jobs := f.jobs(); len(jobs) != 1 {
		t.Fatalf("Jobs = %d, want no deletion in the marker-writing reconciliation", len(jobs))
	}
	condition := meta.FindStatusCondition(f.run().Status.Conditions, nvidiacomv1beta2.DGDRRunConditionCompleted)
	if condition == nil || condition.Status != metav1.ConditionFalse || condition.Reason != dgdrRunReasonInvalidExecution {
		t.Fatalf("cleanup condition = %+v, want False/%s", condition, dgdrRunReasonInvalidExecution)
	}

	t.Log("A later reconciliation observes the marker and stops the Job")
	result, err = f.reconciler.Reconcile(t.Context(), ctrl.Request{NamespacedName: client.ObjectKeyFromObject(f.run())})
	if err != nil || result.RequeueAfter != dgdrRunExecutionCacheRetry {
		t.Fatalf("cleanup result/error = %+v/%v, want deletion-observation retry", result, err)
	}
	if jobs := f.jobs(); len(jobs) != 0 {
		t.Fatalf("Jobs = %d, want the invalid execution stopped", len(jobs))
	}

	t.Log("Publish Completed only after Job absence is observed")
	_, err = f.reconciler.Reconcile(t.Context(), ctrl.Request{NamespacedName: client.ObjectKeyFromObject(f.run())})
	if err == nil || !strings.Contains(err.Error(), "immutable execution ConfigMap") {
		t.Fatalf("reconcile error = %v, want missing immutable execution ConfigMap", err)
	}
}

func TestV1beta2RunControllerWaitsForExecutionConfigCacheLag(t *testing.T) {
	f := newDGDRRunFixture(t, dgdrRunForTest(t, map[string]any{}))
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

func TestV1beta2RunControllerRejectsMutableExecutionConfig(t *testing.T) {
	run := dgdrRunForTest(t, map[string]any{})
	f := newDGDRRunFixture(t, run)
	f.reconcile()
	job := f.job()
	configMap := &corev1.ConfigMap{}
	if err := f.client.Get(t.Context(), client.ObjectKeyFromObject(job), configMap); err != nil {
		t.Fatalf("get execution ConfigMap: %v", err)
	}
	configMap.Immutable = new(false)

	_, err := f.reconciler.existingRunExecution(configMap, run, job)
	if err == nil || !strings.Contains(err.Error(), "must be immutable") {
		t.Fatalf("existing execution error = %v, want mutable ConfigMap rejection", err)
	}
}

func TestV1beta2RunControllerRejectsJobExecutionSkew(t *testing.T) {
	f := newDGDRRunFixture(t, dgdrRunForTest(t, map[string]any{}))
	f.reconcile()

	t.Log("Mutate the observed Job away from the immutable execution identity")
	job := f.job()
	container := &job.Spec.Template.Spec.Containers[0]
	container.Image = "example.com/dynamo-planner:other"
	if err := f.client.Update(t.Context(), job); err != nil {
		t.Fatalf("update Job: %v", err)
	}

	t.Log("Persist the failure marker without deleting the skewed Job")
	result, err := f.reconciler.Reconcile(t.Context(), ctrl.Request{NamespacedName: client.ObjectKeyFromObject(f.run())})
	if err != nil || result.RequeueAfter != dgdrRunExecutionCacheRetry {
		t.Fatalf("marker result/error = %+v/%v, want durable state retry", result, err)
	}
	if jobs := f.jobs(); len(jobs) != 1 {
		t.Fatalf("Jobs = %d, want no deletion in the marker-writing reconciliation", len(jobs))
	}

	t.Log("Observe the marker and stop the skewed Job")
	result, err = f.reconciler.Reconcile(t.Context(), ctrl.Request{NamespacedName: client.ObjectKeyFromObject(f.run())})
	if err != nil || result.RequeueAfter != dgdrRunExecutionCacheRetry {
		t.Fatalf("cleanup result/error = %+v/%v, want deletion-observation retry", result, err)
	}
	if jobs := f.jobs(); len(jobs) != 0 {
		t.Fatalf("Jobs = %d, want the skewed execution stopped", len(jobs))
	}

	_, err = f.reconciler.Reconcile(t.Context(), ctrl.Request{NamespacedName: client.ObjectKeyFromObject(f.run())})
	if err == nil || !strings.Contains(err.Error(), "planner image identities differ") {
		t.Fatalf("terminal reconcile error = %v, want planner identity mismatch", err)
	}
}

func TestValidateDGDRRunJobExecutionRejectsImmutableBindingSkew(t *testing.T) {
	run := dgdrRunForTest(t, map[string]any{})
	f := newDGDRRunFixture(t, run)
	f.reconcile()
	job := f.job()
	configMap := &corev1.ConfigMap{}
	if err := f.client.Get(t.Context(), client.ObjectKeyFromObject(job), configMap); err != nil {
		t.Fatalf("get execution ConfigMap: %v", err)
	}
	execution, err := dgdrRunExecutionFromConfigMap(configMap)
	if err != nil {
		t.Fatalf("decode execution: %v", err)
	}

	tests := []struct {
		name    string
		mutate  func(*batchv1.Job)
		wantErr string
	}{
		{
			name: "backoff limit",
			mutate: func(job *batchv1.Job) {
				job.Spec.BackoffLimit = ptr.To[int32](1)
			},
			wantErr: "Job backoff limit differs",
		},
		{
			name: "parallel sweeps",
			mutate: func(job *batchv1.Job) {
				job.Spec.Parallelism = ptr.To[int32](2)
			},
			wantErr: "Job cardinality differs",
		},
		{
			name: "multiple completions",
			mutate: func(job *batchv1.Job) {
				job.Spec.Completions = ptr.To[int32](2)
			},
			wantErr: "Job cardinality differs",
		},
		{
			name: "indexed completion mode",
			mutate: func(job *batchv1.Job) {
				job.Spec.CompletionMode = ptr.To(batchv1.IndexedCompletion)
			},
			wantErr: "Job completion mode differs",
		},
		{
			name: "suspended execution",
			mutate: func(job *batchv1.Job) {
				job.Spec.Suspend = new(true)
			},
			wantErr: "Job execution policy differs",
		},
		{
			name: "active deadline",
			mutate: func(job *batchv1.Job) {
				job.Spec.ActiveDeadlineSeconds = ptr.To[int64](1)
			},
			wantErr: "Job execution policy differs",
		},
		{
			name: "nondefault pod replacement policy",
			mutate: func(job *batchv1.Job) {
				job.Spec.PodReplacementPolicy = ptr.To(batchv1.Failed)
			},
			wantErr: "Job execution policy differs",
		},
		{
			name: "restart policy",
			mutate: func(job *batchv1.Job) {
				job.Spec.Template.Spec.RestartPolicy = corev1.RestartPolicyAlways
			},
			wantErr: "pod restart policy differs",
		},
		{
			name: "host PID namespace",
			mutate: func(job *batchv1.Job) {
				job.Spec.Template.Spec.HostPID = true
			},
			wantErr: "pod namespace isolation differs",
		},
		{
			name: "host IPC namespace",
			mutate: func(job *batchv1.Job) {
				job.Spec.Template.Spec.HostIPC = true
			},
			wantErr: "pod namespace isolation differs",
		},
		{
			name: "host network namespace",
			mutate: func(job *batchv1.Job) {
				job.Spec.Template.Spec.HostNetwork = true
			},
			wantErr: "pod namespace isolation differs",
		},
		{
			name: "shared process namespace",
			mutate: func(job *batchv1.Job) {
				job.Spec.Template.Spec.ShareProcessNamespace = new(true)
			},
			wantErr: "pod namespace isolation differs",
		},
		{
			name: "init container",
			mutate: func(job *batchv1.Job) {
				job.Spec.Template.Spec.InitContainers = []corev1.Container{{Name: "unexpected", Image: "example.invalid/init"}}
			},
			wantErr: "init containers are not allowed",
		},
		{
			name: "pod runAsNonRoot",
			mutate: func(job *batchv1.Job) {
				job.Spec.Template.Spec.SecurityContext.RunAsNonRoot = new(false)
			},
			wantErr: "pod runAsNonRoot differs",
		},
		{
			name: "pod seccomp",
			mutate: func(job *batchv1.Job) {
				job.Spec.Template.Spec.SecurityContext.SeccompProfile.Type = corev1.SeccompProfileTypeUnconfined
			},
			wantErr: "pod seccomp profile differs",
		},
		{
			name: "pod fsGroup",
			mutate: func(job *batchv1.Job) {
				job.Spec.Template.Spec.SecurityContext.FSGroup = ptr.To[int64](1000)
			},
			wantErr: "pod fsGroup differs",
		},
		{
			name: "pod supplemental group",
			mutate: func(job *batchv1.Job) {
				job.Spec.Template.Spec.SecurityContext.SupplementalGroups = []int64{65532}
			},
			wantErr: "pod supplemental groups differ",
		},
		{
			name: "planner privilege escalation",
			mutate: func(job *batchv1.Job) {
				job.Spec.Template.Spec.Containers[0].SecurityContext.AllowPrivilegeEscalation = new(true)
			},
			wantErr: "planner security context differs",
		},
		{
			name: "planner capabilities",
			mutate: func(job *batchv1.Job) {
				job.Spec.Template.Spec.Containers[0].SecurityContext.Capabilities.Add = []corev1.Capability{"NET_ADMIN"}
			},
			wantErr: "planner security context differs",
		},
		{
			name: "planner user",
			mutate: func(job *batchv1.Job) {
				job.Spec.Template.Spec.Containers[0].SecurityContext.RunAsUser = ptr.To[int64](0)
			},
			wantErr: "planner security context differs",
		},
		{
			name: "planner unmasked proc mount",
			mutate: func(job *batchv1.Job) {
				job.Spec.Template.Spec.Containers[0].SecurityContext.ProcMount = ptr.To(corev1.UnmaskedProcMount)
			},
			wantErr: "planner security context differs",
		},
		{
			name: "publisher security",
			mutate: func(job *batchv1.Job) {
				job.Spec.Template.Spec.Containers[1].SecurityContext.Privileged = new(true)
			},
			wantErr: "publisher security context differs",
		},
		{
			name: "publisher seccomp override",
			mutate: func(job *batchv1.Job) {
				job.Spec.Template.Spec.Containers[1].SecurityContext.SeccompProfile = &corev1.SeccompProfile{
					Type: corev1.SeccompProfileTypeUnconfined,
				}
			},
			wantErr: "publisher security context differs",
		},
		{
			name: "publisher unmasked proc mount",
			mutate: func(job *batchv1.Job) {
				job.Spec.Template.Spec.Containers[1].SecurityContext.ProcMount = ptr.To(corev1.UnmaskedProcMount)
			},
			wantErr: "publisher security context differs",
		},
		{
			name: "automatic service account token",
			mutate: func(job *batchv1.Job) {
				job.Spec.Template.Spec.AutomountServiceAccountToken = new(true)
			},
			wantErr: "publisher service-account identities differ",
		},
		{
			name: "publisher token exposed to planner",
			mutate: func(job *batchv1.Job) {
				job.Spec.Template.Spec.Containers[0].VolumeMounts = append(
					job.Spec.Template.Spec.Containers[0].VolumeMounts,
					corev1.VolumeMount{Name: dgdrRunServiceAccountVolume, MountPath: dgdrRunServiceAccountMountPath},
				)
			},
			wantErr: "planner volume mounts differ",
		},
		{
			name: "planner command",
			mutate: func(job *batchv1.Job) {
				job.Spec.Template.Spec.Containers[0].Command = []string{"python"}
			},
			wantErr: "planner command or arguments differ",
		},
		{
			name: "planner environment secret",
			mutate: func(job *batchv1.Job) {
				job.Spec.Template.Spec.Containers[0].EnvFrom = []corev1.EnvFromSource{{
					SecretRef: &corev1.SecretEnvSource{},
				}}
			},
			wantErr: "planner environment differs",
		},
		{
			name: "planner lifecycle hook",
			mutate: func(job *batchv1.Job) {
				job.Spec.Template.Spec.Containers[0].Lifecycle = &corev1.Lifecycle{
					PostStart: &corev1.LifecycleHandler{Exec: &corev1.ExecAction{Command: []string{"unexpected"}}},
				}
			},
			wantErr: "planner lifecycle differs",
		},
		{
			name: "publisher Run UID",
			mutate: func(job *batchv1.Job) {
				job.Spec.Template.Spec.Containers[1].Args[5] = "other-uid"
			},
			wantErr: "publisher command, arguments or identity differ",
		},
		{
			name: "publisher environment secret",
			mutate: func(job *batchv1.Job) {
				job.Spec.Template.Spec.Containers[1].EnvFrom = []corev1.EnvFromSource{{
					ConfigMapRef: &corev1.ConfigMapEnvSource{},
				}}
			},
			wantErr: "publisher environment differs",
		},
		{
			name: "publisher lifecycle hook",
			mutate: func(job *batchv1.Job) {
				job.Spec.Template.Spec.Containers[1].Lifecycle = &corev1.Lifecycle{
					PreStop: &corev1.LifecycleHandler{Exec: &corev1.ExecAction{Command: []string{"unexpected"}}},
				}
			},
			wantErr: "publisher lifecycle differs",
		},
		{
			name: "input ConfigMap source",
			mutate: func(job *batchv1.Job) {
				job.Spec.Template.Spec.Volumes[0].ConfigMap.Name = "other-input"
			},
			wantErr: "input, snapshot or publisher credential volumes differ",
		},
		{
			name: "snapshot mount",
			mutate: func(job *batchv1.Job) {
				job.Spec.Template.Spec.Containers[0].VolumeMounts[1].MountPath = "/other-output"
			},
			wantErr: "planner volume mounts differ",
		},
		{
			name: "publisher token mount",
			mutate: func(job *batchv1.Job) {
				job.Spec.Template.Spec.Containers[1].VolumeMounts[1].Name = "other-token"
			},
			wantErr: "publisher volume mounts differ",
		},
		{
			name: "service account",
			mutate: func(job *batchv1.Job) {
				job.Spec.Template.Spec.ServiceAccountName = "other-publisher"
			},
			wantErr: "publisher service-account identities differ",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			mutated := job.DeepCopy()
			tt.mutate(mutated)
			err := validateDGDRRunJobExecution(mutated, run, configMap.Name, execution)
			if err == nil || !strings.Contains(err.Error(), tt.wantErr) {
				t.Fatalf("validation error = %v, want %q", err, tt.wantErr)
			}
		})
	}

	t.Run("API defaulted cardinality remains compatible", func(t *testing.T) {
		defaulted := job.DeepCopy()
		defaulted.Spec.Parallelism = ptr.To[int32](1)
		defaulted.Spec.Completions = ptr.To[int32](1)
		defaulted.Spec.CompletionMode = ptr.To(batchv1.NonIndexedCompletion)
		defaulted.Spec.Suspend = new(false)
		defaulted.Spec.ManualSelector = new(false)
		defaulted.Spec.PodReplacementPolicy = ptr.To(batchv1.TerminatingOrFailed)
		defaulted.Spec.Template.Spec.ShareProcessNamespace = new(false)
		defaulted.Spec.Template.Spec.Containers[0].SecurityContext.ProcMount = ptr.To(corev1.DefaultProcMount)
		defaulted.Spec.Template.Spec.Containers[1].SecurityContext.ProcMount = ptr.To(corev1.DefaultProcMount)
		if err := validateDGDRRunJobExecution(defaulted, run, configMap.Name, execution); err != nil {
			t.Fatalf("validation error = %v, want API defaults accepted", err)
		}
	})

	t.Run("a Job from an operator that set fsGroup 65532 remains compatible", func(t *testing.T) {
		legacy := job.DeepCopy()
		legacy.Spec.Template.Spec.SecurityContext.FSGroup = ptr.To[int64](65532)
		if err := validateDGDRRunJobExecution(legacy, run, configMap.Name, execution); err != nil {
			t.Fatalf("validation error = %v, want the legacy fsGroup accepted for a Run without a trace", err)
		}
	})
}

func TestValidateDGDRRunJobExecutionBindsTheTraceMount(t *testing.T) {
	run := dgdrRunWithTraceForTest(t, map[string]any{})
	f := newDGDRRunFixture(t, run, traceClaimForTest())
	f.reconcile()
	job := f.job()
	configMap := &corev1.ConfigMap{}
	if err := f.client.Get(t.Context(), client.ObjectKeyFromObject(job), configMap); err != nil {
		t.Fatalf("get execution ConfigMap: %v", err)
	}
	execution, err := dgdrRunExecutionFromConfigMap(configMap)
	if err != nil {
		t.Fatalf("decode execution: %v", err)
	}

	t.Log("The Job built for a trace Run matches its frozen execution")
	if err := validateDGDRRunJobExecution(job, run, configMap.Name, execution); err != nil {
		t.Fatalf("validation error = %v, want the built trace Job accepted", err)
	}

	t.Log("Any change to the trace claim, its mount or its group is skew")
	tests := []struct {
		name    string
		mutate  func(*batchv1.Job)
		wantErr string
	}{
		{
			name: "missing trace group",
			mutate: func(job *batchv1.Job) {
				job.Spec.Template.Spec.SecurityContext.SupplementalGroups = nil
			},
			wantErr: "pod supplemental groups differ",
		},
		{
			name: "fsGroup in place of the trace group",
			mutate: func(job *batchv1.Job) {
				job.Spec.Template.Spec.SecurityContext.FSGroup = ptr.To[int64](65532)
			},
			wantErr: "pod fsGroup differs",
		},
		{
			name: "writable trace claim",
			mutate: func(job *batchv1.Job) {
				job.Spec.Template.Spec.Volumes[3].PersistentVolumeClaim.ReadOnly = false
			},
			wantErr: "trace volume differs",
		},
		{
			name: "other trace claim",
			mutate: func(job *batchv1.Job) {
				job.Spec.Template.Spec.Volumes[3].PersistentVolumeClaim.ClaimName = "other-traces"
			},
			wantErr: "trace volume differs",
		},
		{
			name: "missing trace claim",
			mutate: func(job *batchv1.Job) {
				job.Spec.Template.Spec.Volumes = job.Spec.Template.Spec.Volumes[:3]
			},
			wantErr: "input, snapshot or publisher credential volumes differ",
		},
		{
			name: "whole claim mounted",
			mutate: func(job *batchv1.Job) {
				job.Spec.Template.Spec.Containers[0].VolumeMounts[2].SubPath = ""
			},
			wantErr: "planner volume mounts differ",
		},
		{
			name: "trace exposed to the publisher",
			mutate: func(job *batchv1.Job) {
				job.Spec.Template.Spec.Containers[1].VolumeMounts = append(
					job.Spec.Template.Spec.Containers[1].VolumeMounts,
					job.Spec.Template.Spec.Containers[0].VolumeMounts[2],
				)
			},
			wantErr: "publisher volume mounts differ",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			mutated := job.DeepCopy()
			tt.mutate(mutated)
			err := validateDGDRRunJobExecution(mutated, run, configMap.Name, execution)
			if err == nil || !strings.Contains(err.Error(), tt.wantErr) {
				t.Fatalf("validation error = %v, want %q", err, tt.wantErr)
			}
		})
	}
}

func TestV1beta2RunControllerStopsAllJobsBeforePermanentCompletion(t *testing.T) {
	run := dgdrRunForTest(t, map[string]any{})
	markerObservedByReconcile := false
	deletedBeforeMarkerObservation := false
	serveStaleMarkerOnce := false
	f := newDGDRRunFixtureWithInterceptor(t, interceptor.Funcs{
		Get: func(ctx context.Context, c client.WithWatch, key client.ObjectKey, object client.Object, opts ...client.GetOption) error {
			if err := c.Get(ctx, key, object, opts...); err != nil {
				return err
			}
			if stored, ok := object.(*nvidiacomv1beta2.DynamoGraphDeploymentRun); ok {
				if serveStaleMarkerOnce {
					meta.RemoveStatusCondition(&stored.Status.Conditions, nvidiacomv1beta2.DGDRRunConditionCompleted)
					serveStaleMarkerOnce = false
				} else if dgdrRunPendingPermanentFailure(stored) != nil {
					markerObservedByReconcile = true
				}
			}
			return nil
		},
		Delete: func(ctx context.Context, c client.WithWatch, object client.Object, opts ...client.DeleteOption) error {
			if _, ok := object.(*batchv1.Job); ok && !markerObservedByReconcile {
				deletedBeforeMarkerObservation = true
			}
			return c.Delete(ctx, object, opts...)
		},
	}, run)
	f.reconcile()
	duplicate := f.job().DeepCopy()
	duplicate.Name += "-duplicate"
	duplicate.ResourceVersion = ""
	duplicate.UID = ""
	if err := f.client.Create(t.Context(), duplicate); err != nil {
		t.Fatalf("create duplicate Job: %v", err)
	}

	t.Log("Record the permanent failure without acting on the locally patched marker")
	result, err := f.reconciler.Reconcile(t.Context(), ctrl.Request{NamespacedName: client.ObjectKeyFromObject(run)})
	if err != nil || result.RequeueAfter != dgdrRunExecutionCacheRetry {
		t.Fatalf("marker result/error = %+v/%v, want durable state retry", result, err)
	}
	if jobs := f.jobs(); len(jobs) != 2 {
		t.Fatalf("Jobs = %d, want no deletion before observing the marker", len(jobs))
	}
	if markerObservedByReconcile || deletedBeforeMarkerObservation {
		t.Fatalf("marker observed/deleted early = %t/%t, want an observed-state barrier", markerObservedByReconcile, deletedBeforeMarkerObservation)
	}
	serveStaleMarkerOnce = true

	t.Log("A reordered cache read cannot skip the marker-observation barrier")
	result, err = f.reconciler.Reconcile(t.Context(), ctrl.Request{NamespacedName: client.ObjectKeyFromObject(run)})
	if err != nil || result.RequeueAfter != dgdrRunExecutionCacheRetry {
		t.Fatalf("stale-cache result/error = %+v/%v, want marker retry", result, err)
	}
	if jobs := f.jobs(); len(jobs) != 2 {
		t.Fatalf("Jobs = %d, want no deletion from a stale marker read", len(jobs))
	}
	if markerObservedByReconcile || deletedBeforeMarkerObservation {
		t.Fatalf("marker observed/deleted early = %t/%t after stale read", markerObservedByReconcile, deletedBeforeMarkerObservation)
	}

	t.Log("A later reconciliation observes the marker and deletes every owned Job")
	result, err = f.reconciler.Reconcile(t.Context(), ctrl.Request{NamespacedName: client.ObjectKeyFromObject(run)})
	if err != nil || result.RequeueAfter != dgdrRunExecutionCacheRetry {
		t.Fatalf("cleanup result/error = %+v/%v, want deletion-observation retry", result, err)
	}
	if !markerObservedByReconcile || deletedBeforeMarkerObservation {
		t.Fatalf("marker observed/deleted early = %t/%t, want deletion only after observation", markerObservedByReconcile, deletedBeforeMarkerObservation)
	}
	if jobs := f.jobs(); len(jobs) != 0 {
		t.Fatalf("Jobs = %d, want all conflicting executions stopped", len(jobs))
	}

	t.Log("Finish the recorded failure after observing Job absence")
	_, err = f.reconciler.Reconcile(t.Context(), ctrl.Request{NamespacedName: client.ObjectKeyFromObject(run)})
	if err == nil || !errors.Is(err, reconcile.TerminalError(nil)) {
		t.Fatalf("terminal reconcile error = %v, want terminal failure", err)
	}
	condition := meta.FindStatusCondition(f.run().Status.Conditions, nvidiacomv1beta2.DGDRRunConditionCompleted)
	if condition == nil || condition.Status != metav1.ConditionTrue {
		t.Fatalf("Completed condition = %+v, want True after cleanup", condition)
	}
}

func TestV1beta2RunControllerConfirmsJobAbsenceThroughAPIReader(t *testing.T) {
	f := newDGDRRunFixture(t, dgdrRunForTest(t, map[string]any{}))
	f.reconcile()

	run := f.run()
	meta.SetStatusCondition(&run.Status.Conditions, metav1.Condition{
		Type:               nvidiacomv1beta2.DGDRRunConditionCompleted,
		Status:             metav1.ConditionFalse,
		Reason:             dgdrRunReasonInvalidExecution,
		Message:            "invalid immutable execution",
		ObservedGeneration: run.Generation,
	})
	if err := f.client.Status().Update(t.Context(), run); err != nil {
		t.Fatalf("record permanent failure marker: %v", err)
	}

	f.reconciler.Client = &staleJobListOnceClient{Client: f.client, hide: true}

	t.Log("The informer misses the Job, but the API reader still observes and stops it")
	result, err := f.reconciler.Reconcile(t.Context(), ctrl.Request{NamespacedName: client.ObjectKeyFromObject(run)})
	if err != nil || result.RequeueAfter != dgdrRunExecutionCacheRetry {
		t.Fatalf("cleanup result/error = %+v/%v, want uncached deletion retry", result, err)
	}
	if jobs := f.jobs(); len(jobs) != 0 {
		t.Fatalf("Jobs = %d, want uncached live execution stopped", len(jobs))
	}
	condition := meta.FindStatusCondition(f.run().Status.Conditions, nvidiacomv1beta2.DGDRRunConditionCompleted)
	if condition == nil || condition.Status != metav1.ConditionFalse {
		t.Fatalf("Completed condition = %+v, want False until uncached absence is observed", condition)
	}

	t.Log("Only a later uncached absence observation permits terminal completion")
	_, err = f.reconciler.Reconcile(t.Context(), ctrl.Request{NamespacedName: client.ObjectKeyFromObject(run)})
	if err == nil || !strings.Contains(err.Error(), "invalid immutable execution") {
		t.Fatalf("terminal reconcile error = %v, want recorded permanent failure", err)
	}
	condition = meta.FindStatusCondition(f.run().Status.Conditions, nvidiacomv1beta2.DGDRRunConditionCompleted)
	if condition == nil || condition.Status != metav1.ConditionTrue {
		t.Fatalf("Completed condition = %+v, want True after uncached absence", condition)
	}
}

func TestV1beta2RunControllerReportsImagePullSecretErrors(t *testing.T) {
	f := newDGDRRunFixture(t, dgdrRunForTest(t, map[string]any{}))
	f.reconciler.DockerSecretRetriever = &fixedRunSecrets{err: errors.New("registry index unavailable")}

	t.Log("Verify execution is not frozen without resolving its image credentials")
	_, err := f.reconciler.Reconcile(t.Context(), ctrl.Request{NamespacedName: client.ObjectKeyFromObject(f.run())})
	if err == nil || !strings.Contains(err.Error(), "registry index unavailable") {
		t.Fatalf("reconcile error = %v, want image pull secret error", err)
	}
}

func TestV1beta2RunControllerSeparatesDigestPinnedPlannerFromRuntimeSource(t *testing.T) {
	f := newDGDRRunFixture(t, dgdrRunForTest(t, map[string]any{}))
	f.reconciler.PlannerImage = "example.com/dynamo-planner@sha256:0123"
	f.reconciler.DynamoVersion = "1.7.0"

	f.reconcile()

	job := f.job()
	planner := containerNamed(t, job.Spec.Template.Spec.Containers, dgdrRunSweeperContainer)
	if planner.Image != f.reconciler.PlannerImage {
		t.Fatalf("planner image = %q, want digest-pinned %q", planner.Image, f.reconciler.PlannerImage)
	}
	configMap := &corev1.ConfigMap{}
	if err := f.client.Get(t.Context(), client.ObjectKeyFromObject(job), configMap); err != nil {
		t.Fatalf("get execution ConfigMap: %v", err)
	}
	if got := configMap.Data[dgdrRunRuntimeImageSourceKey]; got != "example.com/dynamo-planner:1.7.0" {
		t.Fatalf("runtime image source = %q, want normalized release tag", got)
	}
	input := dgdrRunInputFromConfigMap(t, f.client, job)
	if got := input["dgdr_run"].(map[string]any)["runtime_image"]; got != "example.com/dynamo-planner:1.7.0" {
		t.Fatalf("dgdr_run.runtime_image = %v, want normalized release tag", got)
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
	if sweeper.SecurityContext.RunAsUser == nil || *sweeper.SecurityContext.RunAsUser != 1000 {
		t.Fatalf("sweeper runAsUser = %v, want planner image UID 1000", sweeper.SecurityContext.RunAsUser)
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
	assertDGDRRunPublisherIdentity(t, publisher, run, operatorImage)
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
	if hasMount(publisher.VolumeMounts, dgdrRunTraceVolume) {
		t.Fatal("publisher unexpectedly mounts the trace claim")
	}
}

func assertDGDRRunPublisherIdentity(
	t *testing.T,
	publisher corev1.Container,
	run *nvidiacomv1beta2.DynamoGraphDeploymentRun,
	operatorImage string,
) {
	t.Helper()
	if publisher.Image != operatorImage || publisher.ImagePullPolicy != corev1.PullAlways {
		t.Fatalf("publisher image = %q/%s, want configured operator image/Always", publisher.Image, publisher.ImagePullPolicy)
	}
	if want := []string{dgdrRunPublisherBinary}; !reflect.DeepEqual(publisher.Command, want) {
		t.Fatalf("publisher command = %q, want %q", publisher.Command, want)
	}
	if want := []string{
		"--snapshot-dir", dgdrRunSnapshotPath,
		"--run-name", run.Name,
		"--run-uid", string(run.UID),
		"--sweeper-container", dgdrRunSweeperContainer,
	}; !reflect.DeepEqual(publisher.Args, want) {
		t.Fatalf("publisher args = %q, want %q", publisher.Args, want)
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

func assertDGDRRunInput(t *testing.T, c client.Client, job *batchv1.Job) {
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
	if output["num_gpus_per_node"] != float64(8) {
		t.Fatalf("num_gpus_per_node = %v, want typed hardware topology 8", output["num_gpus_per_node"])
	}
	if got := input["engine"].(map[string]any)["hardware"]; got != "h100_sxm" {
		t.Fatalf("engine.hardware = %v, want authoritative input override h100_sxm", got)
	}
	constraints := input["optimization"].(map[string]any)["constraints"].(map[string]any)
	if got := constraints["max_candidate_gpus"]; got != float64(12) {
		t.Fatalf("optimization.constraints.max_candidate_gpus = %v, want authoritative input override 12", got)
	}
}

func dgdrRunInputFromConfigMap(t *testing.T, c client.Client, job *batchv1.Job) map[string]any {
	t.Helper()
	configMap := &corev1.ConfigMap{}
	if err := c.Get(t.Context(), client.ObjectKeyFromObject(job), configMap); err != nil {
		t.Fatalf("get input ConfigMap: %v", err)
	}
	var input map[string]any
	if err := yaml.Unmarshal([]byte(configMap.Data[dgdrRunInputKey]), &input); err != nil {
		t.Fatalf("decode input ConfigMap: %v", err)
	}
	return input
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
		{name: "tag and digest", plannerImage: "example.com/dynamo-planner:v1.7.0-rc.1@sha256:0123", wantOverride: "1.7.0", wantRuntime: "example.com/dynamo-planner:v1.7.0-rc.1"},
		{name: "digest only", plannerImage: "example.com/dynamo-planner@sha256:0123", dynamoVersion: "1.7.0", wantOverride: "1.7.0", wantRuntime: "example.com/dynamo-planner:1.7.0"},
		{name: "opaque CI tag", plannerImage: "example.com/dynamo-planner:git-sha", dynamoVersion: "1.7.0", wantOverride: "1.7.0", wantRuntime: "example.com/dynamo-planner:git-sha"},
		{name: "untagged planner image", plannerImage: "example.com/dynamo-planner", dynamoVersion: "1.7.0", wantErr: true},
		{name: "opaque tag without canonical version", plannerImage: "example.com/dynamo-planner:git-sha", dynamoVersion: "1.7", wantErr: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			version, err := dgdrRunDynamoVersion(tt.plannerImage, tt.dynamoVersion)
			var runtimeImage string
			if err == nil {
				runtimeImage, err = dgdrRunRuntimeImageSource(tt.plannerImage, version)
			}
			if (err != nil) != tt.wantErr {
				t.Fatalf("resolve execution identity error = %v, wantErr %t", err, tt.wantErr)
			}
			if tt.wantErr {
				return
			}
			input, err := dgdrRunInput(dgdrRunForTest(t, map[string]any{}), runtimeImage, version)
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

func assertDGDRRunRunningStatus(t *testing.T, stored *nvidiacomv1beta2.DynamoGraphDeploymentRun, jobName string) {
	t.Helper()
	if stored.Status.Phase != nvidiacomv1beta2.DGDRRunPhaseRunning || stored.Status.JobRef == nil || stored.Status.JobRef.Name != jobName {
		t.Fatalf("Run status = %+v, want Running and JobRef %q", stored.Status, jobName)
	}
	condition := meta.FindStatusCondition(stored.Status.Conditions, nvidiacomv1beta2.DGDRRunConditionCompleted)
	if condition == nil || condition.Status != metav1.ConditionFalse || condition.Reason != dgdrRunReasonRunning {
		t.Fatalf("Completed condition = %+v, want False/%s", condition, dgdrRunReasonRunning)
	}
}

func TestV1beta2RunControllerOwnsOnlyLifecycleStatus(t *testing.T) {
	f := newDGDRRunFixture(t, dgdrRunForTest(t, map[string]any{}))
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
	f := newDGDRRunFixture(t, dgdrRunForTest(t, map[string]any{}))
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
	f := newDGDRRunFixture(t, dgdrRunForTest(t, map[string]any{}))
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

func dgdrRunWithTraceForTest(t *testing.T, inputOverrides map[string]any) *nvidiacomv1beta2.DynamoGraphDeploymentRun {
	t.Helper()
	run := dgdrRunForTest(t, inputOverrides)
	run.Spec.Workload = &nvidiacomv1beta2.WorkloadSpec{Trace: &nvidiacomv1beta2.TraceWorkloadSpec{
		Format: nvidiacomv1beta2.TraceFormatDynamo,
		Source: nvidiacomv1beta2.TraceSourceSpec{PVC: &nvidiacomv1beta2.TracePVCSource{
			ClaimName: "request-traces", Path: "retune/w-1.jsonl.gz",
		}},
	}}
	return run
}

func traceClaimForTest() *corev1.PersistentVolumeClaim {
	return &corev1.PersistentVolumeClaim{
		Name: "request-traces", Namespace: "default",
		Spec: corev1.PersistentVolumeClaimSpec{
			AccessModes: []corev1.PersistentVolumeAccessMode{corev1.ReadWriteMany},
			VolumeMode:  ptr.To(corev1.PersistentVolumeFilesystem),
		},
		Status: corev1.PersistentVolumeClaimStatus{Phase: corev1.ClaimBound},
	}
}

func TestV1beta2RunControllerMountsTheTraceFileReadOnlyIntoTheSweeperOnly(t *testing.T) {
	run := dgdrRunWithTraceForTest(t, map[string]any{})
	f := newDGDRRunFixture(t, run, traceClaimForTest())
	f.reconcile()

	t.Log("The pod mounts the claim read-only as the trace volume")
	job := f.job()
	var trace *corev1.Volume
	for i := range job.Spec.Template.Spec.Volumes {
		if job.Spec.Template.Spec.Volumes[i].Name == dgdrRunTraceVolume {
			trace = &job.Spec.Template.Spec.Volumes[i]
		}
	}
	if trace == nil || trace.PersistentVolumeClaim == nil ||
		trace.PersistentVolumeClaim.ClaimName != "request-traces" || !trace.PersistentVolumeClaim.ReadOnly {
		t.Fatalf("trace volume = %#v, want the read-only request-traces claim", trace)
	}

	t.Log("Only the sweeper mounts the one trace file, read-only through a subPath at the fixed path")
	sweeper := containerNamed(t, job.Spec.Template.Spec.Containers, dgdrRunSweeperContainer)
	want := corev1.VolumeMount{
		Name: dgdrRunTraceVolume, MountPath: "/var/run/aisimulate/trace/trace.jsonl.gz",
		SubPath: "retune/w-1.jsonl.gz", ReadOnly: true,
	}
	var traceMounts []corev1.VolumeMount
	for _, mount := range sweeper.VolumeMounts {
		if mount.Name == dgdrRunTraceVolume {
			traceMounts = append(traceMounts, mount)
		}
	}
	if !reflect.DeepEqual(traceMounts, []corev1.VolumeMount{want}) {
		t.Fatalf("sweeper trace mounts = %#v, want exactly %#v", traceMounts, want)
	}
	assertDGDRRunJobIsolation(t, job, run, f.reconciler.PlannerImage, f.reconciler.OperatorImage)

	t.Log("The pod grants the trace group as a supplemental group and requests no fsGroup")
	security := job.Spec.Template.Spec.SecurityContext
	if security == nil || security.FSGroup != nil || !reflect.DeepEqual(security.SupplementalGroups, []int64{65532}) {
		t.Fatalf("pod securityContext = %#v, want supplemental group 65532 and no fsGroup", security)
	}

	t.Log("The frozen input reads the mounted trace file")
	configMap := &corev1.ConfigMap{}
	if err := f.client.Get(t.Context(), client.ObjectKeyFromObject(job), configMap); err != nil {
		t.Fatalf("get ConfigMap: %v", err)
	}
	var input map[string]any
	if err := yaml.Unmarshal([]byte(configMap.Data[dgdrRunInputKey]), &input); err != nil {
		t.Fatalf("decode input: %v", err)
	}
	source := input["traffic"].(map[string]any)["source"].(map[string]any)
	if !reflect.DeepEqual(source["paths"], []any{"/var/run/aisimulate/trace/trace.jsonl.gz"}) {
		t.Fatalf("traffic.source.paths = %v", source["paths"])
	}
}

func TestV1beta2RunControllerRequestsNoPodGroupsWithoutATrace(t *testing.T) {
	f := newDGDRRunFixture(t, dgdrRunForTest(t, map[string]any{}))
	f.reconcile()

	t.Log("A Run without a trace requests neither an fsGroup nor a supplemental group")
	security := f.job().Spec.Template.Spec.SecurityContext
	if security == nil || security.FSGroup != nil || len(security.SupplementalGroups) != 0 {
		t.Fatalf("pod securityContext = %#v, want no fsGroup and no supplemental groups", security)
	}
}

func TestDGDRRunInputEnforcesTheTypedTraceSource(t *testing.T) {
	t.Log("Lower a Run whose overrides try to replace the trace source and change the load")
	run := dgdrRunWithTraceForTest(t, map[string]any{"traffic": map[string]any{
		"source": map[string]any{"type": "synthetic", "format": "mooncake", "paths": []any{"/etc/passwd"}},
		"load":   map[string]any{"speedup": 4},
	}})
	input, err := dgdrRunInput(run, "example.com/dynamo-planner:1.2.3", "1.2.3")
	if err != nil {
		t.Fatalf("lower input: %v", err)
	}

	t.Log("The typed trace wins for the source; the load override is kept")
	traffic := input["traffic"].(map[string]any)
	wantSource := map[string]any{"type": "trace", "format": "dynamo", "paths": []any{"/var/run/aisimulate/trace/trace.jsonl.gz"}}
	if !reflect.DeepEqual(traffic["source"], wantSource) {
		t.Fatalf("traffic.source = %#v, want %#v", traffic["source"], wantSource)
	}
	load := traffic["load"].(map[string]any)
	if load["type"] != "trace_timestamps" || load["speedup"] != json.Number("4") {
		t.Fatalf("traffic.load = %#v, want trace_timestamps with the override speedup", load)
	}
	if _, found := traffic["stop"]; found {
		t.Fatalf("traffic.stop = %#v, want none for a trace", traffic["stop"])
	}
}

func TestDGDRRunInputRebuildsANonObjectTraceSourceParent(t *testing.T) {
	wantSource := map[string]any{"type": "trace", "format": "dynamo", "paths": []any{"/var/run/aisimulate/trace/trace.jsonl.gz"}}
	tests := []struct {
		name      string
		overrides map[string]any
		wantLoad  map[string]any
	}{
		{
			name:      "scalar traffic",
			overrides: map[string]any{"traffic": "invalid"},
		},
		{
			name:      "scalar traffic.source",
			overrides: map[string]any{"traffic": map[string]any{"source": "invalid"}},
			wantLoad:  map[string]any{"type": "trace_timestamps"},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Log("Lower a Run created without admission whose override replaces a trace source parent")
			input, err := dgdrRunInput(dgdrRunWithTraceForTest(t, tt.overrides), "example.com/dynamo-planner:1.2.3", "1.2.3")
			if err != nil {
				t.Fatalf("lower input: %v", err)
			}

			t.Log("The parent is an object again and carries the typed trace source")
			traffic := input["traffic"].(map[string]any)
			if !reflect.DeepEqual(traffic["source"], wantSource) {
				t.Fatalf("traffic.source = %#v, want %#v", traffic["source"], wantSource)
			}
			if load, _ := traffic["load"].(map[string]any); !reflect.DeepEqual(load, tt.wantLoad) {
				t.Fatalf("traffic.load = %#v, want %#v", traffic["load"], tt.wantLoad)
			}
		})
	}
}

func TestV1beta2RunControllerFailsARunWhoseTraceClaimCannotBeMounted(t *testing.T) {
	tests := []struct {
		name    string
		missing bool
		mutate  func(*corev1.PersistentVolumeClaim)
		reason  string
	}{
		{
			name:    "missing claim",
			missing: true,
			reason:  `trace PersistentVolumeClaim "request-traces" does not exist in namespace "default"`,
		},
		{
			name: "claim being deleted",
			mutate: func(claim *corev1.PersistentVolumeClaim) {
				claim.DeletionTimestamp = new(metav1.Now())
				claim.Finalizers = []string{"kubernetes.io/pvc-protection"}
			},
			reason: `trace PersistentVolumeClaim "request-traces" in namespace "default" is being deleted`,
		},
		{
			name: "raw block claim",
			mutate: func(claim *corev1.PersistentVolumeClaim) {
				claim.Spec.VolumeMode = ptr.To(corev1.PersistentVolumeBlock)
			},
			reason: `trace PersistentVolumeClaim "request-traces" in namespace "default" is a raw block volume`,
		},
		{
			name: "lost claim",
			mutate: func(claim *corev1.PersistentVolumeClaim) {
				claim.Status.Phase = corev1.ClaimLost
			},
			reason: `trace PersistentVolumeClaim "request-traces" in namespace "default" has lost its PersistentVolume`,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			objects := []client.Object{dgdrRunWithTraceForTest(t, map[string]any{})}
			if !tt.missing {
				claim := traceClaimForTest()
				tt.mutate(claim)
				objects = append(objects, claim)
			}
			f := newDGDRRunFixture(t, objects...)

			t.Log("Reconcile twice: the first pass fails the Run, the second sees it completed")
			f.reconcile()
			f.reconcile()

			t.Log("The Run is Failed and Completed with the claim named in the message")
			run := f.run()
			condition := meta.FindStatusCondition(run.Status.Conditions, nvidiacomv1beta2.DGDRRunConditionCompleted)
			if run.Status.Phase != nvidiacomv1beta2.DGDRRunPhaseFailed || condition == nil ||
				condition.Status != metav1.ConditionTrue || condition.Reason != dgdrRunReasonTraceSourceUnavailable {
				t.Fatalf("status = %+v, want Failed and Completed=True/%s", run.Status, dgdrRunReasonTraceSourceUnavailable)
			}
			if !strings.HasPrefix(condition.Message, tt.reason) || run.Status.CompletionTime == nil || run.Status.JobRef != nil {
				t.Fatalf("condition message = %q, completionTime = %v, jobRef = %v; want message prefix %q",
					condition.Message, run.Status.CompletionTime, run.Status.JobRef, tt.reason)
			}
			if condition.ObservedGeneration != run.Generation {
				t.Fatalf("condition observedGeneration = %d, want %d", condition.ObservedGeneration, run.Generation)
			}

			t.Log("Nothing was created for the Run")
			assertNoDGDRRunResources(t, f.client)
		})
	}
}

func assertNoDGDRRunResources(t *testing.T, c client.Client) {
	t.Helper()
	for _, list := range []client.ObjectList{
		&batchv1.JobList{}, &corev1.ConfigMapList{}, &corev1.ServiceAccountList{},
		&rbacv1.RoleList{}, &rbacv1.RoleBindingList{},
	} {
		if err := c.List(t.Context(), list, client.InNamespace("default")); err != nil {
			t.Fatalf("list %T: %v", list, err)
		}
		if n := meta.LenList(list); n != 0 {
			t.Fatalf("%T has %d items, want none", list, n)
		}
	}
}

func TestV1beta2RunControllerAcceptsAnUnboundOrSharedTraceClaim(t *testing.T) {
	tests := []struct {
		name   string
		mutate func(*corev1.PersistentVolumeClaim)
	}{
		{name: "WaitForFirstConsumer claim still Pending", mutate: func(claim *corev1.PersistentVolumeClaim) {
			claim.Status.Phase = corev1.ClaimPending
		}},
		{name: "ReadWriteOnce claim", mutate: func(claim *corev1.PersistentVolumeClaim) {
			claim.Spec.AccessModes = []corev1.PersistentVolumeAccessMode{corev1.ReadWriteOnce}
		}},
		{name: "ReadWriteOncePod claim", mutate: func(claim *corev1.PersistentVolumeClaim) {
			claim.Spec.AccessModes = []corev1.PersistentVolumeAccessMode{corev1.ReadWriteOncePod}
		}},
		{name: "claim without a volume mode", mutate: func(claim *corev1.PersistentVolumeClaim) {
			claim.Spec.VolumeMode = nil
		}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			claim := traceClaimForTest()
			tt.mutate(claim)
			f := newDGDRRunFixture(t, dgdrRunWithTraceForTest(t, map[string]any{}), claim)
			f.reconcile()
			assertDGDRRunPendingStatus(t, f.run(), f.job().Name)
		})
	}
}

func TestV1beta2RunControllerRetriesATransientTraceClaimReadError(t *testing.T) {
	f := newDGDRRunFixture(t, dgdrRunWithTraceForTest(t, map[string]any{}), traceClaimForTest())
	f.reconciler.APIReader = interceptor.NewClient(f.client.(client.WithWatch), interceptor.Funcs{
		Get: func(ctx context.Context, c client.WithWatch, key client.ObjectKey, obj client.Object, opts ...client.GetOption) error {
			if _, ok := obj.(*corev1.PersistentVolumeClaim); ok {
				return errors.New("API server unavailable")
			}
			return c.Get(ctx, key, obj, opts...)
		},
	})

	t.Log("A read error other than NotFound is returned for a retry")
	_, err := f.reconciler.Reconcile(t.Context(), ctrl.Request{NamespacedName: client.ObjectKeyFromObject(f.run())})
	if err == nil || !strings.Contains(err.Error(), "API server unavailable") {
		t.Fatalf("reconcile error = %v, want the claim read error", err)
	}

	t.Log("The Run stays open, and nothing was created for it")
	if run := f.run(); run.Status.Phase != "" || len(run.Status.Conditions) != 0 {
		t.Fatalf("status = %+v, want untouched", run.Status)
	}
	assertNoDGDRRunResources(t, f.client)
}

func TestV1beta2RunControllerKeepsFollowingAStartedJobAfterItsTraceClaimDisappears(t *testing.T) {
	tests := []struct {
		name      string
		status    batchv1.JobStatus
		podNode   string
		podPhase  corev1.PodPhase
		wantPhase nvidiacomv1beta2.DGDRRunPhase
	}{
		{
			name:      "pod scheduled while its images are pulled",
			status:    batchv1.JobStatus{Active: 1, Ready: new(int32(0))},
			podNode:   "node-a",
			podPhase:  corev1.PodPending,
			wantPhase: nvidiacomv1beta2.DGDRRunPhaseRunning,
		},
		{
			name:      "a container exited after the pod was ready",
			status:    batchv1.JobStatus{Active: 1, Ready: new(int32(0))},
			podNode:   "node-a",
			podPhase:  corev1.PodRunning,
			wantPhase: nvidiacomv1beta2.DGDRRunPhaseRunning,
		},
		{
			name:      "pod ready while the pod cache still shows it unscheduled",
			status:    batchv1.JobStatus{Active: 1, Ready: new(int32(1))},
			podPhase:  corev1.PodPending,
			wantPhase: nvidiacomv1beta2.DGDRRunPhaseRunning,
		},
		{
			name:      "Job tracks a succeeded pod it has not counted",
			status:    batchv1.JobStatus{UncountedTerminatedPods: &batchv1.UncountedTerminatedPods{Succeeded: []types.UID{"pod"}}},
			wantPhase: nvidiacomv1beta2.DGDRRunPhaseRunning,
		},
		{
			name:      "Job tracks a failed pod it has not counted",
			status:    batchv1.JobStatus{UncountedTerminatedPods: &batchv1.UncountedTerminatedPods{Failed: []types.UID{"pod"}}},
			wantPhase: nvidiacomv1beta2.DGDRRunPhaseRunning,
		},
		{
			name:      "Job met its success criteria",
			status:    batchv1.JobStatus{Conditions: []batchv1.JobCondition{{Type: batchv1.JobSuccessCriteriaMet, Status: corev1.ConditionTrue}}},
			wantPhase: nvidiacomv1beta2.DGDRRunPhaseRunning,
		},
		{
			name:      "Job reached its failure target",
			status:    batchv1.JobStatus{Conditions: []batchv1.JobCondition{{Type: batchv1.JobFailureTarget, Status: corev1.ConditionTrue}}},
			wantPhase: nvidiacomv1beta2.DGDRRunPhaseRunning,
		},
		{
			name:      "Job counted a succeeded pod",
			status:    batchv1.JobStatus{Succeeded: 1},
			wantPhase: nvidiacomv1beta2.DGDRRunPhaseRunning,
		},
		{
			name:      "Job counted a failed pod",
			status:    batchv1.JobStatus{Failed: 1},
			wantPhase: nvidiacomv1beta2.DGDRRunPhaseRunning,
		},
		{
			name:      "Job complete",
			status:    batchv1.JobStatus{Conditions: []batchv1.JobCondition{{Type: batchv1.JobComplete, Status: corev1.ConditionTrue}}},
			wantPhase: nvidiacomv1beta2.DGDRRunPhaseSucceeded,
		},
		{
			name:      "Job failed",
			status:    batchv1.JobStatus{Conditions: []batchv1.JobCondition{{Type: batchv1.JobFailed, Status: corev1.ConditionTrue}}},
			wantPhase: nvidiacomv1beta2.DGDRRunPhaseFailed,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			claim := traceClaimForTest()
			claim.Finalizers = []string{"kubernetes.io/pvc-protection"}
			f := newDGDRRunFixture(t, dgdrRunWithTraceForTest(t, map[string]any{}), claim)
			f.reconcile()

			t.Log("The Job, and its pod where the row has one, report the pod past the claim")
			job := f.job()
			job.Status = tt.status
			job.Status.StartTime = new(metav1.Now())
			if err := f.client.Status().Update(t.Context(), job); err != nil {
				t.Fatalf("update Job status: %v", err)
			}
			if tt.podPhase != "" {
				createDGDRRunJobPodForTest(t, f.client, job, tt.podNode, corev1.PodStatus{Phase: tt.podPhase})
			}

			t.Log("Delete the claim")
			if err := f.client.Delete(t.Context(), traceClaimForTest()); err != nil {
				t.Fatalf("delete claim: %v", err)
			}

			t.Log("Hide the Job from the cached List, as a lagging informer would")
			cached := f.reconciler.Client
			f.reconciler.Client = interceptor.NewClient(f.client.(client.WithWatch), interceptor.Funcs{
				List: func(ctx context.Context, c client.WithWatch, list client.ObjectList, opts ...client.ListOption) error {
					if _, ok := list.(*batchv1.JobList); ok {
						return nil
					}
					return c.List(ctx, list, opts...)
				},
			})
			_, err := f.reconciler.Reconcile(t.Context(), ctrl.Request{NamespacedName: client.ObjectKeyFromObject(f.run())})
			if !apierrors.IsAlreadyExists(err) {
				t.Fatalf("reconcile error = %v, want the Job create to hit AlreadyExists until the cache catches up", err)
			}

			t.Log("Once the cache shows the Job again, the Job, not the claim, decides the Run")
			f.reconciler.Client = cached
			f.reconcile()
			if kept := f.job(); kept.UID != job.UID {
				t.Fatalf("Job UID = %q, want the started Job %q kept", kept.UID, job.UID)
			}
			run := f.run()
			if run.Status.Phase != tt.wantPhase {
				t.Fatalf("phase = %q, want %q", run.Status.Phase, tt.wantPhase)
			}
			condition := meta.FindStatusCondition(run.Status.Conditions, nvidiacomv1beta2.DGDRRunConditionCompleted)
			if condition == nil || condition.Reason == dgdrRunReasonTraceSourceUnavailable {
				t.Fatalf("Completed condition = %+v, want the Job, not the claim, to decide it", condition)
			}
		})
	}
}

// createDGDRRunJobPodForTest stores the pod the Job controller would create for job, bound to node
// unless node is empty, with status.
func createDGDRRunJobPodForTest(t *testing.T, c client.Client, job *batchv1.Job, node string, status corev1.PodStatus) {
	t.Helper()
	pod := &corev1.Pod{
		Name: job.Name + "-pod", Namespace: job.Namespace,
		Labels:          map[string]string{batchv1.JobNameLabel: job.Name},
		OwnerReferences: []metav1.OwnerReference{*metav1.NewControllerRef(job, batchv1.SchemeGroupVersion.WithKind("Job"))},
		Spec:            corev1.PodSpec{NodeName: node},
	}
	if err := c.Create(t.Context(), pod); err != nil {
		t.Fatalf("create Job pod: %v", err)
	}
	pod.Status = status
	if err := c.Status().Update(t.Context(), pod); err != nil {
		t.Fatalf("update Job pod status: %v", err)
	}
}

// assertDGDRRunFailedForTrace checks that the Run failed for its trace with a message that starts
// with reason and names the Job, and that the completed Run deleted that Job.
func assertDGDRRunFailedForTrace(t *testing.T, c client.Client, run *nvidiacomv1beta2.DynamoGraphDeploymentRun, jobName, reason string) {
	t.Helper()
	jobs := &batchv1.JobList{}
	if err := c.List(t.Context(), jobs, client.InNamespace("default")); err != nil {
		t.Fatalf("list Jobs: %v", err)
	}
	if len(jobs.Items) != 0 {
		t.Fatalf("Jobs = %d, want the Job that could never read the trace deleted", len(jobs.Items))
	}
	condition := meta.FindStatusCondition(run.Status.Conditions, nvidiacomv1beta2.DGDRRunConditionCompleted)
	if run.Status.Phase != nvidiacomv1beta2.DGDRRunPhaseFailed || condition == nil ||
		condition.Status != metav1.ConditionTrue || condition.Reason != dgdrRunReasonTraceSourceUnavailable {
		t.Fatalf("status = %+v, want Failed and Completed=True/%s", run.Status, dgdrRunReasonTraceSourceUnavailable)
	}
	wantSuffix := fmt.Sprintf(`; Job %q is deleted`, jobName)
	if !strings.HasPrefix(condition.Message, reason) || !strings.HasSuffix(condition.Message, wantSuffix) {
		t.Fatalf("condition message = %q, want prefix %q and suffix %q", condition.Message, reason, wantSuffix)
	}
	if run.Status.CompletionTime == nil || run.Status.JobRef == nil || run.Status.JobRef.Name != jobName {
		t.Fatalf("completionTime = %v, jobRef = %+v; want a completion time and the Job's jobRef",
			run.Status.CompletionTime, run.Status.JobRef)
	}
}

func TestV1beta2RunControllerDeletesAnUnscheduledJobWhoseTraceClaimBecomesUnusable(t *testing.T) {
	tests := []struct {
		name         string
		finalizers   []string
		deleteClaim  bool
		mutate       func(*corev1.PersistentVolumeClaim)
		updateStatus bool
		reason       string
	}{
		{
			name:        "claim deleted",
			deleteClaim: true,
			reason:      `trace PersistentVolumeClaim "request-traces" does not exist in namespace "default"`,
		},
		{
			name:        "claim being deleted",
			finalizers:  []string{"kubernetes.io/pvc-protection"},
			deleteClaim: true,
			reason:      `trace PersistentVolumeClaim "request-traces" in namespace "default" is being deleted`,
		},
		{
			name: "claim replaced by a raw block claim",
			mutate: func(claim *corev1.PersistentVolumeClaim) {
				claim.Spec.VolumeMode = ptr.To(corev1.PersistentVolumeBlock)
			},
			reason: `trace PersistentVolumeClaim "request-traces" in namespace "default" is a raw block volume`,
		},
		{
			name:         "claim lost its volume",
			mutate:       func(claim *corev1.PersistentVolumeClaim) { claim.Status.Phase = corev1.ClaimLost },
			updateStatus: true,
			reason:       `trace PersistentVolumeClaim "request-traces" in namespace "default" has lost its PersistentVolume`,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			claim := traceClaimForTest()
			claim.Finalizers = tt.finalizers
			f := newDGDRRunFixture(t, dgdrRunWithTraceForTest(t, map[string]any{}), claim)
			f.reconcile()
			job := f.job()

			t.Log("The Job's pod is created but not scheduled yet")
			job.Status = batchv1.JobStatus{Active: 1, Ready: new(int32(0)), StartTime: new(metav1.Now())}
			if err := f.client.Status().Update(t.Context(), job); err != nil {
				t.Fatalf("update Job status: %v", err)
			}
			createDGDRRunJobPodForTest(t, f.client, job, "", corev1.PodStatus{Phase: corev1.PodPending})
			f.reconcile()
			assertDGDRRunRunningStatus(t, f.run(), job.Name)

			t.Log("The claim becomes unusable before the pod is scheduled")
			if tt.deleteClaim {
				if err := f.client.Delete(t.Context(), traceClaimForTest()); err != nil {
					t.Fatalf("delete claim: %v", err)
				}
			}
			if tt.mutate != nil {
				stored := &corev1.PersistentVolumeClaim{}
				if err := f.client.Get(t.Context(), client.ObjectKeyFromObject(claim), stored); err != nil {
					t.Fatalf("get claim: %v", err)
				}
				tt.mutate(stored)
				var err error
				if tt.updateStatus {
					err = f.client.Status().Update(t.Context(), stored)
				} else {
					err = f.client.Update(t.Context(), stored)
				}
				if err != nil {
					t.Fatalf("update claim: %v", err)
				}
			}

			t.Log("Reconcile twice: the first pass fails the Run, the second deletes the Job of the completed Run")
			f.reconcile()
			f.reconcile()
			assertDGDRRunFailedForTrace(t, f.client, f.run(), job.Name, tt.reason)
		})
	}
}

func TestV1beta2RunControllerDeletesAJobWhoseSweeperCannotMountTheTraceFile(t *testing.T) {
	const prefix = `trace file "retune/w-1.jsonl.gz" on PersistentVolumeClaim "request-traces" cannot be mounted: `
	tests := []struct {
		name       string
		container  string
		reason     string
		message    string
		wantFailed bool
	}{
		{
			name:       "a subPath that resolves outside the claim",
			container:  dgdrRunSweeperContainer,
			reason:     "CreateContainerConfigError",
			message:    `failed to prepare subPath for volumeMount "trace" of container "sweeper"`,
			wantFailed: true,
		},
		{
			name:       "a missing file on a read-only claim",
			container:  dgdrRunSweeperContainer,
			reason:     "CreateContainerConfigError",
			message:    `failed to create subPath directory for volumeMount "trace" of container "sweeper"`,
			wantFailed: true,
		},
		{
			name:      "another volume of the sweeper is left to the Job",
			container: dgdrRunSweeperContainer,
			reason:    "CreateContainerConfigError",
			message:   `failed to prepare subPath for volumeMount "input" of container "sweeper"`,
		},
		{
			name:      "the publisher is left to the Job",
			container: dgdrRunPublisherContainer,
			reason:    "CreateContainerConfigError",
			message:   `failed to prepare subPath for volumeMount "trace" of container "publisher"`,
		},
		{
			name:      "a sweeper image that cannot be pulled is left to the Job",
			container: dgdrRunSweeperContainer,
			reason:    "ErrImagePull",
			message:   `failed to pull image`,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			f := newDGDRRunFixture(t, dgdrRunWithTraceForTest(t, map[string]any{}), traceClaimForTest())
			f.reconcile()
			job := f.job()

			t.Log("The scheduled pod's container waits with the row's reason while the claim is usable")
			job.Status = batchv1.JobStatus{Active: 1, Ready: new(int32(0)), StartTime: new(metav1.Now())}
			if err := f.client.Status().Update(t.Context(), job); err != nil {
				t.Fatalf("update Job status: %v", err)
			}
			createDGDRRunJobPodForTest(t, f.client, job, "node-a", corev1.PodStatus{
				Phase: corev1.PodPending,
				ContainerStatuses: []corev1.ContainerStatus{{
					Name:  tt.container,
					State: corev1.ContainerState{Waiting: &corev1.ContainerStateWaiting{Reason: tt.reason, Message: tt.message}},
				}},
			})

			t.Log("Reconcile twice and expect only a trace mount failure to fail the Run and delete its Job")
			f.reconcile()
			f.reconcile()
			if !tt.wantFailed {
				if kept := f.job(); kept.UID != job.UID {
					t.Fatalf("Job UID = %q, want the Job %q kept", kept.UID, job.UID)
				}
				assertDGDRRunRunningStatus(t, f.run(), job.Name)
				return
			}
			assertDGDRRunFailedForTrace(t, f.client, f.run(), job.Name, prefix+tt.message)
		})
	}
}

func TestV1beta2RunControllerFailsAScheduledJobOnlyForAClaimItCanNeverMount(t *testing.T) {
	tests := []struct {
		name           string
		finalizers     []string
		deleteClaim    bool
		lose           bool
		sweeperStarted bool
		reason         string
	}{
		{
			name:        "claim being deleted while pvc-protection keeps it for the pod",
			finalizers:  []string{"kubernetes.io/pvc-protection"},
			deleteClaim: true,
		},
		{
			name:        "claim gone while the pod was bound",
			deleteClaim: true,
			reason:      `trace PersistentVolumeClaim "request-traces" does not exist in namespace "default"`,
		},
		{
			name:        "claim being deleted without pvc-protection",
			finalizers:  []string{"example.com/hold"},
			deleteClaim: true,
			reason:      `trace PersistentVolumeClaim "request-traces" in namespace "default" is being deleted`,
		},
		{
			name:   "claim lost its volume",
			lose:   true,
			reason: `trace PersistentVolumeClaim "request-traces" in namespace "default" has lost its PersistentVolume`,
		},
		{
			name:           "sweeper started before the claim went",
			deleteClaim:    true,
			sweeperStarted: true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			claim := traceClaimForTest()
			claim.Finalizers = tt.finalizers
			f := newDGDRRunFixture(t, dgdrRunWithTraceForTest(t, map[string]any{}), claim)
			f.reconcile()
			job := f.job()

			t.Log("The Job's pod is bound to a node; its sweeper is created or still waits")
			job.Status = batchv1.JobStatus{Active: 1, Ready: new(int32(0)), StartTime: new(metav1.Now())}
			if err := f.client.Status().Update(t.Context(), job); err != nil {
				t.Fatalf("update Job status: %v", err)
			}
			sweeper := corev1.ContainerStatus{Name: dgdrRunSweeperContainer, State: corev1.ContainerState{
				Waiting: &corev1.ContainerStateWaiting{Reason: "ContainerCreating"},
			}}
			if tt.sweeperStarted {
				sweeper.State = corev1.ContainerState{Running: &corev1.ContainerStateRunning{}}
			}
			createDGDRRunJobPodForTest(t, f.client, job, "node-a", corev1.PodStatus{
				Phase: corev1.PodPending, ContainerStatuses: []corev1.ContainerStatus{sweeper},
			})

			t.Log("The claim changes after the binding")
			if tt.deleteClaim {
				if err := f.client.Delete(t.Context(), traceClaimForTest()); err != nil {
					t.Fatalf("delete claim: %v", err)
				}
			}
			if tt.lose {
				stored := &corev1.PersistentVolumeClaim{}
				if err := f.client.Get(t.Context(), client.ObjectKeyFromObject(claim), stored); err != nil {
					t.Fatalf("get claim: %v", err)
				}
				stored.Status.Phase = corev1.ClaimLost
				if err := f.client.Status().Update(t.Context(), stored); err != nil {
					t.Fatalf("lose claim: %v", err)
				}
			}

			t.Log("Reconcile twice and expect only a claim the pod can never mount to end the Run")
			f.reconcile()
			f.reconcile()
			if tt.reason == "" {
				if kept := f.job(); kept.UID != job.UID {
					t.Fatalf("Job UID = %q, want the Job %q kept", kept.UID, job.UID)
				}
				assertDGDRRunRunningStatus(t, f.run(), job.Name)
				return
			}
			assertDGDRRunFailedForTrace(t, f.client, f.run(), job.Name, tt.reason)
		})
	}
}

func TestV1beta2RunControllerDoesNotLetAStaleRunUndoATraceFailure(t *testing.T) {
	tests := []struct {
		name string
		// jobCondition, when set, finishes the Job after the trace failure was recorded.
		jobCondition batchv1.JobConditionType
		// duplicateJob adds a second, unfinished Job the Run controls, so the stale pass fails the
		// Run permanently and writes the permanent-failure marker.
		duplicateJob bool
	}{
		{name: "the trace is still unreadable"},
		{name: "the Job completes after the failure", jobCondition: batchv1.JobComplete},
		{name: "a second Job appears after the Job completes", jobCondition: batchv1.JobComplete, duplicateJob: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			f := newDGDRRunFixture(t, dgdrRunWithTraceForTest(t, map[string]any{}), traceClaimForTest())
			f.reconcile()
			job := f.job()

			t.Log("The Job's pod waits unscheduled while the Run is Running")
			job.Status = batchv1.JobStatus{Active: 1, Ready: new(int32(0)), StartTime: new(metav1.Now())}
			if err := f.client.Status().Update(t.Context(), job); err != nil {
				t.Fatalf("update Job status: %v", err)
			}
			createDGDRRunJobPodForTest(t, f.client, job, "", corev1.PodStatus{Phase: corev1.PodPending})
			f.reconcile()
			stale := f.run()
			assertDGDRRunRunningStatus(t, stale, job.Name)

			t.Log("The claim goes and the trace failure is recorded")
			if err := f.client.Delete(t.Context(), traceClaimForTest()); err != nil {
				t.Fatalf("delete claim: %v", err)
			}
			f.reconcile()
			failed := f.run()
			if condition := meta.FindStatusCondition(failed.Status.Conditions, nvidiacomv1beta2.DGDRRunConditionCompleted); condition == nil ||
				condition.Reason != dgdrRunReasonTraceSourceUnavailable {
				t.Fatalf("Completed condition = %+v, want %s", condition, dgdrRunReasonTraceSourceUnavailable)
			}
			if tt.jobCondition != "" {
				stored := f.job()
				stored.Status.Conditions = []batchv1.JobCondition{{Type: tt.jobCondition, Status: corev1.ConditionTrue}}
				if err := f.client.Status().Update(t.Context(), stored); err != nil {
					t.Fatalf("finish Job: %v", err)
				}
			}
			if tt.duplicateJob {
				duplicate := f.job().DeepCopy()
				duplicate.Name += "-duplicate"
				duplicate.ResourceVersion, duplicate.UID = "", ""
				duplicate.Status = batchv1.JobStatus{}
				if err := f.client.Create(t.Context(), duplicate); err != nil {
					t.Fatalf("create duplicate Job: %v", err)
				}
			}

			t.Log("A pass that still reads the Running Run conflicts instead of overwriting the failure")
			cached := f.reconciler.Client
			f.reconciler.Client = interceptor.NewClient(f.client.(client.WithWatch), interceptor.Funcs{
				Get: func(ctx context.Context, c client.WithWatch, key client.ObjectKey, obj client.Object, opts ...client.GetOption) error {
					if run, ok := obj.(*nvidiacomv1beta2.DynamoGraphDeploymentRun); ok {
						stale.DeepCopyInto(run)
						return nil
					}
					return c.Get(ctx, key, obj, opts...)
				},
			})
			_, err := f.reconciler.Reconcile(t.Context(), ctrl.Request{NamespacedName: client.ObjectKeyFromObject(stale)})
			if !apierrors.IsConflict(err) {
				t.Fatalf("reconcile error = %v, want a status conflict", err)
			}
			f.reconciler.Client = cached
			if got := f.run(); !reflect.DeepEqual(got.Status, failed.Status) {
				t.Fatalf("status = %+v, want the recorded failure %+v", got.Status, failed.Status)
			}

			t.Log("The next pass deletes the Job only if it has not finished, and keeps the failure")
			f.reconcile()
			if tt.jobCondition == "" {
				assertDGDRRunFailedForTrace(t, f.client, f.run(), job.Name,
					`trace PersistentVolumeClaim "request-traces" does not exist in namespace "default"`)
				return
			}
			if jobs := f.jobs(); len(jobs) != 1 || jobs[0].UID != job.UID {
				t.Fatalf("Jobs = %d, want only the finished Job %q kept", len(jobs), job.UID)
			}
			if got := f.run(); !reflect.DeepEqual(got.Status, failed.Status) {
				t.Fatalf("status = %+v, want the recorded failure %+v", got.Status, failed.Status)
			}
		})
	}
}

func TestV1beta2RunControllerWaitsUntilTheCacheShowsAnActiveJobsPod(t *testing.T) {
	f := newDGDRRunFixture(t, dgdrRunWithTraceForTest(t, map[string]any{}), traceClaimForTest())
	f.reconcile()

	t.Log("The Job counts an active pod that the pod cache does not show yet")
	job := f.job()
	job.Status = batchv1.JobStatus{Active: 1, Ready: new(int32(0)), StartTime: new(metav1.Now())}
	if err := f.client.Status().Update(t.Context(), job); err != nil {
		t.Fatalf("update Job status: %v", err)
	}

	t.Log("The claim disappears; whether the pod is scheduled is unknown, so the Job is kept for now")
	if err := f.client.Delete(t.Context(), traceClaimForTest()); err != nil {
		t.Fatalf("delete claim: %v", err)
	}
	f.reconcile()
	if kept := f.job(); kept.UID != job.UID {
		t.Fatalf("Job UID = %q, want the Job %q kept", kept.UID, job.UID)
	}
	assertDGDRRunRunningStatus(t, f.run(), job.Name)
}

func TestV1beta2RunControllerKeepsTheJobUntilTheTraceFailureIsRecorded(t *testing.T) {
	f := newDGDRRunFixture(t, dgdrRunWithTraceForTest(t, map[string]any{}), traceClaimForTest())
	f.reconcile()
	f.reconcile()
	jobName := f.job().Name
	assertDGDRRunPendingStatus(t, f.run(), jobName)

	t.Log("Delete the claim, then fail the status patch that records the trace failure")
	if err := f.client.Delete(t.Context(), traceClaimForTest()); err != nil {
		t.Fatalf("delete claim: %v", err)
	}
	cached := f.reconciler.Client
	f.reconciler.Client = interceptor.NewClient(f.client.(client.WithWatch), interceptor.Funcs{
		SubResourcePatch: func(context.Context, client.Client, string, client.Object, client.Patch, ...client.SubResourcePatchOption) error {
			return errors.New("injected status patch failure")
		},
	})
	if _, err := f.reconciler.Reconcile(t.Context(), ctrl.Request{NamespacedName: client.ObjectKeyFromObject(f.run())}); err == nil {
		t.Fatal("reconcile error = nil, want the injected status patch failure")
	}

	t.Log("The Job stays until the failure is recorded, so no pass can create another one")
	if job := f.job(); job.Name != jobName {
		t.Fatalf("Job = %q, want %q kept", job.Name, jobName)
	}
	assertDGDRRunPendingStatus(t, f.run(), jobName)

	t.Log("The next passes record the failure and then delete the Job")
	f.reconciler.Client = cached
	f.reconcile()
	f.reconcile()
	assertDGDRRunFailedForTrace(t, f.client, f.run(), jobName,
		`trace PersistentVolumeClaim "request-traces" does not exist in namespace "default"`)
}

func TestV1beta2RunControllerWakesTheRunsThatReadAChangedTraceClaim(t *testing.T) {
	reader := dgdrRunWithTraceForTest(t, map[string]any{})
	otherClaim := dgdrRunWithTraceForTest(t, map[string]any{})
	otherClaim.Name = "search-2"
	otherClaim.Spec.Workload.Trace.Source.PVC.ClaimName = "other-traces"
	otherNamespace := dgdrRunWithTraceForTest(t, map[string]any{})
	otherNamespace.Namespace = "elsewhere"
	withoutTrace := dgdrRunForTest(t, map[string]any{})
	withoutTrace.Name = "search-3"
	f := newDGDRRunFixture(t, reader, otherClaim, otherNamespace, withoutTrace)

	t.Log("An event on request-traces in default wakes only the Run that reads it")
	got := f.reconciler.runsForTraceClaim(t.Context(), traceClaimForTest())
	want := []reconcile.Request{{NamespacedName: client.ObjectKeyFromObject(reader)}}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("requests = %v, want %v", got, want)
	}
}

func TestV1beta2RunControllerWakesTheRunOfAJobPod(t *testing.T) {
	run := dgdrRunWithTraceForTest(t, map[string]any{})
	f := newDGDRRunFixture(t, run, traceClaimForTest())
	f.reconcile()
	job := f.job()
	otherJob := &batchv1.Job{Name: "other-aisimulate", Namespace: "default", UID: "other-job"}
	if err := f.client.Create(t.Context(), otherJob); err != nil {
		t.Fatalf("create Job without a Run: %v", err)
	}
	tests := []struct {
		name      string
		owner     metav1.OwnerReference
		wantWatch bool
		want      []reconcile.Request
	}{
		{
			name:      "a pod of the Run's Job",
			owner:     *metav1.NewControllerRef(job, batchv1.SchemeGroupVersion.WithKind("Job")),
			wantWatch: true,
			want:      []reconcile.Request{{NamespacedName: client.ObjectKeyFromObject(f.run())}},
		},
		{
			name:      "a pod of a Job no Run controls",
			owner:     *metav1.NewControllerRef(otherJob, batchv1.SchemeGroupVersion.WithKind("Job")),
			wantWatch: true,
		},
		{
			name:      "a pod of a Job that no longer exists",
			owner:     metav1.OwnerReference{APIVersion: "batch/v1", Kind: "Job", Name: "gone-aisimulate", UID: "gone", Controller: new(true)},
			wantWatch: true,
		},
		{
			name:      "a pod of a Job with the same name but another UID",
			owner:     metav1.OwnerReference{APIVersion: "batch/v1", Kind: "Job", Name: job.Name, UID: "replaced", Controller: new(true)},
			wantWatch: true,
		},
		{
			name:  "a pod of a Job named unlike a Run's Job",
			owner: metav1.OwnerReference{APIVersion: "batch/v1", Kind: "Job", Name: "nightly", UID: "nightly", Controller: new(true)},
		},
		{
			name:  "a pod of a ReplicaSet",
			owner: metav1.OwnerReference{APIVersion: "apps/v1", Kind: "ReplicaSet", Name: "web-aisimulate", UID: "rs", Controller: new(true)},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			pod := &corev1.Pod{Name: "pod", Namespace: "default", OwnerReferences: []metav1.OwnerReference{tt.owner}}

			t.Log("Only pods of Jobs named like a Run's Job pass the watch filter")
			if got := dgdrRunJobPod(pod); got != tt.wantWatch {
				t.Fatalf("dgdrRunJobPod = %v, want %v", got, tt.wantWatch)
			}

			t.Log("Only a pod of the Job a Run controls wakes that Run")
			if got := f.reconciler.runForJobPod(t.Context(), pod); !reflect.DeepEqual(got, tt.want) {
				t.Fatalf("requests = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestV1beta2RunControllerKeepsAJobWhoseJobRefWasNeverRecorded(t *testing.T) {
	f := newDGDRRunFixture(t, dgdrRunWithTraceForTest(t, map[string]any{}), traceClaimForTest())
	cached := f.reconciler.Client

	t.Log("Create the Job but fail the status patch that would record its jobRef")
	f.reconciler.Client = interceptor.NewClient(f.client.(client.WithWatch), interceptor.Funcs{
		SubResourcePatch: func(context.Context, client.Client, string, client.Object, client.Patch, ...client.SubResourcePatchOption) error {
			return errors.New("injected status patch failure")
		},
	})
	if _, err := f.reconciler.Reconcile(t.Context(), ctrl.Request{NamespacedName: client.ObjectKeyFromObject(f.run())}); err == nil {
		t.Fatal("reconcile error = nil, want the injected status patch failure")
	}
	job := f.job()
	if f.run().Status.JobRef != nil {
		t.Fatalf("jobRef = %+v, want none after the failed status patch", f.run().Status.JobRef)
	}

	t.Log("The Job's pod becomes ready")
	job.Status = batchv1.JobStatus{Active: 1, Ready: new(int32(1)), StartTime: new(metav1.Now())}
	if err := f.client.Status().Update(t.Context(), job); err != nil {
		t.Fatalf("update Job status: %v", err)
	}

	t.Log("Delete the claim and hide the Job from the cached List")
	if err := f.client.Delete(t.Context(), traceClaimForTest()); err != nil {
		t.Fatalf("delete claim: %v", err)
	}
	f.reconciler.Client = interceptor.NewClient(f.client.(client.WithWatch), interceptor.Funcs{
		List: func(ctx context.Context, c client.WithWatch, list client.ObjectList, opts ...client.ListOption) error {
			if _, ok := list.(*batchv1.JobList); ok {
				return nil
			}
			return c.List(ctx, list, opts...)
		},
	})
	_, err := f.reconciler.Reconcile(t.Context(), ctrl.Request{NamespacedName: client.ObjectKeyFromObject(f.run())})
	if !apierrors.IsAlreadyExists(err) {
		t.Fatalf("reconcile error = %v, want the Job create to hit AlreadyExists instead of failing the Run", err)
	}
	if condition := meta.FindStatusCondition(f.run().Status.Conditions, nvidiacomv1beta2.DGDRRunConditionCompleted); condition != nil {
		t.Fatalf("Completed condition = %+v, want none while the Job exists", condition)
	}

	t.Log("Once the cache shows the Job, the Job drives the Run")
	f.reconciler.Client = cached
	f.reconcile()
	assertDGDRRunRunningStatus(t, f.run(), job.Name)
}
