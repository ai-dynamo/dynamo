/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package validation

import (
	"context"
	"fmt"
	"slices"
	"strings"
	"testing"

	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/features"
	grovev1alpha1 "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	corev1 "k8s.io/api/core/v1"
	k8serrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/client-go/rest"
	k8sptr "k8s.io/utils/ptr"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
	ctrlwebhook "sigs.k8s.io/controller-runtime/pkg/webhook"
)

const sglangBackendFramework = "sglang"

func TestDynamoGraphDeploymentConversionFailureIsFatal(t *testing.T) {
	dgd := newBetaDGDForValidation()
	dgd.Spec.Components = append(dgd.Spec.Components, dgd.Spec.Components[0])

	validator := newDynamoGraphDeploymentTestValidator(t)
	ctx := features.WithGate(context.Background(), features.Gates{Grove: true, ElasticEPRayPoC: true})
	_, err := validator.Validate(ctx, dgd, runtimeVersionSourceV1Beta1)
	if err == nil || !strings.Contains(err.Error(), "failed to reconstruct compatibility view") {
		t.Fatalf("Validate() error = %v, want fatal conversion error", err)
	}
	if k8serrors.IsInvalid(err) {
		t.Fatalf("Validate() error = %v, want fatal conversion error rather than field validation error", err)
	}
}

func assertFieldPaths(t *testing.T, errs field.ErrorList, want []string) {
	t.Helper()
	got := make([]string, len(errs))
	for i := range errs {
		got[i] = errs[i].Field
	}
	if !slices.Equal(got, want) {
		t.Fatalf("field paths = %v, want %v", got, want)
	}
}

func newBetaDGDForValidation() *nvidiacomv1beta1.DynamoGraphDeployment {
	return &nvidiacomv1beta1.DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-graph",
			Namespace: "default",
		},
		Spec: nvidiacomv1beta1.DynamoGraphDeploymentSpec{
			BackendFramework: "vllm",
			Components: []nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec{
				{
					ComponentName:          "frontend",
					ComponentType:          nvidiacomv1beta1.ComponentTypeFrontend,
					RuntimeVersionOverride: "1.1.0",
					Replicas:               k8sptr.To(int32(1)),
					PodTemplate: &corev1.PodTemplateSpec{Spec: corev1.PodSpec{
						Containers: []corev1.Container{{Name: consts.MainContainerName, Image: "registry.example/runtime:1.1.0"}},
					}},
				},
				{
					ComponentName:          "worker",
					ComponentType:          nvidiacomv1beta1.ComponentTypeWorker,
					RuntimeVersionOverride: "1.1.0",
					Replicas:               k8sptr.To(int32(2)),
					PodTemplate: &corev1.PodTemplateSpec{Spec: corev1.PodSpec{
						Containers: []corev1.Container{{Name: consts.MainContainerName, Image: "registry.example/runtime:1.1.0"}},
					}},
				},
			},
		},
	}
}

type fakeManager struct {
	ctrl.Manager
	client        client.Client
	config        *rest.Config
	scheme        *runtime.Scheme
	webhookServer ctrlwebhook.Server
}

func (m *fakeManager) GetClient() client.Client             { return m.client }
func (m *fakeManager) GetConfig() *rest.Config              { return m.config }
func (m *fakeManager) GetScheme() *runtime.Scheme           { return m.scheme }
func (m *fakeManager) GetWebhookServer() ctrlwebhook.Server { return m.webhookServer }

func newDynamoGraphDeploymentTestValidator(t *testing.T) *DynamoGraphDeploymentValidator {
	t.Helper()
	return NewDynamoGraphDeploymentValidator(newGroveTopologyTestManager(t))
}

func newGroveTopologyTestManager(t *testing.T) ctrl.Manager {
	t.Helper()
	scheme := runtime.NewScheme()
	if err := grovev1alpha1.AddToScheme(scheme); err != nil {
		t.Fatalf("add Grove scheme: %v", err)
	}
	return &fakeManager{
		client: fake.NewClientBuilder().WithScheme(scheme).Build(),
		config: &rest.Config{},
	}
}

func assertBetaValidationErrors(t *testing.T, err error, wantErrs []string) {
	t.Helper()
	if len(wantErrs) == 0 {
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		return
	}
	if err == nil {
		t.Fatalf("expected errors %v but got nil", wantErrs)
	}
	statusErr, ok := err.(*k8serrors.StatusError)
	if !ok || !k8serrors.IsInvalid(err) {
		t.Fatalf("error = %T %v, want typed Kubernetes invalid error", err, err)
	}
	if statusErr.ErrStatus.Details == nil {
		t.Fatalf("error = %v, want typed field causes", err)
	}

	causes := statusErr.ErrStatus.Details.Causes
	gotErrs := make([]string, len(causes))
	for i, cause := range causes {
		if cause.Field == "" {
			t.Fatalf("error cause = %#v, want an exact field path", cause)
		}
		gotErrs[i] = fmt.Sprintf("%s: %s", cause.Field, cause.Message)
	}
	if !slices.Equal(gotErrs, wantErrs) {
		t.Fatalf("webhook errors = %v, want %v", gotErrs, wantErrs)
	}
}

// elasticEPSharedSpec builds the shape the elastic-EP rules target: a single-node WORKER.
//
// ComponentType is set deliberately. The single-replica rule only fires on shapes that
// actually derive a follower, so a spec without a component type is not a shape the rule
// applies to -- and a fixture that omits it would exercise the early return rather than the
// rule.
func elasticEPSharedSpec(command, args []string) *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec {
	return &nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec{
		ComponentType: consts.ComponentTypeWorker,
		PodTemplate: &corev1.PodTemplateSpec{Spec: corev1.PodSpec{
			Containers: []corev1.Container{{
				Name:    consts.MainContainerName,
				Command: command,
				Args:    args,
			}},
		}},
	}
}

func TestValidateElasticEPRequiresCommand(t *testing.T) {
	const vllm = "vllm"
	rayArgs := []string{"--model", "test", "--data-parallel-backend", "ray", "--enable-elastic-ep"}
	fldPath := field.NewPath("spec")
	const commandPath = "spec.podTemplate.spec.containers[0].command"

	tests := []struct {
		name    string
		backend string
		spec    *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec
		want    []string
	}{
		{
			name:    "vllm elastic-EP ray with empty command is rejected",
			backend: vllm,
			spec:    elasticEPSharedSpec(nil, rayArgs),
			want:    []string{commandPath},
		},
		{
			name:    "vllm elastic-EP with -dpb=ray alias and empty command is rejected",
			backend: vllm,
			spec:    elasticEPSharedSpec(nil, []string{"--model", "test", "-dpb=ray", "--enable-elastic-ep"}),
			want:    []string{commandPath},
		},
		{
			name:    "explicit command is accepted",
			backend: vllm,
			spec:    elasticEPSharedSpec([]string{"python3", "-m", "dynamo.vllm"}, rayArgs),
			want:    nil,
		},
		{
			name:    "elastic-EP flags carried in Command are accepted",
			backend: vllm,
			spec:    elasticEPSharedSpec([]string{"python3", "-m", "dynamo.vllm", "--data-parallel-backend", "ray", "--enable-elastic-ep"}, nil),
			want:    nil,
		},
		{
			name:    "non-vllm backend is not validated",
			backend: sglangBackendFramework,
			spec:    elasticEPSharedSpec(nil, rayArgs),
			want:    nil,
		},
		{
			name:    "vllm without elastic-EP is accepted",
			backend: vllm,
			spec:    elasticEPSharedSpec(nil, []string{"--model", "test"}),
			want:    nil,
		},
		{
			name:    "vllm elastic-EP on a non-ray backend is accepted",
			backend: vllm,
			spec:    elasticEPSharedSpec(nil, []string{"--model", "test", "--data-parallel-backend", "mp", "--enable-elastic-ep"}),
			want:    nil,
		},
		{
			name:    "nil pod template is ignored",
			backend: vllm,
			spec:    &nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec{},
			want:    nil,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assertFieldPaths(t, validateElasticEPRequiresCommand(tt.backend, tt.spec, fldPath), tt.want)
		})
	}
}

// Elastic EP grows by adding followers to one leader's Ray cluster, not by adding
// leaders. Two replicas mean two independent Ray heads, but the operator renders one
// "<component>-ray" Service and one derived follower per component, so a follower cannot
// say which leader it belongs to. Rejecting here is what keeps the operator from silently
// dropping both the Service and the follower and leaving leaders that can never grow.
func TestValidateElasticEPSingleReplica(t *testing.T) {
	const vllm = "vllm"
	rayArgs := []string{"--model", "test", "--data-parallel-backend", "ray", "--enable-elastic-ep"}
	command := []string{"python3", "-m", "dynamo.vllm"}
	fldPath := field.NewPath("spec")
	const replicasPath = "spec.replicas"

	withReplicas := func(spec *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec, n *int32) *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec {
		spec.Replicas = n
		return spec
	}

	tests := []struct {
		name    string
		backend string
		spec    *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec
		want    []string
	}{
		{
			name:    "elastic-EP with two replicas is rejected",
			backend: vllm,
			spec:    withReplicas(elasticEPSharedSpec(command, rayArgs), k8sptr.To(int32(2))),
			want:    []string{replicasPath},
		},
		{
			name:    "a single replica is accepted",
			backend: vllm,
			spec:    withReplicas(elasticEPSharedSpec(command, rayArgs), k8sptr.To(int32(1))),
			want:    nil,
		},
		{
			name:    "unset replicas is accepted: it defaults to one leader",
			backend: vllm,
			spec:    withReplicas(elasticEPSharedSpec(command, rayArgs), nil),
			want:    nil,
		},
		{
			name:    "zero replicas is accepted: a scaled-to-zero leader has no Ray head to confuse",
			backend: vllm,
			spec:    withReplicas(elasticEPSharedSpec(command, rayArgs), k8sptr.To(int32(0))),
			want:    nil,
		},
		{
			name:    "a non-elastic-EP component may scale freely",
			backend: vllm,
			spec:    withReplicas(elasticEPSharedSpec(command, []string{"--model", "test"}), k8sptr.To(int32(4))),
			want:    nil,
		},
		{
			name:    "elastic-EP on a non-ray backend may scale freely",
			backend: vllm,
			spec:    withReplicas(elasticEPSharedSpec(command, []string{"--model", "test", "--data-parallel-backend", "mp", "--enable-elastic-ep"}), k8sptr.To(int32(3))),
			want:    nil,
		},
		{
			name:    "non-vllm backend is not validated",
			backend: sglangBackendFramework,
			spec:    withReplicas(elasticEPSharedSpec(command, rayArgs), k8sptr.To(int32(2))),
			want:    nil,
		},
		{
			// The rule exists because one follower and one "<component>-ray" Service are
			// derived per component, so two leaders would share one identity. A MULTINODE
			// component derives neither: it takes the LWS path, which never renders
			// RoleFollower. Its replicas are LWS groups, not competing Ray heads.
			//
			// Rejecting it would be a regression -- the merge base accepted this shape --
			// and the error would tell the user to "add followers instead" when multinode
			// can never have one.
			name:    "multinode elastic EP may scale: it derives no follower to collide over",
			backend: vllm,
			spec: withReplicas(func() *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec {
				s := elasticEPSharedSpec(command, rayArgs)
				s.Multinode = &nvidiacomv1beta1.MultinodeSpec{NodeCount: 4}
				return s
			}(), k8sptr.To(int32(2))),
			want: nil,
		},
		{
			// Same reasoning by the other axis: synthesis requires IsWorkerComponent, so a
			// frontend carrying the flags never reaches it and has no Ray Service or
			// follower either. Also a regression against the merge base if rejected.
			name:    "a non-worker component carrying the flags may scale freely",
			backend: vllm,
			spec: withReplicas(func() *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec {
				s := elasticEPSharedSpec(command, rayArgs)
				s.ComponentType = consts.ComponentTypeFrontend
				return s
			}(), k8sptr.To(int32(3))),
			want: nil,
		},
		{
			name:    "nil pod template is ignored",
			backend: vllm,
			spec:    &nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec{Replicas: k8sptr.To(int32(2))},
			want:    nil,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assertFieldPaths(t, validateElasticEPSingleReplica(tt.backend, tt.spec, fldPath), tt.want)
		})
	}
}

// TestDynamoGraphDeploymentRejectsElasticEPWithoutCommand proves the rule is
// wired into the DGD admission path end to end, not just callable in isolation.
func TestDynamoGraphDeploymentRejectsElasticEPWithoutCommand(t *testing.T) {
	dgd := newBetaDGDForValidation()
	// components[1] is the worker; make it request elastic-EP Ray with no command.
	dgd.Spec.Components[1].PodTemplate.Spec.Containers[0].Args = []string{
		"--model", "test", "--data-parallel-backend", "ray", "--enable-elastic-ep",
	}

	validator := newDynamoGraphDeploymentTestValidator(t)
	ctx := features.WithGate(context.Background(), features.Gates{Grove: true, ElasticEPRayPoC: true})
	_, err := validator.Validate(ctx, dgd, runtimeVersionSourceV1Beta1)
	if err == nil || !k8serrors.IsInvalid(err) {
		t.Fatalf("Validate() error = %v, want invalid field error", err)
	}
	if !strings.Contains(err.Error(), "requires an explicit container command") {
		t.Fatalf("Validate() error = %v, want elastic-EP command requirement", err)
	}
}

// TestDynamoGraphDeploymentElasticEPRuleGating pins which of the two elastic-EP rules
// the PoC gate governs, and which it must not.
//
// validateElasticEPRequiresCommand shipped in #12943 and guards the Phase 2/3 Ray head,
// which renders at either gate position. injectElasticEPRayLaunchFlags cannot wrap an
// image ENTRYPOINT it cannot see, so without an explicit command it declines and only
// logs -- gating this rule would make that silent no-op reachable, leaving elastic EP
// accepted and simply absent, with no error, event or condition anywhere.
//
// validateElasticEPSingleReplica is new in this PoC and is the only new RESTRICTION it
// adds -- one follower and one <leader>-ray Service are derived per component -- so it is
// gated: an operator that has not opted in must not start rejecting manifests the merge
// base accepted. Generation is ungated, so the follower and the Service are derived at
// either gate position.
//
// Mutation check: moving validateElasticEPRequiresCommand back inside the gate block
// fails the first subtest.
func TestDynamoGraphDeploymentElasticEPRuleGating(t *testing.T) {
	offCtx := features.WithGate(context.Background(), features.Gates{Grove: true})
	onCtx := features.WithGate(context.Background(), features.Gates{Grove: true, ElasticEPRayPoC: true})
	elasticArgs := []string{"--model", "test", "--data-parallel-backend", "ray", "--enable-elastic-ep"}
	validator := newDynamoGraphDeploymentTestValidator(t)

	t.Run("missing command is rejected at either gate position", func(t *testing.T) {
		dgd := newBetaDGDForValidation()
		dgd.Spec.Components[1].PodTemplate.Spec.Containers[0].Command = nil
		dgd.Spec.Components[1].PodTemplate.Spec.Containers[0].Args = elasticArgs

		for name, ctx := range map[string]context.Context{"gate off": offCtx, "gate on": onCtx} {
			_, err := validator.Validate(ctx, dgd, runtimeVersionSourceV1Beta1)
			if err == nil || !k8serrors.IsInvalid(err) {
				t.Fatalf("%s: error = %v, want invalid field error; this rule shipped ungated in #12943 "+
					"and without it elastic EP silently no-ops", name, err)
			}
		}
	})

	t.Run("replicas > 1 is rejected only when the gate is on", func(t *testing.T) {
		dgd := newBetaDGDForValidation()
		// An explicit command, so the ungated RequiresCommand rule is satisfied and this
		// subtest isolates the gated single-replica rule.
		dgd.Spec.Components[1].PodTemplate.Spec.Containers[0].Command = []string{"python3", "-m", "dynamo.vllm"}
		dgd.Spec.Components[1].PodTemplate.Spec.Containers[0].Args = elasticArgs
		dgd.Spec.Components[1].Replicas = k8sptr.To(int32(2))

		t.Log("Gate off: accepted, because the PoC derives nothing for this shape")
		if _, err := validator.Validate(offCtx, dgd, runtimeVersionSourceV1Beta1); err != nil {
			t.Fatalf("gate off rejected a shape the PoC does not manage: %v", err)
		}

		t.Log("Gate on: rejected, so the skip is the gate and not a broken rule")
		_, err := validator.Validate(onCtx, dgd, runtimeVersionSourceV1Beta1)
		if err == nil || !k8serrors.IsInvalid(err) {
			t.Fatalf("gate on error = %v, want invalid field error", err)
		}
	})
}
