/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package enginegroup

import (
	"context"
	"fmt"
	"net/http"
	"slices"
	"strconv"
	"strings"
	"time"

	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
	domain "github.com/ai-dynamo/dynamo/deploy/operator/internal/enginegroup"
	sglangruntime "github.com/ai-dynamo/dynamo/deploy/operator/internal/enginegroup/sglang"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/enginegroup/workload"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

const (
	defaultSGLangControlPort = 9090
	defaultSGLangServingPort = 8000
)

// sglangRuntimeResolver composes physical capacity with native SGLang membership and drain.
// Its traffic projection does not independently observe discovery registration.
type sglangRuntimeResolver struct {
	client     client.Client
	httpClient *http.Client
}

func (p *sglangRuntimeResolver) Resolve(
	ctx context.Context,
	group *nvidiacomv1beta1.DynamoGraphDeploymentEngineGroup,
	world WorkloadBinding,
) (Runtime, error) {
	if group.Labels[consts.KubeLabelDynamoEngineGroupRuntime] != consts.KubeLabelDynamoEngineGroupSGLang {
		return Runtime{}, ErrRuntimeUnavailable
	}
	if group.UID == "" {
		return Runtime{}, fmt.Errorf("Engine Group UID is required")
	}
	capacity := &SGLangCapacity{Physical: world.Capacity, Reader: p.client, Namespace: group.Namespace}
	allocations, err := world.Capacity.Observe(ctx, engineGroupID(group))
	if err != nil {
		return Runtime{}, err
	}
	primary, workloadRevision, err := p.primaryPod(ctx, group.Namespace, allocations)
	if err != nil {
		return Runtime{}, err
	}
	if primary.Status.PodIP == "" {
		return Runtime{}, fmt.Errorf("SGLang primary Pod %s has no Pod IP", primary.Name)
	}
	if primary.Spec.RestartPolicy != corev1.RestartPolicyNever {
		return Runtime{}, fmt.Errorf("the SGLang profile requires restartPolicy Never to prevent unfenced engine restarts")
	}
	main, err := engineGroupMainContainer(primary)
	if err != nil {
		return Runtime{}, err
	}
	if len(main.Command) != 3 || (main.Command[0] != "python" && main.Command[0] != "python3") ||
		main.Command[1] != "-m" || main.Command[2] != "dynamo.sglang" {
		return Runtime{}, fmt.Errorf("SGLang capacity requires the supported template-invariant bootstrap entrypoint")
	}
	if argumentValue(main.Args, "--elastic-ep-join-mode") != "auto" {
		return Runtime{}, fmt.Errorf("SGLang Engine Groups require native automatic bootstrap")
	}
	if workloadRevision == "" {
		return Runtime{}, fmt.Errorf(
			"SGLang primary allocation needs a stable workload template revision",
		)
	}
	gpuLimit := main.Resources.Limits[corev1.ResourceName("nvidia.com/gpu")]
	if gpuLimit.IsZero() {
		return Runtime{}, fmt.Errorf("SGLang primary main container needs an explicit GPU limit")
	}
	resolved, err := dynamo.ResolveSGLangElasticEPProfile(dynamo.SGLangProfileGeometrySource{
		Command:                    main.Command,
		Args:                       main.Args,
		MainContainerGPUs:          gpuLimit.Value(),
		DedicatedMainGPUAllocation: true,
		WorkloadRevisionDigest:     workloadRevision,
	})
	if err != nil {
		return Runtime{}, fmt.Errorf("resolve SGLang Engine Group profile: %w", err)
	}

	controlPort, err := metadataPort(group, primary, consts.KubeAnnotationDynamoEngineGroupControlPort, defaultSGLangControlPort)
	if err != nil {
		return Runtime{}, err
	}
	control, err := sglangruntime.NewClient(
		fmt.Sprintf("http://%s:%d", primary.Status.PodIP, controlPort),
		p.httpClient,
	)
	if err != nil {
		return Runtime{}, err
	}
	verifyURL := metadataValue(group, primary, consts.KubeAnnotationDynamoEngineGroupVerifyURL)
	if verifyURL == "" {
		verifyURL = fmt.Sprintf("http://%s:%d/v1/completions", primary.Status.PodIP, defaultSGLangServingPort)
	}
	model := metadataValue(group, primary, consts.KubeAnnotationDynamoEngineGroupVerifyModel)
	if model == "" {
		model = argumentValue(main.Args, "--served-model-name", "--model-path")
	}
	verifier, err := sglangruntime.NewServingVerifier(verifyURL, model, &http.Client{Timeout: 30 * time.Second})
	if err != nil {
		return Runtime{}, err
	}

	// This profile retains the complete initial world; only appended allocations may retire.
	profile := nvidiacomv1beta1.EngineGroupProfileStatus{
		Backend:                     resolved.Geometry.Backend,
		Fingerprint:                 resolved.Geometry.Fingerprint,
		GPUsPerReplica:              resolved.Geometry.GPUsPerReplica,
		PodsPerReplica:              resolved.Geometry.PodsPerReplica,
		NativeMembersPerReplica:     resolved.NativeMembersPerReplica,
		MinSafeServingNativeMembers: 1,
		MinSupportedReplicas:        resolved.InitialReplicas,
		MaxSupportedReplicas:        resolved.MaximumReplicas,
	}
	capacity.MembersPerReplica = resolved.NativeMembersPerReplica
	if _, err := capacity.Observe(ctx, engineGroupID(group)); err != nil {
		return Runtime{}, err
	}
	membership := &sglangruntime.MembershipAdapter{
		Client:             control,
		Capacity:           capacity,
		ProfileFingerprint: profile.Fingerprint,
		MembersPerReplica:  resolved.NativeMembersPerReplica,
		InitialEPSize:      resolved.InitialReplicas * resolved.NativeMembersPerReplica,
		Journal:            runtimeJournal(p.client, group, world.ID, "sglang-membership"),
	}
	traffic := &sglangruntime.TrafficProjection{
		Client:     control,
		Capacity:   capacity,
		Journal:    runtimeJournal(p.client, group, world.ID, "sglang-traffic"),
		Membership: membership,
	}
	return Runtime{
		Workload:   world,
		Profile:    profile,
		Capacity:   capacity,
		Membership: membership,
		Traffic:    traffic,
		Verifier:   verifier,
		Planner: sglangScalePlanner{
			profileFingerprint: profile.Fingerprint,
			worldUID:           world.ID,
			membersPerReplica:  resolved.NativeMembersPerReplica,
			retirementEnabled:  argumentValue(main.Args, "--elastic-ep-retiree-lifecycle") == "external",
		},
	}, nil
}

func (p *sglangRuntimeResolver) primaryPod(
	ctx context.Context,
	namespace string,
	observation workload.Observation,
) (*corev1.Pod, string, error) {
	// Primary ownership follows the validated physical slot, not a mutable role label.
	for _, allocation := range observation.Allocations {
		if allocation.ReplicaID != "replica-0" {
			continue
		}
		ref := allocation.Pod
		primary := &corev1.Pod{}
		if err := p.client.Get(ctx, client.ObjectKey{Namespace: namespace, Name: ref.Name}, primary); err != nil {
			return nil, "", fmt.Errorf("get SGLang primary Pod: %w", err)
		}
		if domain.PodUID(primary.UID) != ref.UID || primary.DeletionTimestamp != nil {
			return nil, "", fmt.Errorf("SGLang primary Pod changed after capacity observation")
		}
		return primary, allocation.TemplateRevision, nil
	}
	return nil, "", fmt.Errorf("SGLang Engine Group primary allocation is missing")
}

type sglangScalePlanner struct {
	profileFingerprint string
	worldUID           types.UID
	membersPerReplica  int32
	retirementEnabled  bool
}

func (p sglangScalePlanner) ResolveScalePlan(
	_ context.Context,
	_ domain.GroupID,
	targetReplicas int32,
	status domain.GroupStatus,
) (ScalePlanResolution, error) {
	// An unresolved operation owns the current collective; a newer target may only queue behind it.
	if status.Transition != nil && status.Transition.Outcome != domain.TransitionOutcomeCompleted &&
		status.Transition.Outcome != domain.TransitionOutcomeRolledBack {
		return ScalePlanResolution{}, nil
	}

	base := status.Membership.Observed.CommittedTopology
	current := base.ReplicaCount()
	if targetReplicas == current {
		return ScalePlanResolution{}, nil
	}

	// Retiring the primary would destroy control authority and requires whole-world retirement.
	if targetReplicas == 0 {
		return ScalePlanResolution{Rejection: &domain.Failure{
			Classification: domain.FailureClassificationTerminal,
			Reason:         "SGLangRetirementUnsupported",
			Message:        "the SGLang adapter cannot safely retire the primary engine world",
		}}, nil
	}
	if targetReplicas < current {
		if !p.retirementEnabled {
			return ScalePlanResolution{Rejection: &domain.Failure{
				Classification: domain.FailureClassificationTerminal,
				Reason:         "SGLangShrinkUnsupported",
				Message:        "planned shrink requires explicit native external retiree lifecycle and a retention gate",
			}}, nil
		}
		return ScalePlanResolution{Plan: &domain.ResolvedPlan{
			ID:                      fmt.Sprintf("sglang-retire-%s-%d-%d", p.worldUID, base.Generation, current-1),
			ProfileFingerprint:      p.profileFingerprint,
			ProcessLifecycleOwner:   domain.ProcessLifecycleOwnerOrchestrator,
			TrafficRequirement:      domain.TrafficRequirementEngineManagedDrain,
			VerificationRequirement: domain.VerificationRequirementRequired,
			Change: domain.ResolvedChange{Kind: domain.PlanKindRetire,
				Retire: &domain.RetireChange{Replicas: []domain.ReplicaID{domain.ReplicaID(fmt.Sprintf("replica-%d", current-1))}},
			},
		}}, nil
	}
	// Each native append admits one complete allocation; larger absolute targets converge through separate commits.
	joining := domain.ReplicaTarget{
		ReplicaID: domain.ReplicaID(fmt.Sprintf("replica-%d", current)),
		SlotID:    domain.CapacitySlotID(fmt.Sprintf("slot-%d", current)),
		Bootstrap: domain.BootstrapModeJoin,
	}
	width := p.membersPerReplica
	if width < 1 {
		return ScalePlanResolution{}, fmt.Errorf("SGLang growth requires a resolved native-member width")
	}
	for local := range width {
		joining.NativeMembers = append(joining.NativeMembers, domain.NativeMemberID(fmt.Sprintf("dp-%d", current*width+local)))
	}
	return ScalePlanResolution{Plan: &domain.ResolvedPlan{
		ID:                      fmt.Sprintf("sglang-grow-%s-%d-%d", p.worldUID, base.Generation, current+1),
		ProfileFingerprint:      p.profileFingerprint,
		ProcessLifecycleOwner:   domain.ProcessLifecycleOwnerOrchestrator,
		TrafficRequirement:      domain.TrafficRequirementKeepServing,
		VerificationRequirement: domain.VerificationRequirementRequired,
		Change: domain.ResolvedChange{
			Kind: domain.PlanKindGrow,
			Grow: &domain.GrowChange{Replicas: []domain.ReplicaTarget{joining}},
		},
	}}, nil
}

func engineGroupMainContainer(pod *corev1.Pod) (*corev1.Container, error) {
	for i := range pod.Spec.Containers {
		if pod.Spec.Containers[i].Name == consts.MainContainerName {
			return &pod.Spec.Containers[i], nil
		}
	}
	return nil, fmt.Errorf("Pod %s has no %q container", pod.Name, consts.MainContainerName)
}

func metadataValue(
	group *nvidiacomv1beta1.DynamoGraphDeploymentEngineGroup,
	pod *corev1.Pod,
	key string,
) string {
	if value := group.Annotations[key]; value != "" {
		return value
	}
	return pod.Annotations[key]
}

func metadataPort(
	group *nvidiacomv1beta1.DynamoGraphDeploymentEngineGroup,
	pod *corev1.Pod,
	key string,
	defaultValue int,
) (int, error) {
	literal := metadataValue(group, pod, key)
	if literal == "" {
		return defaultValue, nil
	}
	port, err := strconv.Atoi(literal)
	if err != nil || port < 1 || port > 65535 {
		return 0, fmt.Errorf("annotation %s must contain a valid TCP port", key)
	}
	return port, nil
}

func argumentValue(args []string, names ...string) string {
	for _, name := range names {
		for index, argument := range slices.Backward(args) {
			if value, found := strings.CutPrefix(argument, name+"="); found {
				return value
			}
			if argument == name && index+1 < len(args) {
				return args[index+1]
			}
		}
	}
	return ""
}

var _ ScalePlanResolver = sglangScalePlanner{}
