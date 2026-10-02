/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package controller

import (
	"context"
	"fmt"
	"net/http"
	"strconv"
	"strings"
	"time"

	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/enginegroup"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/enginegroup/grovecapacity"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/enginegroup/kubejournal"
	sglangruntime "github.com/ai-dynamo/dynamo/deploy/operator/internal/enginegroup/sglang"
	grovecommon "github.com/ai-dynamo/grove/operator/api/common"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

const (
	defaultSGLangControlPort = 9090
	defaultSGLangServingPort = 8000
)

// sglangGrowthPoCRuntimeProvider enables only the explicitly selected growth proof.
// Its legacy membership bridge and traffic projection are not production contracts.
type sglangGrowthPoCRuntimeProvider struct {
	client     client.Client
	httpClient *http.Client
}

func newEngineGroupRuntimeProvider(kubeClient client.Client) EngineGroupRuntimeProvider {
	return &sglangGrowthPoCRuntimeProvider{
		client:     kubeClient,
		httpClient: &http.Client{},
	}
}

func (p *sglangGrowthPoCRuntimeProvider) Resolve(
	ctx context.Context,
	group *nvidiacomv1beta1.DynamoGraphDeploymentEngineGroup,
) (EngineGroupRuntime, error) {
	if group.Labels[consts.KubeLabelDynamoEngineGroupRuntime] != consts.KubeLabelDynamoEngineGroupSGLang {
		return EngineGroupRuntime{}, ErrEngineGroupRuntimeUnavailable
	}
	if group.UID == "" {
		return EngineGroupRuntime{}, fmt.Errorf("Engine Group UID is required")
	}
	// Bind one live member clique by UID; its PCSG count remains the world-count dimension.
	cliqueName := group.Annotations[consts.KubeAnnotationDynamoEngineGroupPodClique]
	cliqueUID := group.Annotations[consts.KubeAnnotationDynamoEngineGroupPodCliqueUID]
	if cliqueName == "" || cliqueUID == "" {
		return EngineGroupRuntime{}, fmt.Errorf("SGLang runtime requires a Grove member-clique name and UID")
	}
	capacity := &grovecapacity.Adapter{
		Client:    p.client,
		Clique:    types.NamespacedName{Namespace: group.Namespace, Name: cliqueName},
		CliqueUID: types.UID(cliqueUID),
		GroupName: group.Name,
		Journal:   kubejournal.NewStore(p.client, group.Namespace, group.Name, group.UID, "grove-capacity"),
	}
	allocations, err := capacity.Observe(ctx, engineGroupID(group))
	if err != nil {
		return EngineGroupRuntime{}, err
	}
	primary, err := p.primaryPod(ctx, group.Namespace, allocations)
	if err != nil {
		return EngineGroupRuntime{}, err
	}
	if primary.Status.PodIP == "" {
		return EngineGroupRuntime{}, fmt.Errorf("SGLang primary Pod %s has no Pod IP", primary.Name)
	}
	if primary.Spec.RestartPolicy != corev1.RestartPolicyNever {
		return EngineGroupRuntime{}, fmt.Errorf("the SGLang proof requires restartPolicy Never to prevent unfenced engine restarts")
	}
	main, err := engineGroupMainContainer(primary)
	if err != nil {
		return EngineGroupRuntime{}, err
	}
	if len(main.Command) != 3 || (main.Command[0] != "python" && main.Command[0] != "python3") ||
		main.Command[1] != "-m" || main.Command[2] != dynamo.SGLangElasticEPBootstrapModule {
		return EngineGroupRuntime{}, fmt.Errorf("Grove SGLang capacity requires the supported template-invariant bootstrap entrypoint")
	}
	workloadRevision := primary.Labels[grovecommon.LabelPodTemplateHash]
	if workloadRevision == "" {
		return EngineGroupRuntime{}, fmt.Errorf(
			"SGLang primary Pod needs the stable Grove pod-template-hash label",
		)
	}
	gpuLimit := main.Resources.Limits[corev1.ResourceName("nvidia.com/gpu")]
	if gpuLimit.IsZero() {
		return EngineGroupRuntime{}, fmt.Errorf("SGLang primary main container needs an explicit GPU limit")
	}
	resolved, err := dynamo.ResolveSGLangElasticEPProfile(dynamo.SGLangProfileGeometrySource{
		Command:                    main.Command,
		Args:                       main.Args,
		InitialReplicas:            1,
		MainContainerGPUs:          gpuLimit.Value(),
		DedicatedMainGPUAllocation: true,
		WorkloadRevisionDigest:     "grove-pod-template:" + workloadRevision,
	})
	if err != nil {
		return EngineGroupRuntime{}, fmt.Errorf("resolve SGLang Engine Group profile: %w", err)
	}
	if resolved.InitialReplicas != 1 {
		return EngineGroupRuntime{}, fmt.Errorf("the SGLang scale-up proof supports an EP1 primary, got EP%d", resolved.InitialReplicas)
	}
	if resolved.MaximumReplicas != 2 {
		return EngineGroupRuntime{}, fmt.Errorf("the SGLang scale-up proof supports an EP2 maximum, got EP%d", resolved.MaximumReplicas)
	}

	controlPort, err := metadataPort(group, primary, consts.KubeAnnotationDynamoEngineGroupControlPort, defaultSGLangControlPort)
	if err != nil {
		return EngineGroupRuntime{}, err
	}
	control, err := sglangruntime.NewClient(
		fmt.Sprintf("http://%s:%d", primary.Status.PodIP, controlPort),
		p.httpClient,
	)
	if err != nil {
		return EngineGroupRuntime{}, err
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
		return EngineGroupRuntime{}, err
	}

	profile := nvidiacomv1beta1.EngineGroupProfileStatus{
		Backend:                     resolved.Geometry.Backend,
		Fingerprint:                 resolved.Geometry.Fingerprint,
		GPUsPerReplica:              resolved.Geometry.GPUsPerReplica,
		PodsPerReplica:              resolved.Geometry.PodsPerReplica,
		NativeMembersPerReplica:     1,
		MinSafeServingNativeMembers: 1,
		MinSupportedReplicas:        1,
		MaxSupportedReplicas:        resolved.MaximumReplicas,
	}
	membership := &sglangruntime.LegacyGrowthAdapter{
		Client:             control,
		Capacity:           capacity,
		ProfileFingerprint: profile.Fingerprint,
		Journal: kubejournal.NewStore(
			p.client, group.Namespace, group.Name, group.UID, "sglang-membership",
		),
	}
	traffic := &sglangruntime.TrafficProjection{
		Client:   control,
		Capacity: capacity,
		Journal: kubejournal.NewStore(
			p.client, group.Namespace, group.Name, group.UID, "sglang-traffic",
		),
	}
	return EngineGroupRuntime{
		Profile:    profile,
		Capacity:   capacity,
		Membership: membership,
		Traffic:    traffic,
		Verifier:   verifier,
		Planner:    sglangGrowthPlanner{profileFingerprint: profile.Fingerprint},
	}, nil
}

func (p *sglangGrowthPoCRuntimeProvider) primaryPod(
	ctx context.Context,
	namespace string,
	observation enginegroup.CapacityObservation,
) (*corev1.Pod, error) {
	// Primary ownership follows the validated Grove slot, not a mutable role label.
	for _, allocation := range observation.Allocations {
		if allocation.Incarnation.ReplicaID != "replica-0" {
			continue
		}
		ref := allocation.Incarnation.CapacityRefs[0]
		primary := &corev1.Pod{}
		if err := p.client.Get(ctx, client.ObjectKey{Namespace: namespace, Name: ref.Name}, primary); err != nil {
			return nil, fmt.Errorf("get SGLang primary Pod: %w", err)
		}
		if enginegroup.PodUID(primary.UID) != ref.UID || primary.DeletionTimestamp != nil {
			return nil, fmt.Errorf("SGLang primary Pod changed after capacity observation")
		}
		return primary, nil
	}
	return nil, fmt.Errorf("SGLang Engine Group primary allocation is missing")
}

type sglangGrowthPlanner struct {
	profileFingerprint string
}

func (p sglangGrowthPlanner) ResolveScalePlan(
	_ context.Context,
	_ enginegroup.GroupID,
	targetReplicas int32,
	status enginegroup.GroupStatus,
) (ScalePlanResolution, error) {
	base := status.Membership.Observed.CommittedTopology
	current := base.ReplicaCount()
	if targetReplicas == current {
		return ScalePlanResolution{}, nil
	}
	if targetReplicas < current {
		return ScalePlanResolution{Rejection: &enginegroup.Failure{
			Classification: enginegroup.FailureClassificationTerminal,
			Reason:         "SGLangShrinkUnsupported",
			Message:        "the merged SGLang Elastic EP path currently supports growth only",
		}}, nil
	}
	targets := make([]enginegroup.ReplicaTarget, 0, targetReplicas-current)
	for rank := current; rank < targetReplicas; rank++ {
		targets = append(targets, enginegroup.ReplicaTarget{
			ReplicaID:     enginegroup.ReplicaID(fmt.Sprintf("replica-%d", rank)),
			SlotID:        enginegroup.CapacitySlotID(fmt.Sprintf("slot-%d", rank)),
			Bootstrap:     enginegroup.BootstrapModeJoin,
			NativeMembers: []enginegroup.NativeMemberID{enginegroup.NativeMemberID(fmt.Sprintf("dp-%d", rank))},
		})
	}
	return ScalePlanResolution{Plan: &enginegroup.ResolvedPlan{
		ID:                      fmt.Sprintf("sglang-grow-%d-%d", base.Generation, targetReplicas),
		ProfileFingerprint:      p.profileFingerprint,
		ProcessLifecycleOwner:   enginegroup.ProcessLifecycleOwnerOrchestrator,
		TrafficRequirement:      enginegroup.TrafficRequirementKeepServing,
		VerificationRequirement: enginegroup.VerificationRequirementRequired,
		Change: enginegroup.ResolvedChange{
			Kind: enginegroup.PlanKindGrow,
			Grow: &enginegroup.GrowChange{Replicas: targets},
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
		for index := len(args) - 1; index >= 0; index-- {
			if strings.HasPrefix(args[index], name+"=") {
				return strings.TrimPrefix(args[index], name+"=")
			}
			if args[index] == name && index+1 < len(args) {
				return args[index+1]
			}
		}
	}
	return ""
}

var _ EngineGroupRuntimeProvider = (*sglangGrowthPoCRuntimeProvider)(nil)
var _ EngineGroupScalePlanResolver = sglangGrowthPlanner{}
