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
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/enginegroup/kubejournal"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/enginegroup/podcapacity"
	sglangruntime "github.com/ai-dynamo/dynamo/deploy/operator/internal/enginegroup/sglang"
	corev1 "k8s.io/api/core/v1"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

const (
	defaultSGLangControlPort = 9090
	defaultSGLangServingPort = 8000
)

// productionEngineGroupRuntimeProvider enables only explicitly selected,
// profile-conforming runtimes. Every other Engine Group remains fail closed.
type productionEngineGroupRuntimeProvider struct {
	client     client.Client
	httpClient *http.Client
}

func newProductionEngineGroupRuntimeProvider(kubeClient client.Client) EngineGroupRuntimeProvider {
	return &productionEngineGroupRuntimeProvider{
		client: kubeClient,
		httpClient: &http.Client{
			Timeout: 12 * time.Minute,
		},
	}
}

func (p *productionEngineGroupRuntimeProvider) Resolve(
	ctx context.Context,
	group *nvidiacomv1beta1.DynamoGraphDeploymentEngineGroup,
) (EngineGroupRuntime, error) {
	if group.Labels[consts.KubeLabelDynamoEngineGroupRuntime] != consts.KubeLabelDynamoEngineGroupSGLang {
		return EngineGroupRuntime{}, ErrEngineGroupRuntimeUnavailable
	}
	if group.UID == "" {
		return EngineGroupRuntime{}, fmt.Errorf("Engine Group UID is required")
	}
	primary, err := p.primaryPod(ctx, group)
	if err != nil {
		return EngineGroupRuntime{}, err
	}
	if primary.Status.PodIP == "" {
		return EngineGroupRuntime{}, fmt.Errorf("SGLang primary Pod %s has no Pod IP", primary.Name)
	}
	main, err := engineGroupMainContainer(primary)
	if err != nil {
		return EngineGroupRuntime{}, err
	}
	workloadRevision := primary.Labels[consts.KubeLabelDynamoWorkerHash]
	if workloadRevision == "" {
		return EngineGroupRuntime{}, fmt.Errorf(
			"SGLang primary Pod needs the stable %s workload revision label",
			consts.KubeLabelDynamoWorkerHash,
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
		WorkloadRevisionDigest:     "worker-hash:" + workloadRevision,
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

	capacity := podcapacity.NewAdapter(
		p.client,
		group.Namespace,
		group.Name,
		group.UID,
		sglangruntime.JoinerPodBuilder{},
	)
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
	membership := &sglangruntime.MembershipAdapter{
		Client:             control,
		Capacity:           capacity,
		ProfileFingerprint: profile.Fingerprint,
		Journal: kubejournal.NewStore(
			p.client, group.Namespace, group.Name, group.UID, "sglang-membership",
		),
	}
	traffic := &sglangruntime.TrafficAdapter{
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

func (p *productionEngineGroupRuntimeProvider) primaryPod(
	ctx context.Context,
	group *nvidiacomv1beta1.DynamoGraphDeploymentEngineGroup,
) (*corev1.Pod, error) {
	pods := &corev1.PodList{}
	if err := p.client.List(ctx, pods,
		client.InNamespace(group.Namespace),
		client.MatchingLabels{consts.KubeLabelDynamoEngineGroup: group.Name},
	); err != nil {
		return nil, fmt.Errorf("list SGLang Engine Group Pods: %w", err)
	}
	var primary *corev1.Pod
	for i := range pods.Items {
		if pods.Items[i].Labels[consts.KubeLabelDynamoEngineGroupRole] != consts.KubeLabelDynamoEngineGroupRolePrimary {
			continue
		}
		if primary != nil {
			return nil, fmt.Errorf("multiple primary Pods found for SGLang Engine Group")
		}
		primary = pods.Items[i].DeepCopy()
	}
	if primary == nil {
		return nil, fmt.Errorf("SGLang Engine Group primary Pod is missing")
	}
	if primary.Labels[consts.KubeLabelDynamoEngineGroupReplica] != "replica-0" ||
		primary.Labels[consts.KubeLabelDynamoEngineGroupSlot] != "slot-0" {
		return nil, fmt.Errorf("SGLang EP1 primary must identify replica-0 in slot-0")
	}
	if primary.Labels[consts.KubeLabelDynamoScaleRepresentative] != consts.KubeLabelDynamoScaleRepresentativeYes {
		return nil, fmt.Errorf("SGLang EP1 primary must carry the scale representative label")
	}
	return primary, nil
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

var _ EngineGroupRuntimeProvider = (*productionEngineGroupRuntimeProvider)(nil)
var _ EngineGroupScalePlanResolver = sglangGrowthPlanner{}
