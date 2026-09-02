/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package controller

import (
	"context"
	"fmt"
	"maps"
	"sync"

	"emperror.dev/errors"
	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/checkpoint"
	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	commonController "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/features"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/gms"
	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	k8serrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"
	leaderworkersetv1 "sigs.k8s.io/lws/api/leaderworkerset/v1"
)

const dcdWorkloadRoleLabel = "role"

// dcdWorkloadRenderer contains the dependencies required to render the
// workload-facing resources of a DynamoComponentDeployment. It deliberately
// does not own reconciliation, watches, finalizers, or status.
//
// Keeping this concrete and package-private establishes a reusable rendering
// boundary without committing to a public provider framework. Composite
// workload programs can reuse this unit without constructing a
// DynamoComponentDeploymentReconciler.
type dcdWorkloadRenderer struct {
	reader client.Reader
	// nodeReader is uncached; see DynamoComponentDeploymentReconciler.NodeReader.
	nodeReader            client.Reader
	config                *configv1alpha1.OperatorConfiguration
	runtimeConfig         *commonController.RuntimeConfig
	dockerSecretRetriever DockerSecretRetriever
}

func newDCDWorkloadRenderer(
	reader client.Reader,
	nodeReader client.Reader,
	config *configv1alpha1.OperatorConfiguration,
	runtimeConfig *commonController.RuntimeConfig,
	dockerSecretRetriever DockerSecretRetriever,
) *dcdWorkloadRenderer {
	return &dcdWorkloadRenderer{
		reader:                reader,
		nodeReader:            nodeReader,
		config:                config,
		runtimeConfig:         runtimeConfig,
		dockerSecretRetriever: dockerSecretRetriever,
	}
}

func (r *DynamoComponentDeploymentReconciler) workloadRenderer() *dcdWorkloadRenderer {
	return newDCDWorkloadRenderer(r.Client, r.NodeReader, r.Config, r.RuntimeConfig, r.DockerSecretRetriever)
}

// renderMultinodePodTemplateSpecs renders the leader and worker pod templates
// shared by LWS and composite multinode workload resources. The caller remains
// responsible for composing those templates into its provider-native object.
func (r *dcdWorkloadRenderer) renderMultinodePodTemplateSpecs(
	ctx context.Context,
	dcd *nvidiacomv1beta1.DynamoComponentDeployment,
) (*corev1.PodTemplateSpec, *corev1.PodTemplateSpec, error) {
	podLabels, err := r.getDCDWorkloadPodLabels(ctx, dcd)
	if err != nil {
		return nil, nil, err
	}
	containerGPUs := r.containerGPUCount(ctx, dcd)

	leaderLabels := make(map[string]string, len(podLabels))
	maps.Copy(leaderLabels, podLabels)
	leaderPodTemplateSpec, err := r.generateLeaderPodTemplateSpec(ctx, dcd, leaderLabels, containerGPUs)
	if err != nil {
		return nil, nil, err
	}

	workerLabels := make(map[string]string, len(podLabels))
	maps.Copy(workerLabels, podLabels)
	workerPodTemplateSpec, err := r.generateWorkerPodTemplateSpec(ctx, dcd, workerLabels, containerGPUs)
	if err != nil {
		return nil, nil, err
	}

	return leaderPodTemplateSpec, workerPodTemplateSpec, nil
}

func (r *dcdWorkloadRenderer) containerGPUCount(
	ctx context.Context,
	dcd *nvidiacomv1beta1.DynamoComponentDeployment,
) dynamo.ContainerGPUCount {
	return sync.OnceValues(func() (int64, error) {
		return dynamo.ResolveContainerGPUs(ctx, r.reader, dcd.Namespace, &dcd.Spec.DynamoComponentDeploymentSharedSpec)
	})
}

func (r *dcdWorkloadRenderer) generateLeaderPodTemplateSpec(
	ctx context.Context,
	dcd *nvidiacomv1beta1.DynamoComponentDeployment,
	labels map[string]string,
	containerGPUs dynamo.ContainerGPUCount,
) (*corev1.PodTemplateSpec, error) {
	leaderPodTemplateSpec, err := r.generatePodTemplateSpec(ctx, dcd, dynamo.RoleLeader, containerGPUs)
	if err != nil {
		return nil, errors.Wrap(err, "failed to generate leader pod template")
	}

	maps.Copy(leaderPodTemplateSpec.ObjectMeta.Labels, labels)
	leaderPodTemplateSpec.ObjectMeta.Labels[dcdWorkloadRoleLabel] = string(dynamo.RoleLeader)
	delete(leaderPodTemplateSpec.ObjectMeta.Labels, commonconsts.KubeLabelDynamoSelector)

	if err := checkMainContainer(&leaderPodTemplateSpec.Spec); err != nil {
		return nil, errors.Wrap(err, "generateLeaderPodTemplateSpec: failed to check main container")
	}

	return leaderPodTemplateSpec, nil
}

func (r *dcdWorkloadRenderer) generateWorkerPodTemplateSpec(
	ctx context.Context,
	dcd *nvidiacomv1beta1.DynamoComponentDeployment,
	labels map[string]string,
	containerGPUs dynamo.ContainerGPUCount,
) (*corev1.PodTemplateSpec, error) {
	workerPodTemplateSpec, err := r.generatePodTemplateSpec(ctx, dcd, dynamo.RoleWorker, containerGPUs)
	if err != nil {
		return nil, errors.Wrap(err, "failed to generate worker pod template")
	}

	maps.Copy(workerPodTemplateSpec.ObjectMeta.Labels, labels)
	workerPodTemplateSpec.ObjectMeta.Labels[dcdWorkloadRoleLabel] = string(dynamo.RoleWorker)
	delete(workerPodTemplateSpec.ObjectMeta.Labels, commonconsts.KubeLabelDynamoSelector)

	if err := checkMainContainer(&workerPodTemplateSpec.Spec); err != nil {
		return nil, errors.Wrap(err, "generateWorkerPodTemplateSpec: failed to check LWS worker main container")
	}

	return workerPodTemplateSpec, nil
}

func (r *dcdWorkloadRenderer) generatePodTemplateSpec(
	ctx context.Context,
	dcd *nvidiacomv1beta1.DynamoComponentDeployment,
	role dynamo.Role,
	containerGPUs dynamo.ContainerGPUCount,
) (*corev1.PodTemplateSpec, error) {
	component := &dcd.Spec.DynamoComponentDeploymentSharedSpec
	componentType, err := r.getDCDWorkloadComponentType(ctx, dcd)
	if err != nil {
		return nil, err
	}
	podLabels := dynamo.GetDCDKubeLabels(dcd)
	podAnnotations := dynamo.GetDCDKubeAnnotations(dcd)
	kubeName := dcd.Name

	// Convert user-provided metrics annotation into controller-managed label.
	// By default (no annotation), metrics are enabled.
	if podAnnotations[commonconsts.KubeAnnotationEnableMetrics] != commonconsts.KubeLabelValueFalse {
		podLabels[commonconsts.KubeLabelMetricsEnabled] = commonconsts.KubeLabelValueTrue
	}

	if parentName := dcd.GetLabels()[commonconsts.KubeLabelDynamoGraphDeploymentName]; parentName != "" {
		podLabels[commonconsts.KubeLabelDynamoGraphDeploymentName] = parentName
	} else if parentName := dcd.GetParentGraphDeploymentName(); parentName != "" {
		podLabels[commonconsts.KubeLabelDynamoGraphDeploymentName] = parentName
	}
	if componentType != "" {
		podLabels[commonconsts.KubeLabelDynamoComponentType] = componentType
	}
	if componentName := dynamo.GetDCDComponentName(dcd); componentName != "" {
		podLabels[commonconsts.KubeLabelDynamoComponent] = componentName
	}
	if dynamoNamespace := dynamo.GetDCDDynamoNamespace(dcd); dynamoNamespace != "" {
		podLabels[commonconsts.KubeLabelDynamoNamespace] = dynamoNamespace
	}
	if workerHash := dcd.GetLabels()[commonconsts.KubeLabelDynamoWorkerHash]; workerHash != "" {
		podLabels[commonconsts.KubeLabelDynamoWorkerHash] = workerHash
	}

	var checkpointInfo *checkpoint.CheckpointInfo
	if checkpointConfig := dynamo.GetCheckpoint(component); r.runtimeConfig.Gate.Enabled(features.Checkpoint) && checkpointConfig != nil {
		info, err := checkpoint.ResolveCheckpointForService(
			ctx,
			r.reader,
			dcd.Namespace,
			dynamo.ToAlphaCheckpointConfig(checkpointConfig),
		)
		if err != nil {
			return nil, errors.Wrap(err, "failed to resolve checkpoint")
		}
		if dynamo.IsIntraPodFailoverEnabled(&dcd.Spec.DynamoComponentDeploymentSharedSpec) {
			info.RestoreTargetContainers = dynamo.IntraPodFailoverEngineContainerNames()
		}
		if err := gms.OverlayClients(
			&info.GPUMemoryService,
			info.CheckpointName,
			info.Exists,
			dynamo.GetGPUMemoryService(component),
		); err != nil {
			return nil, errors.Wrap(err, "failed to apply checkpoint gpuMemoryService config")
		}
		checkpointInfo = info
	}

	podSpec, err := dynamo.GenerateBasePodSpecForController(
		dcd,
		r.dockerSecretRetriever,
		r.config,
		role,
		commonconsts.MultinodeDeploymentTypeLWS,
		checkpointInfo,
		containerGPUs,
		dynamo.GenerateBasePodSpecForControllerOptions{
			WorkloadComponentType: nvidiacomv1beta1.ComponentType(componentType),
		},
	)
	if err != nil {
		return nil, errors.Wrap(err, "failed to generate base pod spec")
	}
	if r.runtimeConfig.Gate.Enabled(features.Checkpoint) {
		if checkpointInfo == nil ||
			string(checkpointInfo.StartupPolicy) == string(nvidiacomv1beta1.CheckpointStartupPolicyWaitForCheckpoint) {
			if err := checkpoint.InjectCheckpointIntoPodSpecWithStorageConfig(
				ctx,
				r.reader,
				dcd.Namespace,
				podSpec,
				checkpointInfo,
				r.config.Checkpoint.Storage,
				r.config.Checkpoint.EffectiveSeccompProfile(),
			); err != nil {
				return nil, errors.Wrap(err, "failed to inject checkpoint config")
			}
		}
	}

	if len(podSpec.Containers) == 0 {
		return nil, errors.New("no containers found in base pod spec")
	}

	if err := r.applyNVLinkTopologyCapability(ctx, role, dcd, podSpec); err != nil {
		return nil, err
	}

	podLabels[commonconsts.KubeLabelDynamoSelector] = kubeName
	r.applyDiscoveryLabels(role, podAnnotations, podLabels)

	if checkpointInfo != nil &&
		(checkpointInfo.StartupPolicy == "" ||
			string(checkpointInfo.StartupPolicy) == string(nvidiacomv1beta1.CheckpointStartupPolicyImmediate)) {
		if err := checkpoint.ApplyRestoreCandidateMetadata(podLabels, podAnnotations, checkpointInfo); err != nil {
			return nil, errors.Wrap(err, "failed to apply checkpoint candidate metadata")
		}
	} else if err := checkpoint.ApplyRestorePodMetadataWithStorageConfig(
		podLabels,
		podAnnotations,
		checkpointInfo,
		r.config.Checkpoint.Storage,
	); err != nil {
		return nil, errors.Wrap(err, "failed to apply checkpoint metadata")
	}

	if podSpec.ServiceAccountName == "" {
		serviceAccounts := &corev1.ServiceAccountList{}
		err = r.reader.List(ctx, serviceAccounts, client.InNamespace(dcd.Namespace), client.MatchingLabels{
			commonconsts.KubeLabelDynamoComponentPod: commonconsts.KubeLabelValueTrue,
		})
		if err != nil {
			return nil, errors.Wrapf(err, "failed to list service accounts in namespace %s", dcd.Namespace)
		}
		if len(serviceAccounts.Items) > 0 {
			podSpec.ServiceAccountName = serviceAccounts.Items[0].Name
		} else {
			podSpec.ServiceAccountName = DefaultServiceAccountName
		}
	}

	return &corev1.PodTemplateSpec{
		ObjectMeta: metav1.ObjectMeta{
			Labels:      podLabels,
			Annotations: podAnnotations,
		},
		Spec: *podSpec,
	}, nil
}

// applyDiscoveryLabels marks the pod for the Kubernetes discovery backend.
//
// The elastic-EP follower is skipped for the same reason buildCliqueForRole skips
// RoleGMS: it runs a bare Ray join, not the Dynamo runtime, so it never registers a
// DynamoWorkerMetadata CR. Labelling it would only keep it in the discovery daemon's
// reflector store and wake its debounce loop on every scale-up/scale-down.
func (r *dcdWorkloadRenderer) applyDiscoveryLabels(role dynamo.Role, podAnnotations, podLabels map[string]string) {
	if role == dynamo.RoleFollower {
		return
	}
	if !commonController.IsK8sDiscoveryEnabled(r.config.Discovery.Backend, podAnnotations) {
		return
	}
	podLabels[commonconsts.KubeLabelDynamoDiscoveryBackend] = "kubernetes"
	podLabels[commonconsts.KubeLabelDynamoDiscoveryEnabled] = commonconsts.KubeLabelValueTrue
}

// applyNVLinkTopologyCapability conditionally applies NVLink-partition pod affinity to followers.
//
// By default, synthesis requires followers to land on the leader's `nvidia.com/gpu.clique`.
// This guarantees NVLink connectivity and prevents silent runtime failures where the
// follower appears healthy but cannot communicate with the leader.
//
// However, this label is only stamped on specific hardware (like GB200s). If the leader
// lands on a node without this label, the follower's affinity can never be satisfied,
// causing it to hang in a Pending state indefinitely.
//
// To prevent this, we drop the affinity requirement if the *leader's* node lacks the
// label, allowing the follower to schedule normally. (Note: We must check the leader's
// node specifically, as the presence of an unrelated GB200 elsewhere in a mixed cluster
// will not satisfy the affinity.)
func (r *dcdWorkloadRenderer) applyNVLinkTopologyCapability(
	ctx context.Context,
	role dynamo.Role,
	dcd *nvidiacomv1beta1.DynamoComponentDeployment,
	podSpec *corev1.PodSpec,
) error {
	if role != dynamo.RoleFollower || podSpec.Affinity == nil || podSpec.Affinity.PodAffinity == nil {
		return nil
	}

	terms := podSpec.Affinity.PodAffinity.RequiredDuringSchedulingIgnoredDuringExecution
	kept := make([]corev1.PodAffinityTerm, 0, len(terms))
	var dropped bool
	for _, term := range terms {
		// Only the term synthesis added is ours to remove. The follower deep-copies the
		// leader's spec, so a user may have their own gpu.clique term here -- and one written
		// with MatchExpressions rather than MatchLabels would look unsatisfiable to the
		// lookup below and be deleted silently. Match on the generation label synthesis
		// stamps, which a user term does not carry.
		if !isSynthesizedLeaderCliqueTerm(term) {
			kept = append(kept, term)
			continue
		}
		support, err := r.leaderNVLinkSupport(ctx, dcd, term.LabelSelector)
		if err != nil {
			return errors.Wrap(err, "failed to determine whether the elastic-EP leader sits in an NVLink partition")
		}
		// Only a proven-unsupported cluster drops the term. Unknown keeps it: the
		// scheduler evaluates pod affinity when it places the pod, and retries until it
		// can, so a leader that is merely not scheduled yet resolves itself.
		if support != nvlinkUnsupported {
			kept = append(kept, term)
			continue
		}
		dropped = true
	}
	if !dropped {
		return nil
	}

	log.FromContext(ctx).Info(
		"elastic-EP leader is not in an NVLink partition; scheduling the follower without partition affinity",
		"topologyKey", commonconsts.NodeLabelGPUClique,
	)
	podSpec.Affinity.PodAffinity.RequiredDuringSchedulingIgnoredDuringExecution = kept
	// Leave no empty PodAffinity behind: an affinity with no terms is meaningless and
	// only makes the rendered pod spec harder to read.
	if len(kept) == 0 {
		podSpec.Affinity.PodAffinity = nil
	}
	return nil
}

// isSynthesizedLeaderCliqueTerm reports whether a required pod-affinity term is the one
// synthesizeElasticEPFollowerDCD added, rather than one the follower inherited from the
// leader's user-supplied affinity.
//
// The distinguishing mark is KubeLabelDynamoSelector: synthesis pins its term to a single
// leader DCD generation, and a user-authored term has no reason to carry that internal
// label. Topology key alone is not enough -- a user may legitimately write their own
// gpu.clique term, and removing it would silently discard their scheduling intent.
func isSynthesizedLeaderCliqueTerm(term corev1.PodAffinityTerm) bool {
	if term.TopologyKey != commonconsts.NodeLabelGPUClique || term.LabelSelector == nil {
		return false
	}
	_, ours := term.LabelSelector.MatchLabels[commonconsts.KubeLabelDynamoSelector]
	return ours
}

// nvlinkSupport is the three-way answer to "can the follower's partition affinity ever
// be satisfied?". Two-way was the bug: it collapsed "the leader is not scheduled yet"
// into "this cluster has no NVLink", and dropped a term the scheduler would have
// honoured moments later.
type nvlinkSupport int

const (
	// nvlinkUnknown means no leader pod is scheduled yet, so there is nothing to compare
	// against. Keep the term: pod affinity is evaluated by the scheduler at scheduling
	// time, not here, and the scheduler retries a Pending pod until the leader lands.
	// Dropping it would turn a transient the scheduler resolves on its own into a
	// permanent loss of the guarantee.
	nvlinkUnknown nvlinkSupport = iota
	// nvlinkSupported means a scheduled leader sits on a node advertising a partition.
	nvlinkSupported
	// nvlinkUnsupported means a scheduled leader sits on a node with no partition label,
	// so the term can never be satisfied on this cluster. Only this answer drops it.
	nvlinkUnsupported
)

// leaderNVLinkSupport reports whether the node running this follower's leader advertises
// an NVLink partition, or that no leader is placed yet.
//
// Only a *scheduled* leader on an unlabelled node proves the cluster cannot satisfy the
// term. Everything else -- no leader pods, a leader still being placed, a node we may not
// read -- is unknown, and unknown keeps the term.
func (r *dcdWorkloadRenderer) leaderNVLinkSupport(
	ctx context.Context,
	dcd *nvidiacomv1beta1.DynamoComponentDeployment,
	leaderSelector *metav1.LabelSelector,
) (nvlinkSupport, error) {
	if leaderSelector == nil || len(leaderSelector.MatchLabels) == 0 {
		return nvlinkUnknown, nil
	}

	leaderPods := &corev1.PodList{}
	if err := r.reader.List(ctx, leaderPods,
		client.InNamespace(dcd.Namespace),
		client.MatchingLabels(leaderSelector.MatchLabels),
	); err != nil {
		return nvlinkUnknown, err
	}

	for i := range leaderPods.Items {
		nodeName := leaderPods.Items[i].Spec.NodeName
		if nodeName == "" {
			continue
		}
		node := &corev1.Node{}
		if err := r.nodeReader.Get(ctx, types.NamespacedName{Name: nodeName}, node); err != nil {
			// Forbidden is treated like NotFound: both mean "cannot tell whether this is
			// an NVLink node", which is unknown rather than unsupported, so the term
			// stays. Namespace-restricted mode grants the manager a namespaced Role,
			// which cannot carry a cluster-scoped Node rule at all, so returning an error
			// here would wedge the whole reconcile instead of leaving one placement
			// decision undecided.
			if k8serrors.IsNotFound(err) || k8serrors.IsForbidden(err) {
				log.FromContext(ctx).V(1).Info(
					"cannot read the elastic-EP leader's node; scheduling the follower without partition affinity",
					"node", nodeName, "reason", err.Error(),
				)
				continue
			}
			return nvlinkUnknown, err
		}
		if _, ok := node.Labels[commonconsts.NodeLabelGPUClique]; ok {
			return nvlinkSupported, nil
		}
		// A scheduled leader on an unlabelled node is the one positive answer: this
		// cluster has no NVLink partitions, so the term can never be satisfied.
		return nvlinkUnsupported, nil
	}
	// No leader pod is scheduled yet. Unknown, not unsupported.
	return nvlinkUnknown, nil
}

func (r *dcdWorkloadRenderer) generateService(
	ctx context.Context,
	dcd *nvidiacomv1beta1.DynamoComponentDeployment,
) (*corev1.Service, bool, error) {
	deleteStub := &corev1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name:      dynamo.NormalizeKubeResourceName(dcd.Name),
			Namespace: dcd.Namespace,
		},
	}

	annotations := dynamo.GetDCDKubeAnnotations(dcd)
	isK8sDiscovery := commonController.IsK8sDiscoveryEnabled(r.config.Discovery.Backend, annotations)

	if !(isK8sDiscovery || dcd.IsFrontendComponent()) {
		return deleteStub, true, nil
	}

	dynamoNamespace := dynamo.GetDCDDynamoNamespace(dcd)
	if dynamoNamespace == "" {
		return nil, false, fmt.Errorf("expected DynamoComponentDeployment %s to have a dynamoNamespace", dcd.Name)
	}

	componentType, err := r.getDCDWorkloadComponentType(ctx, dcd)
	if err != nil {
		return nil, false, err
	}

	svc, err := dynamo.GenerateComponentService(dynamo.ComponentServiceParams{
		ServiceName:     dcd.Name,
		Namespace:       dcd.Namespace,
		ComponentType:   componentType,
		DynamoNamespace: dynamoNamespace,
		ComponentName:   dynamo.GetDCDComponentName(dcd),
		Labels:          dynamo.GetDCDKubeLabels(dcd),
		Annotations:     annotations,
		IsK8sDiscovery:  isK8sDiscovery,
	})
	if err != nil {
		return nil, false, err
	}
	if dcd.IsMultinode() {
		svc.Spec.Selector[dcdWorkloadRoleLabel] = string(dynamo.RoleLeader)
	}
	return svc, false, nil
}

func (r *dcdWorkloadRenderer) getDCDWorkloadPodLabels(
	ctx context.Context,
	dcd *nvidiacomv1beta1.DynamoComponentDeployment,
) (map[string]string, error) {
	labels := dynamo.GetDCDKubeLabels(dcd)
	componentType, err := r.getDCDWorkloadComponentType(ctx, dcd)
	if err != nil {
		return nil, err
	}
	if componentType == "" {
		return labels, nil
	}
	labels[commonconsts.KubeLabelDynamoComponentType] = componentType
	specType := string(dcd.Spec.ComponentType)
	if componentType == commonconsts.ComponentTypeWorker &&
		(specType == commonconsts.ComponentTypePrefill || specType == commonconsts.ComponentTypeDecode) &&
		labels[commonconsts.KubeLabelDynamoSubComponentType] == "" {
		labels[commonconsts.KubeLabelDynamoSubComponentType] = specType
	}
	return labels, nil
}

// getDCDWorkloadComponentType returns the component type that should be
// rendered into pod metadata, env, and service selectors for this DCD. It keeps
// legacy-compatible worker generations as "worker" even when the v1beta1 DCD
// spec is represented as a more specific prefill/decode worker component.
func (r *dcdWorkloadRenderer) getDCDWorkloadComponentType(
	ctx context.Context,
	dcd *nvidiacomv1beta1.DynamoComponentDeployment,
) (string, error) {
	if dcd == nil {
		return "", nil
	}

	componentType := dynamo.GetDCDWorkloadComponentType(dcd)
	if componentType == commonconsts.ComponentTypeWorker || !dynamo.IsWorkerComponent(componentType) {
		return componentType, nil
	}

	if hasLegacyWorkerSelector(dcd.GetLabels(), componentType) {
		return commonconsts.ComponentTypeWorker, nil
	}

	legacy, err := r.hasExistingLegacyWorkerSelector(ctx, dcd, componentType)
	if err != nil {
		return "", err
	}
	if legacy {
		return commonconsts.ComponentTypeWorker, nil
	}

	return componentType, nil
}

func (r *dcdWorkloadRenderer) hasExistingLegacyWorkerSelector(
	ctx context.Context,
	dcd *nvidiacomv1beta1.DynamoComponentDeployment,
	componentType string,
) (bool, error) {
	if dcd == nil || r == nil || r.reader == nil {
		return false, nil
	}

	deployment := &appsv1.Deployment{}
	if err := r.reader.Get(ctx, types.NamespacedName{Name: dcd.Name, Namespace: dcd.Namespace}, deployment); err == nil {
		if hasLegacyWorkerSelector(deployment.Spec.Template.Labels, componentType) {
			return true, nil
		}
	} else if !k8serrors.IsNotFound(err) {
		return false, fmt.Errorf("failed to get deployment %s/%s: %w", dcd.Namespace, dcd.Name, err)
	}

	if r.runtimeConfig.Gate.Enabled(features.LWS) {
		lwsName := leaderWorkerSetName(dcd)
		leaderWorkerSet := &leaderworkersetv1.LeaderWorkerSet{}
		if err := r.reader.Get(ctx, types.NamespacedName{Name: lwsName, Namespace: dcd.Namespace}, leaderWorkerSet); err == nil {
			template := leaderWorkerSet.Spec.LeaderWorkerTemplate
			if template.LeaderTemplate != nil && hasLegacyWorkerSelector(template.LeaderTemplate.Labels, componentType) {
				return true, nil
			}
			if hasLegacyWorkerSelector(template.WorkerTemplate.Labels, componentType) {
				return true, nil
			}
		} else if !k8serrors.IsNotFound(err) {
			return false, fmt.Errorf("failed to get leaderworkerset %s/%s: %w", dcd.Namespace, lwsName, err)
		}
	}

	serviceName := dynamo.NormalizeKubeResourceName(dcd.Name)
	service := &corev1.Service{}
	if err := r.reader.Get(ctx, types.NamespacedName{Name: serviceName, Namespace: dcd.Namespace}, service); err == nil {
		return hasLegacyWorkerSelector(service.Spec.Selector, componentType), nil
	} else if !k8serrors.IsNotFound(err) {
		return false, fmt.Errorf("failed to get service %s/%s: %w", dcd.Namespace, serviceName, err)
	}

	return false, nil
}
