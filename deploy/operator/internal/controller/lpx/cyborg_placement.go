// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package lpx

import (
	"cmp"
	"context"
	"errors"
	"fmt"
	"maps"
	"slices"
	"strconv"

	v1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx"
	lpxv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx/scheduler/v1alpha1"
	grovecommon "github.com/ai-dynamo/grove/operator/api/common"
	grovev1alpha1 "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	autoscalingv1 "k8s.io/api/autoscaling/v1"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/component-helpers/scheduling/corev1/nodeaffinity"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

const (
	// cyborgPlacementAnnotation on the Pod template of a hybrid conductor role selects
	// how this controller distributes the Cyborg Pods of the role. Its only mode,
	// cyborgPlacementPerFabricPod, keeps the replica count of the role in each LPU
	// fabric pod that holds LPUs of the workload replica.
	cyborgPlacementAnnotation   = "lpx.nvidia.com/cyborg-placement"
	cyborgPlacementPerFabricPod = "per-fabric-pod"
	// cyborgPlacementLabel is on the Cyborg PodCliques and Pods that this controller
	// places. Its value is the placement mode.
	cyborgPlacementLabel = "lpx.nvidia.com/cyborg-placement"
	// cyborgPlacementGate is the scheduling gate that holds a placed Cyborg Pod until
	// its pin. It is in the Cyborg Pod template, and Grove must copy the gates of a
	// template to the Pods that it creates.
	cyborgPlacementGate = "lpx.nvidia.com/fabric-pod-pending"
	// fabricPodLabel is the Node label that names the LPU fabric pod of a Node.
	fabricPodLabel = "network.topology.nvidia.com/lpu-fabric-pod"

	// The pin records the placement of a Cyborg Pod in these labels, and a DGD binds
	// Cyborg variables to them. The rank is unique in the fabric pod. The replica
	// index is unique in the PodClique and congruent to the rank modulo the replica
	// count, so that the Pods of a fabric pod use each I/O FPGA route equally.
	cyborgFabricPodLabel     = "lpx.nvidia.com/fabric-pod"
	cyborgFabricPodRankLabel = "lpx.nvidia.com/fabric-pod-rank"
	cyborgReplicaIndexLabel  = "lpx.nvidia.com/replica-index"

	// grovePodGangGate is the scheduling gate that Grove sets on each Pod it creates.
	grovePodGangGate = "grove.io/podgang-pending-creation"
)

// cyborgPlacementApplies reports whether the conductor role of the non-nil serving
// component selects fabric-pod placement for a non-nil plan that has a Cyborg and
// LPU Agents. An unknown mode is an error.
func cyborgPlacementApplies(component *v1beta1.DynamoComponentDeploymentSharedSpec, plan *lpx.MaterializationPlan) (bool, error) {
	role := component.ComponentRole(v1beta1.ComponentRoleLPXConductor)
	if role == nil || role.PodTemplate == nil {
		return false, nil
	}
	switch mode := role.PodTemplate.Annotations[cyborgPlacementAnnotation]; mode {
	case "":
		return false, nil
	case cyborgPlacementPerFabricPod:
	default:
		return false, fmt.Errorf("component %q: annotation %s must be %q, not %q",
			component.ComponentName, cyborgPlacementAnnotation, cyborgPlacementPerFabricPod, mode)
	}

	// A Cyborg that runs every partition on its GPU has no LPU fabric pod.
	hasAgents := slices.ContainsFunc(plan.Agents, func(agent lpx.ExpectedAgent) bool { return agent.Replicas > 0 })
	return plan.CyborgTemplate != "" && hasAgents, nil
}

// reconcileCyborgPlacement keeps the template replica count of Cyborg Pods in each
// LPU fabric pod of every observed Cyborg PodClique that has the placement label.
// Until every LPUPipelineRequest of the workload replica is Bound or Degraded on a
// committed NodeLocal plan, the fabric pods are all those that the PodClique can
// select: the LPX scheduler places LPUs only where a Cyborg Pod is bound. Then
// they are the fabric pods of the committed plans.
//
// Every input is an observation. The Pods of an observed PodClique have the
// placement gate of its template and only this function removes it, so it must
// not wait for the DGD or for a new revision. pcs is nil before the PodCliqueSet
// exists; otherwise pcsgs holds every configured group. One PodClique that fails
// does not stop the others.
func (r *graphReconciler) reconcileCyborgPlacement(
	ctx context.Context,
	pcs *grovev1alpha1.PodCliqueSet,
	pcsgs map[string]*grovev1alpha1.PodCliqueScalingGroup,
	pclqs map[string]*grovev1alpha1.PodClique,
	requests map[string]*lpxv1alpha1.LPUPipelineRequest,
) error {
	if pcs == nil {
		return nil
	}
	if !slices.ContainsFunc(slices.Collect(maps.Values(pclqs)), func(clique *grovev1alpha1.PodClique) bool { return hasCyborgPlacement(clique) }) {
		return nil
	}

	// The Node cache holds metadata only. Namespace-restricted mode cannot read Nodes,
	// and renderPodCliqueSet rejects the placement there.
	nodes := &metav1.PartialObjectMetadataList{}
	nodes.SetGroupVersionKind(corev1.SchemeGroupVersion.WithKind("NodeList"))
	if err := r.List(ctx, nodes, client.HasLabels{fabricPodLabel}); err != nil {
		return fmt.Errorf("list Nodes with label %s: %w", fabricPodLabel, err)
	}

	var errs []error
	for _, group := range pcs.Spec.Template.PodCliqueScalingGroupConfigs {
		pcsg := pcsgs[grovecommon.GeneratePodCliqueScalingGroupName(grovecommon.ResourceNameReplica{Name: pcs.Name, Replica: 0}, group.Name)]
		for _, template := range pcs.Spec.Template.Cliques {
			if !slices.Contains(group.CliqueNames, template.Name) {
				continue
			}
			for replica := range pcsg.Spec.Replicas {
				clique := pclqs[grovecommon.GeneratePodCliqueName(grovecommon.ResourceNameReplica{Name: pcsg.Name, Replica: int(replica)}, template.Name)]
				if clique == nil || !hasCyborgPlacement(clique) {
					continue
				}

				fabricPods, settled := committedFabricPods(requests, pcsg.Name, replica)
				if !settled {
					var err error
					if fabricPods, err = eligibleFabricPods(nodes.Items, clique); err != nil {
						errs = append(errs, err)
						continue
					}
				}
				if err := r.placeCyborgPods(ctx, clique, fabricPods, template.Spec.Replicas, nodes.Items, dynamo.GroveScalingBlocked(pcs)); err != nil {
					errs = append(errs, err)
				}
			}
		}
	}
	return errors.Join(errs...)
}

// committedFabricPods returns the sorted fabric pods that hold the LPUs of one
// workload replica. settled is false until every request of that replica is
// Bound or Degraded on a committed NodeLocal plan.
func committedFabricPods(requests map[string]*lpxv1alpha1.LPUPipelineRequest, scalingGroup string, replica int32) (fabricPods []string, settled bool) {
	selected := make(map[string]struct{})
	for _, request := range requests {
		target := request.Spec.MaterializationTarget.PodCliqueScalingGroupRef
		if target == nil || target.Name != scalingGroup || target.ReplicaIndex != int64(replica) {
			continue
		}

		// A plan that the scheduler has not realized, or is releasing, can still move.
		status := request.Status
		if !request.DeletionTimestamp.IsZero() || status == nil || status.Committed == nil || status.Committed.Plan.Placement.NodeLocal == nil ||
			(status.Phase != lpxv1alpha1.RequestPhaseBound && status.Phase != lpxv1alpha1.RequestPhaseDegraded) {
			return nil, false
		}
		for _, partition := range status.Committed.Plan.Placement.NodeLocal.PartitionSelections {
			for _, row := range partition.SelectedRows {
				selected[row.Endpoint.FabricID] = struct{}{}
			}
		}
	}
	return slices.Sorted(maps.Keys(selected)), len(selected) > 0
}

// eligibleFabricPods returns the sorted fabric pods with a Node that the node
// selector and required node affinity of the Cyborg PodClique can select. Taints
// and free capacity are not evaluated: a Pod in a fabric pod without capacity
// stays Pending and gives the LPX scheduler no evidence for that fabric pod.
func eligibleFabricPods(nodes []metav1.PartialObjectMetadata, clique *grovev1alpha1.PodClique) ([]string, error) {
	required := nodeaffinity.GetRequiredNodeAffinity(&corev1.Pod{Spec: clique.Spec.PodSpec})
	eligible := make(map[string]struct{})
	for i := range nodes {
		selectable, err := required.Match(&corev1.Node{ObjectMeta: nodes[i].ObjectMeta})
		if err != nil {
			return nil, fmt.Errorf("evaluate the node affinity of Cyborg PodClique %q: %w", clique.Name, err)
		}
		if selectable && nodes[i].Labels[fabricPodLabel] != "" {
			eligible[nodes[i].Labels[fabricPodLabel]] = struct{}{}
		}
	}
	if len(eligible) == 0 {
		return nil, fmt.Errorf("no Node with label %s matches the node selector and affinity of Cyborg PodClique %q", fabricPodLabel, clique.Name)
	}
	return slices.Sorted(maps.Keys(eligible)), nil
}

// placeCyborgPods converges one owned Cyborg PodClique on replicas Pods in each of
// the sorted fabricPods: it pins each new Pod to a fabric pod with an open
// position, removes the Pods that have no position, and then sets the capacity.
// nodes holds the metadata of the Nodes that have a fabric pod.
//
//nolint:gocyclo // Keep the steps of one placement in order: positions, pins, removals, capacity.
func (r *graphReconciler) placeCyborgPods(
	ctx context.Context,
	clique *grovev1alpha1.PodClique,
	fabricPods []string,
	replicas int32,
	nodes []metav1.PartialObjectMetadata,
	scalingBlocked bool,
) error {
	logger := log.FromContext(ctx).WithValues("podClique", clique.Name)
	capacity := replicas * int32(len(fabricPods))

	list := &corev1.PodList{}
	if err := r.List(ctx, list, client.InNamespace(clique.Namespace), client.MatchingLabels{grovecommon.LabelPodClique: clique.Name}); err != nil {
		return fmt.Errorf("list the Cyborg Pods of PodClique %q: %w", clique.Name, err)
	}

	var (
		live                      int32
		placed, unplaced, surplus []*corev1.Pod
	)
	for i := range list.Items {
		pod := &list.Items[i]
		// A Pod of an earlier PodClique with the same name is not a Pod of this one.
		if !metav1.IsControlledBy(pod, clique) || !pod.DeletionTimestamp.IsZero() {
			continue
		}
		live++
		if pod.Labels[cyborgFabricPodLabel] != "" {
			placed = append(placed, pod)
		} else {
			unplaced = append(unplaced, pod)
		}
	}

	number := func(pod *corev1.Pod, label string) int {
		value, _ := strconv.Atoi(pod.Labels[label])
		return value
	}
	byGroveIndex := func(a, b *corev1.Pod) int {
		return cmp.Or(cmp.Compare(number(a, grovecommon.LabelPodCliquePodIndex), number(b, grovecommon.LabelPodCliquePodIndex)), cmp.Compare(a.Name, b.Name))
	}

	// A scale-in keeps the lower ranks, so the ranks of a fabric pod stay below its count.
	slices.SortFunc(placed, func(a, b *corev1.Pod) int {
		return cmp.Or(cmp.Compare(number(a, cyborgFabricPodRankLabel), number(b, cyborgFabricPodRankLabel)), byGroveIndex(a, b))
	})
	slices.SortFunc(unplaced, byGroveIndex)

	// Each fabric pod has replicas positions. A rank is used one time in a fabric
	// pod, and a replica index one time in the PodClique.
	open := make(map[string]int32, len(fabricPods))
	for _, fabricPod := range fabricPods {
		open[fabricPod] = replicas
	}
	ranks, indexes := make(map[string]map[int]bool), make(map[int]bool)
	hold := func(fabricPod string, rank, index int) {
		if ranks[fabricPod] == nil {
			ranks[fabricPod] = make(map[int]bool)
		}
		ranks[fabricPod][rank], indexes[index] = true, true
		open[fabricPod]--
	}

	// A placed Pod keeps its position and its numbers for its life. A Pod without
	// its own numbers has no position: two passes can give one number to two Pods
	// when the second pass does not observe the write of the first one, and the
	// containers of a Pod read its labels only when they start.
	for _, pod := range placed {
		fabricPod := pod.Labels[cyborgFabricPodLabel]
		rank, rankErr := strconv.Atoi(pod.Labels[cyborgFabricPodRankLabel])
		index, indexErr := strconv.Atoi(pod.Labels[cyborgReplicaIndexLabel])
		if rankErr != nil || indexErr != nil || open[fabricPod] == 0 || ranks[fabricPod][rank] || indexes[index] {
			surplus = append(surplus, pod)
			continue
		}
		hold(fabricPod, rank, index)
	}

	for _, pod := range unplaced {
		gated := slices.Contains(pod.Spec.SchedulingGates, corev1.PodSchedulingGate{Name: cyborgPlacementGate})
		if !gated {
			logger.Info("Cyborg Pod has no placement gate, so the scheduler can bind it before its pin; Grove must keep the scheduling gates of a Pod template",
				"pod", pod.Name, "bound", pod.Spec.NodeName != "")
		}

		var fabricPod string
		podIndex := number(pod, grovecommon.LabelPodCliquePodIndex)
		switch {
		// A Grove version that replaces the gates of the template can release a Pod
		// before its pin. That Pod keeps the position that its Node gives it.
		case pod.Spec.NodeName != "":
			if node := slices.IndexFunc(nodes, func(node metav1.PartialObjectMetadata) bool { return node.Name == pod.Spec.NodeName }); node >= 0 {
				fabricPod = nodes[node].Labels[fabricPodLabel]
			}
		// Kubernetes accepts a node selector only on a gated Pod.
		case len(pod.Spec.SchedulingGates) == 0:
			continue
		// Grove removes its gate with a list of the gates that stay, and Kubernetes
		// refuses that write when a listed gate is gone. The placement gate leaves last.
		case gated && slices.Contains(pod.Spec.SchedulingGates, corev1.PodSchedulingGate{Name: grovePodGangGate}):
			continue
		// The Pod with Grove index i takes fabric pod number i / replicas, and
		// otherwise the first fabric pod with an open position. A provision thus gives
		// each Pod a position from its index alone, in whatever order the Pods appear.
		default:
			candidates := fabricPods
			if block := podIndex / max(int(replicas), 1); block < len(fabricPods) {
				candidates = append([]string{fabricPods[block]}, fabricPods...)
			}
			if candidate := slices.IndexFunc(candidates, func(fabricPod string) bool { return open[fabricPod] > 0 }); candidate >= 0 {
				fabricPod = candidates[candidate]
			}
		}

		// A bound Pod without a position holds a GPU that no placement uses. Grove's
		// scale-in removes the other unplaced Pods. While a lower capacity is pending,
		// Grove replaces each removed Pod, so a removal would start a loop.
		if open[fabricPod] == 0 {
			if pod.Spec.NodeName != "" && clique.Spec.Replicas <= capacity {
				surplus = append(surplus, pod)
			}
			continue
		}

		// The Pod takes the rank and the replica index of its Grove index when they
		// are free, and otherwise the lowest free rank and the lowest free replica
		// index of that rank. A replacement Pod thus takes the rank that was lost.
		rank := podIndex % int(replicas)
		for candidate := 0; ranks[fabricPod][rank]; candidate++ {
			rank = candidate
		}
		index := podIndex
		for candidate := rank; indexes[index] || index%int(replicas) != rank; candidate += int(replicas) {
			index = candidate
		}

		// A refused pin keeps its position in this pass: the Pod has a version that
		// this pass did not observe, and that version can already hold the position.
		if err := r.pinCyborgPod(ctx, pod, fabricPod, rank, index); err != nil {
			return err
		}
		hold(fabricPod, rank, index)
	}

	// Replica writes, and the removals that come before them, wait while Grove defers scaling.
	if scalingBlocked {
		return nil
	}

	// A pass can observe a Pod before the cache has its deletion mark, so a second removal is not an error.
	for _, pod := range surplus {
		if err := client.IgnoreNotFound(r.Delete(ctx, pod, client.Preconditions{UID: &pod.UID})); err != nil {
			return fmt.Errorf("delete Cyborg Pod %q outside its placement: %w", pod.Name, err)
		}
		logger.Info("Deleted surplus Cyborg Pod", "pod", pod.Name, "fabricPod", pod.Labels[cyborgFabricPodLabel], "node", pod.Spec.NodeName, "fabricPods", fabricPods)
	}

	// Grove reads Pods and PodCliques through different watches. When it observes a
	// lower capacity before the removals, it selects a Pod to remove by itself, which
	// can be the last Cyborg Pod of a fabric pod. Grove creates a Pod for a removed
	// one only after it observed the removal. So lower the capacity only in a pass
	// that removes no Pod and finds the capacity of live Pods; Grove's scale-in then
	// removes the Pods that it created, which have no Node.
	if clique.Spec.Replicas == capacity || (clique.Spec.Replicas > capacity && (len(surplus) > 0 || live != clique.Spec.Replicas)) {
		return nil
	}

	// The placement is the only owner of this capacity, so the write has no version precondition.
	scale := &autoscalingv1.Scale{
		Name: clique.Name, Namespace: clique.Namespace,
		Spec: autoscalingv1.ScaleSpec{Replicas: capacity},
	}
	if err := r.SubResource("scale").Update(ctx, clique, client.WithSubResourceBody(scale)); err != nil {
		return fmt.Errorf("scale Cyborg PodClique %q: %w", clique.Name, err)
	}
	logger.Info("Scaled Cyborg PodClique for its placement", "replicas", capacity, "fabricPods", fabricPods)
	return nil
}

// pinCyborgPod records fabricPod, rank and index on the cached Pod. For an unbound
// Pod, the same write sets the node selector and removes the placement gate: the
// Pod is never schedulable without its pin, and its containers never start
// without the labels. A conflict or a missing Pod is not an error, because the
// next version of the Pod starts a new pass. Every other refusal is permanent.
func (r *graphReconciler) pinCyborgPod(ctx context.Context, pod *corev1.Pod, fabricPod string, rank, index int) error {
	pinned := pod.DeepCopy()
	if pinned.Labels == nil {
		pinned.Labels = make(map[string]string)
	}
	pinned.Labels[cyborgFabricPodLabel] = fabricPod
	pinned.Labels[cyborgFabricPodRankLabel] = strconv.Itoa(rank)
	pinned.Labels[cyborgReplicaIndexLabel] = strconv.Itoa(index)
	if pod.Spec.NodeName == "" {
		if pinned.Spec.NodeSelector == nil {
			pinned.Spec.NodeSelector = make(map[string]string)
		}
		pinned.Spec.NodeSelector[fabricPodLabel] = fabricPod
		pinned.Spec.SchedulingGates = slices.DeleteFunc(pinned.Spec.SchedulingGates, func(gate corev1.PodSchedulingGate) bool {
			return gate.Name == cyborgPlacementGate
		})
	}

	// The cached Pod is a projection, so the write is a patch. A strategic merge
	// patch removes one gate by name and keeps the gates of other owners, and the
	// resource version refuses a pin from an old observation.
	err := r.Patch(ctx, pinned, client.StrategicMergeFrom(pod, client.MergeFromWithOptimisticLock{}))
	switch {
	case err == nil:
		log.FromContext(ctx).Info("Placed Cyborg Pod in fabric pod", "pod", pod.Name, "fabricPod", fabricPod, "rank", rank, "replicaIndex", index, "bound", pod.Spec.NodeName != "")
		return nil
	case apierrors.IsConflict(err), apierrors.IsNotFound(err):
		return nil
	default:
		return fmt.Errorf("place Cyborg Pod %q in fabric pod %q: %w", pod.Name, fabricPod, err)
	}
}
