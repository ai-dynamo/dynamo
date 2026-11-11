// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package lpx

import (
	"cmp"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"maps"
	"slices"
	"strconv"

	v1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx"
	lpxv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx/scheduler/v1alpha1"
	grovecommon "github.com/ai-dynamo/grove/operator/api/common"
	grovev1alpha1 "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	autoscalingv1 "k8s.io/api/autoscaling/v1"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/component-helpers/scheduling/corev1/nodeaffinity"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

const (
	// cyborgPlacementAnnotation on a hybrid conductor role's Pod template selects how
	// this controller distributes the role's Cyborg Pods. Its value is a placement
	// mode, such as lpx.CyborgPlacementPerFabricPod.
	cyborgPlacementAnnotation = "lpx.nvidia.com/cyborg-placement"
	// fabricPodLabel is the Node label from which the LPX scheduler reads the fabric
	// of LPU Nodes and of Cyborg Nodes.
	fabricPodLabel = "network.topology.nvidia.com/lpu-fabric-pod"
	// cyborgFabricPodLabel records the fabric pod in which this controller placed a Cyborg Pod.
	cyborgFabricPodLabel = "lpx.nvidia.com/fabric-pod"
	// grovePodGangGate is the scheduling gate that Grove sets on each Pod it creates
	// and removes when the Pod's PodGang permits scheduling.
	grovePodGangGate = "grove.io/podgang-pending-creation"
)

// cyborgPlacement is the desired distribution of one Cyborg PodClique: replicas
// Pods in each of fabricPods. fabricPods is sorted and not empty.
type cyborgPlacement struct {
	fabricPods []string
	replicas   int32
	// renumber permits the replacement of a Pod to give it a lower Grove index. It is
	// true when every LPU request of the workload replica is Bound: only then does
	// the LPX scheduler wait for the replacement of a Cyborg Pod, where it otherwise
	// releases the plan of a fabric pod that has no bound Cyborg Pod.
	renumber bool
}

// total returns the PodClique capacity that the distribution needs.
func (p cyborgPlacement) total() int32 {
	return p.replicas * int32(len(p.fabricPods))
}

// openFabricPod selects the fabric pod for an unbound Pod with Grove index podIndex,
// or returns an empty value when no position is open. open holds the open positions
// of each fabric pod. The Pod takes the fabric pod of its index block, where block
// k is the indexes [k*replicas, (k+1)*replicas), when that fabric pod has an open
// position, and otherwise the first fabric pod in name order that has one. A
// provision into an empty PodClique therefore gives each fabric pod one contiguous
// index block, in whatever order the Pods appear. Later Pods get Grove's lowest
// free index and the position that is open. cyborgPodToRenumber describes how the
// Pods that stay after a prune get the indexes below the new total.
func (p cyborgPlacement) openFabricPod(open map[string]int32, podIndex int) string {
	if podIndex >= 0 && p.replicas > 0 {
		if block := podIndex / int(p.replicas); block < len(p.fabricPods) && open[p.fabricPods[block]] > 0 {
			return p.fabricPods[block]
		}
	}
	for _, fabricPod := range p.fabricPods {
		if open[fabricPod] > 0 {
			return fabricPod
		}
	}
	return ""
}

// cyborgPlacementApplies reports whether renderPodCliqueSet marks the workload's
// Cyborg template for fabric-pod placement: the non-nil serving component's
// conductor role selects it, and the non-nil plan has a Cyborg and at least one
// LPU Agent to follow. An unknown mode is an error, never ignored.
func cyborgPlacementApplies(component *v1beta1.DynamoComponentDeploymentSharedSpec, plan *lpx.MaterializationPlan) (bool, error) {
	// The conductor role's Pod template carries the selection.
	role := component.ComponentRole(v1beta1.ComponentRoleLPXConductor)
	if role == nil || role.PodTemplate == nil {
		return false, nil
	}
	switch mode := role.PodTemplate.Annotations[cyborgPlacementAnnotation]; mode {
	case "":
		return false, nil
	case lpx.CyborgPlacementPerFabricPod:
	default:
		return false, fmt.Errorf("component %q: annotation %s must be %q, not %q",
			component.ComponentName, cyborgPlacementAnnotation, lpx.CyborgPlacementPerFabricPod, mode)
	}

	// A Cyborg that runs every partition on its GPU has no LPU fabric pod.
	hasAgents := slices.ContainsFunc(plan.Agents, func(agent lpx.ExpectedAgent) bool { return agent.Replicas > 0 })
	return plan.CyborgTemplate != "" && hasAgents, nil
}

// resolveCyborgPlacements derives the desired distribution of every observed Cyborg
// PodClique that carries the placement label, keyed by PodClique name.
//
// A workload replica whose requests are all Bound or Degraded keeps Cyborg Pods only
// in the fabric pods of its committed LPU plans. Until then it keeps them in every
// fabric pod that its PodClique can select, so the LPX scheduler finds a Cyborg in
// each fabric pod that it can use. The count for each fabric pod is the replica
// count of the PodClique's template in the observed PodCliqueSet.
//
// Every input is an owned observation, so the result needs no desired state. A
// PodClique keeps its placement while a new revision waits, and until Grove
// replaces it after the DGD stops selecting the placement. pcs is nil before the
// PodCliqueSet exists; otherwise pcsgs holds every configured group. requests may
// be nil. A PodClique whose placement cannot be derived has no entry and is named
// in the returned error; the other entries stay valid. Inputs are not mutated.
func (r *graphReconciler) resolveCyborgPlacements(
	ctx context.Context,
	pcs *grovev1alpha1.PodCliqueSet,
	pcsgs map[string]*grovev1alpha1.PodCliqueScalingGroup,
	pclqs map[string]*grovev1alpha1.PodClique,
	requests map[string]*lpxv1alpha1.LPUPipelineRequest,
) (map[string]cyborgPlacement, error) {
	placements := make(map[string]cyborgPlacement)

	// Namespace-restricted mode has no Node cache; renderPodCliqueSet rejects the placement there.
	if pcs == nil || r.config.Namespace.Restricted != "" {
		return placements, nil
	}

	// Visit the PodCliques by scaling group, template and replica, as Grove names them.
	var (
		errs  []error
		nodes *metav1.PartialObjectMetadataList
	)
	for _, group := range pcs.Spec.Template.PodCliqueScalingGroupConfigs {
		pcsg := pcsgs[grovecommon.GeneratePodCliqueScalingGroupName(grovecommon.ResourceNameReplica{Name: pcs.Name, Replica: 0}, group.Name)]
		for _, template := range pcs.Spec.Template.Cliques {
			if !slices.Contains(group.CliqueNames, template.Name) {
				continue
			}

			for index := range pcsg.Spec.Replicas {
				name := grovecommon.GeneratePodCliqueName(grovecommon.ResourceNameReplica{Name: pcsg.Name, Replica: int(index)}, template.Name)
				clique := pclqs[name]
				if clique == nil || !hasCyborgPlacement(clique) {
					continue
				}

				fabricPods, settled, bound := committedFabricPods(requests, pcsg.Name, index)
				if !settled {
					// Read only Node metadata, one time for the PodCliques that still need it;
					// the cache holds no complete Node objects.
					if nodes == nil {
						nodes = &metav1.PartialObjectMetadataList{}
						nodes.SetGroupVersionKind(corev1.SchemeGroupVersion.WithKind("NodeList"))
						if err := r.List(ctx, nodes, client.HasLabels{fabricPodLabel}); err != nil {
							return placements, errors.Join(append(errs, fmt.Errorf("list Nodes with label %s: %w", fabricPodLabel, err))...)
						}
					}

					// One PodClique without a selectable Node must not stop the other placements.
					var err error
					if fabricPods, err = eligibleFabricPods(nodes.Items, clique); err != nil {
						errs = append(errs, err)
						continue
					}
				}
				placements[name] = cyborgPlacement{fabricPods: fabricPods, replicas: template.Spec.Replicas, renumber: bound}
			}
		}
	}

	return placements, errors.Join(errs...)
}

// committedFabricPods returns the sorted fabric pods that hold the LPUs of one
// workload replica. settled is false until every request of that replica is
// Bound or Degraded on a committed NodeLocal plan; a replica without requests
// is never settled. bound is true when the replica is settled and every request
// is Bound. requests may be nil.
func committedFabricPods(requests map[string]*lpxv1alpha1.LPUPipelineRequest, scalingGroup string, replica int32) (fabricPods []string, settled, bound bool) {
	selected := make(map[string]struct{})
	bound = true

	for _, request := range requests {
		target := request.Spec.MaterializationTarget.PodCliqueScalingGroupRef
		if target == nil || target.Name != scalingGroup || target.ReplicaIndex != int64(replica) {
			continue
		}

		// A plan that the scheduler has not realized, or is releasing, can still move.
		status := request.Status
		if !request.DeletionTimestamp.IsZero() || status == nil || status.Committed == nil ||
			(status.Phase != lpxv1alpha1.RequestPhaseBound && status.Phase != lpxv1alpha1.RequestPhaseDegraded) {
			return nil, false, false
		}
		placement := status.Committed.Plan.Placement.NodeLocal
		if placement == nil {
			return nil, false, false
		}
		bound = bound && status.Phase == lpxv1alpha1.RequestPhaseBound

		for _, partition := range placement.PartitionSelections {
			for _, row := range partition.SelectedRows {
				selected[row.Endpoint.FabricID] = struct{}{}
			}
		}
	}

	if len(selected) == 0 {
		return nil, false, false
	}
	return slices.Sorted(maps.Keys(selected)), true, bound
}

// eligibleFabricPods lists the fabric pods that contain one of nodes which the node
// selector and required node affinity of the non-nil Cyborg PodClique can select.
// nodes holds the metadata of the Nodes that have the fabric pod label. Taints and
// free capacity are not evaluated: a Cyborg Pod pinned to a fabric pod without
// capacity stays Pending and gives the LPX scheduler no evidence for that fabric pod.
func eligibleFabricPods(nodes []metav1.PartialObjectMetadata, clique *grovev1alpha1.PodClique) ([]string, error) {
	required := nodeaffinity.GetRequiredNodeAffinity(&corev1.Pod{Spec: clique.Spec.PodSpec})

	// Collect the fabric pod of every Node that the PodClique can select.
	eligible := make(map[string]struct{})
	for i := range nodes {
		node := &corev1.Node{ObjectMeta: nodes[i].ObjectMeta}
		if node.Labels[fabricPodLabel] == "" {
			continue
		}
		selectable, err := required.Match(node)
		if err != nil {
			return nil, fmt.Errorf("evaluate the node affinity of Cyborg PodClique %q: %w", clique.Name, err)
		}
		if selectable {
			eligible[node.Labels[fabricPodLabel]] = struct{}{}
		}
	}

	if len(eligible) == 0 {
		return nil, fmt.Errorf("no Node with label %s matches the node selector and affinity of Cyborg PodClique %q", fabricPodLabel, clique.Name)
	}
	return slices.Sorted(maps.Keys(eligible)), nil
}

// pinCyborgPods pins the unplaced Pods of every placed Cyborg PodClique. It writes
// no capacity and removes no Pod, so it can run before the desired state is known.
// pclqs contains the observation of every placement key. One PodClique that fails
// does not stop the others.
func (r *graphReconciler) pinCyborgPods(ctx context.Context, pclqs map[string]*grovev1alpha1.PodClique, placements map[string]cyborgPlacement) error {
	var errs []error
	for _, name := range slices.Sorted(maps.Keys(placements)) {
		if _, _, _, err := r.fillCyborgPlacement(ctx, pclqs[name], placements[name]); err != nil {
			errs = append(errs, err)
		}
	}
	return errors.Join(errs...)
}

// reconcileCyborgPlacements converges every placed Cyborg PodClique in name order.
// pclqs contains the observation of every placement key. The returned bool reports
// a write, after which the Pod and PodClique watches deliver the next observation.
// One PodClique that fails does not stop the others.
func (r *graphReconciler) reconcileCyborgPlacements(
	ctx context.Context,
	pclqs map[string]*grovev1alpha1.PodClique,
	placements map[string]cyborgPlacement,
	scalingBlocked bool,
) (bool, error) {
	var (
		changed bool
		errs    []error
	)
	for _, name := range slices.Sorted(maps.Keys(placements)) {
		cliqueChanged, err := r.reconcileCyborgPlacement(ctx, pclqs[name], placements[name], scalingBlocked)
		changed = changed || cliqueChanged
		if err != nil {
			errs = append(errs, err)
		}
	}
	return changed, errors.Join(errs...)
}

// reconcileCyborgPlacement converges one non-nil, owned Cyborg PodClique on its
// placement. It pins each unplaced Pod to a fabric pod with an open position and
// deletes the placed Pods that the placement does not keep. Only then does it
// write the PodClique capacity, so that Grove finds no placed Pod to select by
// itself. Capacity writes, and the removals that a lower capacity needs, wait
// while scalingBlocked is true.
//
// While a lower capacity is not yet written, Grove replaces each removed Pod. The
// controller leaves those unplaced Pods to Grove's scale-in, which removes
// unbound and newer Pods first; it never deletes them, so a failed capacity write
// cannot start a create and delete loop.
//
// When the capacity is correct and the placement permits it, the pass removes the
// one Pod that cyborgPodToRenumber selects. Grove then creates the replacement with
// a lower index, and a later pass pins it to the position that the removal opened.
func (r *graphReconciler) reconcileCyborgPlacement(
	ctx context.Context,
	clique *grovev1alpha1.PodClique,
	placement cyborgPlacement,
	scalingBlocked bool,
) (bool, error) {
	logger := log.FromContext(ctx).WithValues("podClique", clique.Name)

	live, surplus, changed, err := r.fillCyborgPlacement(ctx, clique, placement)
	if err != nil {
		return changed, err
	}

	// A lower capacity needs its removals first, and both wait while Grove defers replica writes.
	total := placement.total()
	if clique.Spec.Replicas > total && scalingBlocked {
		return changed, nil
	}
	for _, pod := range surplus {
		if err := client.IgnoreNotFound(r.Delete(ctx, pod, client.Preconditions{UID: &pod.UID})); err != nil {
			return changed, fmt.Errorf("delete Cyborg Pod %q outside its placement: %w", pod.Name, err)
		}
		changed = true
		logger.Info("Deleted surplus Cyborg Pod", "pod", pod.Name, "fabricPod", pod.Labels[cyborgFabricPodLabel], "node", pod.Spec.NodeName, "fabricPods", placement.fabricPods)
	}

	if scalingBlocked {
		return changed, nil
	}

	if clique.Spec.Replicas != total {
		// The placement is the only owner of this capacity. A version precondition would
		// only delay the write that stops Grove from replacing the removed Pods.
		scale := &autoscalingv1.Scale{
			Name: clique.Name, Namespace: clique.Namespace,
			Spec: autoscalingv1.ScaleSpec{Replicas: total},
		}
		if err := r.SubResource("scale").Update(ctx, clique, client.WithSubResourceBody(scale)); err != nil {
			return changed, fmt.Errorf("scale Cyborg PodClique %q: %w", clique.Name, err)
		}
		logger.Info("Scaled Cyborg PodClique for its placement", "replicas", total, "fabricPods", placement.fabricPods)
		return true, nil
	}

	// The capacity is correct: give the Pods that stay the indexes below it.
	if !placement.renumber {
		return changed, nil
	}
	pod := cyborgPodToRenumber(live, placement)
	if pod == nil {
		return changed, nil
	}
	if err := client.IgnoreNotFound(r.Delete(ctx, pod, client.Preconditions{UID: &pod.UID})); err != nil {
		return changed, fmt.Errorf("delete Cyborg Pod %q to give it a lower index: %w", pod.Name, err)
	}
	logger.Info("Deleted Cyborg Pod to give it a lower index", "pod", pod.Name, "index", podCliquePodIndex(pod), "fabricPod", pod.Labels[cyborgFabricPodLabel], "replicas", total)
	return true, nil
}

// cyborgPodToRenumber returns the Pod that the controller removes next so that the
// Pods of a placement get the Grove indexes below its total, or nil when it
// removes none now. live holds every live Pod of the PodClique in index order.
//
// Grove gives a new Pod the lowest free index of its PodClique, and the placement
// gate holds that Pod until the controller pins it to the one open position. So
// the replacement of a Pod whose index is at or above the total gets an index
// below the total, in the same fabric pod. The Pods of a provision have higher
// indexes when their fabric pods are not the first ones in name order.
//
// One Pod goes at a time, and only when every position holds a bound Pod: a
// fabric pod with two or more positions then always keeps a bound Cyborg Pod. A
// Ready Pod stays, because it serves with the index that it has. The first fabric
// pod in name order that has such a Pod goes first, lowest index first, so one
// fabric pod is complete before the next one starts and takes consecutive indexes.
func cyborgPodToRenumber(live []*corev1.Pod, placement cyborgPlacement) *corev1.Pod {
	total := int(placement.total())
	if len(live) != total {
		return nil
	}

	// Every position must hold a bound Pod that has a Grove index.
	held := make(map[string]int32, len(placement.fabricPods))
	for _, pod := range live {
		if pod.Spec.NodeName == "" || podCliquePodIndex(pod) < 0 {
			return nil
		}
		held[pod.Labels[cyborgFabricPodLabel]]++
	}
	for _, fabricPod := range placement.fabricPods {
		if held[fabricPod] != placement.replicas {
			return nil
		}
	}

	// Select the Pod above the range, in fabric pod order, that does not serve.
	var selected *corev1.Pod
	for _, pod := range live {
		if podCliquePodIndex(pod) < total || cyborgPodReady(pod) {
			continue
		}
		if selected == nil || pod.Labels[cyborgFabricPodLabel] < selected.Labels[cyborgFabricPodLabel] {
			selected = pod
		}
	}
	return selected
}

// fillCyborgPlacement observes the live Pods of one non-nil, owned Cyborg PodClique
// and pins each unplaced Pod to a fabric pod of the placement that has an open
// position. It returns the live Pods in Grove index order, the Pods among them
// that hold a GPU outside the placement, which the caller can remove, and whether
// it wrote a pin. It writes no capacity and removes no Pod.
func (r *graphReconciler) fillCyborgPlacement(
	ctx context.Context,
	clique *grovev1alpha1.PodClique,
	placement cyborgPlacement,
) (live, surplus []*corev1.Pod, changed bool, err error) {
	logger := log.FromContext(ctx).WithValues("podClique", clique.Name)

	// Observe the live Pods of this PodClique incarnation in Grove index order.
	pods := &corev1.PodList{}
	if err := r.List(ctx, pods, client.InNamespace(clique.Namespace), client.MatchingLabels{grovecommon.LabelPodClique: clique.Name}); err != nil {
		return nil, nil, false, fmt.Errorf("list the Cyborg Pods of PodClique %q: %w", clique.Name, err)
	}
	for i := range pods.Items {
		if pod := &pods.Items[i]; metav1.IsControlledBy(pod, clique) && pod.DeletionTimestamp.IsZero() {
			live = append(live, pod)
		}
	}
	slices.SortFunc(live, func(a, b *corev1.Pod) int {
		return cmp.Or(cmp.Compare(podCliquePodIndex(a), podCliquePodIndex(b)), cmp.Compare(a.Name, b.Name))
	})

	// Each target fabric pod has replicas open positions.
	shrinking := clique.Spec.Replicas > placement.total()
	open := make(map[string]int32, len(placement.fabricPods))
	for _, fabricPod := range placement.fabricPods {
		open[fabricPod] = placement.replicas
	}

	// Placed Pods keep their positions; bound, ready and older Pods go first.
	placed := slices.DeleteFunc(slices.Clone(live), func(pod *corev1.Pod) bool { return pod.Labels[cyborgFabricPodLabel] == "" })
	slices.SortStableFunc(placed, compareCyborgRetention)
	for _, pod := range placed {
		if fabricPod := pod.Labels[cyborgFabricPodLabel]; open[fabricPod] > 0 {
			open[fabricPod]--
		} else {
			surplus = append(surplus, pod)
		}
	}

	// Fill the open positions with the unplaced Pods, each in the fabric pod of its index block.
	for _, pod := range live {
		if pod.Labels[cyborgFabricPodLabel] != "" {
			continue
		}

		// Where Pod admission did not add this controller's scheduling gate, Grove can
		// release a Pod before the pin. Such a Pod binds without a selector: keep it
		// where its Node has an open position.
		fabricPod, selector := "", map[string]string(nil)
		if pod.Spec.NodeName != "" {
			if fabricPod, err = r.nodeFabricPod(ctx, pod.Spec.NodeName); err != nil {
				return live, surplus, changed, err
			}
		} else {
			// Kubernetes accepts a selector only on a gated Pod. Grove removes its gate with
			// a patch that lists the gates it keeps, and Kubernetes refuses that patch when
			// a listed gate is gone. So this controller's gate leaves last, with the pin.
			held := hasSchedulingGate(pod, lpx.CyborgPlacementGate)
			if len(pod.Spec.SchedulingGates) == 0 || (held && hasSchedulingGate(pod, grovePodGangGate)) {
				continue
			}
			if fabricPod = placement.openFabricPod(open, podCliquePodIndex(pod)); fabricPod != "" {
				selector = map[string]string{fabricPodLabel: fabricPod}
			}
		}

		// Without an open position, a bound Pod holds a GPU that no placement uses.
		// Grove's scale-in removes every other unplaced Pod.
		if open[fabricPod] == 0 {
			if pod.Spec.NodeName != "" && !shrinking {
				surplus = append(surplus, pod)
			}
			continue
		}

		placedPod, err := r.placeCyborgPod(ctx, pod, fabricPod, selector)
		if err != nil {
			return live, surplus, changed, err
		}

		// A refused pin keeps its position in this pass: the Pod has a version that
		// this pass did not observe, and that version can already hold the position.
		open[fabricPod]--
		if placedPod {
			changed = true
			logger.Info("Placed Cyborg Pod in fabric pod", "pod", pod.Name, "fabricPod", fabricPod, "bound", pod.Spec.NodeName != "")
		}
	}

	return live, surplus, changed, nil
}

// placeCyborgPod records fabricPod on the non-nil cached Pod. With a non-nil
// selector it also pins the Pod to that fabric pod and, in the same write, removes
// this controller's scheduling gate, so the Pod is never schedulable without its
// pin. The write carries the observed resource version.
// It returns false without an error when the API server refuses the write for a
// reason that a later observation resolves: a newer Pod version or a deleted Pod.
// Every other refusal is an error: no later Pod version removes it.
func (r *graphReconciler) placeCyborgPod(ctx context.Context, pod *corev1.Pod, fabricPod string, selector map[string]string) (bool, error) {
	// The cached Pod is a projection, so name the changes. A strategic merge patch
	// removes one gate by name and keeps every other gate; the removal does nothing
	// on a Pod that admission did not gate.
	patch := map[string]any{"metadata": map[string]any{
		"resourceVersion": pod.ResourceVersion,
		"labels":          map[string]string{cyborgFabricPodLabel: fabricPod},
	}}
	if selector != nil {
		patch["spec"] = map[string]any{
			"nodeSelector":    selector,
			"schedulingGates": []map[string]string{{"name": lpx.CyborgPlacementGate, "$patch": "delete"}},
		}
	}
	data, err := json.Marshal(patch)
	if err != nil {
		return false, fmt.Errorf("encode the placement of Cyborg Pod %q: %w", pod.Name, err)
	}

	err = r.Patch(ctx, pod, client.RawPatch(types.StrategicMergePatchType, data))
	if err == nil {
		return true, nil
	}

	// A later Pod version resolves these refusals and wakes the controller.
	if apierrors.IsConflict(err) || apierrors.IsNotFound(err) {
		log.FromContext(ctx).V(1).Info("Cyborg Pod placement waits for a newer observation", "pod", pod.Name, "fabricPod", fabricPod, "reason", err.Error())
		return false, nil
	}
	return false, fmt.Errorf("place Cyborg Pod %q in fabric pod %q: %w", pod.Name, fabricPod, err)
}

// nodeFabricPod returns the fabric pod of the named Node from cached Node metadata.
// A Node that is absent or has no fabric label returns an empty value.
func (r *graphReconciler) nodeFabricPod(ctx context.Context, nodeName string) (string, error) {
	node := &metav1.PartialObjectMetadata{}
	node.SetGroupVersionKind(corev1.SchemeGroupVersion.WithKind("Node"))
	if err := r.Get(ctx, client.ObjectKey{Name: nodeName}, node); err != nil {
		if apierrors.IsNotFound(err) {
			return "", nil
		}
		return "", fmt.Errorf("observe Node %q of a Cyborg Pod: %w", nodeName, err)
	}
	return node.Labels[fabricPodLabel], nil
}

// hasSchedulingGate reports whether the non-nil Pod has the named scheduling gate.
func hasSchedulingGate(pod *corev1.Pod, name string) bool {
	return slices.ContainsFunc(pod.Spec.SchedulingGates, func(gate corev1.PodSchedulingGate) bool { return gate.Name == name })
}

// compareCyborgRetention orders Pods by how much a placement loses with them:
// bound before unbound, ready before not ready, then the lower Grove index. The
// Pods that stay after a scale-in then have the lowest indexes, and need no
// replacement to get them.
func compareCyborgRetention(a, b *corev1.Pod) int {
	rank := func(pod *corev1.Pod) int {
		switch {
		case pod.Spec.NodeName == "":
			return 2
		case !cyborgPodReady(pod):
			return 1
		default:
			return 0
		}
	}
	return cmp.Or(cmp.Compare(rank(a), rank(b)), cmp.Compare(podCliquePodIndex(a), podCliquePodIndex(b)))
}

// cyborgPodReady reports whether the non-nil Pod has a true Ready condition.
func cyborgPodReady(pod *corev1.Pod) bool {
	return slices.ContainsFunc(pod.Status.Conditions, func(condition corev1.PodCondition) bool {
		return condition.Type == corev1.PodReady && condition.Status == corev1.ConditionTrue
	})
}

// podCliquePodIndex returns Grove's stable Pod index, or -1 for a Pod without one.
func podCliquePodIndex(pod *corev1.Pod) int {
	index, err := strconv.Atoi(pod.Labels[grovecommon.LabelPodCliquePodIndex])
	if err != nil {
		return -1
	}
	return index
}
