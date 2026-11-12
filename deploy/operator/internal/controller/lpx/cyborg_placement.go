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
	"time"

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
	// cyborgServedLabel records that this controller observed a Cyborg Pod Ready. The
	// Pod then keeps its Grove index for its life, also while it is not Ready.
	cyborgServedLabel = "lpx.nvidia.com/served"
	// grovePodGangGate is the scheduling gate that Grove sets on each Pod it creates
	// and removes when the Pod's PodGang permits scheduling.
	grovePodGangGate = "grove.io/podgang-pending-creation"
)

// cyborgPlacement is the desired distribution of one Cyborg PodClique: replicas
// Pods in each of fabricPods. fabricPods is sorted and not empty.
type cyborgPlacement struct {
	fabricPods []string
	replicas   int32
	// settled is true when the fabric pods are those of the committed LPU plans. The
	// placement is then final, and its Pods can get the lowest Grove indexes.
	settled bool
	// surge is the fabric pod that has one more position while one of its Pods gets
	// a lower Grove index, or empty. See cyborgRenumberStep.
	surge string
	// retire is the UID of the Pod that gives up its position to the Pod with the
	// lower index, or empty. See cyborgRenumberStep.
	retire types.UID
}

// total returns the number of Pods that the distribution keeps. The Grove indexes
// below it are the lowest ones.
func (p cyborgPlacement) total() int32 {
	return p.replicas * int32(len(p.fabricPods))
}

// capacity returns the PodClique capacity that the distribution needs now: its
// total, and one more Pod while it has a surge.
func (p cyborgPlacement) capacity() int32 {
	if p.surge != "" {
		return p.total() + 1
	}
	return p.total()
}

// openFabricPod selects the fabric pod for an unbound Pod with Grove index podIndex,
// or returns an empty value when no position is open. open holds the open positions
// of each fabric pod. The Pod takes the fabric pod of its index block, where block
// k is the indexes [k*replicas, (k+1)*replicas), when that fabric pod has an open
// position, and otherwise the first fabric pod in name order that has one. A
// provision into an empty PodClique therefore gives each fabric pod one contiguous
// index block, in whatever order the Pods appear. Later Pods get Grove's lowest
// free index and the position that is open. cyborgRenumberStep describes how the
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

				fabricPods, settled := committedFabricPods(requests, pcsg.Name, index)
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
				placements[name] = cyborgPlacement{fabricPods: fabricPods, replicas: template.Spec.Replicas, settled: settled}
			}
		}
	}

	return placements, errors.Join(errs...)
}

// committedFabricPods returns the sorted fabric pods that hold the LPUs of one
// workload replica. settled is false until every request of that replica is
// Bound or Degraded on a committed NodeLocal plan; a replica without requests
// is never settled. requests may be nil.
func committedFabricPods(requests map[string]*lpxv1alpha1.LPUPipelineRequest, scalingGroup string, replica int32) (fabricPods []string, settled bool) {
	selected := make(map[string]struct{})

	for _, request := range requests {
		target := request.Spec.MaterializationTarget.PodCliqueScalingGroupRef
		if target == nil || target.Name != scalingGroup || target.ReplicaIndex != int64(replica) {
			continue
		}

		// A plan that the scheduler has not realized, or is releasing, can still move.
		status := request.Status
		if !request.DeletionTimestamp.IsZero() || status == nil || status.Committed == nil ||
			(status.Phase != lpxv1alpha1.RequestPhaseBound && status.Phase != lpxv1alpha1.RequestPhaseDegraded) {
			return nil, false
		}
		placement := status.Committed.Plan.Placement.NodeLocal
		if placement == nil {
			return nil, false
		}

		for _, partition := range placement.PartitionSelections {
			for _, row := range partition.SelectedRows {
				selected[row.Endpoint.FabricID] = struct{}{}
			}
		}
	}

	if len(selected) == 0 {
		return nil, false
	}
	return slices.Sorted(maps.Keys(selected)), true
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
		live, _, err := r.liveCyborgPods(ctx, pclqs[name])
		if err == nil {
			_, err = r.fillCyborgPlacement(ctx, pclqs[name], placements[name], live)
		}
		if err != nil {
			errs = append(errs, err)
		}
	}
	return errors.Join(errs...)
}

// planCyborgRenumbering returns each placement with the step that gives one of its
// Pods a lower Grove index, when one is due (see cyborgRenumberStep). It reads the
// Pods and writes nothing. A pass decides the step one time, so that its scaling
// state, pins, removals and capacity agree. pclqs contains the observation of
// every placement key.
func (r *graphReconciler) planCyborgRenumbering(ctx context.Context, pclqs map[string]*grovev1alpha1.PodClique, placements map[string]cyborgPlacement) (map[string]cyborgPlacement, error) {
	planned := make(map[string]cyborgPlacement, len(placements))
	for name, placement := range placements {
		live, _, err := r.liveCyborgPods(ctx, pclqs[name])
		if err != nil {
			return nil, err
		}
		placement.surge, placement.retire = cyborgRenumberStep(pclqs[name], live, placement)
		planned[name] = placement
	}
	return planned, nil
}

// reconcileCyborgPlacements converges every placed Cyborg PodClique in name order.
// pclqs contains the observation of every placement key. The Pod and PodClique
// watches deliver each write as the next observation. One PodClique that fails
// does not stop the others. A capacity write that waits for a removed Pod needs
// a pass at retryAt, which is the earliest such time, or zero.
func (r *graphReconciler) reconcileCyborgPlacements(
	ctx context.Context,
	pclqs map[string]*grovev1alpha1.PodClique,
	placements map[string]cyborgPlacement,
	scalingBlocked bool,
) (retryAt time.Time, err error) {
	// Keep each error, and the earliest time at which a PodClique needs a pass.
	var errs []error
	for _, name := range slices.Sorted(maps.Keys(placements)) {
		at, err := r.reconcileCyborgPlacement(ctx, pclqs[name], placements[name], scalingBlocked)
		if err != nil {
			errs = append(errs, err)
		}
		if !at.IsZero() && (retryAt.IsZero() || at.Before(retryAt)) {
			retryAt = at
		}
	}
	return retryAt, errors.Join(errs...)
}

// reconcileCyborgPlacement converges one non-nil, owned Cyborg PodClique on its
// placement. It pins each unplaced Pod to a fabric pod with an open position and
// deletes the placed Pods that the placement does not keep. Only when those Pods
// are gone does it lower the PodClique capacity, so that Grove finds no placed
// Pod to select by itself. Capacity writes, and the removals that a lower
// capacity needs, wait while scalingBlocked is true. retryAt is not zero when a
// lower capacity waits for the end of a grace period.
//
// While a lower capacity is not yet written, Grove replaces each removed Pod. The
// controller leaves those unplaced Pods to Grove's scale-in, which removes
// unbound and newer Pods first; it never deletes them, so a failed capacity write
// cannot start a create and delete loop.
func (r *graphReconciler) reconcileCyborgPlacement(
	ctx context.Context,
	clique *grovev1alpha1.PodClique,
	placement cyborgPlacement,
	scalingBlocked bool,
) (retryAt time.Time, err error) {
	logger := log.FromContext(ctx).WithValues("podClique", clique.Name)

	live, stopsAt, err := r.liveCyborgPods(ctx, clique)
	if err != nil {
		return time.Time{}, err
	}
	surplus, err := r.fillCyborgPlacement(ctx, clique, placement, live)
	if err != nil {
		return time.Time{}, err
	}

	// A lower capacity needs its removals first, and both wait while Grove defers replica writes.
	capacity := placement.capacity()
	if clique.Spec.Replicas > capacity && scalingBlocked {
		return time.Time{}, nil
	}

	// A second removal of the same Pod changes nothing: a pass can observe a Pod
	// before the cache has its deletion mark.
	for _, pod := range surplus {
		if err := client.IgnoreNotFound(r.Delete(ctx, pod, client.Preconditions{UID: &pod.UID})); err != nil {
			return time.Time{}, fmt.Errorf("delete Cyborg Pod %q outside its placement: %w", pod.Name, err)
		}
		logger.Info("Deleted surplus Cyborg Pod", "pod", pod.Name, "index", podCliquePodIndex(pod), "fabricPod", pod.Labels[cyborgFabricPodLabel], "node", pod.Spec.NodeName, "fabricPods", placement.fabricPods)
	}

	if clique.Spec.Replicas == capacity || scalingBlocked {
		return time.Time{}, nil
	}

	// Grove reads Pods and PodCliques through different watches, so it can observe a
	// lower capacity before the deletion mark of a Pod. It then selects a Pod to
	// remove by itself: a Pod that is not Ready, which can be the only other Cyborg
	// Pod of the fabric pod. Lower the capacity only when the removed Pods are gone,
	// or when their grace period is over. The Pod watch delivers each removal.
	if capacity < clique.Spec.Replicas {
		if len(surplus) > 0 {
			return time.Time{}, nil
		}
		if stopsAt.After(time.Now()) {
			return stopsAt, nil
		}
	}

	// The placement is the only owner of this capacity. A version precondition would
	// only delay the write that stops Grove from replacing the removed Pods.
	scale := &autoscalingv1.Scale{
		Name: clique.Name, Namespace: clique.Namespace,
		Spec: autoscalingv1.ScaleSpec{Replicas: capacity},
	}
	if err := r.SubResource("scale").Update(ctx, clique, client.WithSubResourceBody(scale)); err != nil {
		return time.Time{}, fmt.Errorf("scale Cyborg PodClique %q: %w", clique.Name, err)
	}
	logger.Info("Scaled Cyborg PodClique for its placement", "replicas", capacity, "fabricPods", placement.fabricPods, "surge", placement.surge)
	return time.Time{}, nil
}

// cyborgRenumberStep returns the step that gives one Pod of the placement a lower
// Grove index: the fabric pod that gets one more position, or the UID of the Pod
// that gives up its position, or neither. clique is the non-nil PodClique, and
// live holds every live Pod of it in index order.
//
// Grove gives a new Pod the lowest free index of its PodClique. The Pods that stay
// after a prune have higher indexes when their fabric pods are not the first ones
// in name order. With one more position, Grove creates one Pod, whose index is
// below the total, and the placement gate holds it until the controller pins it
// to that position. When the kubelet has started the new Pod, the Pod above the
// range gives up its position, and the prune steps remove it and write the total.
// So the fabric pod never has fewer Cyborg Pods that the kubelet started than
// before, and no Pod goes when the new Pod gets no GPU.
//
// The step is for the first fabric pod in name order whose bound Pod with the
// lowest index above the range did not serve, and for that Pod. One fabric pod is
// thus complete before the next one starts. After a provision into index blocks,
// the lowest free index has the FPGA route of that Pod, which is its index modulo
// the count of FPGAs, so the routes of the fabric pod are the same after each
// step. A Pod that is Ready, or that this controller recorded as Ready before,
// uses the index that it has, and the Pods above it in its fabric pod then keep
// theirs: a new Pod would get its route. A Pod in a terminal phase is not a bound
// Pod: the LPX scheduler does not accept it as the Cyborg Pod of its fabric pod.
func cyborgRenumberStep(clique *grovev1alpha1.PodClique, live []*corev1.Pod, placement cyborgPlacement) (surge string, retire types.UID) {
	if !placement.settled {
		return "", ""
	}
	total := placement.total()

	// Collect the bound Pods of each fabric pod, in index order.
	bound := make(map[string][]*corev1.Pod, len(placement.fabricPods))
	for _, pod := range live {
		if cyborgPodBound(pod) {
			bound[pod.Labels[cyborgFabricPodLabel]] = append(bound[pod.Labels[cyborgFabricPodLabel]], pod)
		}
	}

	// Find the first fabric pod whose lowest Pod above the range did not serve.
	var candidate *corev1.Pod
	for _, fabricPod := range placement.fabricPods {
		pods := bound[fabricPod]
		first := slices.IndexFunc(pods, func(pod *corev1.Pod) bool { return podCliquePodIndex(pod) >= int(total) })
		if first < 0 {
			continue
		}
		if !cyborgPodServed(pods[first]) {
			candidate = pods[first]
			break
		}

		// That Pod served while a step added a Pod for it. The added Pod has its
		// route and cannot take the place of a different Pod, so it goes: it is
		// the Pod with the highest index below the range that did not serve.
		if len(pods) == int(placement.replicas)+1 {
			for _, pod := range slices.Backward(pods[:first]) {
				if !cyborgPodServed(pod) {
					return "", pod.UID
				}
			}
		}
	}
	if candidate == nil {
		return "", ""
	}
	fabricPod := candidate.Labels[cyborgFabricPodLabel]

	// The fabric pod keeps its count of Pods that the kubelet started, so the new Pod
	// is one of them: the kubelet accepted it and prepared its devices. A Pod that is
	// only bound can still fail there, and it must not be the last Cyborg Pod of the
	// fabric pod then.
	others := int32(0)
	for _, pod := range bound[fabricPod] {
		if pod != candidate && cyborgPodStarted(pod) {
			others++
		}
	}
	if others >= placement.replicas {
		return "", candidate.UID
	}

	// A position that the capacity already has stays until then.
	if clique.Spec.Replicas == total+1 {
		return fabricPod, ""
	}

	// Add a position only to a complete placement that Grove has synchronized. After
	// a prune, Grove can still hold Pods that it created for the removed ones before
	// it observed the lower capacity. With more capacity at that time, one of them
	// stays, and its index is not the lowest one.
	observed := clique.Status.ObservedGeneration
	if clique.Spec.Replicas != total || observed == nil || *observed != clique.Generation ||
		clique.Status.Replicas != total || int32(len(live)) != total {
		return "", ""
	}
	for _, name := range placement.fabricPods {
		if len(bound[name]) != int(placement.replicas) {
			return "", ""
		}
	}
	return fabricPod, ""
}

// liveCyborgPods lists the Pods of one non-nil, owned Cyborg PodClique incarnation
// that are not in deletion, in Grove index order. stopsAt is the latest end of
// the grace period of its Pods in deletion, or zero when it has no such Pod.
func (r *graphReconciler) liveCyborgPods(ctx context.Context, clique *grovev1alpha1.PodClique) (live []*corev1.Pod, stopsAt time.Time, err error) {
	pods := &corev1.PodList{}
	if err := r.List(ctx, pods, client.InNamespace(clique.Namespace), client.MatchingLabels{grovecommon.LabelPodClique: clique.Name}); err != nil {
		return nil, time.Time{}, fmt.Errorf("list the Cyborg Pods of PodClique %q: %w", clique.Name, err)
	}
	// A Pod of an earlier PodClique with the same name is not a Pod of this one.
	for i := range pods.Items {
		pod := &pods.Items[i]
		switch {
		case !metav1.IsControlledBy(pod, clique):
		case pod.DeletionTimestamp.IsZero():
			live = append(live, pod)
		case pod.DeletionTimestamp.After(stopsAt):
			stopsAt = pod.DeletionTimestamp.Time
		}
	}
	slices.SortFunc(live, func(a, b *corev1.Pod) int {
		return cmp.Or(cmp.Compare(podCliquePodIndex(a), podCliquePodIndex(b)), cmp.Compare(a.Name, b.Name))
	})
	return live, stopsAt, nil
}

// fillCyborgPlacement pins each unplaced Pod of one non-nil, owned Cyborg PodClique
// to a fabric pod of the placement that has an open position. live holds the live
// Pods of the PodClique in Grove index order; a write updates its Pod in place. It
// also records each Ready Pod, so that the Pod keeps its index. It returns the
// Pods that the placement does not keep and that the caller can remove. It
// writes no capacity and removes no Pod.
func (r *graphReconciler) fillCyborgPlacement(
	ctx context.Context,
	clique *grovev1alpha1.PodClique,
	placement cyborgPlacement,
	live []*corev1.Pod,
) (surplus []*corev1.Pod, err error) {
	logger := log.FromContext(ctx).WithValues("podClique", clique.Name)

	// Each target fabric pod has replicas open positions, and one more with the surge.
	shrinking := clique.Spec.Replicas > placement.capacity()
	open := make(map[string]int32, len(placement.fabricPods))
	for _, fabricPod := range placement.fabricPods {
		open[fabricPod] = placement.replicas
	}
	if placement.surge != "" {
		open[placement.surge]++
	}

	// A Pod in a terminal phase cannot run again, and Grove creates no Pod in its
	// place while it is there. It holds no position, as the Pod that gives its
	// position to a Pod with a lower index.
	live = slices.DeleteFunc(slices.Clone(live), func(pod *corev1.Pod) bool {
		if cyborgPodTerminal(pod) || (placement.retire != "" && pod.UID == placement.retire) {
			surplus = append(surplus, pod)
			return true
		}
		return false
	})

	// Placed Pods keep their positions; bound Pods, and then Pods that serve, go first.
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
				return surplus, err
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
			return surplus, err
		}

		// A refused pin keeps its position in this pass: the Pod has a version that
		// this pass did not observe, and that version can already hold the position.
		open[fabricPod]--
		if placedPod {
			logger.Info("Placed Cyborg Pod in fabric pod", "pod", pod.Name, "fabricPod", fabricPod, "bound", pod.Spec.NodeName != "")
		}
	}

	// A Pod that serves keeps its index. The record must be durable: the Pod can
	// lose Ready for a time, and it must not go then.
	for _, pod := range live {
		if !cyborgPodReady(pod) || pod.Labels[cyborgServedLabel] != "" {
			continue
		}
		patch := client.RawPatch(types.MergePatchType, fmt.Appendf(nil, `{"metadata":{"labels":{%q:"true"}}}`, cyborgServedLabel))
		if err := client.IgnoreNotFound(r.Patch(ctx, pod, patch)); err != nil {
			return surplus, fmt.Errorf("record that Cyborg Pod %q served: %w", pod.Name, err)
		}
	}

	return surplus, nil
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
// bound before unbound; then a Ready Pod, a Pod with the served record, and the
// other Pods; then the lower Grove index. The Pods that stay after a scale-in
// thus have the lowest indexes among equal Pods. cyborgRenumberStep, not this
// order, selects the Pod that goes for a Pod with a lower index.
func compareCyborgRetention(a, b *corev1.Pod) int {
	rank := func(pod *corev1.Pod) int {
		switch {
		case pod.Spec.NodeName == "":
			return 3
		case cyborgPodReady(pod):
			return 0
		case pod.Labels[cyborgServedLabel] != "":
			return 1
		default:
			return 2
		}
	}
	return cmp.Or(cmp.Compare(rank(a), rank(b)), cmp.Compare(podCliquePodIndex(a), podCliquePodIndex(b)))
}

// cyborgPodBound reports whether the non-nil Pod is on a Node and can run there.
func cyborgPodBound(pod *corev1.Pod) bool {
	return pod.Spec.NodeName != "" && !cyborgPodTerminal(pod)
}

// cyborgPodStarted reports whether the kubelet started a container of the non-nil
// Pod at some time: it then accepted the Pod and prepared its devices. The Pod
// cache keeps only the waiting and the terminated state of a container, so a
// status without a waiting state is of a container that started. A container that
// waits for a restart has its last termination.
func cyborgPodStarted(pod *corev1.Pod) bool {
	started := func(status corev1.ContainerStatus) bool {
		return status.State.Waiting == nil || status.LastTerminationState.Terminated != nil
	}
	return slices.ContainsFunc(pod.Status.InitContainerStatuses, started) || slices.ContainsFunc(pod.Status.ContainerStatuses, started)
}

// cyborgPodTerminal reports whether the non-nil Pod is in a phase that it cannot leave.
func cyborgPodTerminal(pod *corev1.Pod) bool {
	return pod.Status.Phase == corev1.PodFailed || pod.Status.Phase == corev1.PodSucceeded
}

// cyborgPodServed reports whether the non-nil Pod is Ready or has the record that
// it was Ready before. Such a Pod uses the Grove index that it has.
func cyborgPodServed(pod *corev1.Pod) bool {
	return cyborgPodReady(pod) || pod.Labels[cyborgServedLabel] != ""
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
