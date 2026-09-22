// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package lpx

import (
	"context"
	"fmt"

	v1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	v1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
	grovecommon "github.com/ai-dynamo/grove/operator/api/common"
	grovev1alpha1 "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	autoscalingv1 "k8s.io/api/autoscaling/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

// getDynamoGraphDeployment validates the child's DGD using a cached reader.
// reader and deployment must be non-nil. A nil result means the DGD is missing,
// deleting, no longer selects LPX, or has not delivered its revision.
func getDynamoGraphDeployment(ctx context.Context, reader client.Reader, deployment *v1alpha1.LPXGraphDeployment) (*v1beta1.DynamoGraphDeployment, error) {
	logger := log.FromContext(ctx)

	owner := metav1.GetControllerOf(deployment)
	if owner == nil || owner.APIVersion != v1beta1.GroupVersion.String() || owner.Kind != v1beta1.DynamoGraphDeploymentGVK.Kind {
		return nil, fmt.Errorf("LPXGraphDeployment requires a DynamoGraphDeployment controller owner")
	}

	dgd := &v1beta1.DynamoGraphDeployment{}
	if err := reader.Get(ctx, client.ObjectKey{Namespace: deployment.Namespace, Name: owner.Name}, dgd); err != nil {
		if apierrors.IsNotFound(err) {
			return nil, nil
		}
		return nil, fmt.Errorf("cannot get DynamoGraphDeployment %q for LPXGraphDeployment: %w", owner.Name, err)
	}
	if dgd.UID != owner.UID {
		return nil, fmt.Errorf("DynamoGraphDeployment %q no longer has the referenced UID", owner.Name)
	}

	// The parent deletes the child on deselection; GC owns the remaining workload.
	if !dgd.DeletionTimestamp.IsZero() || !dgd.HasLPXComponent() {
		return nil, nil
	}

	restartToken := dynamo.LPXRestartToken(dgd, deployment.Annotations[dynamo.LPXRestartAnnotation])
	revision, err := dynamo.LPXInputRevision(dgd, restartToken)
	if err != nil {
		return nil, fmt.Errorf("compute DynamoGraphDeployment input revision: %w", err)
	}

	if deployment.Spec.InputRevision != revision {
		logger.V(4).Info("LPXDynamoGraphDeployment is outdated", "expectedRevision", revision, "inputRevision", deployment.Spec.InputRevision)
		return nil, nil
	}

	if actualRestartToken := deployment.Annotations[dynamo.LPXRestartAnnotation]; actualRestartToken != restartToken {
		logger.V(4).Info("LPXDynamoGraphDeployment is restarting", "expectedRestartToken", restartToken, "restartToken", actualRestartToken)
		return nil, nil
	}

	return dgd, nil
}

// getPodCliqueSet returns the deployment's owned PCS, or nil when absent.
// reader and deployment must be non-nil; reader must use the cache.
func getPodCliqueSet(ctx context.Context, reader client.Reader, deployment *v1alpha1.LPXGraphDeployment) (*grovev1alpha1.PodCliqueSet, error) {
	pcs := &grovev1alpha1.PodCliqueSet{}
	if err := reader.Get(ctx, client.ObjectKey{Namespace: deployment.Namespace, Name: dynamo.PCSNameForLPX(deployment)}, pcs); err != nil {
		return nil, client.IgnoreNotFound(err)
	}

	if !metav1.IsControlledBy(pcs, deployment) {
		return nil, fmt.Errorf("PodCliqueSet %q is not controlled by this LPXGraphDeployment", pcs.Name)
	}
	return pcs, nil
}

// deletePodCliqueSet deletes the already-validated, non-nil PCS.
func deletePodCliqueSet(ctx context.Context, cl client.Client, pcs *grovev1alpha1.PodCliqueSet) error {
	return client.IgnoreNotFound(
		cl.Delete(
			ctx,
			pcs,
			// Foreground GC so that blocking LPRs and Grove dependents can be removed.
			client.PropagationPolicy(metav1.DeletePropagationForeground),
			client.Preconditions{UID: &pcs.UID, ResourceVersion: &pcs.ResourceVersion},
		),
	)
}

// getPodCliqueScalingGroups observes all workload groups under PCS ordinal zero.
// reader and pcs must be non-nil; reader uses the cache and pcs is owned by this LPXGD.
// Missing or deleting groups are omitted; a foreign group at an expected name is an
// ownership error, never a cache miss.
func getPodCliqueScalingGroups(ctx context.Context, reader client.Reader, pcs *grovev1alpha1.PodCliqueSet) (map[string]*grovev1alpha1.PodCliqueScalingGroup, error) {
	configs := pcs.Spec.Template.PodCliqueScalingGroupConfigs
	if len(configs) == 0 {
		return nil, fmt.Errorf("LPX PodCliqueSet %q requires at least one scaling-group template", pcs.Name)
	}

	// Select only the current templates under PCS ordinal zero.
	expected := make(map[string]struct{}, len(configs))
	for _, config := range configs {
		name := grovecommon.GeneratePodCliqueScalingGroupName(grovecommon.ResourceNameReplica{Name: pcs.Name, Replica: 0}, config.Name)
		expected[name] = struct{}{}
	}

	observed := &grovev1alpha1.PodCliqueScalingGroupList{}
	if err := reader.List(ctx, observed, client.InNamespace(pcs.Namespace), client.MatchingLabels{grovecommon.LabelPartOfKey: pcs.Name}); err != nil {
		return nil, err
	}

	// Establish each expected group's identity and lifetime once for downstream operations.
	groups := make(map[string]*grovev1alpha1.PodCliqueScalingGroup, len(configs))
	for i := range observed.Items {
		group := &observed.Items[i]
		if _, selected := expected[group.Name]; !selected {
			continue
		}
		if !metav1.IsControlledBy(group, pcs) {
			return nil, fmt.Errorf("PodCliqueScalingGroup %q is not controlled by PodCliqueSet %q", group.Name, pcs.Name)
		}
		if group.DeletionTimestamp.IsZero() {
			groups[group.Name] = group
		}
	}
	return groups, nil
}

// scaleDownPodCliqueScalingGroup lowers capacity before asynchronous request cleanup.
// pcsg must be non-nil, owned and not deleting. Call only for explicitly managed
// capacity. Scale-out must wait until terminating request names disappear.
func scaleDownPodCliqueScalingGroup(ctx context.Context, cl client.Client, pcsg *grovev1alpha1.PodCliqueScalingGroup, replicas int32) error {
	if replicas >= pcsg.Spec.Replicas {
		return nil
	}
	return scalePodCliqueScalingGroup(ctx, cl, pcsg, replicas)
}

// scalePodCliqueScalingGroup updates the non-nil, already-owned PCSG using its
// observed resource version. Call only for explicitly managed capacity.
// A successful write updates the supplied observation for the rest of this pass.
func scalePodCliqueScalingGroup(ctx context.Context, cl client.Client, pcsg *grovev1alpha1.PodCliqueScalingGroup, replicas int32) error {
	if pcsg.Spec.Replicas == replicas {
		return nil
	}
	scale := &autoscalingv1.Scale{
		ObjectMeta: metav1.ObjectMeta{ResourceVersion: pcsg.ResourceVersion},
		Spec:       autoscalingv1.ScaleSpec{Replicas: replicas},
	}
	if err := cl.SubResource("scale").Update(ctx, pcsg, client.WithSubResourceBody(scale)); err != nil {
		return fmt.Errorf("scale LPX PodCliqueScalingGroup %q: %w", pcsg.Name, err)
	}
	pcsg.Spec.Replicas = replicas
	return nil
}
