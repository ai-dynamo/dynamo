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

// getPodCliqueScalingGroup observes the single engine group under PCS ordinal zero.
// reader and pcs must be non-nil; reader must use the cache, and pcs must already
// be owned by this LPXGD. A non-nil result is controlled by that PCS and not
// deleting; nil means the group is absent from the cache or is deleting.
// A group controlled by another PCS is an ownership error.
// Nil does not mean zero replicas, and a zero-replica group is returned normally.
func getPodCliqueScalingGroup(ctx context.Context, reader client.Reader, pcs *grovev1alpha1.PodCliqueSet) (*grovev1alpha1.PodCliqueScalingGroup, error) {
	// Engine replicas belong to one PCSG, not to additional top-level PCS ordinals.
	configs := pcs.Spec.Template.PodCliqueScalingGroupConfigs
	if len(configs) != 1 {
		return nil, fmt.Errorf("LPX PodCliqueSet %q requires exactly one scaling-group template", pcs.Name)
	}
	name := grovecommon.GeneratePodCliqueScalingGroupName(grovecommon.ResourceNameReplica{Name: pcs.Name, Replica: 0}, configs[0].Name)

	// Establish the group's identity and lifetime once for all downstream operations.
	pcsg := &grovev1alpha1.PodCliqueScalingGroup{}
	if err := reader.Get(ctx, client.ObjectKey{Namespace: pcs.Namespace, Name: name}, pcsg); err != nil {
		return nil, client.IgnoreNotFound(err)
	}

	if !metav1.IsControlledBy(pcsg, pcs) {
		return nil, fmt.Errorf("PodCliqueScalingGroup %q is not controlled by PodCliqueSet %q", pcsg.Name, pcs.Name)
	}

	if !pcsg.DeletionTimestamp.IsZero() {
		return nil, nil
	}

	return pcsg, nil
}
