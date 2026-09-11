// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package lpx

import (
	"context"

	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	lpxv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/internal/thirdparty/lpxscheduler/v1alpha1"
	groveconstants "github.com/ai-dynamo/grove/operator/api/common/constants"
	grovev1alpha1 "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

const (
	// DeploymentNameAnnotation routes rendered-object events to their LPXGraphDeployment.
	DeploymentNameAnnotation = "lpx.nvidia.com/deployment-name"
	// DeploymentUIDAnnotation records the owning LPXGraphDeployment UID on rendered objects.
	DeploymentUIDAnnotation = "lpx.nvidia.com/deployment-uid"
)

// OwnsPodClique verifies the LPX owner chain before ordinary DGD watches
// suppress a role-readiness event. Authorable annotations alone are not proof.
func OwnsPodClique(ctx context.Context, reader client.Reader, clique *grovev1alpha1.PodClique) bool {
	if clique == nil || reader == nil || !isMaterializedRole(clique) {
		return false
	}
	owner := metav1.GetControllerOf(clique)
	if owner == nil || owner.Kind != groveconstants.KindPodCliqueScalingGroup || owner.APIVersion != grovev1alpha1.SchemeGroupVersion.String() || owner.UID == "" {
		return false
	}
	group := &grovev1alpha1.PodCliqueScalingGroup{}
	if err := reader.Get(ctx, client.ObjectKey{Namespace: clique.Namespace, Name: owner.Name}, group); err != nil || group.UID != owner.UID {
		return false
	}
	owner = metav1.GetControllerOf(group)
	if owner == nil || owner.Kind != groveconstants.KindPodCliqueSet || owner.APIVersion != grovev1alpha1.SchemeGroupVersion.String() || owner.UID == "" {
		return false
	}
	pcs := &grovev1alpha1.PodCliqueSet{}
	if err := reader.Get(ctx, client.ObjectKey{Namespace: clique.Namespace, Name: owner.Name}, pcs); err != nil || pcs.UID != owner.UID {
		return false
	}
	owner = metav1.GetControllerOf(pcs)
	return owner != nil && owner.Kind == nvidiacomv1alpha1.LPXGraphDeploymentGVK.Kind &&
		owner.APIVersion == nvidiacomv1alpha1.GroupVersion.String() && owner.UID != "" &&
		owner.Name == clique.Annotations[DeploymentNameAnnotation] &&
		string(owner.UID) == clique.Annotations[DeploymentUIDAnnotation]
}

func isMaterializedRole(clique *grovev1alpha1.PodClique) bool {
	role := clique.Annotations[lpxv1alpha1.PodRoleAnnotation]
	return clique.Annotations[WorkloadDigestAnnotation] != "" &&
		(role == lpxv1alpha1.PodRoleAgent || role == lpxv1alpha1.PodRoleConductor || role == lpxv1alpha1.PodRoleCyborgWorker)
}
