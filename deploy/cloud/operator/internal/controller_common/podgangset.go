package controller_common

import (
	"sort"

	grovev1alpha1 "github.com/NVIDIA/grove/operator/api/core/v1alpha1"
)

func CanonicalizePodCliqueSet(gangSet *grovev1alpha1.PodCliqueSet) *grovev1alpha1.PodCliqueSet {
	// sort cliques by name
	sort.Slice(gangSet.Spec.Template.Cliques, func(i, j int) bool {
		return gangSet.Spec.Template.Cliques[i].Name < gangSet.Spec.Template.Cliques[j].Name
	})
	// sort scaling groups by name
	sort.Slice(gangSet.Spec.Template.PodCliqueScalingGroupConfigs, func(i, j int) bool {
		return gangSet.Spec.Template.PodCliqueScalingGroupConfigs[i].Name < gangSet.Spec.Template.PodCliqueScalingGroupConfigs[j].Name
	})
	return gangSet
}
