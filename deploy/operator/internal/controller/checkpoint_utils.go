package controller

import (
	"context"

	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/types"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

func isFailedRestorePod(obj client.Object) bool {
	pod, ok := obj.(*corev1.Pod)
	if !ok {
		return false
	}

	labels := pod.GetLabels()
	if labels[commonconsts.KubeLabelIsRestoreTarget] != commonconsts.KubeLabelValueTrue {
		return false
	}

	annotations := pod.GetAnnotations()
	return annotations[commonconsts.KubeAnnotationRestoreStatus] == "failed"
}

func deleteFailedRestorePods(
	ctx context.Context,
	c client.Client,
	namespace string,
	matchLabels map[string]string,
) ([]string, error) {
	podList := &corev1.PodList{}
	if err := c.List(ctx, podList, client.InNamespace(namespace), client.MatchingLabels(matchLabels)); err != nil {
		return nil, err
	}

	deletedPods := make([]string, 0, len(podList.Items))
	for i := range podList.Items {
		pod := &podList.Items[i]
		if !isFailedRestorePod(pod) {
			continue
		}

		if err := c.Delete(ctx, pod); err != nil && !apierrors.IsNotFound(err) {
			return nil, err
		}

		deletedPods = append(deletedPods, pod.Name)
	}

	return deletedPods, nil
}

func mapFailedRestorePodToOwnerRequests(obj client.Object, ownerLabel string) []ctrl.Request {
	if !isFailedRestorePod(obj) {
		return nil
	}

	ownerName := obj.GetLabels()[ownerLabel]
	if ownerName == "" {
		return nil
	}

	return []ctrl.Request{{
		NamespacedName: types.NamespacedName{
			Namespace: obj.GetNamespace(),
			Name:      ownerName,
		},
	}}
}
