package controller_common

import (
	"context"

	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

const (
	finalizerName = "nvidia.com/finalizer"
)

func AddFinalizer(obj client.Object) {
	controllerutil.AddFinalizer(obj, finalizerName)
}

func RemoveFinalizer(obj client.Object) {
	controllerutil.RemoveFinalizer(obj, finalizerName)
}

func ContainsFinalizer(obj client.Object) bool {
	return controllerutil.ContainsFinalizer(obj, finalizerName)
}

type Finalizer[T client.Object] interface {
	FinalizeResource(ctx context.Context, obj T) error
}

func HandleFinalizer[T client.Object](ctx context.Context, obj T, writer client.Writer, finalizer Finalizer[T]) (bool, error) {
	logger := log.FromContext(ctx)
	// Check if the CR is being deleted
	if obj.GetDeletionTimestamp().IsZero() {
		// object is not being deleted, add the finalizer if it doesn't exist
		if !ContainsFinalizer(obj) {
			logger.Info("Adding finalizer to object", "resourceVersion", obj.GetResourceVersion())
			AddFinalizer(obj)
			err := writer.Update(ctx, obj)
			if err != nil {
				logger.Error(err, "Failed to add finalizer")
				return false, err
			}
			logger.Info("Finalizer added to object", "resourceVersion", obj.GetResourceVersion())
		}
	} else {
		// object is being deleted, if the finalizer exists, call the finalizer and remove the finalizer
		if ContainsFinalizer(obj) {
			logger.Info("Calling finalizer", "resourceVersion", obj.GetResourceVersion())
			err := finalizer.FinalizeResource(ctx, obj)
			if err != nil {
				logger.Error(err, "Failed to call finalizer")
				return false, err
			}
			logger.Info("Removing finalizer from object", "resourceVersion", obj.GetResourceVersion())
			RemoveFinalizer(obj)
			err = writer.Update(ctx, obj)
			if err != nil {
				logger.Error(err, "Failed to remove finalizer")
				return false, err
			}
			logger.Info("Finalizer removed from object", "resourceVersion", obj.GetResourceVersion())
			return true, nil
		}
	}
	return false, nil
}
