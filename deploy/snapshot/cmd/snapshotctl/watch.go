package main

import (
	"context"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/client-go/tools/cache"
	watchtools "k8s.io/client-go/tools/watch"
)

func watchNamedObject(
	ctx context.Context,
	name string,
	objType runtime.Object,
	listFn func(context.Context, metav1.ListOptions) (runtime.Object, error),
	watchFn func(context.Context, metav1.ListOptions) (watch.Interface, error),
	condition func(watch.Event) (bool, error),
) error {
	fieldSelector := fields.OneTermEqualSelector("metadata.name", name).String()
	lw := &cache.ListWatch{
		ListFunc: func(options metav1.ListOptions) (runtime.Object, error) {
			options.FieldSelector = fieldSelector
			return listFn(ctx, options)
		},
		WatchFunc: func(options metav1.ListOptions) (watch.Interface, error) {
			options.FieldSelector = fieldSelector
			return watchFn(ctx, options)
		},
	}

	_, err := watchtools.UntilWithSync(ctx, lw, objType, nil, condition)
	if ctx.Err() != nil {
		return ctx.Err()
	}
	return err
}
