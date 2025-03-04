package controller_common

import (
	"context"
	"strings"

	"k8s.io/apimachinery/pkg/api/meta"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/controller-runtime/pkg/predicate"
)

type Config struct {
	// Enable resources filtering, only the resources belonging to the given namespace will be handled.
	RestrictedNamespace string
}

func EphemeralDeploymentEventFilter(config Config) predicate.Predicate {
	return predicate.NewPredicateFuncs(func(o client.Object) bool {
		l := log.FromContext(context.Background())
		objMeta, err := meta.Accessor(o)
		if err != nil {
			l.Error(err, "Error extracting object metadata")
			return false
		}
		if config.RestrictedNamespace != "" {
			// in case of a restricted namespace, we only want to process the events that are in the restricted namespace
			return objMeta.GetNamespace() == config.RestrictedNamespace
		}
		// in all other cases, discard the event if it is destined to an ephemeral deployment
		if strings.Contains(objMeta.GetNamespace(), "ephemeral") {
			return false
		}
		return true
	})
}
