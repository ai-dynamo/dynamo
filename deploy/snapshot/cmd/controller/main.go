package main

import (
	"os"

	appsv1 "k8s.io/api/apps/v1"
	batchv1 "k8s.io/api/batch/v1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/runtime"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	clientgoscheme "k8s.io/client-go/kubernetes/scheme"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/cache"

	snapshotv1alpha1 "github.com/ai-dynamo/dynamo/deploy/snapshot/api/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/snapshot/pkg/logging"
	"github.com/ai-dynamo/dynamo/deploy/snapshot/pkg/requestcontroller"
)

func main() {
	rootLog := logging.ConfigureLogger("stdout")
	ctrl.SetLogger(rootLog)

	scheme := runtime.NewScheme()
	utilruntime.Must(clientgoscheme.AddToScheme(scheme))
	utilruntime.Must(appsv1.AddToScheme(scheme))
	utilruntime.Must(batchv1.AddToScheme(scheme))
	utilruntime.Must(corev1.AddToScheme(scheme))
	utilruntime.Must(snapshotv1alpha1.AddToScheme(scheme))

	options := ctrl.Options{
		Scheme: scheme,
	}
	if watchNamespace := os.Getenv("WATCH_NAMESPACE"); watchNamespace != "" {
		options.Cache = cache.Options{
			DefaultNamespaces: map[string]cache.Config{
				watchNamespace: {},
			},
		}
	}

	manager, err := ctrl.NewManager(ctrl.GetConfigOrDie(), options)
	if err != nil {
		rootLog.Error(err, "Failed to create snapshot controller manager")
		os.Exit(1)
	}

	if err := (&requestcontroller.Reconciler{
		Client: manager.GetClient(),
		Scheme: manager.GetScheme(),
	}).SetupWithManager(manager); err != nil {
		rootLog.Error(err, "Failed to register SnapshotRequest controller")
		os.Exit(1)
	}

	if err := manager.Start(ctrl.SetupSignalHandler()); err != nil {
		rootLog.Error(err, "Snapshot controller exited with error")
		os.Exit(1)
	}
}
