package system

import (
	"os"
	"sync"

	"github.com/sirupsen/logrus"
)

const (
	// NamespaceEnvKey is the environment variable that specifies the system namespace.
	NamespaceEnvKey = "SYSTEM_NAMESPACE"
	// ResourceLabelEnvKey is the environment variable that specifies the system resource
	// label.
	ResourceLabelEnvKey = "SYSTEM_RESOURCE_LABEL"

	DefaultNamespace = "yatai-deployment"
	MagicDNSEnvKey   = "MAGIC_DNS"
	DefaultMagicDNS  = "sslip.io"
)

var (
	once sync.Once
)

// GetNamespace returns the name of the K8s namespace where our system components
// run.
func GetNamespace() string {
	if ns := os.Getenv(NamespaceEnvKey); ns != "" {
		return ns
	}

	once.Do(func() {
		logrus.Infof("%s environment variable not set, using default namespace %s", NamespaceEnvKey, DefaultNamespace)
	})
	return DefaultNamespace
}

// GetResourceLabel returns the label key identifying K8s objects our system
// components source their configuration from.
func GetResourceLabel() string {
	return os.Getenv(ResourceLabelEnvKey)
}

func GetMagicDNS() string {
	magicDNS := os.Getenv(MagicDNSEnvKey)
	if magicDNS == "" {
		magicDNS = DefaultMagicDNS
	}
	return magicDNS
}
