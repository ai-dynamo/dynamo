module github.com/nvidia/dynamo/deploy/inference-gateway

go 1.24.0

require (
	sigs.k8s.io/controller-runtime v0.22.4
	sigs.k8s.io/gateway-api-inference-extension v1.2.1
)

// Note: Run `go mod tidy` after adding the gateway-api-inference-extension dependency
// to fetch all transitive dependencies.

