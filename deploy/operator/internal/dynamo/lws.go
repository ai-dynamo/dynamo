package dynamo

import "fmt"

type LWSMultinodeDeployer struct {
	MultinodeDeployer
}

// GetLeaderHostname returns the leader address in Kubernetes env-var
// expansion syntax. LWS injects LWS_LEADER_ADDRESS into every pod of a
// LeaderWorkerSet, and the kubelet substitutes $(VAR) references in
// container Args/Command before the container starts, which means the
// same string works whether flags are appended directly to a python
// command or wrapped in sh -c. Returning the bare shell form
// ($LWS_LEADER_ADDRESS) would be passed literally to direct-python
// commands and break distributed init.
func (d *LWSMultinodeDeployer) GetLeaderHostname(serviceName string) string {
	return "$(LWS_LEADER_ADDRESS)"
}

// GetNodeRank returns the current pod's rank within its LWS group in
// Kubernetes env-var expansion syntax. needsShell is false because
// $(LWS_WORKER_INDEX) is substituted by the kubelet in container
// Args/Command before the container starts, so no sh -c wrapper is
// required. This contrasts with Grove, which returns a shell
// arithmetic expression and therefore does need shell interpretation.
func (d *LWSMultinodeDeployer) GetNodeRank() (string, bool) {
	return "$(LWS_WORKER_INDEX)", false
}

func (d *LWSMultinodeDeployer) NeedsDNSWait() bool {
	// LWS needs DNS wait because pods start simultaneously and DNS may not be ready
	return true
}

func (d *LWSMultinodeDeployer) GetHostNames(serviceName string, numberOfNodes int32) []string {
	hostnames := make([]string, numberOfNodes)
	hostnames[0] = d.GetLeaderHostname(serviceName)

	// LWS only provides LWS_LEADER_ADDRESS, LWS_GROUP_SIZE, and LWS_WORKER_INDEX
	// LWS_LEADER_ADDRESS format: <lws-name>-<group-index>-<leader-pod-index>.<service-name>.<namespace>
	// Example: trtllm-disagg-tp8-decode-0-0.trtllm-disagg-tp8-decode-0.jsm
	// Worker pods append their index: trtllm-disagg-tp8-decode-0-0-1, trtllm-disagg-tp8-decode-0-0-2, etc.
	// We derive worker addresses by inserting -{i} before the first dot
	for i := int32(1); i < numberOfNodes; i++ {
		// Use sed to replace first "." with "-{i}." to append worker index
		hostnames[i] = fmt.Sprintf("$(echo \"$LWS_LEADER_ADDRESS\" | sed 's/\\./-%d\\./')", i)
	}
	return hostnames
}
