package dynamo

import "fmt"

type LWSMultinodeDeployer struct {
	MultinodeDeployer
}

func (d *LWSMultinodeDeployer) GetLeaderHostname(serviceName string) string {
	return "$LWS_LEADER_ADDRESS"
}

func (d *LWSMultinodeDeployer) GetNodeRank() (string, bool) {
	// This requires shell expansion for variable substitution
	return "$(LWS_WORKER_INDEX)", true
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
