package dynamo

import "fmt"

type LWSMultinodeDeployer struct {
	MultinodeDeployer
}

func (d *LWSMultinodeDeployer) GetLeaderHostname(serviceName string) string {
	return "$(LWS_LEADER_ADDRESS)"
}

func (d *LWSMultinodeDeployer) GetNodeRank() (string, bool) {
	// This requires shell expansion for variable substitution
	return "$(LWS_WORKER_INDEX)", true
}

func (d *LWSMultinodeDeployer) GetHostNames(serviceName string, numberOfNodes int32) []string {
	hostnames := make([]string, numberOfNodes)
	hostnames[0] = d.GetLeaderHostname(serviceName)
	for i := int32(1); i < numberOfNodes; i++ {
		hostnames[i] = fmt.Sprintf("$(LWS_WORKER_%d_ADDRESS)", i)
	}
	return hostnames
}
