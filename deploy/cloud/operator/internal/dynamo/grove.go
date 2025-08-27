package dynamo

import (
	"fmt"

	commonconsts "github.com/ai-dynamo/dynamo/deploy/cloud/operator/internal/consts"
)

type GroveMultinodeDeployer struct {
	MultinodeDeployer
}

func (d *GroveMultinodeDeployer) GetLeaderHostname(serviceName string) string {
	return fmt.Sprintf("${GROVE_PCSG_NAME}-${GROVE_PCSG_INDEX}-%s-%s-0.${GROVE_HEADLESS_SERVICE}", serviceName, commonconsts.GroveRoleSuffixLeader)
}

func (d *GroveMultinodeDeployer) GetNodeRank() string {
	return "$((GROVE_PCLQ_POD_INDEX + 1))"
}

func (d *GroveMultinodeDeployer) GetHostNames(serviceName string, numberOfNodes int32) []string {
	hostnames := make([]string, 0, numberOfNodes)
	leaderHostname := d.GetLeaderHostname(serviceName)
	hostnames = append(hostnames, leaderHostname)
	// Add worker hostnames
	for i := int32(0); i < numberOfNodes-1; i++ {
		workerHostname := fmt.Sprintf("${GROVE_PCSG_NAME}-${GROVE_PCSG_INDEX}-%s-%s-%d.${GROVE_HEADLESS_SERVICE}",
			serviceName, commonconsts.GroveRoleSuffixWorker, i)
		hostnames = append(hostnames, workerHostname)
	}
	return hostnames
}
