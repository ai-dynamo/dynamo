package dynamo

import (
	"fmt"

	commonconsts "github.com/ai-dynamo/dynamo/deploy/cloud/operator/internal/consts"
)

// generateGroveLeaderHostname generates the hostname for the leader pod in Grove multinode deployments
// The leader hostname follows the pattern: {GROVE_PCSG_NAME}-{GROVE_PCSG_INDEX}-{GroveRoleSuffixLeader}-0.{GROVE_HEADLESS_SERVICE}
func generateGroveLeaderHostname() string {
	return fmt.Sprintf("${GROVE_PCSG_NAME}-${GROVE_PCSG_INDEX}-%s-0.${GROVE_HEADLESS_SERVICE}", commonconsts.GroveRoleSuffixLeader)
}
