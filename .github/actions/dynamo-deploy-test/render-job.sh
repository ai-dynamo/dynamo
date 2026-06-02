#!/usr/bin/env bash
# Render the in-cluster deploy-test Job manifest to stdout.
#
# The Job runs the deploy test suite from inside the host namespace so pytest
# reaches the vCluster API over its stable ClusterIP — no port-forward, no
# cross-cloud hop from the runner. It mounts the vCluster kubeconfig secret
# (vc-<name>), rewrites only the server URL to the in-cluster service, and runs
# the same pytest invocation the runner used to run, writing JUnit + allure to
# an emptyDir the runner copies out afterwards.
#
# Args (positional, all required):
#   1 JOB_NAME        DNS-1123 Job name
#   2 HOST_NAMESPACE  host namespace the Job runs in (and where vc-<name> lives)
#   3 TEST_IMAGE      full per-SHA test image reference
#   4 VCLUSTER_NAME   vCluster name (secret is vc-<name>)
#   5 VC_SERVER       in-cluster API server URL (https://<name>.<ns>.svc:443)
#
# Test inputs are passed via the environment (set by the calling step):
#   FRAMEWORK PROFILE IMAGE EXTRA_PYTEST_ARGS JUNIT_FILENAME
set -euo pipefail

JOB_NAME="$1"
HOST_NAMESPACE="$2"
TEST_IMAGE="$3"
VCLUSTER_NAME="$4"
VC_SERVER="$5"

# Assemble the pytest args the same way the old on-runner step did, so markers,
# verbosity, and artifact paths are unchanged.
PYTEST_ARGS=""
[ -n "${FRAMEWORK:-}" ] && PYTEST_ARGS+=" --framework=${FRAMEWORK}"
[ -n "${PROFILE:-}" ] && PYTEST_ARGS+=" --profile=${PROFILE}"
[ -n "${IMAGE:-}" ] && PYTEST_ARGS+=" --image=${IMAGE}"

# The test image is pure-Python (kr8s/kubernetes_asyncio) and reads KUBECONFIG.
# The mounted secret is read-only, so copy it to a writable path and rewrite
# only the server line — the embedded client cert authenticates over any path.
# Run pytest under `set +e` so a non-zero exit still lets the Job surface its
# status (and the JUnit XML is written regardless).
read -r -d '' POD_SCRIPT <<POD_EOF || true
set -euo pipefail
mkdir -p /results/allure-results /tmp/kube
cp /vc-kubeconfig/config /tmp/kube/config
sed -i -E "s#server: https?://[^[:space:]]+#server: ${VC_SERVER}#" /tmp/kube/config
export KUBECONFIG=/tmp/kube/config

cd /workspace
set +e
pytest tests/deploy/test_deploy.py \\
  --namespace=default \\
 ${PYTEST_ARGS} \\
 ${EXTRA_PYTEST_ARGS:-} \\
  -v -s \\
  --durations=10 \\
  --junitxml=/results/${JUNIT_FILENAME} \\
  --alluredir=/results/allure-results \\
  --log-cli-level=INFO
RC=\$?
echo "pytest exit code: \${RC}"
exit \${RC}
POD_EOF

cat <<JOB_EOF
apiVersion: batch/v1
kind: Job
metadata:
  name: ${JOB_NAME}
  namespace: ${HOST_NAMESPACE}
  labels:
    app.kubernetes.io/managed-by: dynamo-deploy-test
spec:
  backoffLimit: 0
  ttlSecondsAfterFinished: 1800
  template:
    metadata:
      labels:
        job-name: ${JOB_NAME}
    spec:
      restartPolicy: Never
      containers:
        - name: deploy-test
          image: ${TEST_IMAGE}
          imagePullPolicy: IfNotPresent
          command: ["bash", "-c"]
          args:
            - |
$(printf '%s\n' "${POD_SCRIPT}" | sed 's/^/                /')
          resources:
            requests:
              cpu: "500m"
              memory: "1Gi"
            limits:
              cpu: "2"
              memory: "4Gi"
          volumeMounts:
            - name: vc-kubeconfig
              mountPath: /vc-kubeconfig
              readOnly: true
            - name: results
              mountPath: /results
      volumes:
        - name: vc-kubeconfig
          secret:
            secretName: vc-${VCLUSTER_NAME}
        - name: results
          emptyDir: {}
JOB_EOF
