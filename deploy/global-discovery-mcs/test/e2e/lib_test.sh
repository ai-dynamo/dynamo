#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Self-test of the presence and absence helpers of lib.sh (kc_names,
# object_present, object_absent) against a stub API server on localhost, so
# that it needs no cluster: an existing object, an absent one, a server that
# never answers (a timeout), a kubeconfig whose requests are forbidden and a
# server error. An absence assertion must hold only when the API server
# answered that the object does not exist. Run it with test/e2e/lib_test.sh.

# shellcheck source-path=SCRIPTDIR
# shellcheck disable=SC2034  # E2E_* settings are read by lib.sh
set -euo pipefail

SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/e2e-lib-test.XXXXXX")"
SERVER_PID=""
HOLE_PID=""
cleanup() {
  local pid
  for pid in ${SERVER_PID} ${HOLE_PID}; do
    kill "${pid}" 2>/dev/null || true
    wait "${pid}" 2>/dev/null || true
  done
  rm -rf "${WORK}"
}
trap cleanup EXIT

command -v kubectl >/dev/null 2>&1 || { echo "lib_test: kubectl is required" >&2; exit 2; }
command -v python3 >/dev/null 2>&1 || { echo "lib_test: python3 is required" >&2; exit 2; }

# The stub: discovery of core/v1 configmaps, one ConfigMap (present), 404 for
# any other, 500 for "broken", a registry behind the Service proxy and 403 for every non-discovery request made as
# the impersonated user "denied" (client-go sends no bearer token over HTTP). A second socket accepts and never answers.
cat >"${WORK}/stub.py" <<'PY'
import json, socketserver, sys, threading
from http.server import BaseHTTPRequestHandler, HTTPServer

def status(code, reason, message):
    return {"kind": "Status", "apiVersion": "v1", "metadata": {}, "status": "Failure",
            "message": message, "reason": reason, "code": code}

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def reply(self, code, body):
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/api":
            return self.reply(200, {"kind": "APIVersions", "versions": ["v1"]})
        if path == "/apis":
            return self.reply(200, {"kind": "APIGroupList", "apiVersion": "v1", "groups": []})
        if path == "/api/v1":
            return self.reply(200, {"kind": "APIResourceList", "groupVersion": "v1", "resources": [
                {"name": "configmaps", "singularName": "configmap", "namespaced": True,
                 "kind": "ConfigMap", "verbs": ["get", "list"], "shortNames": ["cm"]}]})
        if self.headers.get("Impersonate-User") == "denied":
            return self.reply(403, status(403, "Forbidden", "configmaps is forbidden: User \"denied\" cannot get resource \"configmaps\""))
        proxy = "/api/v1/namespaces/mcs-system/services/mcs-registry:http/proxy/v1/clustersets/e2e"
        if path.startswith("/api/v1/namespaces/mcs-system/services/mcs-registry:http/proxy/"):
            if self.headers.get("Impersonate-User") == "noservice":
                return self.reply(404, status(404, "NotFound", "services \"mcs-registry\" not found"))
            if path == proxy + "/clusters":
                return self.reply(200, {"clusters": []})
            if path == proxy + "/imports/ns/gone":
                return self.reply(404, status(404, "NotFound", "the server could not find the requested resource"))
            return self.reply(200, {})
        base = "/api/v1/namespaces/ns/configmaps"
        cm = {"kind": "ConfigMap", "apiVersion": "v1", "metadata": {"name": "present", "namespace": "ns"}}
        if path == base + "/present":
            return self.reply(200, cm)
        if path == base + "/broken":
            return self.reply(500, status(500, "InternalError", "etcdserver: request timed out"))
        if path.startswith(base + "/"):
            return self.reply(404, status(404, "NotFound", "configmaps \"%s\" not found" % path.rsplit("/", 1)[1]))
        if path == base:
            items = [cm] if "labelSelector=app%3Dpresent" in self.path else []
            return self.reply(200, {"kind": "ConfigMapList", "apiVersion": "v1", "metadata": {}, "items": items})
        return self.reply(404, status(404, "NotFound", "not found"))

class Stub(HTTPServer):
    def server_bind(self):
        # HTTPServer.server_bind resolves the host name, which can take seconds.
        socketserver.TCPServer.server_bind(self)
        self.server_name, self.server_port = "localhost", self.server_address[1]

class Hole(socketserver.BaseRequestHandler):
    def handle(self):
        threading.Event().wait()

class ThreadedHole(socketserver.ThreadingMixIn, socketserver.TCPServer):
    daemon_threads = True
    allow_reuse_address = True

which, portfile = sys.argv[1], sys.argv[2]
if which == "stub":
    srv = Stub(("127.0.0.1", 0), Handler)
else:
    srv = ThreadedHole(("127.0.0.1", 0), Hole)
open(portfile, "w").write(str(srv.server_address[1]))
srv.serve_forever()
PY

# start NAME: runs the server NAME in the background and waits for its port
# file, ${WORK}/NAME.port.
start() {
  python3 -I "${WORK}/stub.py" "$1" "${WORK}/$1.port" >/dev/null 2>&1 &
  if [ "$1" = stub ]; then SERVER_PID=$!; else HOLE_PID=$!; fi
  for _ in $(seq 1 50); do
    [ -s "${WORK}/$1.port" ] && return 0
    sleep 0.1
  done
  echo "lib_test: the $1 server did not start" >&2
  exit 2
}

start stub
start hole
STUB_PORT="$(cat "${WORK}/stub.port")"
HOLE_PORT="$(cat "${WORK}/hole.port")"

# One kubeconfig per cluster key, named as lib.sh expects them.
export E2E_PREFIX="selftest" E2E_WORK_DIR="${WORK}/work" E2E_PHASES="3" KUBECACHEDIR="${WORK}/cache"
mkdir -p "${E2E_WORK_DIR}/kubeconfig"
write_kubeconfig() {  # write_kubeconfig KEY PORT [USER]: the user is impersonated
  cat >"${E2E_WORK_DIR}/kubeconfig/${E2E_PREFIX}-$1" <<YAML
apiVersion: v1
kind: Config
clusters:
- name: kind-${E2E_PREFIX}-$1
  cluster: {server: "http://127.0.0.1:$2"}
users:
- name: u
  user: {as: "${3:-ok}"}
contexts:
- name: kind-${E2E_PREFIX}-$1
  context: {cluster: kind-${E2E_PREFIX}-$1, user: u}
current-context: kind-${E2E_PREFIX}-$1
YAML
}
write_kubeconfig ok "${STUB_PORT}"
write_kubeconfig denied "${STUB_PORT}" denied
write_kubeconfig hole "${HOLE_PORT}"
write_kubeconfig hub "${STUB_PORT}"

# shellcheck source=lib.sh
source "${SELF_DIR}/lib.sh"
# shellcheck source=checks.sh
source "${SELF_DIR}/checks.sh"
E2E_REQUEST_TIMEOUT="3s"

FAILED=0
PASSED=0
check() {  # check NAME EXPECTED_STATUS COMMAND ARGS... : also compares LAST_OBSERVED with $OBSERVED_RE
  local name="$1" want="$2" rc=0
  shift 2
  LAST_OBSERVED=""
  "$@" || rc=$?
  if [ "${rc}" != "${want}" ]; then
    printf 'FAIL  %s: status %s, want %s (observed: %s)\n' "${name}" "${rc}" "${want}" "${LAST_OBSERVED}"
    FAILED=$((FAILED + 1))
    return 0
  fi
  if [ -n "${OBSERVED_RE:-}" ] && ! grep -Eq -- "${OBSERVED_RE}" <<<"${LAST_OBSERVED}"; then
    printf 'FAIL  %s: observed "%s", want /%s/\n' "${name}" "${LAST_OBSERVED}" "${OBSERVED_RE}"
    FAILED=$((FAILED + 1))
    return 0
  fi
  printf 'PASS  %s\n' "${name}"
  PASSED=$((PASSED + 1))
}

OBSERVED_RE="still exists"
check "an existing object is not absent" 1 object_absent ok ns configmap/present
OBSERVED_RE=""
check "an existing object is present" 0 object_present ok ns configmap/present
OBSERVED_RE="absent in ok/ns"
check "an absent object is absent" 0 object_absent ok ns configmap/gone
check "an absent object is not present" 1 object_present ok ns configmap/gone
OBSERVED_RE=""
check "an empty list is an empty KC_NAMES" 0 kc_names ok -n ns configmaps -l app=none
[ -z "${KC_NAMES}" ] || { echo "FAIL  an empty list left KC_NAMES=${KC_NAMES}"; FAILED=$((FAILED + 1)); }
check "a non-empty list returns the names" 0 kc_names ok -n ns configmaps -l app=present
[ "${KC_NAMES}" = "configmap/present" ] || { echo "FAIL  list names '${KC_NAMES}'"; FAILED=$((FAILED + 1)); }

# A failure of the call is never absence: the assertion fails and says why.
OBSERVED_RE="failed: .*(forbidden|Forbidden)"
check "a forbidden request does not satisfy object_absent" 1 object_absent denied ns configmap/gone
check "a forbidden request does not satisfy object_present" 1 object_present denied ns configmap/present
check "a forbidden list does not return an empty list" 1 kc_names denied -n ns configmaps -l app=none
OBSERVED_RE="failed: .*(InternalError|timed out)"
check "a server error does not satisfy object_absent" 1 object_absent ok ns configmap/broken
OBSERVED_RE="failed: "
E2E_REQUEST_TIMEOUT="1ms"
check "a timeout against a blackholed server does not satisfy object_absent" 1 object_absent hole ns configmap/gone
check "a timeout against a blackholed server does not return an empty list" 1 kc_names hole -n ns configmaps -l app=none
E2E_REQUEST_TIMEOUT="3s"

# registry_import_absent: the registry's own NotFound, with the registry
# serving; not the API server's NotFound for a missing registry Service.
OBSERVED_RE="is NotFound"
check "the registry's NotFound satisfies registry_import_absent" 0 registry_import_absent ns gone
OBSERVED_RE=""
check "an import that exists is not absent" 1 registry_import_absent ns present
write_kubeconfig hub "${STUB_PORT}" noservice
OBSERVED_RE="not found"
check "a missing registry Service does not satisfy registry_import_absent" 1 registry_import_absent ns gone
write_kubeconfig hub "${HOLE_PORT}"
E2E_REQUEST_TIMEOUT="1ms"
OBSERVED_RE=""
check "a timeout does not satisfy registry_import_absent" 1 registry_import_absent ns gone
E2E_REQUEST_TIMEOUT="3s"

# Through expect: the poll keeps failing with the error text, never passes.
OBSERVED_RE=""
RESULTS=()
E2E_TIMEOUT=3
expect "deleted object" object_absent denied ns configmap/gone >/dev/null || true
if [ "${FAIL_COUNT}" = 1 ] && grep -q 'last observed: kubectl get .*failed' <<<"${RESULTS[0]:-}"; then
  printf 'PASS  expect fails with the error text on a forbidden kubeconfig\n'
  PASSED=$((PASSED + 1))
else
  printf 'FAIL  expect on a forbidden kubeconfig: pass=%s fail=%s results=%s\n' "${PASS_COUNT}" "${FAIL_COUNT}" "${RESULTS[0]:-}"
  FAILED=$((FAILED + 1))
fi

printf '%d passed, %d failed\n' "${PASSED}" "${FAILED}"
[ "${FAILED}" = 0 ]
