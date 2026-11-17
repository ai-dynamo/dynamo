---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Trace Sandboxed Agents
subtitle: Connect Pi tool events to Dynamo request capture through Substrate egress
---

Capture Pi tool execution alongside Dynamo inference, then export the records
to Perfetto. This procedure extends a DGD from the selected model recipe. It
requires a frontend build with native request tracing and the tool-event receiver.
See [Agent tracing](agent-tracing.md)
for the record format and available settings.

## Prerequisites

- Complete the first trial in [Sandboxed Coding Agents](harbor-substrate.mdx).
- Use a Dynamo frontend you control. An external inference API cannot provide
  this server-side capture.
- Stop active Harbor jobs before updating the frontend.
- Run commands from `examples/agents/harbor-substrate` in your Dynamo checkout.

<Steps>
<Step title="Prepare the tool-event transport" id="prepare-the-transport">

Pi publishes tool events over ZMQ. Substrate's TLS-passthrough egress carries
that connection through a local stunnel client to a sidecar in the frontend Pod.
The native receiver listens on loopback. HTTP model traffic uses its separate
endpoint and egress rule.

```mermaid
flowchart LR
    P[Pi in actor] --> C[Local TLS client]
    C -->|TLS through Substrate egress| S[Frontend TLS sidecar]
    S --> F[Dynamo tool-event receiver]
    P -->|Model requests| F
    F --> J[Request and tool records]
    J --> X[Perfetto exporter]
```

Set `MODEL_NAMESPACE` and `DGD_NAME` to the namespace and name from your model
recipe. These shell variables select resources for CLI commands. Work from
the example directory with the intended Kubernetes context already selected.

Edit [tool-trace-service.yaml](https://github.com/ai-dynamo/dynamo/blob/main/examples/agents/harbor-substrate/tool-trace-service.yaml) in place:

- Set both `metadata.namespace` fields to the model namespace (`inference` in
  the example).
- Set `spec.selector.nvidia.com/dynamo-namespace` to the frontend Pod's label
  value, normally `<namespace>-<DGD name>`. Replace `your-model-deployment` in
  the supplied selector with your DGD name. Keep the `Frontend` component selector.

In [stunnel-client.conf](https://github.com/ai-dynamo/dynamo/blob/main/examples/agents/harbor-substrate/stunnel-client.conf), keep port 20391 and set the hostname in `connect`,
`sni`, and `checkHost` to `harbor-tool-events.<namespace>.svc.cluster.local`.
The supplied hostname uses `inference`. It must match the Service namespace
and the certificate's DNS name below.

Create the certificate and apply the edited Service and ConfigMap:

```bash
export MODEL_NAMESPACE=inference
export DGD_NAME=your-model-deployment
export TRACE_HOST="harbor-tool-events.$MODEL_NAMESPACE.svc.cluster.local"
mkdir -p .trace/private .trace/public
chmod 700 .trace/private
openssl req -x509 -newkey rsa:2048 -nodes -days 7 \
  -keyout .trace/private/tls.key -out .trace/public/tls.crt \
  -subj "/CN=$TRACE_HOST" -addext "subjectAltName=DNS:$TRACE_HOST"
chmod 600 .trace/private/tls.key
kubectl -n "$MODEL_NAMESPACE" create secret tls harbor-tool-trace-tls \
  --cert=.trace/public/tls.crt --key=.trace/private/tls.key
kubectl apply -f tool-trace-service.yaml
```

Only the public certificate and client configuration go into actors. The
private key stays in the frontend Secret. The client verifies the certificate
chain and hostname.

<Note>
This hand-made certificate is a tutorial shortcut, not a production practice.
It expires in seven days and nothing rotates it. For a lasting installation,
issue the sidecar's certificate with cert-manager or with Substrate's
`servicedns.podcert.ate.dev/identity` signer, and give actors the CA bundle
instead of the leaf certificate. The TLS hop exists only because Pi publishes
tool events over ZMQ, which the egress gateway cannot authorize. An HTTP
transport would need neither stunnel nor the certificate.
</Note>

</Step>
<Step title="Enable frontend capture" id="enable-frontend-capture">

Set `DEPLOYMENT_YAML` to the manifest you apply for the model recipe. Use an
absolute path because commands here run from `examples/agents/harbor-substrate`.
Save the current rendered manifest before adding capture:

```bash
export DEPLOYMENT_YAML=/path/to/your/model-deployment.yaml
mkdir -p .build
cp "$DEPLOYMENT_YAML" .build/frontend-before.yaml
```

Choose where to edit according to the recipe:

- **Plain manifest:** edit the file named by `DEPLOYMENT_YAML` in place.
- **Kustomize:** edit the source DGD or your cluster overlay, then rerun the
  recipe's render command from the repository root to regenerate
  `DEPLOYMENT_YAML`. Return with `cd examples/agents/harbor-substrate` before
  continuing here. Editing only a generated `/tmp` manifest loses capture on
  the next render. Keep a copy of any source files you change for cleanup.

In the `Frontend` entry under `spec.components`, merge the following fields
into `podTemplate.spec`. Append to the existing
`main` container's `env` and `volumeMounts` lists, add the sidecar to
`containers`, and add the volumes. Preserve the recipe's images, arguments,
other environment variables, worker components, and accompanying resources.
This fragment is not a standalone DGD:

```yaml
containers:
  - name: main
    env:
      - name: DYN_REQUEST_TRACE
        value: "1"
      - name: DYN_REQUEST_TRACE_TOOL_EVENTS_ZMQ_ENDPOINT
        value: tcp://127.0.0.1:20390
      - name: DYN_REQUEST_TRACE_OUTPUT_PATH
        value: /traces/request-trace
      - name: DYN_REQUEST_TRACE_FILE_FORMAT
        value: jsonl
    volumeMounts:
      - name: harbor-traces
        mountPath: /traces
  - name: tool-events-tls
    image: alpine:3.22
    command:
      - sh
      - -ec
      - apk add --no-cache stunnel; exec stunnel /config/stunnel.conf
    ports:
      - name: tls-zmq
        containerPort: 20391
    volumeMounts:
      - name: tool-trace-tls
        mountPath: /tls
        readOnly: true
      - name: tool-trace-config
        mountPath: /config
        readOnly: true
    resources:
      requests:
        cpu: 50m
        memory: 32Mi
      limits:
        cpu: "1"
        memory: 128Mi
volumes:
  - name: harbor-traces
    emptyDir:
      sizeLimit: 2Gi
  - name: tool-trace-tls
    secret:
      secretName: harbor-tool-trace-tls
  - name: tool-trace-config
    configMap:
      name: harbor-tool-trace-tls
```

The main container writes native JSONL records under `/traces`. The sidecar
terminates TLS on port 20391 and forwards events to the receiver on loopback
port 20390. It installs stunnel from Alpine repositories at startup. Use a
prebuilt approved image if Pod egress does not permit that.

`alpine:3.22` includes x86-64 and ARM64 images. Kubernetes selects the variant
for the frontend's node; keep this reference for either architecture. A custom
sidecar image must match the frontend node, independently of the actor's
architecture.

If the frontend already has trace settings or these volume names, merge the
configuration without duplicate entries. For Kustomize, regenerate the manifest
first. Inspect the rendered diff, then apply it:

```bash
diff -u .build/frontend-before.yaml "$DEPLOYMENT_YAML"
kubectl -n "$MODEL_NAMESPACE" apply -f "$DEPLOYMENT_YAML"
kubectl -n "$MODEL_NAMESPACE" wait \
  --for=condition=Ready "dgd/$DGD_NAME" --timeout=1800s
```

This example uses one frontend replica and an ephemeral 2-GiB trace volume.
For more replicas, collect every frontend's trace. For longer runs, use a
persistent volume or your normal trace export system. Copy all segments before
replacing or deleting the frontend Pod.

</Step>
<Step title="Connect actor tool events" id="connect-actor-tool-events">

Append a rule to `rules` in `egress.yaml`. Replace the example hostname
with the value of `TRACE_HOST` from the certificate step:

```yaml
  - tlsPassthrough:
      hostnames:
        - harbor-tool-events.inference.svc.cluster.local
      ports:
        numbers: [20391]
```

In `harbor-job.yaml`, append these entries to the existing
`agents[0].env` mapping:

```yaml
DYN_REQUEST_TRACE: "1"
DYN_REQUEST_TRACE_TOOL_EVENTS_ZMQ_ENDPOINT: tcp://127.0.0.1:20390
DYN_TOOL_TRACE_TLS_CONFIG: /opt/tool-trace/stunnel-client.conf
```

Add this mapping under `environment.kwargs`:

```yaml
uploads:
  /etc/harbor-tool-trace: /opt/tool-trace
```

The upload supplies the public certificate and stunnel configuration. The Pi
adapter starts stunnel, and the provider publishes tool events to its loopback
listener. These settings require the Dynamo frontend capture configured above.
Omit them when using an external endpoint.

Create the ConfigMap that the runner mounts at `/etc/harbor-tool-trace`.
Include only the public certificate and client configuration; keep the server's
private key in its Secret:

```bash
kubectl -n ate-env create configmap harbor-tool-trace \
  --from-file=tls.crt=.trace/public/tls.crt \
  --from-file=stunnel-client.conf \
  --dry-run=client -o yaml | kubectl apply -f -
```

</Step>
</Steps>

## Verify configuration and run a trial

Require the updated DGD to be Ready, then confirm that the tool-event Service
selects a frontend endpoint:

```bash
kubectl -n "$MODEL_NAMESPACE" get endpointslices \
  -l kubernetes.io/service-name=harbor-tool-events -o yaml
```

The endpoint must identify the frontend Pod and port 20391. Inspect the
`tool-events-tls` sidecar logs if the endpoint is missing or the Pod is not ready.

Continue with [Run an observed trial](harbor-substrate.mdx#run-an-observed-trial).
That step applies the updated egress and agent configuration before starting
Harbor. The tutorial then covers trace export and Perfetto inspection.

A successful model request does not prove that tool-event delivery works.
After the trial, require native tool records with Pi's `toolCallId` values
and the same session ID as the model requests. If those records are missing,
check the Pi log, TLS sidecar, and frontend logs before running a batch.

For record fields and the difference between inferred waits and measured
execution, see [Agent Tracing](agent-tracing.md#choose-the-tool-detail-you-need)
and [Request Trace Reference](../../reference/observability/request-traces.mdx).
