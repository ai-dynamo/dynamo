---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: etcd Authentication
subtitle: Connect Dynamo components on Kubernetes to an etcd server that requires TLS and client authentication
---

This page shows how to connect the components of a DynamoGraphDeployment (DGD)
to an etcd server that encrypts traffic with TLS and accepts only authenticated
clients, with a Dynamo user that can read and write only Dynamo's own keys.

By default, the Dynamo Operator uses Kubernetes-native discovery and needs no
etcd. This page applies only when you install the platform chart with etcd
discovery (`dynamo-operator.discoveryBackend: etcd`).

## How Dynamo reads etcd credentials

The operator sets `ETCD_ENDPOINTS` in each component from the platform chart's
`dynamo-operator.etcdAddr`. Each component reads its credentials from these
environment variables:

| Variable | Contents |
|----------|----------|
| `ETCD_AUTH_CA` | PEM contents of the CA certificate that signed the etcd server certificate. |
| `ETCD_AUTH_CLIENT_CERT` | PEM contents of the client certificate Dynamo presents to etcd. |
| `ETCD_AUTH_CLIENT_KEY` | PEM contents of the private key for that client certificate. |
| `ETCD_AUTH_USERNAME` | etcd user name for password authentication. |
| `ETCD_AUTH_PASSWORD` | etcd password for password authentication. |

The client applies these rules:

- The three TLS variables hold **PEM contents, not file paths**. Load them from
  a Secret as environment variables, not as mounted files.
- Dynamo uses TLS only when **all three** TLS variables are set. A CA alone is
  ignored.
- When `ETCD_AUTH_USERNAME` and `ETCD_AUTH_PASSWORD` are both set, Dynamo uses
  password authentication and **ignores the TLS variables**. In that mode it
  trusts no CA, so it cannot connect to an `https://` endpoint, and over an
  `http://` endpoint the password crosses the network unencrypted.

For these reasons, authenticate Dynamo with a **client certificate** (mutual
TLS). etcd maps the certificate's common name (CN) to an etcd user and checks
that user's role on every request.

## Create the client certificate

Sign a client certificate with `CN=dynamo` using the CA that signed the etcd
server certificate. If you do not have a CA yet, create one first; the
[bundled etcd](#configure-etcd) setup signs its server certificate with the
same `ca.pem` and `ca-key.pem`.

```bash
# Certificate authority, only if you do not have one yet
openssl req -x509 -newkey rsa:4096 -nodes -days 365 \
  -keyout ca-key.pem -out ca.pem -subj "/CN=etcd-ca"

# Client certificate. The CN is the etcd user the certificate authenticates as.
openssl req -newkey rsa:2048 -nodes -keyout dynamo-key.pem -out dynamo.csr \
  -subj "/CN=dynamo"
openssl x509 -req -in dynamo.csr -CA ca.pem -CAkey ca-key.pem -CAcreateserial \
  -days 365 -out dynamo.pem -extfile <(printf "extendedKeyUsage=clientAuth")
chmod 600 ./*-key.pem
```

Keep `ca-key.pem` out of the cluster.

## Configure etcd

etcd must require TLS and a client certificate signed by your CA, turn on
authentication, and have a `dynamo` user whose role can use only keys under
`v1/`. Dynamo stores all of its discovery data, model cards, and planner state
under that prefix.

### Bundled etcd

The bundled etcd runs without authentication or TLS by default. To require TLS and a client certificate from every client, and to protect administration with a `root` password, first create two Secrets in the release namespace.

The server certificate must be signed by your CA, list the etcd service names as subject alternative names, and allow both server and client use, because the etcd chart's own scripts also present it as a client certificate. For a release named `dynamo-platform` in namespace `dynamo-system`:

```bash
openssl req -newkey rsa:2048 -nodes -keyout server-key.pem -out server.csr \
  -subj "/CN=etcd-server"
openssl x509 -req -in server.csr -CA ca.pem -CAkey ca-key.pem -CAcreateserial \
  -days 365 -out server.pem \
  -extfile <(printf "subjectAltName=DNS:dynamo-platform-etcd.dynamo-system.svc.cluster.local,DNS:*.dynamo-platform-etcd-headless.dynamo-system.svc.cluster.local,DNS:localhost,IP:127.0.0.1\nextendedKeyUsage=serverAuth,clientAuth")

chmod 600 server-key.pem

kubectl create secret generic dynamo-etcd-server-tls -n dynamo-system \
  --from-file=tls.crt=server.pem --from-file=tls.key=server-key.pem \
  --from-file=ca.crt=ca.pem
kubectl create secret generic dynamo-etcd-root -n dynamo-system \
  --from-literal=root-password="$(openssl rand -base64 24)"
```

Then install the chart with these values:

```yaml
global:
  etcd:
    install: true

dynamo-operator:
  discoveryBackend: etcd
  etcdAddr: "https://dynamo-platform-etcd.dynamo-system.svc.cluster.local:2379"

etcd:
  auth:
    rbac:
      create: true
      allowNoneAuthentication: false
      existingSecret: dynamo-etcd-root
      existingSecretPasswordKey: root-password
    client:
      secureTransport: true
      enableAuthentication: true
      existingSecret: dynamo-etcd-server-tls
      certFilename: tls.crt
      certKeyFilename: tls.key
      caFilename: ca.crt
    peer:
      secureTransport: true
      enableAuthentication: true
      existingSecret: dynamo-etcd-server-tls
      certFilename: tls.crt
      certKeyFilename: tls.key
      caFilename: ca.crt
```

`auth.client.enableAuthentication` makes etcd reject clients without a certificate signed by `ca.crt`, and `auth.rbac.create` creates the `root` user and turns on authentication. The `auth.peer` values apply the same certificate and checks to the peer port `2380`, which etcd members use to talk to each other. Keep them even with one replica: the peer port still listens, and client certificates and RBAC on port `2379` do not cover it. The server certificate already lists the headless service names the peer URLs use.

Create a `dynamo` user that can use only the `v1/` keys Dynamo stores. etcd maps the common name of a client certificate to the user, so give Dynamo components a client certificate with `CN=dynamo`, signed by the same CA:

```bash
kubectl exec -n dynamo-system dynamo-platform-etcd-0 -- bash -c '
  etcd_admin() {
    etcdctl --endpoints=https://127.0.0.1:2379 --cacert="$ETCD_TRUSTED_CA_FILE" \
      --cert="$ETCD_CERT_FILE" --key="$ETCD_KEY_FILE" \
      --user=root --password="$(cat "$ETCD_ROOT_PASSWORD_FILE")" "$@"
  }
  etcd_admin user add dynamo --no-password
  etcd_admin role add dynamo
  etcd_admin role grant-permission dynamo readwrite v1/ --prefix=true
  etcd_admin user grant-role dynamo dynamo
'
```

### External etcd

Start it with `--client-cert-auth` and
`--trusted-ca-file` set to your CA. Then, as the etcd `root` user, create the
`dynamo` user and role and turn on authentication if it is not on yet:

```bash
etcdctl user add dynamo --no-password
etcdctl role add dynamo
etcdctl role grant-permission dynamo readwrite v1/ --prefix=true
etcdctl user grant-role dynamo dynamo
```

Pass your usual `--endpoints`, `--cacert`, `--cert`, `--key`, and `--user`
options to each command.

## Point the operator at etcd

Select etcd discovery and set the etcd address to an `https://` URL whose host
name is in the etcd server certificate when you install the platform chart:

```yaml
dynamo-operator:
  discoveryBackend: etcd
  etcdAddr: "https://my-etcd.etcd.svc.cluster.local:2379"
```

## Store the client credentials in a Secret

Create the Secret in the namespace of your DGD. Each Secret key is the name of
the environment variable it fills.

```bash
kubectl create secret generic dynamo-etcd-client -n <namespace> \
  --from-file=ETCD_AUTH_CA=ca.pem \
  --from-file=ETCD_AUTH_CLIENT_CERT=dynamo.pem \
  --from-file=ETCD_AUTH_CLIENT_KEY=dynamo-key.pem
```

## Inject the Secret into each component

Add an `envFrom` reference to the `main` container of every component in the
DGD, including the frontend, workers, planner, and endpoint picker (EPP).

The DynamoGraphDeploymentRequest (DGDR) profiler Job also receives
`ETCD_ENDPOINTS`, but the operator does not inject these credentials into that
Job. DGDR profiling cannot connect to an authenticated etcd yet.

```yaml
apiVersion: nvidia.com/v1beta1
kind: DynamoGraphDeployment
metadata:
  name: my-graph
spec:
  backendFramework: vllm
  components:
    - name: frontend
      type: frontend
      replicas: 1
      podTemplate:
        spec:
          containers:
            - name: main
              image: nvcr.io/nvidia/ai-dynamo/dynamo-frontend:<version>
              envFrom:
                - secretRef:
                    name: dynamo-etcd-client
    - name: worker
      type: worker
      replicas: 1
      podTemplate:
        spec:
          containers:
            - name: main
              image: nvcr.io/nvidia/ai-dynamo/vllm-runtime:<version>
              envFrom:
                - secretRef:
                    name: dynamo-etcd-client
```

### Dynamo sidecar mode

In [Dynamo sidecar mode](../../reference/kubernetes-api/dynamo-component-deployment.mdx#dynamo-sidecar-mode),
the Dynamo runtime runs in the `runtime` init container and `main` runs the
engine. The operator does not copy the `envFrom` of `main` to `runtime`, so add
the same reference to the `runtime` init container of each such component.
Keep the reference on `main` as well.

This DGD fragment shows the references for a worker. Replace the image
placeholders and keep the launch commands and probes from your sidecar
deployment:

```yaml
spec:
  components:
    - name: worker
      type: worker
      podTemplate:
        spec:
          initContainers:
            - name: runtime
              image: <dynamo-sidecar-image>
              restartPolicy: Always
              envFrom:
                - secretRef:
                    name: dynamo-etcd-client
          containers:
            - name: main
              image: <engine-image>
              envFrom:
                - secretRef:
                    name: dynamo-etcd-client
```

If `spec.components[*].frontendSidecar` selects a co-located frontend, add the
same `envFrom` reference to that container too.

## Check the setup

Once the pods are running, list the keys Dynamo registered, as the etcd `root`
user:

```bash
etcdctl get --prefix v1/ --keys-only
```

The output includes keys under `v1/instances/` and `v1/mdc/`. A component
whose container lacks the `ETCD_AUTH_*` variables keeps logging
`etcd not reachable yet; retrying startup connection` and does not become
ready.
