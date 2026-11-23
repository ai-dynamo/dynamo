---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

## 1. Create the client certificate

Sign a client certificate with `CN=dynamo` using the CA that signed the etcd
server certificate. If you do not have a CA yet, create one first; the
[bundled etcd](#2-configure-etcd) setup signs its server certificate with the
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

## 2. Configure etcd

etcd must require TLS and a client certificate signed by your CA, turn on
authentication, and have a `dynamo` user whose role can use only keys under
`v1/`. Dynamo stores all of its discovery data, model cards, and planner state
under that prefix.

- **The etcd bundled with the platform chart.** Follow
  [Enable authentication and TLS on the bundled etcd](https://github.com/ai-dynamo/dynamo/blob/main/deploy/helm/charts/platform/README.md#enable-authentication-and-tls-on-the-bundled-etcd)
  in the chart README. It covers the server certificate, the chart values, and
  the `dynamo` user.
- **An etcd you run yourself.** Start it with `--client-cert-auth` and
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

## 3. Point the operator at etcd

Select etcd discovery and set the etcd address to an `https://` URL whose host
name is in the etcd server certificate when you install the platform chart:

```yaml
dynamo-operator:
  discoveryBackend: etcd
  etcdAddr: "https://my-etcd.etcd.svc.cluster.local:2379"
```

## 4. Store the client credentials in a Secret

Create the Secret in the namespace of your DGD. Each Secret key is the name of
the environment variable it fills.

```bash
kubectl create secret generic dynamo-etcd-client -n <namespace> \
  --from-file=ETCD_AUTH_CA=ca.pem \
  --from-file=ETCD_AUTH_CLIENT_CERT=dynamo.pem \
  --from-file=ETCD_AUTH_CLIENT_KEY=dynamo-key.pem
```

## 5. Inject the Secret into each component

Add an `envFrom` reference to the `main` container of every component in the
DGD: the frontend, every worker, and the planner if you run one.

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
