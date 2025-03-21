## CAI K8s Setup

### CAI System
The CAI API Server runs on the `dynamo-system` namespace. It consists
of the `dynamo-server` and `postgresql` pods. The API server pod
has an init container that waits for Postgres to start.

There are currently two urls that can be used for the API server.
- Authenticated URL: `https://cai-api.dev.llm.ngc.nvidia.com`
- Unauthenticated URL: `https://cai-api.dev.aire.nvidia.com`

### Dynamo NIM Deployments
All CRDs are created in the `dynamo` namespace. These are
reconciled by the NeMo operator, and image builder jobs and deployments
are created in this namespace.

The API spec allows users to
specify the namespace their Dynamo NIMs are deployed to. However,
the CLI and V2 APIs default currently to `dynamo`.

Note: currently every namespace needs a secret called `dynamo-deployment-shared-env` with content similar
to this:

```yaml
apiVersion: v1
data:
  BENTO_DEPLOYMENT_ALL_NAMESPACES: ZmFsc2U=
  BENTO_DEPLOYMENT_NAMESPACES: Y29tcG91bmRhaQ== # replace to match current namespace
  YATAI_DEPLOYMENT_NAMESPACE: Y29tcG91bmRhaQ== # replace to match current namespace
kind: Secret
metadata:
  name: dynamo-deployment-shared-env
  namespace: dynamo
type: Opaque
```
