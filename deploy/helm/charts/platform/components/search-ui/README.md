# dynamo-search-ui

A browser UI for the DGDR v1beta2 search workflow. It watches
`DynamoGraphDeploymentRequest`, `DynamoGraphDeploymentRun` and
`DynamoGraphDeploymentCandidate` resources, keeps its own durable history of everything it
observed, and offers one write action: creating an independent `DynamoGraphDeployment`
from a Candidate.

The UI is read-only for Requests, Runs and Candidates. It runs as one Deployment with a
ClusterIP Service and needs no ingress:

```bash
kubectl -n dynamo-system port-forward svc/dynamo-search-ui 8080:8080
# open http://localhost:8080
```

## What it shows

- Requests grouped with their Runs, filtered by namespace.
- Run lifecycle: phase, conditions, effective AISimulate input, published-candidate and
  Pareto-front counts, and round and evaluation counters when the publisher reports them.
- Search-space exploration: a parallel-coordinates plot over the resolved evaluation
  parameters of every published point, including points the publisher later superseded.
- The Pareto front (objective scatter) for Pareto Runs or a score ranking for scalar Runs.
- A ranked candidate table with topology, GPUs, objectives and validity.
- Candidate detail with the deployable DGD as YAML, evaluation parameters and metrics.
- Container logs of the Run's Job (`sweeper` and `publisher`), followed live while the pod
  exists and retained afterwards.
- A timeline of observed Run status changes.

## History

Kubernetes keeps only the current bounded projection: the operator retains a few terminal
Runs and the publisher deletes Candidates that fall off the front. The UI records what it
observes so that history stays available:

| Key | Content |
|-----|---------|
| `runs/{ns}/{uid}/run.json` | latest copy of the Run, with first/last seen and deletion time |
| `runs/{ns}/{uid}/status/{ts}.json` | one sample per observed status change |
| `runs/{ns}/{uid}/candidates/{name}.json` | every Candidate ever observed |
| `runs/{ns}/{uid}/logs/{container}.log` | Job container output |

Backends (`history.backend`):

- `none`: in-memory only; everything is lost on restart.
- `local`: files under `history.local.path`; set `history.local.existingClaim` to a PVC to
  survive restarts.
- `azblob`: an Azure Blob Storage container, authenticated with
  [AKS Workload Identity](https://learn.microsoft.com/azure/aks/workload-identity-overview).

### Azure Blob with Workload Identity

Run once per cluster. It creates a StorageV2 account and container, a user-assigned
managed identity with `Storage Blob Data Contributor` on the account, and a federated
credential that trusts the `dynamo-search-ui` ServiceAccount in the release namespace.

```bash
RG=<resource-group> LOC=<region> CLUSTER=<aks-name> NS=dynamo-system
SA=<storage-account-name> CONTAINER=search-ui-history UMI=dynamo-search-ui

az storage account create -g $RG -n $SA -l $LOC --sku Standard_LRS --kind StorageV2 \
  --allow-blob-public-access false --min-tls-version TLS1_2
SA_ID=$(az storage account show -g $RG -n $SA --query id -o tsv)
az identity create -g $RG -n $UMI -l $LOC
PRINCIPAL=$(az identity show -g $RG -n $UMI --query principalId -o tsv)
az role assignment create --assignee-object-id $PRINCIPAL --assignee-principal-type ServicePrincipal \
  --role "Storage Blob Data Contributor" --scope $SA_ID
ISSUER=$(az aks show -g $RG -n $CLUSTER --query oidcIssuerProfile.issuerUrl -o tsv)
az identity federated-credential create -g $RG --identity-name $UMI -n dynamo-search-ui-$NS \
  --issuer "$ISSUER" --subject system:serviceaccount:$NS:dynamo-search-ui --audiences api://AzureADTokenExchange
KEY=$(az storage account keys list -g $RG -n $SA --query '[0].value' -o tsv)
az storage container create -n $CONTAINER --account-name $SA --account-key "$KEY"
CLIENT_ID=$(az identity show -g $RG -n $UMI --query clientId -o tsv)
```

Then install with:

```bash
helm upgrade --install dynamo-search-ui deploy/helm/charts/platform/components/search-ui -n $NS \
  --set image.repository=<registry>/dynamo-search-ui --set image.tag=<tag> \
  --set history.backend=azblob \
  --set history.azblob.accountUrl=https://$SA.blob.core.windows.net \
  --set history.azblob.container=$CONTAINER \
  --set history.azblob.workloadIdentityClientId=$CLIENT_ID
```

The cluster must have the OIDC issuer and Workload Identity enabled
(`az aks update --enable-oidc-issuer --enable-workload-identity`).

## Creating a DGD from a Candidate

`POST /api/v1/dgd` copies the Candidate's DGD spec (evaluation parameters are omitted), runs a
server-side dry run, and creates the DGD unless `dryRun` is set. A Candidate with
`Valid=False` is refused. Candidates written by publishers that predate validation carry
no `Valid` condition; for those the dry run is the only check. Creating a DGD does not
change traffic routing or touch the Candidate or its Run.

Set `dgdCreate.enabled=false` for a read-only installation; the create permission is then
not granted.

## Permissions

| Resource | Verbs | Purpose |
|----------|-------|---------|
| `dynamographdeploymentrequests`, `...runs`, `...candidates` | get, list, watch | the views |
| `pods` | get, list, watch | find the Run's Job pod |
| `pods/log` | get | stream container logs |
| `dynamographdeployments` | get, create | `POST /api/v1/dgd` (only with `dgdCreate.enabled`) |

With `watchNamespace` set the grants are a namespaced Role; otherwise a ClusterRole.
