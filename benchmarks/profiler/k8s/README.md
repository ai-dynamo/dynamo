## Running profile sweep in k8s

```bash
NAMESPACE=sla-planner-2
k apply -f profile-sla.yaml -n $NAMESPACE

# Run port forward to access code-server from the browser
kubectl port-forward devcontainer-profile-sla 8080:8080 -n $NAMESPACE

# access code-server at http://localhost:8080 using password "password"
```