use super::*;
use async_trait::async_trait;
use futures::StreamExt;
use k8s_openapi::api::discovery::v1::EndpointSlice;
use kube::{
    Client, ResourceExt,
    api::{Api, ListParams},
    runtime::{WatchStreamExt, watcher},
};
use parking_lot::Mutex;
use serde_json::Value;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use tokio::sync::broadcast;

/// Kubernetes-based implementation of ServiceDiscovery using EndpointSlices
#[derive(Clone)]
pub struct KubernetesServiceDiscovery {
    client: Client,
    event_senders: Arc<Mutex<HashMap<String, broadcast::Sender<InstanceEvent>>>>,
}

impl KubernetesServiceDiscovery {
    /// Create a new KubernetesServiceDiscovery
    pub async fn new() -> Result<Self> {
        let client = Client::try_default().await.map_err(|e| {
            DiscoveryError::RegistrationError(format!("Failed to create k8s client: {}", e))
        })?;

        Ok(Self {
            client,
            event_senders: Arc::new(Mutex::new(HashMap::new())),
        })
    }

    /// Get or create an event sender for a specific namespace/component pair
    fn get_or_create_sender(&self, key: String) -> broadcast::Sender<InstanceEvent> {
        let mut senders = self.event_senders.lock();
        senders
            .entry(key)
            .or_insert_with(|| {
                let (tx, _) = broadcast::channel(100);
                tx
            })
            .clone()
    }

    /// Build label selector for namespace and component
    fn build_label_selector(namespace: &str, component: &str) -> String {
        format!(
            "dynamo.namespace={},dynamo.component={}",
            namespace, component
        )
    }

    /// Watch key for a namespace/component pair
    fn watch_key(namespace: &str, component: &str) -> String {
        format!("{}:{}", namespace, component)
    }

    /// Extract pod names and their ready status from EndpointSlices
    fn extract_instances(endpoint_slices: &[EndpointSlice]) -> Vec<Instance> {
        let mut instances = Vec::new();

        for slice in endpoint_slices {
            for endpoint in &slice.endpoints {
                // Only include ready endpoints
                if let Some(conditions) = &endpoint.conditions
                    && conditions.ready != Some(true) {
                        continue;
                    }

                // Extract pod name from targetRef
                if let Some(target_ref) = &endpoint.target_ref
                    && target_ref.kind.as_deref() == Some("Pod")
                        && let Some(pod_name) = &target_ref.name {
                            instances.push(Instance::new(pod_name.clone(), Value::Null));
                        }
            }
        }

        instances
    }

    /// Start watching EndpointSlices for a specific namespace/component
    fn start_watch(&self, namespace: &str, component: &str, watch_namespace: &str) {
        let client = self.client.clone();
        let label_selector = Self::build_label_selector(namespace, component);
        let event_tx = self.get_or_create_sender(Self::watch_key(namespace, component));
        let watch_namespace = watch_namespace.to_string();
        let namespace_copy = namespace.to_string();
        let component_copy = component.to_string();

        tokio::spawn(async move {
            println!(
                "[K8s Discovery] Starting EndpointSlice watch: namespace={}, component={}, k8s_namespace={}, labels={}",
                namespace_copy, component_copy, watch_namespace, label_selector
            );
            let api: Api<EndpointSlice> = Api::namespaced(client, &watch_namespace);
            let watch_config = watcher::Config::default().labels(&label_selector);

            let mut stream = watcher(api, watch_config).applied_objects().boxed();

            // Track known ready instances across all slices
            // Key: pod name, Value: slice name (for tracking which slice it came from)
            let mut known_ready: HashMap<String, String> = HashMap::new();
            // Track current state of all slices
            let mut slice_instances: HashMap<String, HashSet<String>> = HashMap::new();

            while let Some(result) = stream.next().await {
                match result {
                    Ok(endpoint_slice) => {
                        let slice_name = endpoint_slice.name_any();

                        // Extract ready instances from this slice
                        let mut slice_ready = HashSet::new();

                        for endpoint in &endpoint_slice.endpoints {
                            // Check if endpoint is ready
                            let is_ready = endpoint
                                .conditions
                                .as_ref()
                                .and_then(|c| c.ready)
                                .unwrap_or(false);

                            if is_ready
                                && let Some(target_ref) = &endpoint.target_ref
                                    && target_ref.kind.as_deref() == Some("Pod")
                                        && let Some(pod_name) = &target_ref.name {
                                            slice_ready.insert(pod_name.clone());
                                        }
                        }

                        // Update slice_instances map
                        slice_instances.insert(slice_name.clone(), slice_ready);

                        // Rebuild the complete set of ready instances across all slices
                        // TODO: First pass, entire set of instances is rebuilt across Dynamo.
                        let mut current_ready: HashMap<String, String> = HashMap::new();
                        for (slice, pods) in &slice_instances {
                            for pod in pods {
                                current_ready.insert(pod.clone(), slice.clone());
                            }
                        }

                        // Find newly ready instances (Added events)
                        for pod_name in current_ready.keys() {
                            if !known_ready.contains_key(pod_name) {
                                println!(
                                    "[K8s Discovery] ✅ Instance ADDED: pod_name={}, slice={}",
                                    pod_name, slice_name
                                );
                                let instance = Instance::new(pod_name.clone(), Value::Null);
                                let _ = event_tx.send(InstanceEvent::Added(instance));
                            }
                        }

                        // Find no-longer-ready instances (Removed events)
                        for pod_name in known_ready.keys() {
                            if !current_ready.contains_key(pod_name) {
                                println!(
                                    "[K8s Discovery] ❌ Instance REMOVED: pod_name={}",
                                    pod_name
                                );
                                let _ = event_tx.send(InstanceEvent::Removed(pod_name.clone()));
                            }
                        }

                        known_ready = current_ready;
                    }
                    Err(e) => {
                        eprintln!("[K8s Discovery] ⚠️  Error watching EndpointSlices: {}", e);
                        // Continue watching despite errors
                    }
                }
            }
        });
    }
}

/// Handle for a Kubernetes-registered instance
pub struct KubernetesInstanceHandle {
    instance_id: String,
}

impl KubernetesInstanceHandle {
    /// Read pod name from environment variable
    fn read_pod_name() -> Result<String> {
        std::env::var("POD_NAME").map_err(|_| {
            DiscoveryError::RegistrationError("POD_NAME environment variable not set".to_string())
        })
    }
}

#[async_trait]
impl InstanceHandle for KubernetesInstanceHandle {
    fn instance_id(&self) -> &str {
        &self.instance_id
    }

    async fn set_metadata(&self, _metadata: Value) -> Result<()> {
        // Metadata changes are not supported in this implementation
        // The Kubernetes operator manages the pod metadata
        Ok(())
    }

    async fn set_ready(&self, _status: InstanceStatus) -> Result<()> {
        // Readiness is controlled by Kubernetes readiness probes
        // The operator and pod's readiness probe determine the actual status
        // This is a no-op as the pod's readiness is reflected in EndpointSlices
        Ok(())
    }
}

#[async_trait]
impl ServiceDiscovery for KubernetesServiceDiscovery {
    async fn register_instance(
        &self,
        namespace: &str,
        component: &str,
    ) -> Result<Box<dyn InstanceHandle>> {
        // Read pod name from environment
        let instance_id = KubernetesInstanceHandle::read_pod_name()?;

        println!(
            "[K8s Discovery] 📝 Registered instance: namespace={}, component={}, pod_name={}",
            namespace, component, instance_id
        );

        Ok(Box::new(KubernetesInstanceHandle { instance_id }))
    }

    async fn list_instances(&self, namespace: &str, component: &str) -> Result<Vec<Instance>> {
        // Query all EndpointSlices with matching labels
        let label_selector = Self::build_label_selector(namespace, component);

        // Get the current namespace from env var, or use "default"
        let current_namespace =
            std::env::var("POD_NAMESPACE").unwrap_or_else(|_| "default".to_string());

        let api: Api<EndpointSlice> = Api::namespaced(self.client.clone(), &current_namespace);
        let lp = ListParams::default().labels(&label_selector);

        let slices = api.list(&lp).await.map_err(|e| {
            DiscoveryError::MetadataError(format!("Failed to list EndpointSlices: {}", e))
        })?;

        let instances = Self::extract_instances(&slices.items);

        println!(
            "[K8s Discovery] 📋 Listed {} instances: namespace={}, component={}, pods={:?}",
            instances.len(),
            namespace,
            component,
            instances.iter().map(|i| &i.instance_id).collect::<Vec<_>>()
        );

        Ok(instances)
    }

    async fn watch(
        &self,
        namespace: &str,
        component: &str,
    ) -> Result<broadcast::Receiver<InstanceEvent>> {
        let key = Self::watch_key(namespace, component);

        // Get or create event sender for this namespace/component
        let event_tx = self.get_or_create_sender(key.clone());

        // Check if we need to start a watcher
        let needs_watch = {
            let senders = self.event_senders.lock();
            senders.get(&key).map(|tx| tx.receiver_count()).unwrap_or(0) == 0
        };

        if needs_watch {
            // Get the current namespace from env var, or use "default"
            let watch_namespace =
                std::env::var("POD_NAMESPACE").unwrap_or_else(|_| "default".to_string());

            println!(
                "[K8s Discovery] 👀 Starting new EndpointSlice watcher: namespace={}, component={}",
                namespace, component
            );

            self.start_watch(namespace, component, &watch_namespace);
        }

        Ok(event_tx.subscribe())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_build_label_selector() {
        let selector = KubernetesServiceDiscovery::build_label_selector("my-ns", "my-comp");
        assert_eq!(selector, "dynamo.namespace=my-ns,dynamo.component=my-comp");
    }

    #[test]
    fn test_watch_key() {
        let key = KubernetesServiceDiscovery::watch_key("ns1", "comp1");
        assert_eq!(key, "ns1:comp1");
    }

    /// Integration test for Kubernetes service discovery
    ///
    /// Prerequisites:
    /// 1. Have a Kubernetes cluster accessible via kubectl
    /// 2. Set KUBECONFIG environment variable to point to your kubeconfig file:
    ///    export KUBECONFIG=/path/to/your/kubeconfig
    /// 3. Set POD_NAMESPACE environment variable (defaults to "default"):
    ///    export POD_NAMESPACE=default
    /// 4. Create EndpointSlices in your cluster with the following labels:
    ///    dynamo.namespace=test
    ///    dynamo.component=worker
    ///
    /// Example EndpointSlice creation (see kubernetes/endpoint-slice-test.yaml):
    /// kubectl apply -f kubernetes/endpoint-slice-test.yaml
    ///
    /// Run this test with:
    /// cargo test --package dynamo-runtime test_kubernetes_discovery -- --ignored --nocapture
    #[tokio::test]
    #[ignore] // Ignore by default since it requires a running cluster
    async fn test_kubernetes_discovery_list_and_watch() {
        // Initialize tracing for debugging
        let _ = tracing_subscriber::fmt()
            .with_env_filter("debug")
            .try_init();

        // Create discovery client
        let discovery = KubernetesServiceDiscovery::new()
            .await
            .expect("Failed to create Kubernetes discovery client. Make sure KUBECONFIG is set.");

        println!("✓ Successfully connected to Kubernetes cluster");

        // Test list_instances
        println!("\n--- Testing list_instances ---");
        let instances = discovery
            .list_instances("test", "worker")
            .await
            .expect("Failed to list instances");

        println!("Found {} instances:", instances.len());
        for instance in &instances {
            println!("  - {}", instance.instance_id);
        }

        // Test watch
        println!("\n--- Testing watch ---");
        let mut watch = discovery
            .watch("test", "worker")
            .await
            .expect("Failed to create watch");

        println!("Watching for changes (will wait 30 seconds)...");
        println!("Now you can:");
        println!(
            "  1. Scale up/down your deployment: kubectl scale deployment <name> --replicas=N"
        );
        println!("  2. Delete pods: kubectl delete pod <pod-name>");
        println!("  3. Create new EndpointSlices with matching labels");

        // Wait for events for 30 seconds
        let timeout = Duration::from_secs(30);
        let start = tokio::time::Instant::now();

        let mut event_count = 0;
        while start.elapsed() < timeout {
            match tokio::time::timeout(Duration::from_secs(1), watch.recv()).await {
                Ok(Ok(event)) => {
                    event_count += 1;
                    match event {
                        InstanceEvent::Added(instance) => {
                            println!("  [ADDED] Instance: {}", instance.instance_id);
                        }
                        InstanceEvent::Removed(instance_id) => {
                            println!("  [REMOVED] Instance: {}", instance_id);
                        }
                    }
                }
                Ok(Err(e)) => {
                    println!("  Watch error: {:?}", e);
                    break;
                }
                Err(_) => {
                    // Timeout - no event received, continue waiting
                }
            }
        }

        println!("\n--- Test Summary ---");
        println!("Total events received: {}", event_count);

        // Re-list to see final state
        let final_instances = discovery
            .list_instances("test", "worker")
            .await
            .expect("Failed to list instances");

        println!("Final instance count: {}", final_instances.len());
        for instance in &final_instances {
            println!("  - {}", instance.instance_id);
        }
    }

    /// Quick smoke test to verify connection to cluster
    /// Run with: cargo test --package dynamo-runtime test_kubernetes_connection -- --ignored
    #[tokio::test]
    #[ignore]
    async fn test_kubernetes_connection() {
        let discovery = KubernetesServiceDiscovery::new()
            .await
            .expect("Failed to create Kubernetes discovery client");

        // Just try to list - even if there are no results, connection works
        let result = discovery.list_instances("test", "worker").await;

        match result {
            Ok(instances) => {
                println!(
                    "✓ Connected successfully! Found {} instances",
                    instances.len()
                );
                for instance in &instances {
                    println!("  - {}", instance.instance_id);
                }
            }
            Err(e) => {
                panic!("Failed to list instances: {:?}", e);
            }
        }
    }
}
