use super::*;
use serde_json::json;

/// Example showing basic usage of the ServiceDiscovery interface
pub async fn basic_usage_example(discovery: &dyn ServiceDiscovery) -> Result<()> {
    // Register a new instance
    let handle = discovery.register_instance("myapp", "backend").await?;
    
    // Set transport metadata
    handle.set_metadata(json!({
        "transport": {
            "protocol": "nats",
            "subjects": {
                "endpoints": {
                    "generate": "myapp.backend.generate"
                }
            }
        },
        "endpoints": {
            "generate": {
                "type": "push",
                "input_type": "String",
                "output_type": "String"
            }
        }
    })).await?;
    
    // Mark instance as ready
    handle.set_ready(InstanceStatus::Ready).await?;
    
    Ok(())
}

/// Example showing how to discover and watch instances
pub async fn discovery_example(discovery: &dyn ServiceDiscovery) -> Result<()> {
    // List current instances
    let instances = discovery.list_instances("myapp", "backend").await?;
    for instance in instances {
        println!("Found instance: {}", instance.instance_id);
    }
    
    // Watch for changes
    let mut watch = discovery.watch("myapp", "backend").await?;
    
    tokio::spawn(async move {
        while let Ok(event) = watch.recv().await {
            match event {
                InstanceEvent::Added(instance) => {
                    println!("New instance: {}", instance.instance_id);
                }
                InstanceEvent::Removed(id) => {
                    println!("Instance removed: {}", id);
                }
            }
        }
    });
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::discovery::mock::MockServiceDiscovery;

    #[tokio::test]
    async fn test_examples() {
        let discovery = MockServiceDiscovery::new();
        
        // Test basic usage
        basic_usage_example(&discovery).await.unwrap();
        
        // Test discovery
        discovery_example(&discovery).await.unwrap();
    }
}
