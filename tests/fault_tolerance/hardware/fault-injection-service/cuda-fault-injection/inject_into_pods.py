#!/usr/bin/env python3
"""
Inject CUDA Fault Library into Running vLLM Pods

This script:
1. Copies fake_cuda_xid79.so into target pods
2. Restarts pods with LD_PRELOAD environment variable set
3. Monitors pod crash/restart behavior

Usage:
    # Inject into all vLLM worker pods
    python inject_into_pods.py --deployment vllm-v1-disagg-router --namespace dynamo-oviya
    
    # Inject into specific node
    python inject_into_pods.py --deployment vllm-v1-disagg-router --node aks-a100a-36888584-vmss000003
    
    # Remove injection (restore normal operation)
    python inject_into_pods.py --deployment vllm-v1-disagg-router --remove
"""

import argparse
import os
import sys
import time
import subprocess
from pathlib import Path
from kubernetes import client, config
from kubernetes.client.rest import ApiException


def get_library_path():
    """Get path to the compiled library."""
    script_dir = Path(__file__).parent
    lib_path = script_dir / "fake_cuda_xid79.so"
    
    if not lib_path.exists():
        print(f"❌ Library not found: {lib_path}")
        print("   Run: make")
        sys.exit(1)
    
    return lib_path


def copy_library_to_pod(pod_name, namespace, lib_path, target_path="/tmp/fake_cuda_xid79.so"):
    """Copy library to pod using kubectl cp."""
    print(f"[→] Copying library to {pod_name}...")
    
    cmd = [
        "kubectl", "cp",
        str(lib_path),
        f"{namespace}/{pod_name}:{target_path}"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"    ⚠ Failed to copy: {result.stderr}")
        return False
    
    print(f"    ✓ Copied to {target_path}")
    return True


def create_cuda_fault_configmap(namespace, lib_path=None):
    """Create ConfigMap with CUDA fault injection library source code."""
    import base64
    
    if lib_path is None:
        lib_path = get_library_path()
    
    print(f"[→] Creating ConfigMap with CUDA fault library source...")
    
    try:
        # Get the source file path
        lib_dir = os.path.dirname(lib_path)
        source_file = os.path.join(lib_dir, "fake_cuda_xid79.c")
        
        if not os.path.exists(source_file):
            print(f"❌ Source file not found: {source_file}")
            return False
        
        # Read source code
        with open(source_file, 'r') as f:
            source_code = f.read()
        
        print(f"    → Source code size: {len(source_code)} bytes")
        
        # Create ConfigMap with source code
        # We'll compile it in the init container to ensure Linux compatibility
        core_api = client.CoreV1Api()
        
        configmap = client.V1ConfigMap(
            metadata=client.V1ObjectMeta(name="cuda-fault-injection-lib"),
            data={"fake_cuda_xid79.c": source_code}
        )
        
        # Delete existing if present
        try:
            core_api.delete_namespaced_config_map("cuda-fault-injection-lib", namespace)
            print(f"    → Deleted existing ConfigMap")
            time.sleep(2)
        except:
            pass
        
        # Create new
        core_api.create_namespaced_config_map(namespace, configmap)
        print(f"[✓] ConfigMap created: cuda-fault-injection-lib")
        print(f"    → Source code will be compiled in init container (Linux-compatible)")
        return True
        
    except Exception as e:
        print(f"❌ Failed to create ConfigMap: {e}")
        import traceback
        traceback.print_exc()
        return False


def delete_cuda_fault_configmap(namespace):
    """Delete the CUDA fault injection ConfigMap."""
    try:
        core_api = client.CoreV1Api()
        core_api.delete_namespaced_config_map("cuda-fault-injection-lib", namespace)
        print(f"[✓] ConfigMap deleted: cuda-fault-injection-lib")
        return True
    except:
        return False


def patch_deployment_env(deployment_name, namespace, enable=True, use_configmap=True, target_node=None):
    """Patch deployment to add/remove LD_PRELOAD environment variable.
    
    Args:
        deployment_name: Name of the deployment
        namespace: Kubernetes namespace
        enable: Whether to enable (True) or disable (False) CUDA fault injection
        use_configmap: Whether to use ConfigMap for library distribution
        target_node: If provided, adds node affinity to pin pods to this node
                    (simulates real XID 79 where pods crash on the faulty node)
    """
    custom_api = client.CustomObjectsApi()
    apps_api = client.AppsV1Api()
    
    # Try DynamoGraphDeployment first
    is_dgd = False
    dgd = None
    try:
        print(f"    → Attempting to get DynamoGraphDeployment: {deployment_name}")
        dgd = custom_api.get_namespaced_custom_object(
            group="nvidia.com",
            version="v1alpha1",
            namespace=namespace,
            plural="dynamographdeployments",
            name=deployment_name
        )
        is_dgd = True
        print(f"    ✓ Found DynamoGraphDeployment")
        
    except ApiException as e:
        if e.status == 404:
            # Not a DynamoGraphDeployment, will try standard Deployment below
            print(f"    → DynamoGraphDeployment not found (404), will try standard Deployment")
            is_dgd = False
        else:
            print(f"❌ Failed to get DynamoGraphDeployment: {e}")
            return False
    except Exception as e:
        print(f"❌ Unexpected error getting DynamoGraphDeployment: {e}")
        return False
    
    if is_dgd:
        # Process DynamoGraphDeployment
        try:
            print(f"    → Processing DynamoGraphDeployment...")
            
            # Determine library path based on ConfigMap usage
            lib_path = "/cuda-fault/fake_cuda_xid79.so" if use_configmap else "/tmp/fake_cuda_xid79.so"
            
            # Prepare environment variables
            new_envs = []
            if enable:
                new_envs = [
                    {"name": "LD_PRELOAD", "value": lib_path},
                    {"name": "CUDA_FAULT_INJECTION_ENABLED", "value": "1"}
                ]
            
            # Patch worker services (VllmDecodeWorker and VllmPrefillWorker)
            services_to_patch = ["VllmDecodeWorker", "VllmPrefillWorker"]
            patched_services = []
            
            available_services = list(dgd.get("spec", {}).get("services", {}).keys())
            print(f"    → Available services: {available_services}")
            
            for service_name in services_to_patch:
                if service_name in dgd.get("spec", {}).get("services", {}):
                    print(f"    → Patching service: {service_name}")
                    service = dgd["spec"]["services"][service_name]
                    
                    # Ensure extraPodSpec exists
                    if "extraPodSpec" not in service:
                        service["extraPodSpec"] = {}
                    if "mainContainer" not in service["extraPodSpec"]:
                        service["extraPodSpec"]["mainContainer"] = {}
                    if "env" not in service["extraPodSpec"]["mainContainer"]:
                        service["extraPodSpec"]["mainContainer"]["env"] = []
                    
                    # Remove existing LD_PRELOAD and CUDA_FAULT_INJECTION_ENABLED if present
                    service["extraPodSpec"]["mainContainer"]["env"] = [
                        env for env in service["extraPodSpec"]["mainContainer"]["env"]
                        if env.get("name") not in ["LD_PRELOAD", "CUDA_FAULT_INJECTION_ENABLED"]
                    ]
                    
                    # Add new environment variables if enabling
                    if enable:
                        service["extraPodSpec"]["mainContainer"]["env"].extend(new_envs)
                    
                    # Handle ConfigMap volume mount
                    if use_configmap and enable:
                        # Add emptyDir volume for decoded library
                        if "volumes" not in service["extraPodSpec"]:
                            service["extraPodSpec"]["volumes"] = []
                        
                        # Remove existing volumes if present
                        service["extraPodSpec"]["volumes"] = [
                            v for v in service["extraPodSpec"]["volumes"]
                            if v.get("name") not in ["cuda-fault-lib-source", "cuda-fault-lib"]
                        ]
                        
                        # Add ConfigMap volume (source - base64 encoded)
                        service["extraPodSpec"]["volumes"].append({
                            "name": "cuda-fault-lib-source",
                            "configMap": {
                                "name": "cuda-fault-injection-lib"
                            }
                        })
                        
                        # Add emptyDir volume (destination - decoded binary)
                        service["extraPodSpec"]["volumes"].append({
                            "name": "cuda-fault-lib",
                            "emptyDir": {}
                        })
                        
                        # Add init container to decode base64
                        if "initContainers" not in service["extraPodSpec"]:
                            service["extraPodSpec"]["initContainers"] = []
                        
                        # Remove existing init container if present
                        service["extraPodSpec"]["initContainers"] = [
                            ic for ic in service["extraPodSpec"]["initContainers"]
                            if ic.get("name") not in ["decode-cuda-fault-lib", "compile-cuda-fault-lib"]
                        ]
                        
                        # Add init container to compile the library
                        service["extraPodSpec"]["initContainers"].append({
                            "name": "compile-cuda-fault-lib",
                            "image": "gcc:latest",
                            "command": ["sh", "-c"],
                            "args": [
                                "gcc -shared -fPIC -Wall -Wextra /source/fake_cuda_xid79.c -o /dest/fake_cuda_xid79.so -ldl && "
                                "chmod 755 /dest/fake_cuda_xid79.so && "
                                "echo 'Compiled CUDA fault library for Linux' && "
                                "ls -lh /dest/fake_cuda_xid79.so && "
                                "file /dest/fake_cuda_xid79.so"
                            ],
                            "volumeMounts": [
                                {
                                    "name": "cuda-fault-lib-source",
                                    "mountPath": "/source",
                                    "readOnly": True
                                },
                                {
                                    "name": "cuda-fault-lib",
                                    "mountPath": "/dest"
                                }
                            ]
                        })
                        
                        # Add volume mount to main container
                        if "volumeMounts" not in service["extraPodSpec"]["mainContainer"]:
                            service["extraPodSpec"]["mainContainer"]["volumeMounts"] = []
                        
                        # Remove existing mount if present
                        service["extraPodSpec"]["mainContainer"]["volumeMounts"] = [
                            vm for vm in service["extraPodSpec"]["mainContainer"]["volumeMounts"]
                            if vm.get("name") != "cuda-fault-lib"
                        ]
                        
                        # Add mount
                        service["extraPodSpec"]["mainContainer"]["volumeMounts"].append({
                            "name": "cuda-fault-lib",
                            "mountPath": "/cuda-fault",
                            "readOnly": True
                        })
                        
                        print(f"      ✓ Added init container to decode library")
                        print(f"      ✓ Added ConfigMap volume mount")
                    
                    # Add node affinity to pin pods to target node (simulates real XID 79 behavior)
                    if target_node and enable:
                        if "affinity" not in service["extraPodSpec"]:
                            service["extraPodSpec"]["affinity"] = {}
                        
                        service["extraPodSpec"]["affinity"]["nodeAffinity"] = {
                            "requiredDuringSchedulingIgnoredDuringExecution": {
                                "nodeSelectorTerms": [{
                                    "matchExpressions": [{
                                        "key": "kubernetes.io/hostname",
                                        "operator": "In",
                                        "values": [target_node]
                                    }]
                                }]
                            }
                        }
                        print(f"      ✓ Added node affinity to pin pods to {target_node}")
                    
                    elif not enable:
                        # Remove ConfigMap volume and mount when disabling
                        if "volumes" in service["extraPodSpec"]:
                            service["extraPodSpec"]["volumes"] = [
                                v for v in service["extraPodSpec"]["volumes"]
                                if v.get("name") not in ["cuda-fault-lib", "cuda-fault-lib-source"]
                            ]
                        
                        if "volumeMounts" in service["extraPodSpec"].get("mainContainer", {}):
                            service["extraPodSpec"]["mainContainer"]["volumeMounts"] = [
                                vm for vm in service["extraPodSpec"]["mainContainer"]["volumeMounts"]
                                if vm.get("name") != "cuda-fault-lib"
                            ]
                        
                        # Remove init container
                        if "initContainers" in service["extraPodSpec"]:
                            service["extraPodSpec"]["initContainers"] = [
                                ic for ic in service["extraPodSpec"]["initContainers"]
                                if ic.get("name") not in ["decode-cuda-fault-lib", "compile-cuda-fault-lib"]
                            ]
                        
                        # Remove node affinity
                        # Must explicitly set to None and ensure it's in the patch
                        # (simply deleting the key doesn't remove it from K8s)
                        service["extraPodSpec"]["affinity"] = None
                        print(f"      ✓ Removed node affinity")
                    
                    patched_services.append(service_name)
            
            print(f"    → Applying patch to DynamoGraphDeployment...")
            
            # Apply the patch
            custom_api.patch_namespaced_custom_object(
                group="nvidia.com",
                version="v1alpha1",
                namespace=namespace,
                plural="dynamographdeployments",
                name=deployment_name,
                body=dgd
            )
            
            # For disable, use JSON patch to explicitly remove affinity (None doesn't always work)
            if not enable and patched_services:
                print(f"    → Applying JSON patch to ensure affinity removal...")
                json_patches = []
                for service_name in patched_services:
                    json_patches.append({
                        "op": "replace",
                        "path": f"/spec/services/{service_name}/extraPodSpec/affinity",
                        "value": None
                    })
                
                if json_patches:
                    try:
                        custom_api.patch_namespaced_custom_object(
                            group="nvidia.com",
                            version="v1alpha1",
                            namespace=namespace,
                            plural="dynamographdeployments",
                            name=deployment_name,
                            body=json_patches,
                            _content_type="application/json-patch+json"
                        )
                        print(f"      ✓ JSON patch applied for affinity removal")
                    except Exception as e:
                        print(f"      ⚠ JSON patch failed (affinity may not exist): {e}")
            
            action = "enabled" if enable else "disabled"
            print(f"[✓] DynamoGraphDeployment patched - CUDA fault injection {action}")
            print(f"    Services patched: {', '.join(patched_services)}")
            if use_configmap and enable:
                print(f"    Library mounted at: {lib_path}")
            return True
            
        except ApiException as e:
            print(f"❌ Failed to patch DynamoGraphDeployment (ApiException): {e}")
            print(f"    Status: {e.status}")
            print(f"    Reason: {e.reason}")
            return False
        except Exception as e:
            print(f"❌ Failed to patch DynamoGraphDeployment (Exception): {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    # Try standard Deployment
    print(f"    → Not a DynamoGraphDeployment, trying standard Deployment...")
    
    try:
        deployment = apps_api.read_namespaced_deployment(deployment_name, namespace)
    except ApiException as e:
        print(f"❌ Failed to read deployment {deployment_name}: {e}")
        return False
    
    # Find all containers and patch their env
    for container in deployment.spec.template.spec.containers:
        if container.env is None:
            container.env = []
        
        # Remove existing LD_PRELOAD and CUDA_FAULT_INJECTION_ENABLED
        container.env = [
            e for e in container.env 
            if e.name not in ["LD_PRELOAD", "CUDA_FAULT_INJECTION_ENABLED"]
        ]
        
        if enable:
            # Add new env vars
            container.env.append(
                client.V1EnvVar(name="LD_PRELOAD", value="/tmp/fake_cuda_xid79.so")
            )
            container.env.append(
                client.V1EnvVar(name="CUDA_FAULT_INJECTION_ENABLED", value="1")
            )
    
    try:
        apps_api.patch_namespaced_deployment(
            deployment_name,
            namespace,
            deployment
        )
        action = "enabled" if enable else "disabled"
        print(f"[✓] Deployment patched - CUDA fault injection {action}")
        return True
    except ApiException as e:
        print(f"❌ Failed to patch deployment: {e}")
        return False


def get_worker_pods(deployment_name, namespace, node_name=None):
    """Get worker pods for a deployment."""
    core_api = client.CoreV1Api()
    
    label_selector = f"nvidia.com/dynamo-graph-deployment-name={deployment_name},nvidia.com/dynamo-component-type=worker"
    field_selector = f"spec.nodeName={node_name}" if node_name else None
    
    try:
        pods = core_api.list_namespaced_pod(
            namespace=namespace,
            label_selector=label_selector,
            field_selector=field_selector
        )
        return pods.items
    except ApiException as e:
        print(f"❌ Failed to list pods: {e}")
        return []


def inject_into_running_pods(deployment_name, namespace, node_name=None):
    """Inject library into currently running pods (temporary - lost on restart)."""
    print("\n" + "="*80)
    print("Method 1: Inject into Running Pods (Temporary)")
    print("="*80)
    print("⚠ This method is TEMPORARY - injection lost when pods restart")
    print("   Use Method 2 (deployment patch) for persistent injection")
    print()
    
    lib_path = get_library_path()
    pods = get_worker_pods(deployment_name, namespace, node_name)
    
    if not pods:
        print(f"❌ No worker pods found for deployment {deployment_name}")
        return False
    
    print(f"[→] Found {len(pods)} worker pods")
    
    success_count = 0
    for pod in pods:
        pod_name = pod.metadata.name
        print(f"\n[→] Processing pod: {pod_name}")
        
        # Copy library
        if copy_library_to_pod(pod_name, namespace, lib_path):
            success_count += 1
            print(f"    ✓ Library injected")
            print(f"    ⚠ To activate, pod must be restarted with LD_PRELOAD set")
        else:
            print(f"    ✗ Failed to inject")
    
    print(f"\n[→] Injected into {success_count}/{len(pods)} pods")
    print("\n⚠ IMPORTANT: Pods must be restarted with LD_PRELOAD environment variable!")
    print("   Use Method 2 (--patch-deployment) to set LD_PRELOAD and restart pods")
    
    return success_count > 0


def inject_via_deployment_patch(deployment_name, namespace):
    """Inject by patching deployment and rolling restart (persistent)."""
    print("\n" + "="*80)
    print("Method 2: Patch Deployment (Persistent)")
    print("="*80)
    print("This method:")
    print("  1. Copies library to existing pods")
    print("  2. Patches deployment to set LD_PRELOAD")
    print("  3. Triggers rolling restart")
    print("  4. New pods will have CUDA fault injection enabled")
    print()
    
    lib_path = get_library_path()
    
    # Step 1: Copy library to all current pods
    print("[1/3] Copying library to existing pods...")
    pods = get_worker_pods(deployment_name, namespace)
    
    if not pods:
        print(f"❌ No worker pods found")
        return False
    
    for pod in pods:
        copy_library_to_pod(pod.metadata.name, namespace, lib_path)
    
    # Step 2: Patch deployment
    print("\n[2/3] Patching deployment to enable CUDA fault injection...")
    if not patch_deployment_env(deployment_name, namespace, enable=True):
        return False
    
    # Step 3: Trigger rolling restart
    print("\n[3/3] Triggering pod restart...")
    
    # For DynamoGraphDeployment, we need to delete pods to trigger restart
    # (kubectl rollout restart doesn't work with custom resources)
    core_api = client.CoreV1Api()
    
    print(f"[→] Deleting {len(pods)} worker pods to trigger restart with new env vars...")
    for pod in pods:
        try:
            core_api.delete_namespaced_pod(
                name=pod.metadata.name,
                namespace=namespace,
                grace_period_seconds=30  # Graceful shutdown
            )
            print(f"    ✓ Deleted: {pod.metadata.name}")
        except ApiException as e:
            print(f"    ✗ Failed to delete {pod.metadata.name}: {e}")
    
    print(f"\n[→] Waiting for new pods to start with CUDA fault injection...")
    time.sleep(10)
    
    # Wait for pods to be recreated
    max_wait = 300  # 5 minutes
    start_time = time.time()
    
    while time.time() - start_time < max_wait:
        new_pods = get_worker_pods(deployment_name, namespace)
        
        # Check if we have the expected number of pods
        if len(new_pods) >= len(pods):
            # Check if all pods are running
            all_running = all(pod.status.phase == "Running" for pod in new_pods)
            
            if all_running:
                print(f"\n[✓] All {len(new_pods)} pods restarted successfully")
                print("[✓] CUDA fault injection is now ACTIVE")
                print("\n⚠ Pods will crash with 'No CUDA-capable device' error")
                print("   This simulates XID 79 (GPU falls off bus)")
                return True
        
        time.sleep(5)
    
    print("\n⚠ Timeout waiting for pods to restart")
    return False


def remove_injection(deployment_name, namespace):
    """Remove CUDA fault injection from deployment."""
    print("\n" + "="*80)
    print("Removing CUDA Fault Injection")
    print("="*80)
    
    print("[→] Removing LD_PRELOAD from deployment...")
    if not patch_deployment_env(deployment_name, namespace, enable=False):
        return False
    
    print("\n[→] Deleting pods to restore normal operation...")
    
    # Get current pods
    pods = get_worker_pods(deployment_name, namespace)
    
    if not pods:
        print(f"⚠ No worker pods found")
        return True
    
    # Delete pods to trigger restart without fault injection
    core_api = client.CoreV1Api()
    
    for pod in pods:
        try:
            core_api.delete_namespaced_pod(
                name=pod.metadata.name,
                namespace=namespace,
                grace_period_seconds=30
            )
            print(f"    ✓ Deleted: {pod.metadata.name}")
        except ApiException as e:
            print(f"    ✗ Failed to delete {pod.metadata.name}: {e}")
    
    print("[✓] Injection removed - pods will restart normally")
    return True


def verify_injection(deployment_name, namespace):
    """Verify that CUDA fault injection is active in pods."""
    print("\n" + "="*80)
    print("Verifying CUDA Fault Injection")
    print("="*80)
    
    pods = get_worker_pods(deployment_name, namespace)
    
    if not pods:
        print("❌ No pods found")
        return
    
    for pod in pods[:3]:  # Check first 3 pods
        pod_name = pod.metadata.name
        print(f"\n[→] Checking pod: {pod_name}")
        
        # Check if library exists
        cmd = [
            "kubectl", "exec", "-n", namespace, pod_name, "--",
            "ls", "-lh", "/tmp/fake_cuda_xid79.so"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"    ✓ Library present: {result.stdout.strip()}")
        else:
            print(f"    ✗ Library not found")
        
        # Check environment variables
        cmd = [
            "kubectl", "exec", "-n", namespace, pod_name, "--",
            "env"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if "LD_PRELOAD=/tmp/fake_cuda_xid79.so" in result.stdout:
            print(f"    ✓ LD_PRELOAD set correctly")
        else:
            print(f"    ✗ LD_PRELOAD not set")
        
        if "CUDA_FAULT_INJECTION_ENABLED=1" in result.stdout:
            print(f"    ✓ Fault injection enabled")
        else:
            print(f"    ✗ Fault injection not enabled")
        
        # Check pod logs for injection messages
        cmd = [
            "kubectl", "logs", "-n", namespace, pod_name,
            "--tail=50"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if "[CUDA FAULT INJECTION] ENABLED" in result.stderr or "[CUDA FAULT INJECTION] ENABLED" in result.stdout:
            print(f"    ✓ Fault injection active (found in logs)")
        elif "CrashLoopBackOff" in pod.status.phase or pod.status.container_statuses:
            if any(cs.state.waiting and "CrashLoopBackOff" in cs.state.waiting.reason 
                   for cs in pod.status.container_statuses or []):
                print(f"    ✓ Pod crashing (expected with fault injection)")
        else:
            print(f"    ? Cannot confirm injection status from logs")


def main():
    parser = argparse.ArgumentParser(
        description="Inject CUDA fault library into vLLM pods",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Patch deployment (persistent, recommended)
  python inject_into_pods.py --deployment vllm-v1-disagg-router --namespace dynamo-oviya --patch-deployment
  
  # Inject into running pods on specific node (temporary)
  python inject_into_pods.py --deployment vllm-v1-disagg-router --node aks-a100a-36888584-vmss000003
  
  # Remove injection
  python inject_into_pods.py --deployment vllm-v1-disagg-router --namespace dynamo-oviya --remove
  
  # Verify injection
  python inject_into_pods.py --deployment vllm-v1-disagg-router --namespace dynamo-oviya --verify
        """
    )
    
    parser.add_argument("--deployment", required=True, help="Deployment name (e.g., vllm-v1-disagg-router)")
    parser.add_argument("--namespace", default="dynamo-oviya", help="Kubernetes namespace")
    parser.add_argument("--node", help="Target specific node (optional)")
    parser.add_argument("--patch-deployment", action="store_true", help="Patch deployment (persistent)")
    parser.add_argument("--remove", action="store_true", help="Remove injection")
    parser.add_argument("--verify", action="store_true", help="Verify injection status")
    
    args = parser.parse_args()
    
    # Load kubeconfig
    try:
        config.load_kube_config()
    except:
        print("❌ Failed to load kubeconfig")
        sys.exit(1)
    
    print("="*80)
    print("CUDA Fault Injection - Pod Injector")
    print("="*80)
    print(f"Deployment: {args.deployment}")
    print(f"Namespace:  {args.namespace}")
    if args.node:
        print(f"Node:       {args.node}")
    print()
    
    if args.verify:
        verify_injection(args.deployment, args.namespace)
    elif args.remove:
        remove_injection(args.deployment, args.namespace)
    elif args.patch_deployment:
        inject_via_deployment_patch(args.deployment, args.namespace)
    else:
        inject_into_running_pods(args.deployment, args.namespace, args.node)


if __name__ == "__main__":
    main()

