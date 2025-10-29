from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import httpx
import uvicorn
from kubernetes import client, config
from kubernetes.client.rest import ApiException
import json
import logging
import os
import asyncio
from typing import List, Dict, Tuple, Union, AsyncIterator, Optional
from itertools import count
from datetime import datetime

# Configure logging level from environment
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI()

# Configuration
NAMESPACE = os.getenv("NAMESPACE", "default")
K8S_NAMESPACE = os.getenv("K8S_NAMESPACE", NAMESPACE)  # Kubernetes namespace to search for services
FRONTEND_LABEL_SELECTOR = os.getenv("FRONTEND_LABEL_SELECTOR", "")  # e.g., "app=dynamo-frontend"
FRONTEND_SERVICE_PATTERN = os.getenv("FRONTEND_SERVICE_PATTERN", "frontend")  # Pattern in service name
STATIC_FRONTENDS = os.getenv("STATIC_FRONTENDS", "").split(",") if os.getenv("STATIC_FRONTENDS") else []
DISCOVERY_INTERVAL = int(os.getenv("DISCOVERY_INTERVAL", "5"))  # seconds

# Initialize Kubernetes client and httpx client
k8s_client = None
httpx_client = None
request_counter = count()  # For round-robin routing

# Shared frontend list (updated by background task)
cached_frontends: List[str] = []
discovery_task = None
last_discovery_time: Optional[datetime] = None
discovery_error_count = 0
total_requests = 0

class RequestData(BaseModel):
    target: str
    # Add any other fields your request might contain
    message: str = None

async def update_frontends_periodically():
    """Background task that periodically updates the cached frontends list."""
    global cached_frontends, last_discovery_time, discovery_error_count
    
    while True:
        try:
            logger.debug(f"[Discovery] Starting frontend discovery (interval: {DISCOVERY_INTERVAL}s)")
            start_time = datetime.now()
            
            frontends = discover_frontends(K8S_NAMESPACE, STATIC_FRONTENDS)
            
            discovery_duration = (datetime.now() - start_time).total_seconds()
            last_discovery_time = datetime.now()
            
            # Update the cached list
            if frontends != cached_frontends:
                old_count = len(cached_frontends)
                new_count = len(frontends)
                logger.info(f"[Discovery] Frontend list updated: {old_count} -> {new_count} frontends (took {discovery_duration:.2f}s)")
                logger.info(f"[Discovery] New frontends: {frontends}")
                cached_frontends = frontends
                discovery_error_count = 0  # Reset error count on success
            else:
                logger.debug(f"[Discovery] Frontend list unchanged: {len(frontends)} frontends (took {discovery_duration:.2f}s)")
            
        except Exception as e:
            discovery_error_count += 1
            logger.error(f"[Discovery] Error updating frontend cache (error #{discovery_error_count}): {e}", exc_info=True)
        
        # Wait before next update
        await asyncio.sleep(DISCOVERY_INTERVAL)

@app.on_event("startup")
async def startup_event():
    global k8s_client, httpx_client, discovery_task, cached_frontends
    try:
        logger.info("[Startup] Initializing Kubernetes client")
        # Try to load in-cluster config first (when running in a pod)
        try:
            config.load_incluster_config()
            logger.info("[Startup] Loaded in-cluster Kubernetes configuration")
        except config.ConfigException:
            # Fall back to kubeconfig (for local development)
            config.load_kube_config()
            logger.info("[Startup] Loaded Kubernetes configuration from kubeconfig")
        
        k8s_client = client.CoreV1Api()
        logger.info("[Startup] Successfully initialized Kubernetes client")
    except Exception as e:
        logger.error(f"[Startup] Failed to initialize Kubernetes client: {e}", exc_info=True)
        logger.warning("[Startup] Continuing with static frontends only")
        k8s_client = None
    
    # Initialize persistent httpx client for proxying
    logger.info("[Startup] Initializing httpx client for request proxying")
    httpx_client = httpx.AsyncClient(
        timeout=httpx.Timeout(300.0, connect=10.0),
        follow_redirects=False
    )
    logger.info("[Startup] httpx client initialized")
    
    # Do initial frontend discovery synchronously
    logger.info(f"[Startup] Performing initial frontend discovery for namespace: {K8S_NAMESPACE}")
    cached_frontends = discover_frontends(K8S_NAMESPACE, STATIC_FRONTENDS)
    logger.info(f"[Startup] Initial frontend discovery complete: {len(cached_frontends)} frontends found")
    if cached_frontends:
        logger.info(f"[Startup] Discovered frontends: {cached_frontends}")
    else:
        logger.warning("[Startup] No frontends discovered! Requests will fail until frontends are available.")
    
    # Start background task to periodically update frontends
    discovery_task = asyncio.create_task(update_frontends_periodically())
    logger.info(f"[Startup] Started background frontend discovery task (interval: {DISCOVERY_INTERVAL}s)")

@app.on_event("shutdown")
async def shutdown_event():
    global httpx_client, discovery_task
    
    logger.info("[Shutdown] Starting graceful shutdown...")
    
    # Cancel background task
    if discovery_task:
        logger.info("[Shutdown] Cancelling background discovery task")
        discovery_task.cancel()
        try:
            await discovery_task
        except asyncio.CancelledError:
            logger.info("[Shutdown] Background discovery task cancelled")
    
    if httpx_client:
        try:
            logger.info("[Shutdown] Closing httpx client")
            await httpx_client.aclose()
            logger.info("[Shutdown] httpx client closed")
        except Exception as e:
            logger.error(f"[Shutdown] Error closing httpx client: {e}")
    
    logger.info(f"[Shutdown] Shutdown complete. Total requests processed: {total_requests}")

def discover_frontends(k8s_namespace: str, static_frontends: List[str] = None) -> List[str]:
    """Discover frontend services from Kubernetes API.
    
    Args:
        k8s_namespace: Kubernetes namespace to search for services
        static_frontends: List of static frontend URLs as fallback
    
    Returns:
        List of frontend URLs (discovered + static)
    """
    if not k8s_client:
        logger.warning("Kubernetes client not available, returning static frontends only")
        return static_frontends or []
    
    discovered_frontends = []
    
    try:
        # List all services in the namespace
        services = k8s_client.list_namespaced_service(namespace=k8s_namespace)
        
        for service in services.items:
            try:
                service_name = service.metadata.name
                
                # Filter services based on label selector if provided
                if FRONTEND_LABEL_SELECTOR:
                    labels = service.metadata.labels or {}
                    # Parse label selector (simple format: "key=value")
                    selector_parts = FRONTEND_LABEL_SELECTOR.split("=")
                    if len(selector_parts) == 2:
                        key, value = selector_parts
                        if labels.get(key.strip()) != value.strip():
                            continue
                
                # Filter services based on name pattern
                if FRONTEND_SERVICE_PATTERN and FRONTEND_SERVICE_PATTERN.lower() not in service_name.lower():
                    continue
                
                # Extract ports and build frontend URLs
                if service.spec and service.spec.ports:
                    for port in service.spec.ports:
                        # Look for HTTP port (typically named 'http' or port 8000)
                        if port.name == 'http' or port.port == 8000:
                            # Build service DNS name
                            # Format: <service-name>.<namespace>.svc.cluster.local:<port>
                            service_dns = f"{service_name}.{k8s_namespace}.svc.cluster.local"
                            frontend_url = f"http://{service_dns}:{port.port}"
                            
                            if frontend_url not in discovered_frontends:
                                discovered_frontends.append(frontend_url)
                                logger.debug(f"Discovered frontend service: {frontend_url} (service: {service_name})")
                            break  # Only use the first matching port
                
            except Exception as e:
                logger.warning(f"Failed to process service {service.metadata.name}: {e}")
                continue
        
        # Combine static frontends with discovered ones
        all_frontends = list(set((static_frontends or []) + discovered_frontends))
        logger.info(f"Discovered {len(discovered_frontends)} frontends from Kubernetes, total: {len(all_frontends)}")
        return all_frontends
        
    except ApiException as e:
        logger.error(f"Kubernetes API error discovering frontends: {e.status} - {e.reason}")
        return static_frontends or []
    except Exception as e:
        logger.error(f"Error discovering frontends from Kubernetes: {e}", exc_info=True)
        return static_frontends or []

async def route_request(method: str, path: str, headers: Dict[str, str], 
                       content: bytes, frontend_index: Optional[int] = None) -> Tuple[int, Dict[str, str], Union[bytes, AsyncIterator[bytes]]]:
    """Route a request to a frontend.
    
    Args:
        method: HTTP method
        path: Request path
        headers: Request headers
        content: Request body
        frontend_index: Optional specific frontend index to use. If None, uses round-robin.
    
    Returns either:
    - For non-streaming: tuple[int, Dict[str, str], bytes] (status, headers, body)
    - For streaming: tuple[int, Dict[str, str], AsyncIterator[bytes]] (status, headers, stream)
    """
    global total_requests
    total_requests += 1
    
    # Use cached frontends list (updated by background task)
    frontends = cached_frontends
    
    logger.debug(f"[Route] Request #{total_requests}: {method} {path}")
    logger.debug(f"[Route] Available frontends: {len(frontends)}")
    
    if not frontends:
        logger.error(f"[Route] No frontend endpoints available for request {method} {path}")
        raise Exception("No frontend endpoints available")
    
    # Select frontend either by specified index or round-robin
    if frontend_index is not None:
        if frontend_index < 0 or frontend_index >= len(frontends):
            logger.error(f"[Route] Invalid frontend index {frontend_index}, must be 0-{len(frontends)-1}")
            raise Exception(f"Invalid frontend index {frontend_index}, must be 0-{len(frontends)-1}")
        logger.info(f"[Route] Using specified frontend index: {frontend_index}")
        target_frontend = frontends[frontend_index]
    else:
        frontend_index = next(request_counter) % len(frontends)
        target_frontend = frontends[frontend_index]
    
    # Construct target URL
    target_url = f"{target_frontend.rstrip('/')}{path}"
    
    logger.info(f"[Route] #{total_requests} Routing {method} {path} -> {target_url} (frontend {frontend_index + 1}/{len(frontends)})")
    
    # Forward the request
    try:
        logger.debug(f"[Route] Forwarding request to backend...")
        # Stream everything with minimal latency
        async def zero_buffer_stream():
            async with httpx_client.stream(
                method=method,
                url=target_url,
                headers=headers,
                content=content
            ) as response:
                content_type = response.headers.get('content-type', 'unknown')
                logger.info(f"[Route] Response from backend: {response.status_code}, content-type: {content_type}")
                logger.debug(f"[Route] Response headers: {dict(response.headers)}")
                
                chunk_count = 0
                # Stream with the smallest possible chunks for minimal TTFT
                async for chunk in response.aiter_bytes(chunk_size=64):
                    if chunk:
                        chunk_count += 1
                        if chunk_count == 1:
                            logger.debug(f"[Route] First chunk received ({len(chunk)} bytes)")
                        yield chunk
                
                logger.debug(f"[Route] Streaming complete: {chunk_count} chunks sent")
        
        # Always return as streaming - this eliminates all buffering
        # For simplicity, returning 200 here but ideally should get from response
        return 200, {"content-type": "text/event-stream"}, zero_buffer_stream()
    
    except Exception as e:
        logger.error(f"[Route] Error routing request to {target_url}: {e}", exc_info=True)
        raise


async def route_request_func(method: str, path: str, headers: Dict[str, str], 
                            content: bytes, frontend_index: Optional[int] = None) -> Tuple[int, Dict[str, str], Union[bytes, AsyncIterator[bytes]]]:
    """Wrapper function for route_request to avoid naming conflicts."""
    return await route_request(method, path, headers, content, frontend_index)

async def _handle_request(request: Request, path: str, frontend_index: Optional[int] = None) -> Response:
    """Handle incoming HTTP requests by proxying them.
    
    Args:
        request: The incoming FastAPI request
        path: The path to proxy
        frontend_index: Optional specific frontend index to route to. If None, uses round-robin.
    """
    try:
        # Extract request details
        method = request.method
        path_with_query = str(request.url.path)
        if request.url.query:
            path_with_query += f"?{request.url.query}"
        
        headers = dict(request.headers)
        body = await request.body()
        
        # Remove hop-by-hop headers (except transfer-encoding for streaming)
        headers_to_remove = ['host', 'connection', 'upgrade', 'proxy-connection',
                           'proxy-authenticate', 'proxy-authorization', 'te', 'trailers']
        for header in headers_to_remove:
            headers.pop(header.lower(), None)
        
        # Route the request (using round-robin or specific index)
        result = await route_request(
            method, path_with_query, headers, body, frontend_index=frontend_index
        )
        status, response_headers, response_data = result
        
        # Always treat as streaming response to eliminate any buffering
        is_streaming_response = hasattr(response_data, '__aiter__')
        
        # Clean up headers but preserve streaming-related ones
        headers_to_remove_from_response = [h for h in headers_to_remove if h != 'transfer-encoding']
        for header in headers_to_remove_from_response:
            response_headers.pop(header.lower(), None)
        
        if is_streaming_response:
            # Stream the response immediately 
            logger.info(f"Streaming response for {method} {path_with_query}")
            
            return StreamingResponse(
                content=response_data,
                status_code=status,
                headers=response_headers
            )
        else:
            # Fallback for non-streaming (shouldn't happen with our new approach)
            logger.debug(f"Non-streaming response for {method} {path_with_query}")
            return Response(
                content=response_data,
                status_code=status,
                headers=response_headers
            )
    
    except Exception as e:
        logger.error(f"Error handling request: {e}")
        return Response(
            content=f"Proxy error: {str(e)}",
            status_code=500,
            media_type="text/plain"
        )


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    k8s_status = "connected" if k8s_client else "disconnected"
    has_frontends = len(cached_frontends) > 0
    
    return {
        "status": "healthy" if has_frontends else "degraded",
        "kubernetes": k8s_status,
        "namespace": NAMESPACE,
        "frontends_available": len(cached_frontends),
        "total_requests": total_requests
    }

@app.get("/frontends")
async def list_frontends():
    """List all discovered frontends for debugging."""
    return {
        "namespace": NAMESPACE,
        "total": len(cached_frontends),
        "frontends": cached_frontends,
        "discovery_interval": DISCOVERY_INTERVAL,
        "last_discovery": last_discovery_time.isoformat() if last_discovery_time else None,
        "discovery_errors": discovery_error_count
    }

@app.get("/debug/state")
async def debug_state():
    """Comprehensive debug endpoint showing all internal state."""
    import sys
    
    return {
        "app_info": {
            "namespace": NAMESPACE,
            "k8s_namespace": K8S_NAMESPACE,
            "frontend_label_selector": FRONTEND_LABEL_SELECTOR,
            "frontend_service_pattern": FRONTEND_SERVICE_PATTERN,
            "discovery_interval": DISCOVERY_INTERVAL,
            "log_level": LOG_LEVEL,
            "python_version": sys.version,
        },
        "connections": {
            "k8s_connected": k8s_client is not None,
            "httpx_client_active": httpx_client is not None,
            "discovery_task_running": discovery_task is not None and not discovery_task.done(),
        },
        "frontends": {
            "cached_count": len(cached_frontends),
            "cached_list": cached_frontends,
            "static_frontends": STATIC_FRONTENDS,
            "last_discovery": last_discovery_time.isoformat() if last_discovery_time else None,
            "discovery_errors": discovery_error_count,
        },
        "requests": {
            "total_processed": total_requests,
            "next_frontend_index": next(request_counter) % len(cached_frontends) if cached_frontends else None,
        },
        "timestamp": datetime.now().isoformat()
    }

@app.get("/debug/k8s")
async def debug_k8s():
    """Debug endpoint to inspect Kubernetes services."""
    if not k8s_client:
        return {"error": "Kubernetes client not connected"}
    
    try:
        services = k8s_client.list_namespaced_service(namespace=K8S_NAMESPACE)
        
        service_list = []
        for service in services.items:
            ports_info = []
            if service.spec and service.spec.ports:
                for port in service.spec.ports:
                    ports_info.append({
                        "name": port.name,
                        "port": port.port,
                        "target_port": str(port.target_port) if port.target_port else None,
                        "protocol": port.protocol
                    })
            
            service_list.append({
                "name": service.metadata.name,
                "namespace": service.metadata.namespace,
                "labels": service.metadata.labels or {},
                "type": service.spec.type if service.spec else None,
                "cluster_ip": service.spec.cluster_ip if service.spec else None,
                "ports": ports_info
            })
        
        return {
            "namespace": K8S_NAMESPACE,
            "total_services": len(service_list),
            "services": service_list,
            "timestamp": datetime.now().isoformat()
        }
    except ApiException as e:
        logger.error(f"Kubernetes API error: {e.status} - {e.reason}")
        return {"error": f"API error: {e.status} - {e.reason}"}
    except Exception as e:
        logger.error(f"Error reading Kubernetes services: {e}", exc_info=True)
        return {"error": str(e)}

@app.post("/debug/test-route")
async def test_route(payload: Dict = None):
    """Test endpoint to simulate a routed request without actually calling backends."""
    frontends = cached_frontends
    
    if not frontends:
        return {
            "error": "No frontends available",
            "frontends": frontends,
            "namespace": NAMESPACE
        }
    
    frontend_index = next(request_counter) % len(frontends)
    target_frontend = frontends[frontend_index]
    
    return {
        "message": "Test routing (no actual backend call)",
        "would_route_to": target_frontend,
        "frontend_index": frontend_index,
        "total_frontends": len(frontends),
        "all_frontends": frontends,
        "payload_received": payload,
        "request_number": total_requests + 1
    }

@app.api_route("/route/{frontend_index:int}/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"])
async def route_to_specific_index(request: Request, frontend_index: int, path: str):
    """Route a request to a specific frontend by index.
    
    The request body should be in standard OpenAI format (or any other format the backend expects).
    
    Example:
    ```
    curl -X POST http://localhost:8080/route/0/v1/chat/completions \\
      -H "Content-Type: application/json" \\
      -d '{
        "model": "nvidia/Llama-3.1-8B-Instruct-FP8",
        "messages": [{"role": "user", "content": "hello"}],
        "stream": false,
        "max_tokens": 100
      }'
    ```
    """
    frontends = cached_frontends
    
    # Validate frontend index
    if not frontends:
        logger.error(f"[Route/{frontend_index}] No frontends available")
        raise HTTPException(status_code=503, detail="No frontend endpoints available")
    
    if frontend_index < 0 or frontend_index >= len(frontends):
        logger.error(f"[Route/{frontend_index}] Invalid index {frontend_index}, must be 0-{len(frontends)-1}")
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid frontend index {frontend_index}, must be 0-{len(frontends)-1}"
        )
    
    logger.info(f"[Route/{frontend_index}] Routing {request.method} /{path} to frontend index {frontend_index}: {frontends[frontend_index]}")
    
    try:
        # Use the existing _handle_request function which properly handles the request
        return await _handle_request(request, path, frontend_index=frontend_index)
    
    except Exception as e:
        logger.error(f"[Route/{frontend_index}] Error routing to index {frontend_index}: {e}", exc_info=True)
        raise HTTPException(status_code=502, detail=f"Backend error: {str(e)}")

@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"])
async def proxy_handler(request: Request, path: str):
    """Catch-all route handler that proxies all requests to discovered frontends."""
    return await _handle_request(request, path)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
