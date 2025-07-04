import logging
import uvicorn
from fastapi import FastAPI, HTTPException
import sglang as sgl
import asyncio
from sglang.srt.server_args import ServerArgs
from dynamo.runtime import DistributedRuntime

class SglangHttpServer:
    def __init__(self, server_args: ServerArgs, drt: DistributedRuntime, port: int = 9001):
        self.port = port
        self.app = FastAPI()
        self.server_args = server_args
        self.drt = drt
        self.worker_client = None
        self.decode_flush_client = None
        self._clients_initialized = False
        self.setup_routes()
        
    async def _ensure_clients(self):
        if not self._clients_initialized:
            self.worker_client = await self.drt.namespace("dynamo").component("worker").endpoint("flush_cache").client()
            if self.server_args.disaggregation_mode != "null":
                self.decode_flush_client = await self.drt.namespace("dynamo").component("decode").endpoint("flush_cache").client()
            self._clients_initialized = True

    def setup_routes(self):
        @self.app.post("/flush_cache")
        async def flush_cache():
            """Flush the radix cache."""
            await self._ensure_clients()
            try:
                # Flush worker instances
                await self.worker_client.wait_for_instances()
                worker_ids = self.worker_client.instance_ids()
                print(f"DEBUG: Found {len(worker_ids)} worker instances: {worker_ids}")
                
                for inst_id in worker_ids:
                    try:
                        print(f"DEBUG: Calling worker instance {inst_id}")
                        stream = await self.worker_client.direct("{}", inst_id)
                        print(f"DEBUG: Got stream for worker {inst_id}")
                        
                        async for payload in stream:
                            print(f"Worker[{inst_id}] -> {payload}")
                    except Exception as e:
                        print(f"DEBUG: Exception for worker {inst_id}: {e}")
                        logging.error(f"Worker[{inst_id}] flush error: {e}")
                
                # Handle decode workers if in disaggregation mode
                if self.server_args.disaggregation_mode != "null":
                    await self.decode_flush_client.wait_for_instances()
                    decode_ids = self.decode_flush_client.instance_ids()
                    
                    for inst_id in decode_ids:
                        stream = await self.decode_flush_client.direct("{}", inst_id)
                        async for payload in stream:
                            print(f"Decode[{inst_id}] -> {payload}")

                return {"message": "Cache flush initiated", "success": True}
            except Exception as e:
                logging.error(f"Cache flush error: {e}")
                return {"message": f"Cache flush failed: {str(e)}", "success": False}
        
    async def start_server(self):
        """Start the HTTP server"""
        config = uvicorn.Config(
            self.app, 
            host="0.0.0.0", 
            port=self.port, 
            log_config=None  # Disable all uvicorn logging
        )
        server = uvicorn.Server(config)
        
        # Single nice log with available endpoints
        logging.info(f"🚀 SGL engine HTTP server running on http://0.0.0.0:{self.port} - Endpoints: POST /flush_cache")
        
        await server.serve()