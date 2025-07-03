import logging
import uvicorn
from fastapi import FastAPI, HTTPException
import sglang as sgl
import asyncio
from dynamo.runtime import DistributedRuntime

class SglangHttpServer:
    # TODO: add decode client conditional as well
    def __init__(self, engine: sgl.Engine, drt: DistributedRuntime, port: int = 8888):
        self.engine = engine
        self.port = port
        self.app = FastAPI()
        self.setup_routes()
        
    def setup_routes(self):
        @self.app.post("/flush_cache")
        async def flush_cache():
            """Flush the radix cache."""
            try:
                # this targets the prefill/aggregated worker
                asyncio.create_task(self.engine.tokenizer_manager.flush_cache())
                
                # then we target the decode worker over nats
                await self.drt.namespace("dynamo").component("decode").endpoint("generate").client()
                
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
        logging.info(f"🚀 Admin server running on http://0.0.0.0:{self.port} - Endpoints: POST /flush_cache")
        
        await server.serve()