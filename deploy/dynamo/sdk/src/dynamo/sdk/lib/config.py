from typing import Dict, Any, Optional
import json
import os

class ServiceConfig:
    """Configuration store that can be passed between services"""
    
    _instance = None
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = ServiceConfig()
        return cls._instance
    
    def __init__(self):
        # Load from environment variable if present
        self.configs = {}
        env_config = os.environ.get("DYNAMO_SERVICE_CONFIG")
        if env_config:
            try:
                self.configs = json.loads(env_config)
            except json.JSONDecodeError:
                print(f"Failed to parse DYNAMO_SERVICE_CONFIG: {env_config}")
    
    def get_config(self, service_name: str, key: str, default: Any = None) -> Any:
        """Get config for a specific service and key"""
        service_config = self.configs.get(service_name, {})
        return service_config.get(key, default)
    
    def get_service_config(self, service_name: str) -> Dict[str, Any]:
        """Get all configs for a specific service"""
        return self.configs.get(service_name, {})
    
    def require_config(self, service_name: str, key: str, error_msg: str = None) -> Any:
        """Get a required config value, raising error if not found"""
        value = self.get_config(service_name, key)
        if value is None:
            msg = error_msg or f"{service_name}.{key} must be specified in configuration"
            raise ValueError(msg)
        return value