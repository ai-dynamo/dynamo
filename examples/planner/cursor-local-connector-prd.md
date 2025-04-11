# Product Requirements Document: LocalConnector for Dynamo Planner

## Overview
The LocalConnector enables automatic scaling of Dynamo components in a local development environment, with special focus on GPU resource management for workers.

## Core Requirements

### 1. Component Management
- Discover and track components in namespace
- Scale components up/down with proper resource handling
- Monitor component health and status
- Handle process lifecycle (start/stop/restart)

### 2. Resource Management
- Track GPU allocations and availability
- Manage GPU assignments for workers
- Update state file with resource allocations
- Prevent resource conflicts and overallocation

### 3. State Management
- Read/write state file for persistence
- Track component configurations
- Maintain resource allocation state
- Handle environment variables

## API Design

### Core APIs
```python
class LocalConnector(PlannerConnector):
    # Base Component Operations
    async def get_component_replicas(self, component_name: str) -> int
    async def list_components(self) -> List[Dict[str, Any]]
    async def get_component_logs(self, component_name: str, lines: int = 100) -> List[str]
    async def restart_component(self, component_name: str) -> bool

    # Resource Management
    async def get_available_gpus(self) -> List[str]
    async def allocate_gpus(self, component_name: str, num_gpus: int) -> List[str]
    async def release_gpus(self, component_name: str) -> bool
    
    # State Management
    async def load_state(self) -> Dict[str, Any]
    async def save_state(self, state: Dict[str, Any]) -> bool
    async def update_component_resources(self, component_name: str, resources: Dict) -> bool

    # Enhanced Scaling Operations
    async def add_gpu_worker(self, component_name: str, num_gpus: int = 1) -> bool
    async def remove_gpu_worker(self, component_name: str) -> bool
    async def scale_component(self, component_name: str, replicas: int) -> bool
```

## Implementation Strategy

### Phase 1: State Management
1. Implement state file operations
2. Add component configuration tracking
3. Handle resource state persistence

### Phase 2: Resource Management
1. Implement GPU tracking
2. Add allocation/deallocation logic
3. Handle resource conflicts
4. Update state file on changes

### Phase 3: Enhanced Scaling
1. Add GPU worker-specific scaling
2. Implement resource-aware scaling
3. Handle environment updates
4. Add validation and error handling

### Phase 4: Testing & Validation
1. Test GPU allocation scenarios
2. Validate state persistence
3. Test scaling operations
4. Add comprehensive error handling

## Success Criteria
1. Successfully scale GPU workers up/down
2. Maintain accurate resource allocation state
3. Prevent GPU conflicts
4. Handle component lifecycle properly
5. Persist state across restarts
