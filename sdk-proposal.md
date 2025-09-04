# Dynamo SDK Redesign

Here are three areas in which I propose making improvements to the Dynamo SDK

1. Top-level DynamoGraph component
2. Re-thinking depends and client-creation
3. Serving components with python3

# Top-level DynamoGraph component

## Current Design

- Right now, there is no top level abstraction for a dynamo graph/pipeline.
- The link syntax tries to fill the gap but is an insuffient and misleading abstraction.

```python
# agg.py
Frontend.link(Processor).link(Backend) # Functionally, this just gets them to be served together. Does nothing to couple the components.
```

We then serve this by just pointing to the frontend component.

```
dynamo serve agg:Frontend
```

There's a few issues with this:
- Dynamo models are more DAGs than pipelines. As the edges between the components increase, the top level definition will get clunky with a lot of lines of links.
- As a user, the top level command should suggest I'm serving a graph, not a component.
- Cannot be extended to naturally support polymorphism, abstract classes, dependency injection. 

Let's assume for the following example that we have added support for abstract classes into the link/depend syntax allowing linking to different concrete impls. An attempt to compose the same model from different components might look like this:

```python
# Define a fast agg pipeline
g1 = Frontend.link(FastProcessor).link(FastBackend)

# Define a slow agg pipeline
g2 = Frontend.link(SlowProcessor).link(SlowBackend)

# Ideally, we'd like to do something like the following, but in this case what happens is that Frontend just gets overwritten by the last one. g1 does not exist, g2 does and is just a pointer to the frontend!
```

All this motivates the need for some meta container to encode the components of the actual model/pipeline being deployed!

## Proposed Design

Introducing a DynamoGraph entity. (Name TBD, other candidates: DynamoContainer, DynamoDeployment, DynamoPipeline)
Represents a container for a set of components that should be served or deployed together to compose some larger model.

Examples of DynamoGraphs:
- LLMAgg - Frontend, Processor, Backend
- LLMAggRouter - Frontend, Processor, Router, Backend
- LLMDisagg - Frontend, Processor, Disaggregator, Backend
- LLMDisaggRouter - Frontend, Processor, Disaggregator, Router, Backend
- LLaVA - Frontend, EmbeddingWorker, Backend

Notes:
- A DynamoGraph is a container for a set of components that should be served or deployed together to compose some larger model.
- Topology of graph is not defined at the top level, just the components composing the pipeline (shown below). There is no reason the edges between the components should be visible at the top level (as is currently the case in the link/depends syntax).
- Allows for more natural polymorphism and dependency injection (shown below)

Syntax to define your DynamoGraph. Open to suggestions
```python
from dynamo import DynamoGraph

# This encodes a spec for an aggregate LLM model composed of a frontend, processor, and backend
# The type hints inform what components can be passed in
# The defaults suggest what will be served if nothing is passed in
class LLMAgg(DynamoGraph):
    def __init__(self, frontend: Frontend = Frontend, processor: AbstractProcessor = Processor, backend: AbstractBackend = Backend):
        pass
```

Now, let's see how this spec can be used to compose multiple models composed of different concrete components
```python
from graphs import LLMAgg
from components import Frontend, FastProcessor, FastBackend, SlowProcessor, SlowBackend

default_agg = LLMAgg() # Will use default components (vanilla frontend, processor, backend)

custom_agg = LLMAgg(processor=FastProcessor) # defaults to vanilla frontend and backend

# Define a fast agg pipeline
fast_agg = LLMAgg(frontend=Frontend, processor=FastProcessor, backend=FastBackend)

# Define a slow agg pipeline
slow_agg = LLMAgg(frontend=Frontend, processor=SlowProcessor, backend=SlowBackend)

# Error! Components need to be compatible with the type hint
g4 = LLMAgg(frontend=Backend, processor=SlowProcessor, backend=FastBackend)
```

Finally, we serve the graph with the following:

```bash
dynamo serve graph:fast_agg
# or dynamo serve graph:slow_agg
```

Additionally, having a container entity for the components also allows us to do the following:

```python
g = LLMAgg(frontend=Frontend, processor=FastProcessor, backend=FastBackend)
g.add_sidecar_components([Planner, Metrics]) # augment the graph with components that are not part of the main pipeline, alternate serve and deploy setting may be imposed on these 'sidecar' components
```

# Re-thinking depends and client-creation

## Current Design

Right now, depends() construct fulfills two purposes
1. Ensure that the dependent service is also served
2. Return a client to the dependency

Current usage:
```python
from dynamo import Depends, service
@service
class Frontend():
    processor = depends(Processor)
    
    ...

    @dynamo_entrypoint
    def generate(self, request: str):
        for response in self.processor.generate(request):
            yield response
```

Some feedback about depends()
- Valid feedback has been raised that client creation should be opt-in and should not get in the way. 
- If the user wants to create their own client without hand-holding, they should be able to do so via the runtime directly.

How can we make this an opt-in feature without getting in the way, but allowing a seamless experience with dependency injection?

## Proposed Design

To re-iterate, depends() construct fulfills two purposes
1. Ensure that the dependent service is also served
2. Return a client to the dependency


With the new DynamoGraph entity, we can propose a new more flexible model that might remove depends altogether.

```python
# graph.py
from dynamo import DynamoGraph, service, async_on_start, init_clients
from components import Frontend, Backend, AbstractFrontend, AbstractBackend, AbstractProcessor

# Define an LLM Agg graph using our the new DynamoGraph entity
# Processor needs to be passed to the graph, no concrete impl is defined as default, Frontend and Backend are optional and have concrete impls defined as defaults
class LLMAgg(DynamoGraph):
    def __init__(self, processor: AbstractProcessor, frontend: AbstractFrontend = Frontend, backend: AbstractBackend = Backend):
        pass

# Define a custom processor component
@service
class MyProcessor():
    def __init__(self):
        ...

    # Pattern 1:User defines their own clients
    @async_on_start
    def async_init(self):
        # Existing pattern of explicitly creating a client from the runtime
        processor_client = await runtime.namespace("dynamo")
            .component("Backend")
            .endpoint("generate")
            .client()

    # Pattern 2: Dynamo serve injects clients as part of this startup hook
    @init_clients # NEW! can also be named @require_clients
    def require_clients(self, backend_client: Backend):
        """
        @init_clients is a startup hook (similar to async_init) that runs before a component is served. It allows clients of dependencies to be injected into the component without the developer having to write logic to locate and create the clients.

        Note that it doesn't cause these components to be served. If the required Dynamo Components are not served as part of the current DynamoGraph, an error will be thrown. The developer can assign a default value to indicate that the client is optional (in case the component is not part of the current graph).
        """
        # Do whatever you want with the clients, most likely store them on self for later use
        self.backend_client = backend_client

    @dynamo_endpoint
    def generate(self, request: str):
        # Use the client generated in the init_clients hook
        for response in self.backend_client.generate(request):
            yield response

agg = LLMAgg(processor=MyProcessor) # frontend and backend will be served as part of the graph since defaults are defined on their spec
```

Note:
- There's other syntaxes we can propose to inject the clients. We can add them as arguments to the __init__ method, or direclty to the signature of the methods that are decorated with @dynamo_endpoint. I think the goal is to keep this door open without getting in the way.

Lastly serve the model:

```
dynamo serve graph:agg
```

Let's recap what is happening here:
1. Define a DynamoGraph entity that encodes the components that need to be served
2. Define your own processor component, require a backend client to be injected using the @init_clients hook

THe user can leave out the init_clients startup hook if they want to manage client creation themselves. 

# Serving components directly with python (instead of specifying dynamo serve)

WIP
