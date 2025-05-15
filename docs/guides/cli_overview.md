# About the Dynamo Command Line Interface
The Dynamo CLI is a powerful tool for serving, containerizing, and deploying Dynamo applications. It leverages core pieces of the BentoML deployment stack and provides a range of commands to manage your Dynamo services.

The Dynamo CLI lets you:
- `run` - quickly chat with a model
- `serve` - run a set of services locally (via `depends()` or `.link()`)
- `build` - create an archive of your services (called a `bento`)
- `deploy` - create a pipeline on Dynamo Cloud

## Commands

### `run`

The `run` command allows you to quickly chat with a model. Under the hood - it is running the `dynamo-run` Rust binary. You can find the arguments that it takes here: [dynamo-run docs](dynamo_run.md)

**Example**
```bash
dynamo run deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
```

### `serve`

The `serve` command lets you run a defined inference graph locally. You must point toward your file and intended class using file:Class syntax

**Usage**
```bash
dynamo serve [SERVICE]
```

**Arguments**
- `SERVICE` - The service to start. You use file:Class syntax to specify the service.

**Flags**
- `--file`/`-f` - Path to optional YAML configuration file. An example of the YAML file can be found in the configuration section of the [SDK docs](../API/sdk.md)
- `--dry-run` - Print out the dependency graph and values without starting any services.
- `--service-name` - Only serve the specified service name. The rest of the discoverable components in the graph are not started.
- `--working-dir` - Specify the directory to find the Service instance
- Any additional flags that follow Class.key=value will be passed to the service constructor for the target service and parsed. Please see the configuration section of the [SDK docs](../API/sdk.md) for more details.

**Example**
```bash
cd examples
# Spin up Frontend, Middle, and Backend components
dynamo serve hello_world:Frontend

# Spin up only the Middle component in the graph that is discoverable from the Frontend service
dynamo serve  --service-name Middle hello_world:Frontend
```

### `build`

The `build` commmand allows you to package up your inference graph and its dependancies and create an archive of it. This is commonly paired with the `--containerize` flag to create a single docker container that runs your inference graph. As with `serve`, you point toward the first service in your dependency graph.

**Usage**
```bash
dynamo build [SERVICE]
```

**Arguments**
- `SERVICE` - The service to build. You use file:Class syntax to specify the service.

**Flags**
- `--working-dir` - Specify the directory to find the Service instance
- `--containerize` - Whether to containerize the Bento after building

**Example**
```bash
cd examples/hello_world
dynamo build hello_world:Frontend
```

### `deploy`

The `deploy` commmand creates a pipeline on Dynamo Cloud using parameters at the prompt or using a YAML configuration file.

**Usage**
```bash
dynamo deploy [PIPELINE]
```

**Arguments**
- `pipeline` - The pipeline to deploy. Defaults to *None*; required.

**Flags**
- `--name` or `-n` - Deployment name. Defaults to *None*; required.
- `--config-file` or `-f` - Configuration file path. Defaults to *None*; required.
- `--wait` - Whether or not to wait for deployment to be ready. Defaults to wait.
  `--no-wait`
- `--timeout` - The number of seconds that can elapse before deployment times out; measured in seconds. Defaults to 3600. 
- `--endpoint` or `-e` - The Dynamo Cloud endpoint where the pipeline should be deployed. Defaults to *None*; required.
- `--help` or `-h` - Display in-line help for `dynamo deploy`.


**Example**

For a detailed example, see [Operator Deployment](dynamo_deploy/operator_deployment.md).
