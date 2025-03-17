# Dynamo CLI Documentation
The Dynamo CLI is a powerful tool for serving, containerizing, and deploying Dynamo applications. It leverages core pieces of the BentoML deployment stack and provides a range of commands to manage your Dynamo services.

Overview
At a high level, the Dynamo CLI allows you to:
- `run` - quickly chat with a model 
- `start` - run an individual service locally
- `serve` - run a set of services locally (via `depends()` or `.link()`)
- `build` - create an archive of your services (called a `bento`)
- `containerize` - containerize your services for deployment
- `deploy` - deploy your services to a Kubernetes cluster running the Dynamo Server
- `server` - interact with your Dynamo Server

# Commands

## `run`

The `run` command allows you to quickly chat with a model. Under the hood - it is running the `dynamo-run` Rust binary. You can find the arguments that it takes here: [dynamo-run docs](../../../../../launch/README.md)

## `start`

The `start` command allows you to run an individual service locally. It is useful for development and testing and is similar to just running `python3 service.py`

**Usage**
```bash
dynamo start [SERVICE]
```

**Arguments**
- `SERVICE` - The service to start. You use file:Class syntax to specify the service.

**Flags**
- `--file`/`-f` - Path to optional YAML configuration file.
- `--dry-run` - Print out the dependency graph and values without starting any services.
- `--working-dir` - Specify the directory to find the Service instance
- Any additional flags that follow Class.key=value will be passed to the service constructor and parsed. Please see the [SDK docs](../sdk/README.md) for more details.

## `serve`

Spin up a dynamo dependancy graph locally. You must point toward your file and intended class using file:Class syntax

**Usage**
```bash
dynamo serve [SERVICE]
```

**Arguments**
- `SERVICE` - The service to start. You use file:Class syntax to specify the service.

**Flags**
- `--file`/`-f` - Path to optional YAML configuration file.
- `--dry-run` - Print out the dependency graph and values without starting any services.
- `--working-dir` - Specify the directory to find the Service instance
- Any additional flags that follow Class.key=value will be passed to the service constructor for the target service and parsed. Please see the [SDK docs](../sdk/README.md) for more details.

## `build`

Build a bento from your services. A bento is a deployment archive that contains your service(s) and all of their dependencies.

**Usage**
```bash
dynamo build [SERVICE]
```

**Arguments**
- `SERVICE` - The service to build. You use file:Class syntax to specify the service.

**Flags**
build
Build a new Bento from current directory.

bentoml build [OPTIONS] [BUILD_CTX]
Options

-f, --bentofile <bentofile>
Path to bentofile. Default to ‘bentofile.yaml’

--version <version>
Bento version. By default the version will be generated.

--label <KEY=VALUE>
(multiple)Bento labels

-o, --output <output>
Output log format. ‘-o tag’ to display only bento tag.

Default:
'default'

Options:
tag | default

--containerize
Whether to containerize the Bento after building. ‘–containerize’ is the shortcut of ‘bentoml build && bentoml containerize’.

## `containerize`

Containerize your services for deployment. 

## `deploy`

Deploy your services to a Kubernetes cluster running the Dynamo Server. 

## `server`

Authenticate and connect to your Dynamo Server running on a cluster. 





