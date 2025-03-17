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

## `serve`

Spin up a dynamo dependancy graph locally. You must point toward your file and intended class using file:Class syntax

## `build`

Build a bento from your services. 

## `containerize`

Containerize your services for deployment. 

## `deploy`

Deploy your services to a Kubernetes cluster running the Dynamo Server. 

## `server`

Authenticate and connect to your Dynamo Server running on a cluster. 





