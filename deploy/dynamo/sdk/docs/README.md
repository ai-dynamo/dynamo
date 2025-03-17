# Documentation for the Dynamo SDK and CLI 

# Introduction 

Dynamo aims to be the flexible and performant distributed inferencing solution for large-scale deployments. It is an ecosystems of tools, frameworks, and abstractions that makes the design, customization, and deployment of frontier-level models onto datacenter-scale infrastructure easy to reason about and optimized for your specific inferencing workloads. Dynamo's core is written in Rust and contains a set of well-define Python bindings. Examples for those can be found [here](../../../../README.md). 

Dynamo SDK is a layer on top of the core. It is a Python framework that aims to make it easy to create inference graphs and deploy them locally and onto a target K8s cluster. The SDK was heavily inspired by [BentoML](https://github.com/bentoml/BentoML) and we leverage many of its core primatives throughout the SDK. The Dynamo CLI is a companion tool that allows you to spin up an inference pipeline locally, containerize it, and then also deploy it. You can find a toy hello-world example [here](../README.md)

# Installation 

The SDK can be installed using pip:

```bash
pip install ai-dynamo
```

# Core Concepts 

### Defining a Service 
servicedecorator, decorator values, dynamo section 
endpoint decorator link this and above to the python bindings docs
class based (similar to bento and ray)
async on start (working on adding support for all other hooks)
passing in arguments + configuring with a YAML (and how to read those args in the service class)

### Composing services into an inference graph
2 ways
- using the depends. this is the most supported way and is the recommended way when you are looking to deploy. 
- depends for us means edges that will spin up BUT ALSO are an astraction over creating a client for a dynemo endpoint (from above)
- deploy is the maximal set of edges. link lets you define that at runtime for experimentation. our examples using this experimental link syntax

### Learning by example. Aggregated KV routing. 

### CLI flows

1. serve is your go to 
2. then you build a `bento`. explain slightly and share the structure 
3. this is what is used to deploy onto a cluster 