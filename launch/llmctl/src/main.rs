// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use clap::{Parser, Subcommand};

use dynamo_llm::discovery::{ModelManager, ModelWatcher};
use dynamo_llm::local_model::{LocalModel, ModelNetworkName};
use dynamo_llm::model_type::ModelType;
use dynamo_runtime::component::Endpoint;
use dynamo_runtime::pipeline::RouterMode;
use dynamo_runtime::{
    distributed::DistributedConfig, logging, DistributedRuntime, Result, Runtime, Worker,
};

// Macro to define model types and associated commands
macro_rules! define_type_subcommands {
    ($(($variant:ident, $primary_name:expr, [$($alias:expr),*], $help:expr)),* $(,)?) => {
        #[derive(Subcommand)]
        enum AddCommands {
            $(
                #[doc = $help]
                #[command(name = $primary_name, aliases = [$($alias),*])]
                $variant(AddModelArgs),
            )*
        }

        #[derive(Subcommand)]
        enum ListCommands {
            $(
                #[doc = concat!("List ", $primary_name, " models")]
                #[command(name = $primary_name, aliases = [$($alias),*])]
                $variant,
            )*
        }

        #[derive(Subcommand)]
        enum RemoveCommands {
            $(
                #[doc = concat!("Remove ", $primary_name, " model")]
                #[command(name = $primary_name, aliases = [$($alias),*])]
                $variant(RemoveModelArgs),
            )*
        }

        impl AddCommands {
            fn into_parts(self) -> (ModelType, String, String) {
                match self {
                    $(Self::$variant(args) => (ModelType::$variant, args.model_name, args.endpoint_name)),*
                }
            }
        }

        impl RemoveCommands {
            fn into_parts(self) -> (ModelType, String) {
                match self {
                    $(Self::$variant(args) => (ModelType::$variant, args.model_name)),*
                }
            }
        }

        impl ListCommands {
            fn model_type(&self) -> ModelType {
                match self {
                    $(Self::$variant => ModelType::$variant),*
                }
            }
        }
    }
}

define_type_subcommands!(
    (
        Chat,
        "chat",
        ["chat-model", "chat-models"],
        "Add a chat model"
    ),
    (
        Completion,
        "completion",
        ["completions", "completion-model"],
        "Add a completion model"
    ),
    // Add new model types here:
    (
        Embedding,
        "embedding",
        ["embeddings", "embedding-model"],
        "Add an embedding model"
    )
);

#[derive(Parser)]
#[command(
    author="NVIDIA",
    version="0.2.1",
    about="LLMCTL - Deprecated. Do not use.",
    long_about = None,
    disable_help_subcommand = true,
)]
struct Cli {
    /// Public Namespace to operate in
    /// Do not use this. In fact don't use anything about this file.
    #[arg(short = 'n', long)]
    public_namespace: Option<String>,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// HTTP service related commands
    Http {
        #[command(subcommand)]
        command: HttpCommands,
    },
}

#[derive(Subcommand)]
enum HttpCommands {
    /// Add models
    Add {
        #[command(subcommand)]
        model_type: AddCommands,
    },

    /// List models (all types if no specific type provided)
    List {
        #[command(subcommand)]
        model_type: Option<ListCommands>,
    },

    /// Remove models
    Remove {
        #[command(subcommand)]
        model_type: RemoveCommands,
    },
}

#[derive(Parser)]
struct AddModelArgs {
    /// Model name (e.g. foo/v1)
    #[arg(name = "model-name")]
    model_name: String,
    /// Endpoint name (format: component.endpoint or namespace.component.endpoint)
    #[arg(name = "endpoint-name")]
    endpoint_name: String,
}

/// Common fields for removing any model type
#[derive(Parser)]
struct RemoveModelArgs {
    /// Name of the model to remove
    #[arg(name = "model-name")]
    model_name: String,
}

fn main() -> Result<()> {
    logging::init();
    let cli = Cli::parse();

    // Default namespace to "dynamo" if not specified
    let namespace = cli.public_namespace.unwrap_or_else(|| "dynamo".to_string());

    let worker = Worker::from_settings()?;
    worker.execute(|runtime| async move { handle_command(runtime, namespace, cli.command).await })
}

async fn handle_command(runtime: Runtime, namespace: String, command: Commands) -> Result<()> {
    let settings = DistributedConfig::for_cli();
    let distributed = DistributedRuntime::new(runtime, settings).await?;

    match command {
        Commands::Http { command } => {
            match command {
                HttpCommands::Add { model_type } => {
                    let (model_type, model_name, endpoint_name) = model_type.into_parts();
                    add_model(
                        &distributed,
                        namespace.to_string(),
                        model_type,
                        model_name,
                        &endpoint_name,
                    )
                    .await?;
                }
                HttpCommands::List { model_type } => {
                    match model_type {
                        Some(model_type) => {
                            list_models(
                                &distributed,
                                namespace.clone(),
                                Some(model_type.model_type()),
                            )
                            .await?;
                        }
                        None => {
                            // List all model types
                            list_models(&distributed, namespace.clone(), None).await?;
                        }
                    }
                }
                HttpCommands::Remove { model_type } => {
                    let (_, name) = model_type.into_parts();
                    remove_model(&distributed, &name).await?;
                }
            }
        }
    }
    Ok(())
}

async fn add_model(
    distributed: &DistributedRuntime,
    namespace: String,
    model_type: ModelType,
    model_name: String,
    endpoint_name: &str,
) -> Result<()> {
    tracing::debug!("Adding model {model_name} with endpoint {endpoint_name}");
    if model_name.starts_with('/') {
        anyhow::bail!("Model name '{model_name}' cannot start with a slash");
    }

    let endpoint = endpoint_from_name(distributed, &namespace, endpoint_name)?;

    let mut model = LocalModel::with_name_only(&model_name);
    model.attach(&endpoint, model_type).await?;

    Ok(())
}

#[derive(tabled::Tabled)]
struct ModelRow {
    #[tabled(rename = "MODEL TYPE")]
    model_type: String,
    #[tabled(rename = "MODEL NAME")]
    name: String,
    #[tabled(rename = "NAMESPACE")]
    namespace: String,
    #[tabled(rename = "COMPONENT")]
    component: String,
    #[tabled(rename = "ENDPOINT")]
    endpoint: String,
}

async fn list_models(
    distributed: &DistributedRuntime,
    namespace: String,
    model_type: Option<ModelType>,
) -> Result<()> {
    // We only need a ModelWatcher to call it's all_entries. llmctl is going away so no need to
    // refactor for this.
    let watcher = ModelWatcher::new(
        distributed.clone(),
        Arc::new(ModelManager::new()),
        RouterMode::Random,
    );

    let mut models = Vec::new();
    for entry in watcher.all_entries().await? {
        match (model_type, entry.model_type) {
            (None, _) => {
                // list all
            }
            (Some(want), got) if want == got => {
                // match
            }
            _ => {
                // no match
                continue;
            }
        }
        models.push(ModelRow {
            model_type: entry.model_type.as_str().to_string(),
            name: entry.name,
            namespace: entry.endpoint.namespace,
            component: entry.endpoint.component,
            endpoint: entry.endpoint.name,
        });
    }

    if models.is_empty() {
        match &model_type {
            Some(mt) => println!(
                "No {} models found in namespace: {}",
                mt.as_str(),
                namespace
            ),
            None => println!("No models found in namespace: {}", namespace),
        }
    } else {
        let table = tabled::Table::new(models);
        match &model_type {
            Some(mt) => println!("Listing {} models in namespace: {}", mt.as_str(), namespace),
            None => println!("Listing all models in namespace: {}", namespace),
        }
        println!("{}", table);
    }
    Ok(())
}

async fn remove_model(distributed: &DistributedRuntime, model_name: &str) -> Result<()> {
    // We have to do this manually because normally the etcd lease system does it for us
    let watcher = ModelWatcher::new(
        distributed.clone(),
        Arc::new(ModelManager::new()),
        RouterMode::Random,
    );
    let Some(etcd_client) = distributed.etcd_client() else {
        anyhow::bail!("llmctl is only useful with dynamic workers");
    };
    let active_instances = watcher.entries_for_model(model_name).await?;
    for entry in active_instances {
        let network_name = ModelNetworkName::from_entry(&entry, 0);
        tracing::debug!("deleting key: {network_name}");
        etcd_client
            .kv_delete(network_name.to_string(), None)
            .await?;
    }

    Ok(())
}

fn endpoint_from_name(
    distributed: &DistributedRuntime,
    namespace: &str,
    endpoint_name: &str,
) -> anyhow::Result<Endpoint> {
    let parts: Vec<&str> = endpoint_name.split('.').collect();

    if parts.len() < 2 {
        anyhow::bail!("Endpoint name '{}' is too short. Format should be 'component.endpoint' or 'namespace.component.endpoint'", endpoint_name);
    } else if parts.len() > 3 {
        anyhow::bail!("Endpoint name '{}' is too long. Format should be 'component.endpoint' or 'namespace.component.endpoint'", endpoint_name);
    }

    // TODO previous version sometime hardcoded this to "http", so maybe adjust
    let component_name = parts[parts.len() - 2].to_string();
    let endpoint_name = parts[parts.len() - 1].to_string();

    let component = distributed
        .namespace(namespace)?
        .component(component_name)?;

    Ok(component.endpoint(endpoint_name))
}
