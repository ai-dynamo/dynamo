// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use clap::{Parser, Subcommand};
use tracing as log;

use dynamo_llm::{http::service::discovery::ModelEntry, model_type::ModelType};
use dynamo_runtime::{
    distributed::DistributedConfig, logging, protocols::Endpoint, raise, DistributedRuntime,
    Result, Runtime, Worker,
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
);

#[derive(Parser)]
#[command(
    author="NVIDIA",
    version="0.2.1",
    about="LLMCTL - Control and manage Dynamo Components",
    long_about = None,
    disable_help_subcommand = true,
)]
struct Cli {
    /// Public Namespace to operate in
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

    // Default namespace to "public" if not specified
    let namespace = cli.public_namespace.unwrap_or_else(|| "public".to_string());

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
                    let (model_type, name) = model_type.into_parts();
                    remove_model(&distributed, namespace.to_string(), model_type, &name).await?;
                }
            }
        }
    }
    Ok(())
}

// Helper functions to handle the actual operations
async fn add_model(
    distributed: &DistributedRuntime,
    namespace: String,
    model_type: ModelType,
    model_name: String,
    endpoint_name: &str,
) -> Result<()> {
    log::debug!(
        "Adding model {} with endpoint {}",
        model_name,
        endpoint_name
    );

    if model_name.starts_with('/') {
        raise!("Model name '{}' cannot start with a slash", model_name);
    }

    let parts: Vec<&str> = endpoint_name.split('.').collect();

    if parts.len() < 2 {
        raise!("Endpoint name '{}' is too short. Format should be 'component.endpoint' or 'namespace.component.endpoint'", endpoint_name);
    } else if parts.len() > 3 {
        raise!("Endpoint name '{}' is too long. Format should be 'component.endpoint' or 'namespace.component.endpoint'", endpoint_name);
    }

    // create model entry
    let endpoint = Endpoint {
        namespace: if parts.len() == 3 {
            parts[0].to_string()
        } else {
            println!(
                "Using the public namespace: {} for model: {}",
                namespace, model_name
            );
            namespace.clone()
        },
        component: parts[parts.len() - 2].to_string(),
        name: parts[parts.len() - 1].to_string(),
    };

    let model = ModelEntry {
        name: model_name.to_string(),
        endpoint,
        model_type,
    };

    // add model to etcd
    let component = distributed.namespace(&namespace)?.component("http")?;
    let path = format!(
        "{}/models/{}/{}",
        component.etcd_path(),
        model_type.as_str(),
        model_name
    );
    let etcd_client = distributed
        .etcd_client()
        .expect("unreachable: llmctl is only useful with dynamic workers");

    // check if model already exists
    let kvs = etcd_client.kv_get_prefix(&path).await?;

    if !kvs.is_empty() {
        println!(
            "{} model {} already exists, please remove it before changing the endpoint.",
            model_type.as_str(),
            model_name,
        );
        list_single_model(distributed, namespace, model_type, model_name).await?;
    } else {
        etcd_client
            .kv_create(path, serde_json::to_vec_pretty(&model)?, None)
            .await?;
        println!("Added new {} model {}", model_type.as_str(), model_name,);
        list_single_model(distributed, namespace, model_type, model_name).await?;
    }

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

async fn list_single_model(
    distributed: &DistributedRuntime,
    namespace: String,
    model_type: ModelType,
    model_name: String,
) -> Result<()> {
    let component = distributed.namespace(&namespace)?.component("http")?;
    let path = format!(
        "{}/models/{}/{}",
        component.etcd_path(),
        model_type.as_str(),
        model_name
    );

    let mut models = Vec::new();
    let etcd_client = distributed
        .etcd_client()
        .expect("llmctl is only useful for dynamic workers");
    let kvs = etcd_client.kv_get_prefix(&path).await?;

    for kv in kvs {
        if let (Ok(_key), Ok(model)) = (
            kv.key_str(),
            serde_json::from_slice::<ModelEntry>(kv.value()),
        ) {
            models.push(ModelRow {
                model_type: model_type.as_str().to_string(),
                name: model_name.clone(),
                namespace: model.endpoint.namespace,
                component: model.endpoint.component,
                endpoint: model.endpoint.name,
            });
        }
    }

    if models.is_empty() {
        println!("Something went wrong, no model was found.");
    } else {
        let table = tabled::Table::new(models);
        println!("{}", table);
    }
    Ok(())
}

async fn list_models(
    distributed: &DistributedRuntime,
    namespace: String,
    model_type: Option<ModelType>,
) -> Result<()> {
    let component = distributed.namespace(&namespace)?.component("http")?;

    let mut models = Vec::new();
    let model_types = match model_type {
        Some(mt) => vec![mt],
        None => vec![ModelType::Chat, ModelType::Completion],
    };

    // TODO: Do we need the model_type in etcd key?

    for mt in model_types {
        let prefix = format!("{}/models/{}/", component.etcd_path(), mt.as_str(),);

        let etcd_client = distributed
            .etcd_client()
            .expect("llmctl is only useful with dynamic workers");
        let kvs = etcd_client.kv_get_prefix(&prefix).await?;

        for kv in kvs {
            if let (Ok(key), Ok(model)) = (
                kv.key_str(),
                serde_json::from_slice::<ModelEntry>(kv.value()),
            ) {
                models.push(ModelRow {
                    model_type: mt.as_str().to_string(),
                    name: key.trim_start_matches(&prefix).to_string(),
                    namespace: model.endpoint.namespace,
                    component: model.endpoint.component,
                    endpoint: model.endpoint.name,
                });
            }
        }
    }

    if models.is_empty() {
        match &model_type {
            Some(mt) => println!(
                "No {} models found in the public namespace: {}",
                mt.as_str(),
                namespace
            ),
            None => println!("No models found in the public namespace: {}", namespace),
        }
    } else {
        let table = tabled::Table::new(models);
        match &model_type {
            Some(mt) => println!(
                "Listing {} models in the public namespace: {}",
                mt.as_str(),
                namespace
            ),
            None => println!("Listing all models in the public namespace: {}", namespace),
        }
        println!("{}", table);
    }
    Ok(())
}

async fn remove_model(
    distributed: &DistributedRuntime,
    namespace: String,
    model_type: ModelType,
    name: &str,
) -> Result<()> {
    let component = distributed.namespace(&namespace)?.component("http")?;
    let prefix = format!(
        "{}/models/{}/{}",
        component.etcd_path(),
        model_type.as_str(),
        name
    );

    log::debug!("deleting key: {}", prefix);

    // get the kvs from etcd
    let mut kv_client = distributed
        .etcd_client()
        .expect("llmctl is only useful with dynamic workers")
        .etcd_client()
        .kv_client();
    match kv_client.delete(prefix.as_bytes(), None).await {
        Ok(_response) => {
            println!(
                "{} model {} removed from the public namespace: {}",
                model_type.as_str(),
                name,
                namespace
            );
        }
        Err(e) => {
            log::error!("Error removing model {}: {}", name, e);
        }
    }
    Ok(())
}
