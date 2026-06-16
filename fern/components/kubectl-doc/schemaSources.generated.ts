export type SchemaSource = {
  initial: string;
  full?: string;
};

export const schemaBaseURL = "https://raw.githubusercontent.com/ai-dynamo/dynamo/main/fern/kubectl-doc-schemas";

export const schemaSources: Record<string, SchemaSource> = {
  "DynamoCheckpointSchema0": { initial: "dynamo-checkpoint-schema-0.json", full: "dynamo-checkpoint-schema-0-full.json" },
  "DynamoComponentDeploymentSchema0": { initial: "dynamo-component-deployment-schema-0.json", full: "dynamo-component-deployment-schema-0-full.json" },
  "DynamoComponentDeploymentSchema1": { initial: "dynamo-component-deployment-schema-1.json", full: "dynamo-component-deployment-schema-1-full.json" },
  "DynamoGraphDeploymentRequestSchema0": { initial: "dynamo-graph-deployment-request-schema-0.json", full: "dynamo-graph-deployment-request-schema-0-full.json" },
  "DynamoGraphDeploymentRequestSchema1": { initial: "dynamo-graph-deployment-request-schema-1.json", full: "dynamo-graph-deployment-request-schema-1-full.json" },
  "DynamoGraphDeploymentScalingAdapterSchema0": { initial: "dynamo-graph-deployment-scaling-adapter-schema-0.json", full: "dynamo-graph-deployment-scaling-adapter-schema-0-full.json" },
  "DynamoGraphDeploymentScalingAdapterSchema1": { initial: "dynamo-graph-deployment-scaling-adapter-schema-1.json", full: "dynamo-graph-deployment-scaling-adapter-schema-1-full.json" },
  "DynamoGraphDeploymentSchema0": { initial: "dynamo-graph-deployment-schema-0.json", full: "dynamo-graph-deployment-schema-0-full.json" },
  "DynamoGraphDeploymentSchema1": { initial: "dynamo-graph-deployment-schema-1.json", full: "dynamo-graph-deployment-schema-1-full.json" },
  "DynamoModelSchema0": { initial: "dynamo-model-schema-0.json", full: "dynamo-model-schema-0-full.json" },
  "DynamoWorkerMetadataSchema0": { initial: "dynamo-worker-metadata-schema-0.json", full: "dynamo-worker-metadata-schema-0-full.json" },
};
