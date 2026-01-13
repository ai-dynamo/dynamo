import type {SidebarsConfig} from '@docusaurus/plugin-content-docs';

/**
 * NVIDIA Dynamo Documentation Sidebars
 * 
 * Structure matches https://docs.nvidia.com/dynamo/latest/
 */
const sidebars: SidebarsConfig = {
  docs: [
    // ==================== Getting Started ====================
    {
      type: 'category',
      label: 'Getting Started',
      collapsed: false,
      items: [
        { type: 'doc', id: 'intro', label: 'Quickstart' },
        { type: 'doc', id: 'installation', label: 'Installation' },
        { type: 'doc', id: 'reference/support-matrix', label: 'Support Matrix' },
        { type: 'doc', id: 'examples', label: 'Examples' },
      ],
    },

    // ==================== Kubernetes Deployment ====================
    {
      type: 'category',
      label: 'Kubernetes Deployment',
      collapsed: false,
      items: [
        {
          type: 'category',
          label: 'Deployment Guide',
          items: [
            { type: 'doc', id: 'kubernetes/README', label: 'Kubernetes Quickstart' },
            { type: 'doc', id: 'kubernetes/installation_guide', label: 'Detailed Installation Guide' },
            { type: 'doc', id: 'kubernetes/dynamo_operator', label: 'Dynamo Operator' },
            { type: 'doc', id: 'kubernetes/deployment/minikube', label: 'Minikube Setup' },
            { type: 'doc', id: 'kubernetes/deployment/dynamomodel-guide', label: 'Managing Models with DynamoModel' },
          ],
        },
        {
          type: 'category',
          label: 'Observability (K8s)',
          items: [
            { type: 'doc', id: 'kubernetes/observability/metrics', label: 'Metrics' },
            { type: 'doc', id: 'kubernetes/observability/logging', label: 'Logging' },
          ],
        },
        {
          type: 'category',
          label: 'Multinode',
          items: [
            { type: 'doc', id: 'kubernetes/deployment/multinode-deployment', label: 'Multinode Deployments' },
            { type: 'doc', id: 'kubernetes/grove', label: 'Grove' },
          ],
        },
      ],
    },

    // ==================== User Guides ====================
    {
      type: 'category',
      label: 'User Guides',
      collapsed: false,
      items: [
        { type: 'doc', id: 'agents/tool-calling', label: 'Tool Calling' },
        { type: 'doc', id: 'multimodal/index', label: 'Multimodality Support' },
        { type: 'doc', id: 'performance/aiconfigurator', label: 'Finding Best Initial Configs' },
        { type: 'doc', id: 'benchmarks/benchmarking', label: 'Dynamo Benchmarking Guide' },
        { type: 'doc', id: 'performance/tuning', label: 'Tuning Disaggregated Performance' },
        { type: 'doc', id: 'development/runtime-guide', label: 'Writing Python Workers in Dynamo' },
        {
          type: 'category',
          label: 'Observability (Local)',
          items: [
            { type: 'doc', id: 'observability/README', label: 'Overview' },
            { type: 'doc', id: 'observability/prometheus-grafana', label: 'Prometheus + Grafana Setup' },
            { type: 'doc', id: 'observability/metrics', label: 'Metrics' },
            { type: 'doc', id: 'observability/metrics-developer-guide', label: 'Metrics Developer Guide' },
            { type: 'doc', id: 'observability/health-checks', label: 'Health Checks' },
            { type: 'doc', id: 'observability/tracing', label: 'Tracing' },
            { type: 'doc', id: 'observability/logging', label: 'Logging' },
          ],
        },
        { type: 'doc', id: 'reference/glossary', label: 'Glossary' },
      ],
    },

    // ==================== Components ====================
    {
      type: 'category',
      label: 'Components',
      collapsed: false,
      items: [
        {
          type: 'category',
          label: 'Backends',
          items: [
            {
              type: 'category',
              label: 'vLLM',
              items: [
                { type: 'doc', id: 'backends/vllm/README', label: 'Overview' },
                { type: 'doc', id: 'backends/vllm/deepseek-r1', label: 'DeepSeek-R1' },
                { type: 'doc', id: 'backends/vllm/gpt-oss', label: 'GPT-OSS' },
                { type: 'doc', id: 'backends/vllm/multi-node', label: 'Multi-Node' },
                { type: 'doc', id: 'backends/vllm/speculative_decoding', label: 'Speculative Decoding' },
                { type: 'doc', id: 'backends/vllm/prompt-embeddings', label: 'Prompt Embeddings' },
                { type: 'doc', id: 'backends/vllm/LMCache_Integration', label: 'LMCache Integration' },
                { type: 'doc', id: 'backends/vllm/prometheus', label: 'Prometheus' },
              ],
            },
            {
              type: 'category',
              label: 'SGLang',
              items: [
                { type: 'doc', id: 'backends/sglang/README', label: 'Overview' },
                { type: 'doc', id: 'backends/sglang/gpt-oss', label: 'GPT-OSS' },
                { type: 'doc', id: 'backends/sglang/sglang-disaggregation', label: 'Disaggregation' },
                { type: 'doc', id: 'backends/sglang/expert-distribution-eplb', label: 'Expert Distribution (EPLB)' },
                { type: 'doc', id: 'backends/sglang/sgl-hicache-example', label: 'HiCache Example' },
                { type: 'doc', id: 'backends/sglang/profiling', label: 'Profiling' },
                { type: 'doc', id: 'backends/sglang/prometheus', label: 'Prometheus' },
              ],
            },
            {
              type: 'category',
              label: 'TensorRT-LLM',
              items: [
                { type: 'doc', id: 'backends/trtllm/README', label: 'Overview' },
                { type: 'doc', id: 'backends/trtllm/gpt-oss', label: 'GPT-OSS' },
                { type: 'doc', id: 'backends/trtllm/kv-cache-transfer', label: 'KV Cache Transfer' },
                { type: 'doc', id: 'backends/trtllm/gemma3_sliding_window_attention', label: 'Gemma3 Sliding Window' },
                { type: 'doc', id: 'backends/trtllm/llama4_plus_eagle', label: 'Llama4 + Eagle' },
                { type: 'doc', id: 'backends/trtllm/multinode/multinode-examples', label: 'Multinode Examples' },
                { type: 'doc', id: 'backends/trtllm/prometheus', label: 'Prometheus' },
              ],
            },
          ],
        },
        { type: 'doc', id: 'router/README', label: 'Router' },
        {
          type: 'category',
          label: 'Planner',
          items: [
            { type: 'doc', id: 'planner/planner_intro', label: 'Overview' },
            { type: 'doc', id: 'planner/sla_planner_quickstart', label: 'SLA Planner Quick Start' },
            { type: 'doc', id: 'benchmarks/sla_driven_profiling', label: 'SLA-Driven Profiling' },
            { type: 'doc', id: 'planner/sla_planner', label: 'SLA-based Planner' },
          ],
        },
        {
          type: 'category',
          label: 'KVBM',
          items: [
            { type: 'doc', id: 'kvbm/kvbm_intro', label: 'Overview' },
            { type: 'doc', id: 'kvbm/kvbm_motivation', label: 'Motivation' },
            { type: 'doc', id: 'kvbm/kvbm_architecture', label: 'Architecture' },
            { type: 'doc', id: 'kvbm/kvbm_components', label: 'Components' },
            { type: 'doc', id: 'kvbm/kvbm_design_deepdive', label: 'Design Deep Dive' },
            { type: 'doc', id: 'kvbm/kvbm_integrations', label: 'Integrations' },
            { type: 'doc', id: 'kvbm/vllm-setup', label: 'KVBM in vLLM' },
            { type: 'doc', id: 'kvbm/trtllm-setup', label: 'KVBM in TRTLLM' },
            { type: 'doc', id: 'backends/vllm/LMCache_Integration', label: 'LMCache Integration' },
            { type: 'doc', id: 'kvbm/kvbm_reading', label: 'Further Reading' },
          ],
        },
      ],
    },

    // ==================== Design Docs ====================
    {
      type: 'category',
      label: 'Design Docs',
      collapsed: false,
      items: [
        { type: 'doc', id: 'design_docs/architecture', label: 'Overall Architecture' },
        { type: 'doc', id: 'design_docs/dynamo_flow', label: 'Architecture Flow' },
        { type: 'doc', id: 'design_docs/disagg_serving', label: 'Disaggregated Serving' },
        { type: 'doc', id: 'design_docs/distributed_runtime', label: 'Distributed Runtime' },
      ],
    },

    // ==================== Additional Resources ====================
    {
      type: 'category',
      label: 'Additional Resources',
      collapsed: true,
      items: [
        {
          type: 'category',
          label: 'Advanced Kubernetes',
          items: [
            { type: 'doc', id: 'kubernetes/deployment/create_deployment', label: 'Create Deployment' },
            { type: 'doc', id: 'kubernetes/autoscaling', label: 'Autoscaling' },
            { type: 'doc', id: 'kubernetes/service_discovery', label: 'Service Discovery' },
            { type: 'doc', id: 'kubernetes/model_caching_with_fluid', label: 'Model Caching with Fluid' },
            { type: 'doc', id: 'kubernetes/fluxcd', label: 'FluxCD' },
            { type: 'doc', id: 'kubernetes/webhooks', label: 'Webhooks' },
            { type: 'doc', id: 'kubernetes/api_reference', label: 'API Reference' },
          ],
        },
        {
          type: 'category',
          label: 'Multimodal Details',
          items: [
            { type: 'doc', id: 'multimodal/vllm', label: 'vLLM' },
            { type: 'doc', id: 'multimodal/sglang', label: 'SGLang' },
            { type: 'doc', id: 'multimodal/trtllm', label: 'TensorRT-LLM' },
          ],
        },
        {
          type: 'category',
          label: 'Router Details',
          items: [
            { type: 'doc', id: 'router/kv_cache_routing', label: 'KV Cache Routing' },
          ],
        },
        {
          type: 'category',
          label: 'Fault Tolerance',
          items: [
            { type: 'doc', id: 'fault_tolerance/request_cancellation', label: 'Request Cancellation' },
            { type: 'doc', id: 'fault_tolerance/request_migration', label: 'Request Migration' },
          ],
        },
        {
          type: 'category',
          label: 'Benchmarks',
          items: [
            { type: 'doc', id: 'benchmarks/kv-router-ab-testing', label: 'KV Router A/B Testing' },
          ],
        },
        {
          type: 'category',
          label: 'Frontends',
          items: [
            { type: 'doc', id: 'frontends/kserve', label: 'KServe' },
          ],
        },
        {
          type: 'category',
          label: 'Development',
          items: [
            { type: 'doc', id: 'development/backend-guide', label: 'Backend Guide' },
          ],
        },
        {
          type: 'category',
          label: 'Guides',
          items: [
            { type: 'doc', id: 'guides/request_plane', label: 'Request Plane' },
            { type: 'doc', id: 'guides/jail_stream_readme', label: 'Jail Stream' },
          ],
        },
        { type: 'doc', id: 'planner/load_planner', label: 'Load Planner' },
        { type: 'doc', id: 'reference/cli', label: 'CLI Reference' },
        {
          type: 'category',
          label: 'API Reference',
          items: [
            {
              type: 'category',
              label: 'NIXL Connect',
              items: [
                { type: 'doc', id: 'api/nixl_connect/README', label: 'Overview' },
                { type: 'doc', id: 'api/nixl_connect/connector', label: 'Connector' },
                { type: 'doc', id: 'api/nixl_connect/device', label: 'Device' },
                { type: 'doc', id: 'api/nixl_connect/device_kind', label: 'Device Kind' },
                { type: 'doc', id: 'api/nixl_connect/descriptor', label: 'Descriptor' },
                { type: 'doc', id: 'api/nixl_connect/read_operation', label: 'Read Operation' },
                { type: 'doc', id: 'api/nixl_connect/write_operation', label: 'Write Operation' },
                { type: 'doc', id: 'api/nixl_connect/readable_operation', label: 'Readable Operation' },
                { type: 'doc', id: 'api/nixl_connect/writable_operation', label: 'Writable Operation' },
                { type: 'doc', id: 'api/nixl_connect/operation_status', label: 'Operation Status' },
                { type: 'doc', id: 'api/nixl_connect/rdma_metadata', label: 'RDMA Metadata' },
              ],
            },
          ],
        },
      ],
    },
  ],
};

export default sidebars;
