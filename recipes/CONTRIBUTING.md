#  Recipes Contributing Guide

When adding new model recipes, ensure they follow the standard structure:
```text
<model-name>/
├── model-cache/
│   ├── model-cache.yaml
│   └── model-download.yaml
├── <framework>/
│   └── <deployment-mode>/
│       ├── deploy.yaml
│       └── perf.yaml (optional)
└── README.md (optional)
```

## Validation
The `run.sh` script expects this exact directory structure and will validate that the directories and files exist before deployment:
- Model directory exists in `recipes/<model>/`
- Framework is one of the supported frameworks (vllm, sglang, trtllm)
- Framework directory exists in `recipes/<model>/<framework>/`
- Deployment directory exists in `recipes/<model>/<framework>/<deployment>/`
- Required files (`deploy.yaml`) exist in the deployment directory
- If present, performance benchmarks (`perf.yaml`) will be automatically executed