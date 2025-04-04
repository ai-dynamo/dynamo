## Provision S3-compatible cloud object storage:
The Dynamo API Server requires a s3-compatible object store to store Dynamo NIMs.

## Provision PostgreSQL Database
The Dynamo API Server requires a PostgreSQL database to store data entity and version metadata.


## Contributing
### Initialize a new virtual environment with uv
uv venv

### Activate the virtual environment
source .venv/bin/activate

### Install service
uv pip install .

### Start the service
ai-dynamo-store

### (Optional) Development workflow
#### Install dev dependencies
uv pip install -e ".[dev]"

#### Run docker container locally
earthly +docker && docker run -it my-registry/ai-dynamo-store:latest