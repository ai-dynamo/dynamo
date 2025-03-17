# Initialize a new virtual environment with uv
uv venv

# Activate the virtual environment
source .venv/bin/activate 

# Install service
uv pip install . 

# Start the service
ai-dynamo-store

# (Optional) Development workflow 
## Install dev dependencies
uv pip install -e ".[dev]"

## Run docker container locally
earthly +docker && dk run -it my-registry/ai-dynamo-store:latest