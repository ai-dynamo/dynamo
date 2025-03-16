# Initialize a new virtual environment with uv
uv venv

# Activate the virtual environment
source .venv/bin/activate 

# Install from the lock file
uv pip install -r uv.lock

# (Optional) Install dev dependencies
uv pip install -e ".[dev]"

# Start the service
ai-dynamo-store