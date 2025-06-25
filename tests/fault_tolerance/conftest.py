import pytest


def pytest_addoption(parser):
    parser.addoption("--requests-per-client", type=int, default=100)
    parser.addoption("--clients", type=int, default=10)
    parser.addoption("--no-respawn", action="store_true", default=False)
    parser.addoption("--input-token-length", type=int, default=100)
    parser.addoption("--output-token-length", type=int, default=100)
    parser.addoption("--max-num-seqs", type=int, default=None)
    parser.addoption("--max-retries", type=int, default=1)
    parser.addoption("--display-dynamo-output", action="store_true", default=False)
    parser.addoption("--combine-process-logs", action="store_true", default=False)
    parser.addoption("--hf-hub-offline", action="store_true", default=False)


@pytest.fixture
def display_dynamo_output(request):
    return request.config.getoption("--display-dynamo-output")


@pytest.fixture
def max_retries(request):
    return request.config.getoption("--max-retries")


@pytest.fixture
def max_num_seqs(request):
    return request.config.getoption("--max-num-seqs")


@pytest.fixture
def num_clients(request):
    return request.config.getoption("--clients")


@pytest.fixture
def input_token_length(request):
    return request.config.getoption("--input-token-length")


@pytest.fixture
def output_token_length(request):
    return request.config.getoption("--output-token-length")


@pytest.fixture
def requests_per_client(request):
    return request.config.getoption("--requests-per-client")


@pytest.fixture
def respawn(request):
    return not request.config.getoption("--no-respawn")


@pytest.fixture
def separate_process_logs(request):
    return not request.config.getoption("--combine-process-logs")


@pytest.fixture
def hf_hub_offline(request):
    return request.config.getoption("--hf-hub-offline")
