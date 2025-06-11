def pytest_addoption(parser):
    parser.addoption("--requests-per-client", type=int, default=10)
    parser.addoption("--clients", type=int, default=10)
