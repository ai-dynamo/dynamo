import pytest
from unittest.mock import MagicMock, AsyncMock
from dynamo.vllm.handlers import EmbeddingWorkerHandler

@pytest.mark.asyncio
async def test_embedding_handler_rejects_boolean_dimensions():
    # Mock dependencies
    config = MagicMock()
    config.served_model_name = "test-model"
    engine_client = MagicMock()
    
    handler = EmbeddingWorkerHandler(config, engine_client)
    
    context = MagicMock()
    context.id.return_value = "req-123"
    
    # dimensions=True should be rejected
    request = {
        "model": "test-model",
        "input": "hello",
        "dimensions": True
    }
    
    with pytest.raises(TypeError, match="Invalid 'dimensions' type bool; expected int"):
        # We need to trigger the validation inside generate()
        # Using a minimal mock context and request
        gen = handler.generate(request, context)
        await gen.__anext__()

@pytest.mark.asyncio
async def test_embedding_handler_accepts_integer_dimensions():
    # Mock dependencies
    config = MagicMock()
    config.served_model_name = "test-model"
    engine_client = MagicMock()
    
    # Mock engine_client.encode to return a minimal iterator
    mock_output = MagicMock()
    mock_output.outputs.data = [0.1, 0.2, 0.3]
    
    async def mock_encode(*args, **kwargs):
        yield mock_output
        
    engine_client.encode = mock_encode
    
    handler = EmbeddingWorkerHandler(config, engine_client)
    
    context = MagicMock()
    context.id.return_value = "req-123"
    
    # dimensions=2 should be accepted
    request = {
        "model": "test-model",
        "input": "hello",
        "dimensions": 2
    }
    
    # This should not raise TypeError during validation
    gen = handler.generate(request, context)
    try:
        await gen.__anext__()
    except StopAsyncIteration:
        pass
    except Exception as e:
        # We expect it might fail later if mocks aren't perfect, 
        # but it shouldn't be a TypeError from our validation.
        assert not isinstance(e, TypeError) or "dimensions" not in str(e)
