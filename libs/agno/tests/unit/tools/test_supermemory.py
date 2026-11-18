import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from supermemory import DocumentRef, ForgetResult, ProfileResponse, SearchResponse

from agno.run import RunContext
from agno.tools.supermemory import SupermemoryTools

MOCK_API_KEY = "test_api_key"

DOCUMENT_REF = {"id": "doc-123", "status": "queued"}
SEARCH_RESULTS = [
    {
        "id": "mem-1",
        "memory": "Lives in NYC",
        "metadata": {},
        "similarity": 0.9,
        "isLatest": True,
        "isInference": False,
        "system": {"updatedAt": "2026-01-01T00:00:00Z"},
    },
    {
        "id": "chunk-1",
        "chunk": "I live in NYC",
        "metadata": {"source": "chat"},
        "similarity": 0.7,
        "isLatest": True,
        "isInference": False,
        "system": {"updatedAt": "2026-01-01T00:00:00Z"},
    },
]
PROFILE = {
    "static": [{"id": "mem-1", "memory": "Lives in NYC"}],
    "dynamic": [{"id": "mem-2", "memory": "Going to a concert tomorrow"}],
    "buckets": {},
}
FORGET_RESULT = {"count": 1, "errors": [], "matches": [{"id": "mem-1", "memory": "Lives in NYC"}]}


def _configure_client(client):
    """Return real Supermemory response models from a mocked client."""
    client.add.return_value = DocumentRef.model_validate(DOCUMENT_REF)
    client.search.return_value = SearchResponse.model_validate({"results": SEARCH_RESULTS, "searchTime": 12.5})
    client.profile.return_value = ProfileResponse.model_validate({"profile": PROFILE})
    client.memories.forget.return_value = ForgetResult.model_validate(FORGET_RESULT)
    return client


@pytest.fixture(autouse=True)
def clear_env(monkeypatch):
    monkeypatch.delenv("SUPERMEMORY_API_KEY", raising=False)


@pytest.fixture
def mock_supermemory():
    with patch("agno.tools.supermemory.Supermemory") as mock:
        mock.return_value = _configure_client(MagicMock())
        yield mock


@pytest.fixture
def mock_async_supermemory():
    with patch("agno.tools.supermemory.AsyncSupermemory") as mock:
        client = MagicMock()
        client.add = AsyncMock()
        client.search = AsyncMock()
        client.profile = AsyncMock()
        client.memories.forget = AsyncMock()
        mock.return_value = _configure_client(client)
        yield mock


@pytest.fixture
def toolkit(mock_supermemory, mock_async_supermemory):
    return SupermemoryTools(api_key=MOCK_API_KEY, namespace="user_alex")


@pytest.fixture
def toolkit_no_namespace(mock_supermemory, mock_async_supermemory):
    return SupermemoryTools(api_key=MOCK_API_KEY)


@pytest.fixture
def dummy_run_context():
    """Return a minimal RunContext without a user_id."""
    return RunContext(run_id="test-run-id", session_id="test-session-id", user_id=None)


@pytest.fixture
def user_run_context():
    """Return a RunContext that carries a user_id."""
    return RunContext(run_id="test-run-id", session_id="test-session-id", user_id="context_user_123")


class TestSupermemoryToolkit:
    def test_init_with_api_key(self, toolkit, mock_supermemory, mock_async_supermemory):
        mock_supermemory.assert_called_once_with(api_key=MOCK_API_KEY)
        mock_async_supermemory.assert_called_once_with(api_key=MOCK_API_KEY)
        assert toolkit.client == mock_supermemory.return_value
        assert toolkit.async_client == mock_async_supermemory.return_value

    def test_init_with_env_api_key(self, monkeypatch, mock_supermemory, mock_async_supermemory):
        monkeypatch.setenv("SUPERMEMORY_API_KEY", "env_api_key")
        toolkit = SupermemoryTools()
        assert toolkit.api_key == "env_api_key"
        mock_supermemory.assert_called_once_with(api_key="env_api_key")

    def test_init_no_api_key(self):
        with pytest.raises(ValueError, match="No Supermemory API key provided"):
            SupermemoryTools()

    def test_default_tools_registered(self, toolkit):
        expected = ["add_memory", "search_memory", "get_profile", "forget_memories"]
        assert list(toolkit.functions.keys()) == expected
        assert list(toolkit.async_functions.keys()) == expected

    def test_enable_flags(self, mock_supermemory, mock_async_supermemory):
        toolkit = SupermemoryTools(
            api_key=MOCK_API_KEY,
            enable_get_profile=False,
            enable_forget_memories=False,
        )
        assert list(toolkit.functions.keys()) == ["add_memory", "search_memory"]
        assert list(toolkit.async_functions.keys()) == ["add_memory", "search_memory"]

    def test_all_overrides_enable_flags(self, mock_supermemory, mock_async_supermemory):
        toolkit = SupermemoryTools(
            api_key=MOCK_API_KEY,
            enable_add_memory=False,
            enable_search_memory=False,
            enable_get_profile=False,
            enable_forget_memories=False,
            all=True,
        )
        assert len(toolkit.functions) == 4
        assert len(toolkit.async_functions) == 4

    def test_get_namespace_from_constructor(self, toolkit, dummy_run_context):
        assert toolkit._get_namespace(dummy_run_context) == "user_alex"

    def test_get_namespace_from_run_context(self, toolkit_no_namespace, user_run_context):
        assert toolkit_no_namespace._get_namespace(user_run_context) == "context_user_123"

    def test_get_namespace_constructor_takes_priority(self, toolkit, user_run_context):
        assert toolkit._get_namespace(user_run_context) == "user_alex"

    def test_get_namespace_not_provided(self, toolkit_no_namespace, dummy_run_context):
        assert toolkit_no_namespace._get_namespace(dummy_run_context) is None

    def test_add_memory(self, toolkit, dummy_run_context):
        result = toolkit.add_memory(dummy_run_context, content="I live in NYC")
        toolkit.client.add.assert_called_once_with("user_alex", content="I live in NYC")
        assert json.loads(result) == DOCUMENT_REF

    def test_add_memory_with_run_context_namespace(self, toolkit_no_namespace, user_run_context):
        result = toolkit_no_namespace.add_memory(user_run_context, content="I live in NYC")
        toolkit_no_namespace.client.add.assert_called_once_with("context_user_123", content="I live in NYC")
        assert json.loads(result) == DOCUMENT_REF

    def test_add_memory_with_dreaming(self, mock_supermemory, mock_async_supermemory, dummy_run_context):
        toolkit = SupermemoryTools(api_key=MOCK_API_KEY, namespace="user_alex", dreaming="instant")
        toolkit.add_memory(dummy_run_context, content="I live in NYC")
        toolkit.client.add.assert_called_once_with("user_alex", content="I live in NYC", dreaming="instant")

    def test_add_memory_no_namespace(self, toolkit_no_namespace, dummy_run_context):
        result = toolkit_no_namespace.add_memory(dummy_run_context, content="I live in NYC")
        assert result.startswith("Error in add_memory: A namespace must be provided")
        toolkit_no_namespace.client.add.assert_not_called()

    def test_add_memory_error(self, toolkit, dummy_run_context):
        toolkit.client.add.side_effect = Exception("Test add error")
        result = toolkit.add_memory(dummy_run_context, content="I live in NYC")
        assert result == "Error adding memory: Test add error"

    def test_search_memory(self, toolkit, dummy_run_context):
        result = toolkit.search_memory(dummy_run_context, query="where do I live?")
        toolkit.client.search.assert_called_once_with("user_alex", query="where do I live?")
        assert json.loads(result) == SEARCH_RESULTS

    def test_search_memory_with_run_context_namespace(self, toolkit_no_namespace, user_run_context):
        result = toolkit_no_namespace.search_memory(user_run_context, query="where do I live?")
        toolkit_no_namespace.client.search.assert_called_once_with("context_user_123", query="where do I live?")
        assert json.loads(result) == SEARCH_RESULTS

    def test_search_memory_no_results(self, toolkit, dummy_run_context):
        toolkit.client.search.return_value = SearchResponse.model_validate({"results": [], "searchTime": 1.0})
        result = toolkit.search_memory(dummy_run_context, query="unknown topic")
        assert json.loads(result) == []

    def test_search_memory_no_namespace(self, toolkit_no_namespace, dummy_run_context):
        result = toolkit_no_namespace.search_memory(dummy_run_context, query="where do I live?")
        assert result.startswith("Error in search_memory: A namespace must be provided")
        toolkit_no_namespace.client.search.assert_not_called()

    def test_search_memory_error(self, toolkit, dummy_run_context):
        toolkit.client.search.side_effect = Exception("Test search error")
        result = toolkit.search_memory(dummy_run_context, query="where do I live?")
        assert result == "Error searching memory: Test search error"

    def test_get_profile(self, toolkit, dummy_run_context):
        result = toolkit.get_profile(dummy_run_context)
        toolkit.client.profile.assert_called_once_with("user_alex")
        assert json.loads(result) == PROFILE

    def test_get_profile_with_run_context_namespace(self, toolkit_no_namespace, user_run_context):
        result = toolkit_no_namespace.get_profile(user_run_context)
        toolkit_no_namespace.client.profile.assert_called_once_with("context_user_123")
        assert json.loads(result) == PROFILE

    def test_get_profile_no_namespace(self, toolkit_no_namespace, dummy_run_context):
        result = toolkit_no_namespace.get_profile(dummy_run_context)
        assert result.startswith("Error in get_profile: A namespace must be provided")
        toolkit_no_namespace.client.profile.assert_not_called()

    def test_get_profile_error(self, toolkit, dummy_run_context):
        toolkit.client.profile.side_effect = Exception("Test profile error")
        result = toolkit.get_profile(dummy_run_context)
        assert result == "Error getting profile: Test profile error"

    def test_forget_memories(self, toolkit, dummy_run_context):
        result = toolkit.forget_memories(dummy_run_context, memory_ids=["mem-1"])
        toolkit.client.memories.forget.assert_called_once_with("user_alex", ids=["mem-1"])
        assert json.loads(result) == FORGET_RESULT

    def test_forget_memories_with_run_context_namespace(self, toolkit_no_namespace, user_run_context):
        result = toolkit_no_namespace.forget_memories(user_run_context, memory_ids=["mem-1"])
        toolkit_no_namespace.client.memories.forget.assert_called_once_with("context_user_123", ids=["mem-1"])
        assert json.loads(result) == FORGET_RESULT

    def test_forget_memories_no_ids(self, toolkit, dummy_run_context):
        result = toolkit.forget_memories(dummy_run_context, memory_ids=[])
        assert result == "Error forgetting memories: At least one memory id must be provided."
        toolkit.client.memories.forget.assert_not_called()

    def test_forget_memories_no_namespace(self, toolkit_no_namespace, dummy_run_context):
        result = toolkit_no_namespace.forget_memories(dummy_run_context, memory_ids=["mem-1"])
        assert result.startswith("Error in forget_memories: A namespace must be provided")
        toolkit_no_namespace.client.memories.forget.assert_not_called()

    def test_forget_memories_error(self, toolkit, dummy_run_context):
        toolkit.client.memories.forget.side_effect = Exception("Test forget error")
        result = toolkit.forget_memories(dummy_run_context, memory_ids=["mem-1"])
        assert result == "Error forgetting memories: Test forget error"


class TestSupermemoryToolkitAsync:
    @pytest.mark.asyncio
    async def test_aadd_memory(self, toolkit, dummy_run_context):
        result = await toolkit.aadd_memory(dummy_run_context, content="I live in NYC")
        toolkit.async_client.add.assert_awaited_once_with("user_alex", content="I live in NYC")
        toolkit.client.add.assert_not_called()
        assert json.loads(result) == DOCUMENT_REF

    @pytest.mark.asyncio
    async def test_aadd_memory_with_run_context_namespace(self, toolkit_no_namespace, user_run_context):
        await toolkit_no_namespace.aadd_memory(user_run_context, content="I live in NYC")
        toolkit_no_namespace.async_client.add.assert_awaited_once_with("context_user_123", content="I live in NYC")

    @pytest.mark.asyncio
    async def test_aadd_memory_no_namespace(self, toolkit_no_namespace, dummy_run_context):
        result = await toolkit_no_namespace.aadd_memory(dummy_run_context, content="I live in NYC")
        assert result.startswith("Error in add_memory: A namespace must be provided")
        toolkit_no_namespace.async_client.add.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_aadd_memory_error(self, toolkit, dummy_run_context):
        toolkit.async_client.add.side_effect = Exception("Test add error")
        result = await toolkit.aadd_memory(dummy_run_context, content="I live in NYC")
        assert result == "Error adding memory: Test add error"

    @pytest.mark.asyncio
    async def test_asearch_memory(self, toolkit, dummy_run_context):
        result = await toolkit.asearch_memory(dummy_run_context, query="where do I live?")
        toolkit.async_client.search.assert_awaited_once_with("user_alex", query="where do I live?")
        assert json.loads(result) == SEARCH_RESULTS

    @pytest.mark.asyncio
    async def test_asearch_memory_error(self, toolkit, dummy_run_context):
        toolkit.async_client.search.side_effect = Exception("Test search error")
        result = await toolkit.asearch_memory(dummy_run_context, query="where do I live?")
        assert result == "Error searching memory: Test search error"

    @pytest.mark.asyncio
    async def test_aget_profile(self, toolkit, dummy_run_context):
        result = await toolkit.aget_profile(dummy_run_context)
        toolkit.async_client.profile.assert_awaited_once_with("user_alex")
        assert json.loads(result) == PROFILE

    @pytest.mark.asyncio
    async def test_aget_profile_error(self, toolkit, dummy_run_context):
        toolkit.async_client.profile.side_effect = Exception("Test profile error")
        result = await toolkit.aget_profile(dummy_run_context)
        assert result == "Error getting profile: Test profile error"

    @pytest.mark.asyncio
    async def test_aforget_memories(self, toolkit, dummy_run_context):
        result = await toolkit.aforget_memories(dummy_run_context, memory_ids=["mem-1"])
        toolkit.async_client.memories.forget.assert_awaited_once_with("user_alex", ids=["mem-1"])
        assert json.loads(result) == FORGET_RESULT

    @pytest.mark.asyncio
    async def test_aforget_memories_no_ids(self, toolkit, dummy_run_context):
        result = await toolkit.aforget_memories(dummy_run_context, memory_ids=[])
        assert result == "Error forgetting memories: At least one memory id must be provided."
        toolkit.async_client.memories.forget.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_aforget_memories_error(self, toolkit, dummy_run_context):
        toolkit.async_client.memories.forget.side_effect = Exception("Test forget error")
        result = await toolkit.aforget_memories(dummy_run_context, memory_ids=["mem-1"])
        assert result == "Error forgetting memories: Test forget error"
