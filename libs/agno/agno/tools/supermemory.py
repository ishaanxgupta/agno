import json
from os import getenv
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

from agno.run import RunContext
from agno.tools import Toolkit
from agno.utils.log import log_debug, log_error

try:
    from supermemory import AsyncSupermemory, Supermemory
except ImportError:
    raise ImportError("`supermemory` package not found. Please install it with `pip install supermemory`")


class SupermemoryTools(Toolkit):
    """Store, search and forget memories with Supermemory.

    Supermemory is a hosted memory API for AI agents. It stores content, extracts
    memories from it, keeps a per-user profile and returns relevant context on search.

    Every call is scoped to a namespace: a user id, project id or any string that
    isolates memories. The toolkit uses the ``namespace`` passed to the constructor
    and falls back to the ``user_id`` of the current run.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        namespace: Optional[str] = None,
        dreaming: Optional[Literal["dynamic", "instant"]] = None,
        enable_add_memory: bool = True,
        enable_search_memory: bool = True,
        enable_get_profile: bool = True,
        enable_forget_memories: bool = True,
        all: bool = False,
        **kwargs,
    ):
        """Initialize SupermemoryTools.

        Args:
            api_key: Supermemory API key. Falls back to the SUPERMEMORY_API_KEY env var.
            namespace: Namespace that scopes every call. Falls back to the run's user_id.
            dreaming: Processing mode for added content. Supermemory defaults to "dynamic",
                where memories can take minutes to appear. "instant" skips the batching,
                so memories are usually searchable within about a minute.
            enable_add_memory: Register the add_memory tool (sync and async).
            enable_search_memory: Register the search_memory tool (sync and async).
            enable_get_profile: Register the get_profile tool (sync and async).
            enable_forget_memories: Register the forget_memories tool (sync and async).
            all: Register all tools regardless of individual flags.
            **kwargs: Additional arguments passed to Toolkit.
        """
        self.api_key = api_key or getenv("SUPERMEMORY_API_KEY")
        if not self.api_key:
            raise ValueError(
                "No Supermemory API key provided. Please set the SUPERMEMORY_API_KEY environment variable or pass it to the SupermemoryTools constructor."
            )
        self.namespace = namespace
        self.dreaming = dreaming

        self.client: Supermemory = Supermemory(api_key=self.api_key)
        self.async_client: AsyncSupermemory = AsyncSupermemory(api_key=self.api_key)

        tools: List[Any] = []
        async_tools: List[Tuple[Callable[..., Any], str]] = []
        if enable_add_memory or all:
            tools.append(self.add_memory)
            async_tools.append((self.aadd_memory, "add_memory"))
        if enable_search_memory or all:
            tools.append(self.search_memory)
            async_tools.append((self.asearch_memory, "search_memory"))
        if enable_get_profile or all:
            tools.append(self.get_profile)
            async_tools.append((self.aget_profile, "get_profile"))
        if enable_forget_memories or all:
            tools.append(self.forget_memories)
            async_tools.append((self.aforget_memories, "forget_memories"))

        super().__init__(name="supermemory_tools", tools=tools, async_tools=async_tools, **kwargs)

    def _get_namespace(self, run_context: RunContext) -> Optional[str]:
        """Resolve the namespace: the constructor value first, then the run's user_id."""
        resolved_namespace = self.namespace
        if not resolved_namespace:
            try:
                resolved_namespace = run_context.user_id
            except Exception:
                pass
        return resolved_namespace or None

    @staticmethod
    def _namespace_error(method_name: str) -> str:
        error_msg = f"Error in {method_name}: A namespace must be provided in the constructor or as the run's user_id."
        log_error(error_msg)
        return error_msg

    def _add_kwargs(self, content: str) -> Dict[str, Any]:
        add_kwargs: Dict[str, Any] = {"content": content}
        if self.dreaming:
            add_kwargs["dreaming"] = self.dreaming
        return add_kwargs

    def add_memory(self, run_context: RunContext, content: str) -> str:
        """Add content to the user's memory. Supermemory extracts memories from it in the background.

        Args:
            content (str): The facts or text that should be remembered.
                Example: "I live in NYC and prefer morning meetings"

        Returns:
            str: JSON with the id and processing status of the stored document, or an error message.
        """
        namespace = self._get_namespace(run_context)
        if not namespace:
            return self._namespace_error("add_memory")
        try:
            log_debug(f"Adding memory to Supermemory namespace: {namespace}")
            result = self.client.add(namespace, **self._add_kwargs(content))
            return json.dumps(result.to_dict(mode="json"))
        except Exception as e:
            log_error(f"Error adding memory: {str(e)}")
            return f"Error adding memory: {e}"

    async def aadd_memory(self, run_context: RunContext, content: str) -> str:
        """Add content to the user's memory. Supermemory extracts memories from it in the background.

        Args:
            content (str): The facts or text that should be remembered.
                Example: "I live in NYC and prefer morning meetings"

        Returns:
            str: JSON with the id and processing status of the stored document, or an error message.
        """
        namespace = self._get_namespace(run_context)
        if not namespace:
            return self._namespace_error("add_memory")
        try:
            log_debug(f"Adding memory to Supermemory namespace: {namespace}")
            result = await self.async_client.add(namespace, **self._add_kwargs(content))
            return json.dumps(result.to_dict(mode="json"))
        except Exception as e:
            log_error(f"Error adding memory: {str(e)}")
            return f"Error adding memory: {e}"

    def search_memory(self, run_context: RunContext, query: str) -> str:
        """Search the user's stored memories and source passages for *query*.

        Args:
            query (str): Natural-language question, topic or phrase to look up.

        Returns:
            str: JSON list of results, or an error message. Results with a `memory` field are
                learned memories; results with a `chunk` field are passages of stored content.
        """
        namespace = self._get_namespace(run_context)
        if not namespace:
            return self._namespace_error("search_memory")
        try:
            log_debug(f"Searching Supermemory namespace: {namespace}")
            response = self.client.search(namespace, query=query)
            return json.dumps([result.to_dict(mode="json") for result in response.results])
        except Exception as e:
            log_error(f"Error searching memory: {str(e)}")
            return f"Error searching memory: {e}"

    async def asearch_memory(self, run_context: RunContext, query: str) -> str:
        """Search the user's stored memories and source passages for *query*.

        Args:
            query (str): Natural-language question, topic or phrase to look up.

        Returns:
            str: JSON list of results, or an error message. Results with a `memory` field are
                learned memories; results with a `chunk` field are passages of stored content.
        """
        namespace = self._get_namespace(run_context)
        if not namespace:
            return self._namespace_error("search_memory")
        try:
            log_debug(f"Searching Supermemory namespace: {namespace}")
            response = await self.async_client.search(namespace, query=query)
            return json.dumps([result.to_dict(mode="json") for result in response.results])
        except Exception as e:
            log_error(f"Error searching memory: {str(e)}")
            return f"Error searching memory: {e}"

    def get_profile(self, run_context: RunContext) -> str:
        """Get the user's profile: stable facts, recent context and custom buckets.

        Returns:
            str: JSON with `static`, `dynamic` and `buckets` lists of `{id, memory}` entries,
                or an error message.
        """
        namespace = self._get_namespace(run_context)
        if not namespace:
            return self._namespace_error("get_profile")
        try:
            log_debug(f"Getting profile for Supermemory namespace: {namespace}")
            response = self.client.profile(namespace)
            return json.dumps(response.profile.to_dict(mode="json"))
        except Exception as e:
            log_error(f"Error getting profile: {str(e)}")
            return f"Error getting profile: {e}"

    async def aget_profile(self, run_context: RunContext) -> str:
        """Get the user's profile: stable facts, recent context and custom buckets.

        Returns:
            str: JSON with `static`, `dynamic` and `buckets` lists of `{id, memory}` entries,
                or an error message.
        """
        namespace = self._get_namespace(run_context)
        if not namespace:
            return self._namespace_error("get_profile")
        try:
            log_debug(f"Getting profile for Supermemory namespace: {namespace}")
            response = await self.async_client.profile(namespace)
            return json.dumps(response.profile.to_dict(mode="json"))
        except Exception as e:
            log_error(f"Error getting profile: {str(e)}")
            return f"Error getting profile: {e}"

    def forget_memories(self, run_context: RunContext, memory_ids: List[str]) -> str:
        """Forget specific memories so they are no longer recalled.

        Args:
            memory_ids (List[str]): Ids of the memories to forget. Use the `id` of results
                from `search_memory` that have a `memory` field, or of `get_profile` entries.

        Returns:
            str: JSON with the `count` of matched memories, the `matches` that were forgotten
                and per-id `errors`, or an error message.
        """
        namespace = self._get_namespace(run_context)
        if not namespace:
            return self._namespace_error("forget_memories")
        if not memory_ids:
            return "Error forgetting memories: At least one memory id must be provided."
        try:
            log_debug(f"Forgetting {len(memory_ids)} memories in Supermemory namespace: {namespace}")
            result = self.client.memories.forget(namespace, ids=memory_ids)
            return json.dumps(result.to_dict(mode="json"))
        except Exception as e:
            log_error(f"Error forgetting memories: {str(e)}")
            return f"Error forgetting memories: {e}"

    async def aforget_memories(self, run_context: RunContext, memory_ids: List[str]) -> str:
        """Forget specific memories so they are no longer recalled.

        Args:
            memory_ids (List[str]): Ids of the memories to forget. Use the `id` of results
                from `search_memory` that have a `memory` field, or of `get_profile` entries.

        Returns:
            str: JSON with the `count` of matched memories, the `matches` that were forgotten
                and per-id `errors`, or an error message.
        """
        namespace = self._get_namespace(run_context)
        if not namespace:
            return self._namespace_error("forget_memories")
        if not memory_ids:
            return "Error forgetting memories: At least one memory id must be provided."
        try:
            log_debug(f"Forgetting {len(memory_ids)} memories in Supermemory namespace: {namespace}")
            result = await self.async_client.memories.forget(namespace, ids=memory_ids)
            return json.dumps(result.to_dict(mode="json"))
        except Exception as e:
            log_error(f"Error forgetting memories: {str(e)}")
            return f"Error forgetting memories: {e}"
