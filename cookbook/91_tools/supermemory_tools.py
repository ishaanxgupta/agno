"""
This example demonstrates how to use the Supermemory toolkit with Agno agents.

Supermemory is a hosted memory API for AI agents. It stores content, extracts memories from it,
keeps a per-user profile and returns relevant context on search.

To get started, please export your Supermemory API key as an environment variable. You can get your Supermemory API key from https://console.supermemory.ai

export SUPERMEMORY_API_KEY=<your-supermemory-api-key>

Run `uv pip install supermemory` to install the dependency.

Memories are scoped to a namespace. With no namespace set, the toolkit uses the agent's user_id.
dreaming="instant" makes added content searchable within about a minute instead of a few.
"""

import asyncio
import time
from textwrap import dedent

from agno.agent import Agent
from agno.models.openai import OpenAIResponses
from agno.tools.supermemory import SupermemoryTools

# ---------------------------------------------------------------------------
# Create Agent
# ---------------------------------------------------------------------------


USER_ID = "jane_doe"
SESSION_ID = "agno_session"

memory_tools = SupermemoryTools(dreaming="instant")
agent = Agent(
    model=OpenAIResponses(id="gpt-5.6-luna"),
    tools=[memory_tools],
    user_id=USER_ID,
    session_id=SESSION_ID,
    markdown=True,
    instructions=dedent(
        """
        You have an evolving memory of this user. Proactively capture new personal details,
        preferences, plans, and relevant context the user shares, and naturally bring them up
        in later conversation. Before answering questions about past details, recall from your memory
        to provide precise and personalized responses. Keep your memory concise: store only
        meaningful information that enhances long-term dialogue. If the user asks you to forget
        something, search for the matching memories and forget them by id.
        """
    ),
)

agent_specific = Agent(
    model=OpenAIResponses(id="gpt-5.6-luna"),
    tools=[
        SupermemoryTools(
            namespace=USER_ID,
            enable_add_memory=False,
            enable_search_memory=True,
            enable_get_profile=True,
            enable_forget_memories=False,
        )
    ],
    markdown=True,
    instructions=dedent(
        """
        You can read the user's profile and search their memories, but cannot add or forget memories.
        Use what you know about the user to give personalized answers.
        """
    ),
)


def wait_for_profile(namespace: str, timeout: float = 120) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        profile = memory_tools.client.profile(namespace).profile
        if profile.static or profile.dynamic:
            return
        time.sleep(5)


# ---------------------------------------------------------------------------
# Run Agent
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Example 1: Using all Supermemory functions ===")
    agent.print_response("I live in NYC and work as a software engineer")
    agent.print_response("I'm going to a concert tomorrow")

    print("Waiting for Supermemory to process the new memories...")
    wait_for_profile(USER_ID)
    agent.print_response("What do you remember about me?")

    print(
        "\n=== Example 2: Using specific Supermemory functions (profile + search only) ==="
    )
    agent_specific.print_response("What do you know about me from my profile?")

    print("\n=== Example 3: Async usage ===")
    asyncio.run(agent.aprint_response("Where do I live?"))
