# Mistral plugin for LiveKit Agents

Support for the LLMs from Mistral 🇫🇷

See [https://docs.livekit.io/agents/integrations/llm/mistral/](https://docs.livekit.io/agents/integrations/llm/mistral/) for more information.

## Installation

```bash
git clone https://github.com/frntn/livekit-plugins-mistral livekit-plugins-mistral
pip install ./livekit-plugins-mistral
```

## Pre-requisites

You'll need an API key from Mistral. It can be set as an environment variable: `MISTRAL_API_KEY`

## Usage

This plugin supports two modes:

### Normal Mode (Default)
Use Mistral's standard chat completion API with any available model:

```python
from livekit.plugins.mistral import LLM
from livekit.agents.llm import ChatContext

# Create LLM instance (normal mode is default)
llm = LLM(
    model="mistral-small-latest",
    temperature=0.7
)

# Or explicitly specify normal mode
llm = LLM(
    model="mistral-small-latest", 
    mode="normal",
    temperature=0.7
)

# Use with ChatContext
chat_ctx = ChatContext()
chat_ctx.append(role="user", text="Hello!")

stream = llm.chat(chat_ctx=chat_ctx)
async for chunk in stream:
    if chunk.delta.content:
        print(chunk.delta.content, end="")
```

### Agent Mode
Use Mistral's specialized agents with their agent_id:

```python
from livekit.plugins.mistral import LLM, AgentError
from livekit.agents.llm import ChatContext

# Create LLM instance in agent mode
llm = LLM(
    model="ag:51eea425:20250527:dune:6eaddef8",  # Your agent ID
    mode="agent",
    temperature=0.3
)

# Use the same way as normal mode
chat_ctx = ChatContext()
chat_ctx.append(role="user", text="Hello!")

try:
    stream = llm.chat(chat_ctx=chat_ctx)
    async for chunk in stream:
        if chunk.delta.content:
            print(chunk.delta.content, end="")
except AgentError as e:
    print(f"Agent error: {e}")
```

## Key Features

- **Unified Interface**: Same API for both normal models and specialized agents
- **Automatic Routing**: The plugin automatically uses the appropriate Mistral API based on the mode
- **Stateless Operation**: Both modes work stateless, rebuilding conversation context on each call (consistent with LiveKit patterns)
- **Error Handling**: Specific `AgentError` exception for agent-related issues
- **Full Compatibility**: Maintains backward compatibility with existing code

## Agent Mode Details

When using agent mode:
- The `model` parameter should contain your agent ID (format: `ag:*`)
- Function calls are handled internally by the agent (tools parameter is ignored)
- Conversation history is sent as inputs to maintain context
- Each call starts a new conversation (stateless, like normal mode)
- Agent-specific errors are wrapped in `AgentError` exceptions

## Examples

See `example_agent_usage.py` for complete usage examples of both modes.
