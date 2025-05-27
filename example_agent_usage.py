#!/usr/bin/env python3
"""
Example usage of the Mistral LiveKit plugin with agent mode.

This example demonstrates how to use both normal and agent modes
with the updated Mistral plugin.
"""

import asyncio
import os
from livekit.agents.llm import ChatContext, ChatMessage
from livekit.plugins.mistral import LLM, AgentError

async def example_normal_mode():
    """Example using normal mode (existing functionality)"""
    print("=== Normal Mode Example ===")
    
    # Create LLM instance in normal mode (default)
    llm = LLM(
        model="mistral-small-latest",
        mode="normal",  # explicit, but this is the default
        temperature=0.7
    )
    
    # Create a chat context
    chat_ctx = ChatContext()
    chat_ctx.append(role="user", text="Hello! What's the weather like today?")
    
    try:
        # Get response
        stream = llm.chat(chat_ctx=chat_ctx)
        
        print("Normal mode response:")
        async for chunk in stream:
            if chunk.delta.content:
                print(chunk.delta.content, end="", flush=True)
        print("\n")
        
    except Exception as e:
        print(f"Error in normal mode: {e}")

async def example_agent_mode():
    """Example using agent mode (new functionality)"""
    print("=== Agent Mode Example ===")
    
    # Create LLM instance in agent mode
    # Note: Replace with your actual agent ID
    agent_id = "ag:51eea425:20250527:dune:6eaddef8"  # Example agent ID
    
    llm = LLM(
        model=agent_id,
        mode="agent",
        temperature=0.3
    )
    
    # Create a chat context
    chat_ctx = ChatContext()
    chat_ctx.append(role="user", text="Hello! Can you help me with some research?")
    
    try:
        # Get response
        stream = llm.chat(chat_ctx=chat_ctx)
        
        print("Agent mode response:")
        async for chunk in stream:
            if chunk.delta.content:
                print(chunk.delta.content, end="", flush=True)
        print("\n")
        
    except AgentError as e:
        print(f"Agent-specific error: {e}")
    except Exception as e:
        print(f"General error in agent mode: {e}")

async def example_conversation():
    """Example of a multi-turn conversation in agent mode"""
    print("=== Multi-turn Agent Conversation ===")
    
    # Note: Replace with your actual agent ID
    agent_id = "ag:51eea425:20250527:dune:6eaddef8"  # Example agent ID
    
    llm = LLM(
        model=agent_id,
        mode="agent"
    )
    
    # Create a chat context for conversation
    chat_ctx = ChatContext()
    
    # First message
    chat_ctx.append(role="user", text="What is machine learning?")
    
    try:
        stream = llm.chat(chat_ctx=chat_ctx)
        
        print("User: What is machine learning?")
        print("Agent: ", end="")
        
        response_content = ""
        async for chunk in stream:
            if chunk.delta.content:
                content = chunk.delta.content
                print(content, end="", flush=True)
                response_content += content
        print("\n")
        
        # Add agent response to context
        chat_ctx.append(role="assistant", text=response_content)
        
        # Follow-up question
        chat_ctx.append(role="user", text="Can you give me a simple example?")
        
        stream = llm.chat(chat_ctx=chat_ctx)
        
        print("User: Can you give me a simple example?")
        print("Agent: ", end="")
        
        async for chunk in stream:
            if chunk.delta.content:
                print(chunk.delta.content, end="", flush=True)
        print("\n")
        
    except AgentError as e:
        print(f"Agent-specific error: {e}")
    except Exception as e:
        print(f"Error in conversation: {e}")

async def main():
    """Main function to run examples"""
    
    # Check if API key is set
    if not os.environ.get("MISTRAL_API_KEY"):
        print("Please set MISTRAL_API_KEY environment variable")
        return
    
    print("Mistral LiveKit Plugin - Agent Mode Examples")
    print("=" * 50)
    
    # Run normal mode example
    await example_normal_mode()
    
    print("\n" + "=" * 50 + "\n")
    
    # Run agent mode example
    # Note: This will only work if you have a valid agent ID
    # await example_agent_mode()
    
    # Run conversation example
    # await example_conversation()
    
    print("Examples completed!")

if __name__ == "__main__":
    asyncio.run(main())
