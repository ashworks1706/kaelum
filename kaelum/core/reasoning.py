"""Reasoning generation interface and LLM client abstraction."""

import os
from typing import Any, List, Optional

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from pydantic import BaseModel, Field
from kaelum.core.config import LLMConfig


class Message(BaseModel):
    """A message in the conversation."""
    role: str = Field(description="Role: 'user', 'assistant', or 'system'")
    content: str = Field(description="Message content")


class LLMClient:
    """LLM client supporting Ollama, vLLM, and custom providers."""

    def __init__(self, config: LLMConfig):
        self.config = config
        self._client: Optional[Any] = None
        self._init_client()

    def _init_client(self) -> None:
        """Initialize the LLM client."""
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI library required. Install: pip install openai")
        
        # All providers use OpenAI-compatible API
        if self.config.provider == "ollama":
            base_url = self.config.base_url or "http://localhost:11434/v1"
            api_key = "ollama"
        elif self.config.provider == "vllm":
            base_url = self.config.base_url or "http://localhost:8000/v1"
            api_key = self.config.api_key or "vllm"
        elif self.config.provider == "custom":
            base_url = self.config.base_url
            if not base_url:
                raise ValueError("base_url required for custom provider")
            api_key = self.config.api_key or "custom"
        else:
            raise ValueError(f"Unsupported provider: {self.config.provider}")
        
        self._client = OpenAI(base_url=base_url, api_key=api_key)

    def generate(self, messages: List[Message], stream: bool = False):
        """Generate a response from the LLM.
        
        Args:
            messages: List of messages
            stream: If True, returns a generator that yields chunks. If False, returns complete string.
        """
        response = self._client.chat.completions.create(
            model=self.config.model,
            messages=[{"role": m.role, "content": m.content} for m in messages],
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            stream=stream,
        )
        
        if stream:
            # Return generator for streaming
            def stream_generator():
                for chunk in response:
                    if chunk.choices and chunk.choices[0].delta.content:
                        yield chunk.choices[0].delta.content
            return stream_generator()
        else:
            # Return complete response
            if not response.choices or not response.choices[0].message.content:
                raise RuntimeError("LLM returned empty response")
            return response.choices[0].message.content


class ReasoningGenerator:
    """Generates reasoning traces using an LLM."""

    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    def generate_reasoning(self, query: str, stream: bool = False):
        """Generate a reasoning trace for a query.
        
        Args:
            query: User query
            stream: If True, yields chunks as they're generated
        """
        system_prompt = """You are a reasoning assistant. Break down problems into clear, logical steps.
Present your reasoning as a numbered list."""

        messages = [
            Message(role="system", content=system_prompt),
            Message(role="user", content=query),
        ]

        response = self.llm.generate(messages, stream=stream)

        if stream:
            # Stream chunks directly
            return response
        else:
            # Parse reasoning trace
            trace = []
            for line in response.strip().split("\n"):
                line = line.strip()
                if line and (line[0].isdigit() or line.startswith("-") or line.startswith("•")):
                    step = line.lstrip("0123456789.-•) ").strip()
                    if step:
                        trace.append(step)

            return trace if trace else [response.strip()]

    def generate_answer(self, query: str, reasoning_trace: List[str], stream: bool = False):
        """Generate final answer based on reasoning trace.
        
        Args:
            query: User query
            reasoning_trace: List of reasoning steps
            stream: If True, yields chunks as they're generated
        """
        trace_text = "\n".join(f"{i+1}. {step}" for i, step in enumerate(reasoning_trace))

        messages = [
            Message(
                role="system",
                content="You are a reasoning assistant. Provide a clear, concise final answer.",
            ),
            Message(
                role="user",
                content=f"Query: {query}\n\nReasoning:\n{trace_text}\n\nFinal answer:",
            ),
        ]

        return self.llm.generate(messages, stream=stream)
