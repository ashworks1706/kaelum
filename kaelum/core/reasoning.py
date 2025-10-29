"""Reasoning generation interface and LLM client abstraction."""

import os
from typing import Any, Dict, List, Optional

from anthropic import Anthropic
from openai import OpenAI
from pydantic import BaseModel, Field

from kaelum.core.config import LLMConfig


class Message(BaseModel):
    """A message in the conversation."""

    role: str = Field(description="Role: 'user', 'assistant', or 'system'")
    content: str = Field(description="Message content")


class ReasoningResult(BaseModel):
    """Result of a reasoning operation."""

    final: str = Field(description="Final reasoning answer")
    trace: List[str] = Field(default_factory=list, description="Reasoning trace steps")
    verified: bool = Field(default=False, description="Whether reasoning was verified")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="Confidence score")
    diagnostics: Dict[str, Any] = Field(
        default_factory=dict, description="Diagnostic information"
    )


class LLMClient:
    """Abstraction for LLM clients supporting OpenAI and Anthropic."""

    def __init__(self, config: LLMConfig):
        """Initialize LLM client with configuration."""
        self.config = config
        self._client: Optional[Any] = None
        self._init_client()

    def _init_client(self) -> None:
        """Initialize the appropriate LLM client."""
        api_key = self.config.api_key or os.getenv(
            f"{self.config.provider.upper()}_API_KEY"
        )

        if self.config.provider == "openai":
            self._client = OpenAI(api_key=api_key)
        elif self.config.provider == "anthropic":
            self._client = Anthropic(api_key=api_key)
        else:
            raise ValueError(f"Unsupported provider: {self.config.provider}")

    def generate(self, messages: List[Message]) -> str:
        """Generate a response from the LLM."""
        if self.config.provider == "openai":
            return self._generate_openai(messages)
        elif self.config.provider == "anthropic":
            return self._generate_anthropic(messages)
        else:
            raise ValueError(f"Unsupported provider: {self.config.provider}")

    def _generate_openai(self, messages: List[Message]) -> str:
        """Generate response using OpenAI API."""
        response = self._client.chat.completions.create(
            model=self.config.model,
            messages=[{"role": m.role, "content": m.content} for m in messages],
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
        return response.choices[0].message.content or ""

    def _generate_anthropic(self, messages: List[Message]) -> str:
        """Generate response using Anthropic API."""
        # Separate system messages from conversation
        system_msg = ""
        conv_messages = []

        for msg in messages:
            if msg.role == "system":
                system_msg += msg.content + "\n"
            else:
                conv_messages.append({"role": msg.role, "content": msg.content})

        response = self._client.messages.create(
            model=self.config.model,
            messages=conv_messages,
            system=system_msg.strip() if system_msg else None,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
        return response.content[0].text if response.content else ""


class ReasoningGenerator:
    """Generates reasoning traces using an LLM."""

    def __init__(self, llm_client: LLMClient):
        """Initialize with an LLM client."""
        self.llm = llm_client

    def generate_reasoning(self, query: str, context: Optional[str] = None) -> List[str]:
        """Generate a reasoning trace for a query."""
        system_prompt = """You are a reasoning assistant. Break down the problem into clear, logical steps.
Present your reasoning as a numbered list of steps, each on its own line."""

        user_prompt = f"Query: {query}"
        if context:
            user_prompt = f"Context: {context}\n\n{user_prompt}"

        messages = [
            Message(role="system", content=system_prompt),
            Message(role="user", content=user_prompt),
        ]

        response = self.llm.generate(messages)

        # Parse reasoning trace into steps
        trace = []
        for line in response.strip().split("\n"):
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith("-") or line.startswith("•")):
                # Remove numbering/bullets
                step = line.lstrip("0123456789.-•) ").strip()
                if step:
                    trace.append(step)

        return trace if trace else [response.strip()]

    def generate_answer(self, query: str, reasoning_trace: List[str]) -> str:
        """Generate a final answer based on reasoning trace."""
        trace_text = "\n".join(f"{i+1}. {step}" for i, step in enumerate(reasoning_trace))

        messages = [
            Message(
                role="system",
                content="You are a reasoning assistant. Provide a clear, concise final answer based on the reasoning trace.",
            ),
            Message(
                role="user",
                content=f"Query: {query}\n\nReasoning trace:\n{trace_text}\n\nProvide the final answer:",
            ),
        ]

        return self.llm.generate(messages)
