"""Reasoning generation interface and LLM client abstraction."""

import os
from typing import Any, Dict, List, Optional

import google.generativeai as genai
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
    """Abstraction for LLM clients supporting Google Gemini."""

    def __init__(self, config: LLMConfig):
        """Initialize LLM client with configuration."""
        self.config = config
        self._client: Optional[Any] = None
        self._init_client()

    def _init_client(self) -> None:
        """Initialize the appropriate LLM client."""
        api_key = self.config.api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        
        if not api_key:
            raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY environment variable must be set")

        if self.config.provider == "gemini":
            genai.configure(api_key=api_key)
            self._client = genai.GenerativeModel(self.config.model)
        else:
            raise ValueError(f"Unsupported provider: {self.config.provider}")

    def generate(self, messages: List[Message]) -> str:
        """Generate a response from the LLM."""
        if self.config.provider == "gemini":
            return self._generate_gemini(messages)
        else:
            raise ValueError(f"Unsupported provider: {self.config.provider}")

    def _generate_gemini(self, messages: List[Message]) -> str:
        """Generate response using Google Gemini API."""
        # Combine messages into a single prompt
        # Gemini handles system messages as part of the prompt context
        full_prompt = ""
        
        for msg in messages:
            if msg.role == "system":
                full_prompt += f"{msg.content}\n\n"
            elif msg.role == "user":
                full_prompt += f"User: {msg.content}\n"
            elif msg.role == "assistant":
                full_prompt += f"Assistant: {msg.content}\n"
        
        try:
            response = self._client.generate_content(
                full_prompt,
                generation_config={
                    "temperature": self.config.temperature,
                    "max_output_tokens": self.config.max_tokens,
                }
            )
            return response.text if response.text else ""
        except Exception as e:
            # Handle potential content filtering or other errors
            return f"Error generating response: {str(e)}"


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
