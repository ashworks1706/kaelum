"""Reasoning generation interface and LLM client abstraction."""

import os
import json
from typing import Any, Dict, List, Optional

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

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
    """Abstraction for LLM clients supporting multiple providers."""

    def __init__(self, config: LLMConfig):
        """Initialize LLM client with configuration."""
        self.config = config
        self._client: Optional[Any] = None
        self._init_client()

    def _init_client(self) -> None:
        """Initialize the appropriate LLM client."""
        if self.config.provider in ["ollama", "vllm", "openai", "custom"]:
            if not OPENAI_AVAILABLE:
                raise ImportError(
                    "OpenAI library required for ollama/vllm/openai/custom providers. "
                    "Install with: pip install openai"
                )
            
            # Ollama, vLLM, and custom models use OpenAI-compatible API
            if self.config.provider == "ollama":
                base_url = self.config.base_url or "http://localhost:11434/v1"
                api_key = "ollama"  # Ollama doesn't need a real key
            elif self.config.provider == "vllm":
                base_url = self.config.base_url or "http://localhost:8000/v1"
                api_key = self.config.api_key or "vllm"
            elif self.config.provider == "custom":
                # Custom local model - you provide base_url and optional api_key
                base_url = self.config.base_url
                if not base_url:
                    raise ValueError(
                        "base_url must be provided for custom provider. "
                        "Point it to your model's OpenAI-compatible endpoint."
                    )
                api_key = self.config.api_key or "custom-model"
            else:  # openai
                base_url = self.config.base_url
                api_key = self.config.api_key or os.getenv("OPENAI_API_KEY")
                if not api_key:
                    raise ValueError("OPENAI_API_KEY environment variable must be set")
            
            self._client = OpenAI(base_url=base_url, api_key=api_key)
            
        elif self.config.provider == "gemini":
            if not GEMINI_AVAILABLE:
                raise ImportError(
                    "Google Generative AI library required for Gemini. "
                    "Install with: pip install google-generativeai"
                )
            
            api_key = self.config.api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY environment variable must be set")
            
            genai.configure(api_key=api_key)
            self._client = genai.GenerativeModel(self.config.model)
        else:
            raise ValueError(f"Unsupported provider: {self.config.provider}")

    def generate(self, messages: List[Message]) -> str:
        """Generate a response from the LLM."""
        if self.config.provider in ["ollama", "vllm", "openai", "custom"]:
            return self._generate_openai_compatible(messages)
        elif self.config.provider == "gemini":
            return self._generate_gemini(messages)
        else:
            raise ValueError(f"Unsupported provider: {self.config.provider}")

    def _generate_openai_compatible(self, messages: List[Message]) -> str:
        """Generate response using OpenAI-compatible API (Ollama, vLLM, OpenAI). Fails loudly."""
        response = self._client.chat.completions.create(
            model=self.config.model,
            messages=[{"role": m.role, "content": m.content} for m in messages],
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
        
        if not response.choices or not response.choices[0].message.content:
            raise RuntimeError("LLM returned empty response")
        
        return response.choices[0].message.content

    def _generate_gemini(self, messages: List[Message]) -> str:
        """Generate response using Google Gemini API. Fails loudly."""
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
        
        response = self._client.generate_content(
            full_prompt,
            generation_config={
                "temperature": self.config.temperature,
                "max_output_tokens": self.config.max_tokens,
            }
        )
        
        if not response.text:
            raise RuntimeError("Gemini returned empty response (possibly content filtered)")
        
        return response.text


class ReasoningGenerator:
    """Generates reasoning traces using an LLM."""

    def __init__(self, llm_client: LLMClient):
        """Initialize with an LLM client."""
        self.llm = llm_client

    def generate_reasoning(self, query: str, context: Optional[str] = None) -> List[str]:
        """Generate a reasoning trace for a query."""
        # Try structured output first (for compatible providers)
        if self.llm.config.provider in ["openai", "ollama", "vllm"]:
            try:
                return self._generate_structured_reasoning(query, context)
            except Exception:
                # Fall back to text parsing if JSON mode fails
                pass
        
        # Fall back to traditional text parsing
        return self._generate_text_reasoning(query, context)
    
    def _generate_structured_reasoning(self, query: str, context: Optional[str] = None) -> List[str]:
        """Generate reasoning using JSON mode for structured output."""
        system_prompt = """You are a reasoning assistant. Break down problems into clear, logical steps.
Respond in JSON format with this structure:
{
    "steps": ["step 1", "step 2", "step 3"],
    "reasoning": "brief explanation"
}"""

        user_prompt = f"Query: {query}"
        if context:
            user_prompt = f"Context: {context}\n\n{user_prompt}"

        messages = [
            Message(role="system", content=system_prompt),
            Message(role="user", content=user_prompt),
        ]

        # For OpenAI-compatible APIs that support JSON mode
        try:
            response = self.llm._client.chat.completions.create(
                model=self.llm.config.model,
                messages=[{"role": m.role, "content": m.content} for m in messages],
                temperature=self.llm.config.temperature,
                max_tokens=self.llm.config.max_tokens,
                response_format={"type": "json_object"}
            )
            result = json.loads(response.choices[0].message.content)
            return result.get("steps", [])
        except Exception:
            # If JSON mode not supported (e.g., some custom models), fall back to text parsing
            raise
    
    def _generate_text_reasoning(self, query: str, context: Optional[str] = None) -> List[str]:
        """Generate reasoning using traditional text parsing."""
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
