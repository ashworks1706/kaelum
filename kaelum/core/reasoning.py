"""Reasoning generation interface and LLM client abstraction."""

import json
from typing import Any, List, Optional

import httpx
from pydantic import BaseModel, Field
from kaelum.core.config import LLMConfig


class Message(BaseModel):
    """A message in the conversation."""
    role: str = Field(description="Role: 'user', 'assistant', or 'system'")
    content: str = Field(description="Message content")


class LLMClient:
    """Client for any OpenAI-compatible API endpoint."""

    def __init__(self, config: LLMConfig):
        self.config = config
        self.base_url = config.base_url
        self.headers = self._get_headers()

    def _get_headers(self) -> dict:
        """Get headers for API requests."""
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        return headers

    def generate(self, messages: List[Message], stream: bool = False):
        """Generate a response from the LLM.
        
        Args:
            messages: List of messages
            stream: If True, returns a generator that yields chunks. If False, returns complete string.
        """
        url = f"{self.base_url}/chat/completions"
        
        payload = {
            "model": self.config.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "stream": stream,
        }
        
        if stream:
            # Return generator for streaming
            def stream_generator():
                with httpx.stream("POST", url, json=payload, headers=self.headers, timeout=60.0) as response:
                    response.raise_for_status()
                    for line in response.iter_lines():
                        if line.startswith("data: "):
                            data_str = line[6:]  # Remove "data: " prefix
                            if data_str.strip() == "[DONE]":
                                break
                            try:
                                data = json.loads(data_str)
                                if "choices" in data and len(data["choices"]) > 0:
                                    delta = data["choices"][0].get("delta", {})
                                    content = delta.get("content")
                                    if content:
                                        yield content
                            except json.JSONDecodeError:
                                continue
            return stream_generator()
        else:
            # Return complete response
            with httpx.Client(timeout=60.0) as client:
                response = client.post(url, json=payload, headers=self.headers)
                response.raise_for_status()
                data = response.json()
                
                if not data.get("choices") or not data["choices"][0].get("message", {}).get("content"):
                    raise RuntimeError("LLM returned empty response")
                
                return data["choices"][0]["message"]["content"]


class ReasoningGenerator:
    """Generates reasoning traces using an LLM."""

    def __init__(self, llm_client: LLMClient, system_prompt=None, user_template=None):
        self.llm = llm_client
        self.system_prompt = system_prompt or """You are a reasoning assistant. Break down problems into clear, logical steps.
Present your reasoning as a numbered list."""
        self.user_template = user_template or "{query}"

    def generate_reasoning(self, query: str, stream: bool = False):
        """Generate a reasoning trace for a query.
        
        Args:
            query: User query
            stream: If True, yields chunks as they're generated
        """
        # Format the user message using template
        user_message = self.user_template.format(query=query)

        messages = [
            Message(role="system", content=self.system_prompt),
            Message(role="user", content=user_message),
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
