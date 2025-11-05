import json
import re
from typing import Any, List, Optional

import httpx
from pydantic import BaseModel, Field
from core.config import LLMConfig


class Message(BaseModel):
    role: str = Field(description="Role: 'user', 'assistant', or 'system'")
    content: str = Field(description="Message content")


class LLMClient:
    def __init__(self, config: LLMConfig):
        self.config = config
        self.base_url = config.base_url
        self.headers = self._get_headers()

    def _get_headers(self) -> dict:
        headers = {"Content-Type": "application/json"}
        
        # Add API key if provided (required for vLLM and other OpenAI-compatible servers)
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        
        return headers

    def generate(self, messages: List[Message], stream: bool = False):
        import logging
        logger = logging.getLogger("kaelum.llm")
        
        url = f"{self.base_url}/chat/completions"
        
        payload = {
            "model": self.config.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "stream": stream,
        }
        
        logger.debug(f"LLM: Calling {url}")
        logger.debug(f"LLM: Model = {self.config.model}")
        logger.debug(f"LLM: Temperature = {self.config.temperature}")
        logger.debug(f"LLM: Max tokens = {payload['max_tokens']}")
        logger.debug(f"LLM: Stream = {stream}")
        logger.debug(f"LLM: Messages count = {len(messages)}")
        
        try:
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
                import time
                start_time = time.time()
                
                # Log the request
                logger.info(f"\n{'─' * 80}")
                logger.info(f"LLM REQUEST:")
                logger.info(f"  Model: {self.config.model}")
                logger.info(f"  Messages: {len(messages)} message(s)")
                if len(messages) > 0:
                    # Log system prompt if present
                    if messages[0].role == "system":
                        system_preview = messages[0].content[:100].replace('\n', ' ')
                        logger.info(f"  System: {system_preview}...")
                    # Log user message
                    user_msg = next((m for m in messages if m.role == "user"), None)
                    if user_msg:
                        user_preview = user_msg.content[:200].replace('\n', ' ')
                        logger.info(f"  User: {user_preview}...")
                
                with httpx.Client(timeout=60.0) as client:
                    response = client.post(url, json=payload, headers=self.headers)
                    response.raise_for_status()
                    data = response.json()
                    
                    elapsed = time.time() - start_time
                    
                    if not data.get("choices") or not data["choices"][0].get("message", {}).get("content"):
                        raise RuntimeError("LLM returned empty response")
                    
                    content = data["choices"][0]["message"]["content"]
                    
                    # Log the response
                    logger.info(f"\nLLM RESPONSE ({elapsed:.2f}s):")
                    logger.info(f"  Length: {len(content)} chars")
                    # Show first 500 chars of response
                    preview = content[:500].replace('\n', '\n  ')
                    logger.info(f"  Content:\n  {preview}")
                    if len(content) > 500:
                        logger.info(f"  ... ({len(content) - 500} more chars)")
                    logger.info(f"{'─' * 80}\n")
                    
                    return content
        
        except httpx.ConnectError as e:
            raise ConnectionError(
                f"Failed to connect to LLM server at {self.base_url}. "
                f"Is the server running? For vLLM, start with: "
                f"python -m vllm.entrypoints.openai.api_server --model {self.config.model} --port <port>"
            ) from e
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                raise PermissionError(
                    f"Authentication failed. For vLLM, you may need to set an API key. "
                    f"Add --api-key parameter or set LLM_API_KEY environment variable."
                ) from e
            elif e.response.status_code == 404:
                raise ValueError(
                    f"Model '{self.config.model}' not found at {self.base_url}. "
                    f"Check that the model name is correct and the server has loaded it."
                ) from e
            else:
                raise RuntimeError(
                    f"LLM server returned error {e.response.status_code}: {e.response.text}"
                ) from e
        except httpx.TimeoutException as e:
            raise TimeoutError(
                f"Request to LLM server timed out after 60 seconds. "
                f"The model may be too slow or overloaded."
            ) from e


class ReasoningGenerator:

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
            trace = self._parse_structured_list(response)
            return trace if trace else [response.strip()]

    def generate_answer(self, query: str, reasoning_trace: List[str], stream: bool = False):
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
    
    def _parse_structured_list(self, text: str) -> List[str]:
        patterns = [
            r'^\d+[\.\)]\s+(.+)',
            r'^[-•*]\s+(.+)',
            r'^[a-zA-Z][\.\)]\s+(.+)',
            r'^(?:Step\s+\d+:)\s*(.+)'
        ]
        
        items = []
        current_item = []
        
        for line in text.split('\n'):
            line = line.strip()
            if not line:
                continue
            
            is_new_item = False
            matched_content = None
            
            for pattern in patterns:
                match = re.match(pattern, line, re.IGNORECASE)
                if match:
                    is_new_item = True
                    matched_content = match.group(1)
                    break
            
            if is_new_item:
                if current_item:
                    items.append(' '.join(current_item))
                current_item = [matched_content]
            else:
                current_item.append(line)
        
        if current_item:
            items.append(' '.join(current_item))
        
        return items

