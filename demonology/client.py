"""
Demonology API Client - Handles communication with llama.cpp backend.
"""

import asyncio
import json
import logging
from typing import AsyncIterator, Dict, List, Optional, Any
import httpx


logger = logging.getLogger(__name__)


class DemonologyClient:
    """Client for communicating with llama.cpp OpenAI-compatible API."""
    
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8080/v1",
        model: str = "Qwen-3-Coder-30B",
        max_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.95,
        timeout: float = 60.0
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.timeout = timeout
        # Create timeout configuration with longer read timeout for streaming
        timeout_config = httpx.Timeout(
            connect=10.0,  # Connection timeout
            read=timeout * 2,  # Read timeout for streaming responses
            write=30.0,  # Write timeout
            pool=10.0  # Pool timeout
        )
        self._client = httpx.AsyncClient(timeout=timeout_config)
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self._client.aclose()
    
    async def close(self):
        """Close the HTTP client."""
        await self._client.aclose()
    
    def _build_request_payload(
        self,
        messages: List[Dict[str, str]],
        stream: bool = True,
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Build the request payload for the API."""
        payload = {
            "model": kwargs.get("model", self.model),
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "temperature": kwargs.get("temperature", self.temperature),
            "top_p": kwargs.get("top_p", self.top_p),
            "stream": stream
        }
        
        # Add tools if provided
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        
        # Add any additional parameters
        for key in ["frequency_penalty", "presence_penalty", "stop"]:
            if key in kwargs:
                payload[key] = kwargs[key]
        
        return payload
    
    async def stream_chat_completion(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Stream chat completion responses from the API.
        
        Yields response chunks (content or tool calls) as they arrive.
        """
        url = f"{self.base_url}/chat/completions"
        payload = self._build_request_payload(messages, stream=True, tools=tools, **kwargs)
        
        try:
            async with self._client.stream(
                "POST",
                url,
                json=payload,
                headers={"Content-Type": "application/json"}
            ) as response:
                response.raise_for_status()
                
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    
                    if line.startswith("data: "):
                        data_str = line[6:]  # Remove "data: " prefix
                        
                        if data_str.strip() == "[DONE]":
                            break
                        
                        try:
                            data = json.loads(data_str)
                            if "choices" in data and len(data["choices"]) > 0:
                                choice = data["choices"][0]
                                if "delta" in choice:
                                    delta = choice["delta"]
                                    
                                    # Yield the entire delta for processing
                                    if delta:  # Only yield if delta has content
                                        yield delta
                                        
                        except json.JSONDecodeError as e:
                            logger.warning(f"Failed to parse streaming response: {e}")
                            logger.debug(f"Raw line was: {line}")
                            continue
        
        except httpx.HTTPStatusError as e:
            try:
                # Try to read the response content for error details
                error_content = await e.response.aread()
                error_text = error_content.decode('utf-8', errors='replace') if error_content else 'No error details'
                # Limit error text length to prevent overwhelming output
                if len(error_text) > 200:
                    error_text = error_text[:200] + "..."
                error_msg = f"HTTP error {e.response.status_code}: {error_text}"
            except Exception:
                # If we can't read the response, just use the status code
                error_msg = f"HTTP error {e.response.status_code}: Unable to read error details"
            raise DemonologyAPIError(error_msg) from e
        except httpx.TimeoutException as e:
            error_msg = f"Request timeout: The server took too long to respond"
            raise DemonologyAPIError(error_msg) from e
        except httpx.RequestError as e:
            error_msg = f"Network error: {str(e)}"
            raise DemonologyAPIError(error_msg) from e
        except Exception as e:
            error_msg = f"Unexpected error during streaming: {str(e)}"
            raise DemonologyAPIError(error_msg) from e
    
    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> str:
        """
        Get a complete chat completion response.
        
        Returns the full response content.
        """
        url = f"{self.base_url}/chat/completions"
        payload = self._build_request_payload(messages, stream=False, **kwargs)
        
        try:
            response = await self._client.post(
                url,
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            
            data = response.json()
            if (
                "choices" in data
                and len(data["choices"]) > 0
                and "message" in data["choices"][0]
                and "content" in data["choices"][0]["message"]
            ):
                return data["choices"][0]["message"]["content"]
            else:
                raise DemonologyAPIError("Invalid response format from API")
        
        except httpx.HTTPStatusError as e:
            try:
                # Try to read the response content for error details
                error_content = await e.response.aread()
                error_text = error_content.decode('utf-8', errors='replace') if error_content else 'No error details'
                # Limit error text length to prevent overwhelming output
                if len(error_text) > 200:
                    error_text = error_text[:200] + "..."
                error_msg = f"HTTP error {e.response.status_code}: {error_text}"
            except Exception:
                # If we can't read the response, just use the status code
                error_msg = f"HTTP error {e.response.status_code}: Unable to read error details"
            raise DemonologyAPIError(error_msg) from e
        except httpx.TimeoutException as e:
            error_msg = f"Request timeout: The server took too long to respond"
            raise DemonologyAPIError(error_msg) from e
        except httpx.RequestError as e:
            error_msg = f"Network error: {str(e)}"
            raise DemonologyAPIError(error_msg) from e
        except Exception as e:
            error_msg = f"Unexpected error during streaming: {str(e)}"
            raise DemonologyAPIError(error_msg) from e
    
    async def test_connection(self) -> bool:
        """Test if the API endpoint is accessible."""
        try:
            # Try a simple completion request
            messages = [{"role": "user", "content": "Hello"}]
            response = await self.chat_completion(messages, max_tokens=1)
            return True
        except Exception:
            return False


class DemonologyAPIError(Exception):
    """Custom exception for Demonology API errors."""
    pass