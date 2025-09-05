"""
Demonology API Client - Handles communication with llama.cpp backend.
"""

import asyncio
import json
import logging
import re
from typing import AsyncIterator, Dict, List, Optional, Any
import httpx
import time
import random


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
        timeout: float = 60.0,
        max_retries: int = 3,
        retry_delay: float = 2.0
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # Create timeout configuration with much longer timeouts for stability
        timeout_config = httpx.Timeout(
            connect=30.0,  # Increased connection timeout
            read=timeout * 4,  # Much longer read timeout for streaming
            write=60.0,  # Longer write timeout
            pool=30.0  # Longer pool timeout
        )
        
        # Create client with connection pooling and keep-alive
        limits = httpx.Limits(
            max_keepalive_connections=10,
            max_connections=20,
            keepalive_expiry=300.0  # 5 minutes keep-alive
        )
        
        # Try HTTP/2 if available, fall back to HTTP/1.1
        try:
            self._client = httpx.AsyncClient(
                timeout=timeout_config,
                limits=limits,
                http2=True,  # Enable HTTP/2 for better connection reuse
                headers={"Connection": "keep-alive"}
            )
        except ImportError:
            # Fall back to HTTP/1.1 if h2 package not installed
            logger.warning("HTTP/2 not available (h2 package not installed), using HTTP/1.1")
            self._client = httpx.AsyncClient(
                timeout=timeout_config,
                limits=limits,
                headers={"Connection": "keep-alive"}
            )
    
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
        
        # Add tools if provided - Re-enabled with grammar workaround
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
            # Add stop sequences to prevent infinite generation
            payload["stop"] = ["<|endoftext|>", "<|im_end|>", "</s>", "\n\n---"]
            # Lower max_tokens for tool calls to prevent grammar issues
            payload["max_tokens"] = min(kwargs.get("max_tokens", self.max_tokens), 2048)
        
        # Add any additional parameters
        for key in ["frequency_penalty", "presence_penalty", "stop"]:
            if key in kwargs:
                payload[key] = kwargs[key]
        
        return payload
    
    async def _exponential_backoff_delay(self, attempt: int) -> None:
        """Apply exponential backoff with jitter for retry attempts."""
        delay = self.retry_delay * (2 ** attempt) + random.uniform(0, 1)
        await asyncio.sleep(delay)
    
    def _detect_and_convert_xml_tool_calls(self, content: str) -> tuple[str, Optional[List[Dict[str, Any]]]]:
        """
        Detect and convert XML-style tool calls (Qwen3 format) to OpenAI format.
        
        Handles multiple malformed Qwen3 formats:
        - <tool_call><function=name><parameter=p>v</parameter></function></tool_call>
        - <function=name><parameter=p>v</parameter></function> (missing tool_call wrapper)
        - <tool_call>name<parameter=p>v</parameter></function></tool_call> (missing function= wrapper)
        - <function=name (missing closing >, malformed)
        - Simple format: <function=name> with no parameters
        
        Returns:
            tuple: (cleaned_content, tool_calls_if_found)
        """
        if not content or ("<function=" not in content and "<tool_call>" not in content):
            return content, None
            
        tool_calls = []
        cleaned_content = content
        
        # First, try to match the malformed format: <tool_call>name<parameter=...></parameter></function></tool_call>
        malformed_pattern = r'<tool_call>\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\n?(.*?)</function>\s*</tool_call>'
        malformed_matches = list(re.finditer(malformed_pattern, content, re.DOTALL))
        
        # Then try standard Qwen3 format: <tool_call><function=name>...</function></tool_call>
        tool_call_pattern = r'<tool_call>\s*<function=([a-zA-Z_][a-zA-Z0-9_]*)\s*>?(.*?)</function>\s*</tool_call>'
        full_matches = list(re.finditer(tool_call_pattern, content, re.DOTALL))
        
        # Process malformed matches first (priority over other formats)
        if malformed_matches:
            for i, match in enumerate(reversed(malformed_matches)):
                function_name = match.group(1)
                params_content = match.group(2).strip()
                
                # Map common malformed function names to correct ones
                if function_name in ["_reader", "file", "_explorer", "explorer", "list_files", "directory_list"]:
                    function_name = "file_operations"
                
                # Parse parameters from format: <parameter=name>value</parameter>
                arguments = {}
                if params_content:
                    param_pattern = r'<parameter=([a-zA-Z_][a-zA-Z0-9_]*)\s*>(.*?)</parameter>'
                    param_matches = re.findall(param_pattern, params_content, re.DOTALL)
                    for param_name, param_value in param_matches:
                        # Map parameter names for file operations
                        if function_name == "file_operations":
                            if param_name == "file_path":
                                arguments["operation"] = "read"
                                arguments["path"] = param_value.strip()
                            elif param_name == "path":
                                # For _explorer or directory listing, use list operation
                                path_val = param_value.strip()
                                if path_val in [".", "./"]:
                                    arguments["operation"] = "list"
                                    arguments["path"] = "."
                                else:
                                    # Check if it's likely a file or directory
                                    arguments["operation"] = "list" if not path_val.endswith(('.txt', '.md', '.py', '.js', '.json', '.xml', '.html')) else "read"
                                    arguments["path"] = path_val
                            elif param_name == "files":
                                # Handle array of files - just take the first one for now
                                files_str = param_value.strip()
                                if files_str.startswith('[') and files_str.endswith(']'):
                                    # Parse simple JSON array
                                    try:
                                        files_list = json.loads(files_str)
                                        if files_list and isinstance(files_list, list):
                                            arguments["operation"] = "read"
                                            arguments["path"] = files_list[0]
                                    except:
                                        arguments["operation"] = "read"
                                        arguments["path"] = files_str
                                else:
                                    arguments["operation"] = "read"
                                    arguments["path"] = files_str
                            else:
                                arguments[param_name] = param_value.strip()
                        else:
                            arguments[param_name] = param_value.strip()
                
                # Create OpenAI-style tool call
                tool_call = {
                    "id": f"call_{function_name}_{len(malformed_matches)-i}",
                    "type": "function",
                    "function": {
                        "name": function_name,
                        "arguments": json.dumps(arguments)
                    }
                }
                tool_calls.append(tool_call)
                
                # Remove the malformed tool call from content
                cleaned_content = cleaned_content[:match.start()] + cleaned_content[match.end():]
        
        # If no malformed matches and no full matches, look for partial/malformed patterns
        elif not full_matches:
            # Look for standalone function calls or malformed ones
            # Pattern handles: <function=name>, <function=name>, <function=name (missing >)
            # Also handle extremely malformed: <function=name without closing > at end
            function_pattern = r'<function=([a-zA-Z_][a-zA-Z0-9_]*)\s*>?([^<]*?)(?=<function=|</function>|$)'
            partial_matches = list(re.finditer(function_pattern, content, re.DOTALL))
            
            # If still no matches, try to catch incomplete function calls at end of content
            if not partial_matches:
                incomplete_pattern = r'<function=([a-zA-Z_][a-zA-Z0-9_]*)'
                incomplete_matches = list(re.finditer(incomplete_pattern, content))
                
                if incomplete_matches:
                    partial_matches = incomplete_matches
            
            if not partial_matches:
                return content, None
            
            # Process partial matches
            for i, match in enumerate(reversed(partial_matches)):
                function_name = match.group(1)
                params_content = match.group(2).strip() if len(match.groups()) > 1 and match.group(2) else ""
                
                # Map common function names to correct tool names
                if function_name == "system_info":
                    function_name = "code_execution"
                    arguments = {"language": "bash", "code": "uname -a && lscpu | head -10 && free -h && df -h"}
                elif function_name == "python_environment_check":
                    function_name = "code_execution"
                    arguments = {"language": "python", "code": "import sys; print('Python version:', sys.version)"}
                elif function_name == "run_code":
                    function_name = "code_execution"
                    # Parse parameters if available
                    arguments = {"language": "python", "code": "print('Hello World')"}
                elif function_name in ["_explorer", "explorer", "list_files", "directory_list"]:
                    function_name = "file_operations"
                    arguments = {"operation": "list", "path": "."}
                else:
                    # Simple parsing for malformed cases - assume no parameters for now
                    arguments = {}
                
                # Create OpenAI-style tool call
                tool_call = {
                    "id": f"call_{function_name}_{len(partial_matches)-i}",
                    "type": "function", 
                    "function": {
                        "name": function_name,
                        "arguments": json.dumps(arguments)
                    }
                }
                tool_calls.append(tool_call)
                
                # Remove the malformed function call from content
                cleaned_content = cleaned_content[:match.start()] + cleaned_content[match.end():]
        
        else:
            # Process full Qwen3 format matches
            for i, match in enumerate(reversed(full_matches)):
                function_name = match.group(1)
                params_content = match.group(2).strip()
                
                # Parse parameters from Qwen3 format: <parameter=name>value</parameter>
                arguments = {}
                if params_content:
                    param_pattern = r'<parameter=([a-zA-Z_][a-zA-Z0-9_]*)\s*>(.*?)</parameter>'
                    param_matches = re.findall(param_pattern, params_content, re.DOTALL)
                    for param_name, param_value in param_matches:
                        arguments[param_name] = param_value.strip()
                
                # Create OpenAI-style tool call
                tool_call = {
                    "id": f"call_{function_name}_{len(full_matches)-i}",
                    "type": "function",
                    "function": {
                        "name": function_name,
                        "arguments": json.dumps(arguments)
                    }
                }
                tool_calls.append(tool_call)
                
                # Remove the full tool call from content
                cleaned_content = cleaned_content[:match.start()] + cleaned_content[match.end():]
        
        # Clean up any remaining whitespace
        cleaned_content = cleaned_content.strip()
        
        # Return in proper order (first match = first tool call)
        tool_calls.reverse()
        
        if tool_calls:
            logger.debug(f"Converted {len(tool_calls)} XML tool calls to OpenAI format: {[tc['function']['name'] for tc in tool_calls]}")
        
        return cleaned_content, tool_calls if tool_calls else None
    
    def _preprocess_delta_for_xml_tools(self, delta: Dict[str, Any]) -> Dict[str, Any]:
        """
        Preprocess delta to convert XML-style tool calls to OpenAI format.
        This handles Qwen3's malformed XML tool calls.
        """
        if "content" not in delta or not delta["content"]:
            return delta
            
        content = delta["content"]
        cleaned_content, tool_calls = self._detect_and_convert_xml_tool_calls(content)
        
        if tool_calls:
            # Create a new delta with tool_calls instead of content
            new_delta = {}
            if cleaned_content.strip():
                new_delta["content"] = cleaned_content
            new_delta["tool_calls"] = []
            
            # Convert to streaming format (individual deltas per tool call)
            for i, tool_call in enumerate(tool_calls):
                new_delta["tool_calls"].append({
                    "index": i,
                    "id": tool_call["id"],
                    "type": tool_call["type"],
                    "function": tool_call["function"]
                })
            
            logger.debug(f"Preprocessed XML content to OpenAI tool calls: {[tc['function']['name'] for tc in tool_calls]}")
            return new_delta
        
        # No XML tool calls found, return original delta
        return delta
    
    async def stream_chat_completion(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Stream chat completion responses from the API with retry logic.
        
        Yields response chunks (content or tool calls) as they arrive.
        """
        url = f"{self.base_url}/chat/completions"
        payload = self._build_request_payload(messages, stream=True, tools=tools, **kwargs)
        
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                async with self._client.stream(
                    "POST",
                    url,
                    json=payload,
                    headers={"Content-Type": "application/json", "Connection": "keep-alive"}
                ) as response:
                    response.raise_for_status()
                    
                    # Add heartbeat tracking to detect stalled connections
                    last_data_time = time.time()
                    heartbeat_timeout = 120.0  # 2 minutes without data
                    
                    async for line in response.aiter_lines():
                        if not line.strip():
                            continue
                        
                        last_data_time = time.time()  # Reset heartbeat timer
                        
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
                                        
                                        # Preprocess delta to handle XML-style tool calls (Qwen3 compatibility)
                                        if delta:  # Only process if delta has content
                                            preprocessed_delta = self._preprocess_delta_for_xml_tools(delta)
                                            yield preprocessed_delta
                                            
                            except json.JSONDecodeError as e:
                                logger.warning(f"Failed to parse streaming response: {e}")
                                logger.debug(f"Raw line was: {line}")
                                continue
                        
                        # Check for heartbeat timeout
                        if time.time() - last_data_time > heartbeat_timeout:
                            raise DemonologyAPIError("Connection heartbeat timeout - server may be stalled")
                
                # If we reach here, the request was successful
                return
            
            except (httpx.HTTPStatusError, httpx.TimeoutException, httpx.RequestError, DemonologyAPIError) as e:
                last_exception = e
                
                # Check if this is a retryable error
                retryable = False
                if isinstance(e, httpx.HTTPStatusError):
                    # Retry on server errors (5xx) but not client errors (4xx)
                    retryable = 500 <= e.response.status_code < 600
                elif isinstance(e, (httpx.TimeoutException, httpx.RequestError)):
                    retryable = True
                elif isinstance(e, DemonologyAPIError) and ("500" in str(e) or "timeout" in str(e).lower()):
                    retryable = True
                
                if retryable and attempt < self.max_retries:
                    logger.warning(f"Attempt {attempt + 1} failed, retrying in {self.retry_delay * (2 ** attempt):.1f}s: {e}")
                    await self._exponential_backoff_delay(attempt)
                    continue
                else:
                    # Not retryable or max attempts reached
                    break
            
            except Exception as e:
                last_exception = e
                break
        
        # All retries exhausted, raise the last exception with appropriate formatting
        if isinstance(last_exception, httpx.HTTPStatusError):
            try:
                error_content = await last_exception.response.aread()
                error_text = error_content.decode('utf-8', errors='replace') if error_content else 'No error details'
                if len(error_text) > 200:
                    error_text = error_text[:200] + "..."
                error_msg = f"HTTP error {last_exception.response.status_code}: {error_text}"
            except Exception:
                error_msg = f"HTTP error {last_exception.response.status_code}: Unable to read error details"
            raise DemonologyAPIError(error_msg) from last_exception
        elif isinstance(last_exception, httpx.TimeoutException):
            error_msg = f"Request timeout: The server took too long to respond (after {self.max_retries + 1} attempts)"
            raise DemonologyAPIError(error_msg) from last_exception
        elif isinstance(last_exception, httpx.RequestError):
            error_msg = f"Network error: {str(last_exception)} (after {self.max_retries + 1} attempts)"
            raise DemonologyAPIError(error_msg) from last_exception
        else:
            error_msg = f"Unexpected error during streaming: {str(last_exception)} (after {self.max_retries + 1} attempts)"
            raise DemonologyAPIError(error_msg) from last_exception
    
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
        """Test if the API endpoint is accessible and ready for chat."""
        try:
            # First check if server is responding at all
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.base_url}/models")
                if response.status_code != 200:
                    return False
                    
            # Then test with a minimal chat completion
            messages = [{"role": "user", "content": "Hi"}]
            response = await self.chat_completion(messages, max_tokens=1)
            return True
        except Exception as e:
            logger.debug(f"Connection test failed: {e}")
            return False
    
    async def restart_server_and_reconnect(self) -> bool:
        """
        Kill the llama server process, clear VRAM, and wait for restart.
        Returns True if server becomes available, False otherwise.
        """
        import subprocess
        import asyncio
        
        logger.info("Attempting to restart llama server...")
        
        try:
            # Kill any existing llama-server processes
            logger.info("Killing existing llama-server processes...")
            try:
                result = subprocess.run(['pkill', '-f', 'llama-server'], 
                                      capture_output=True, text=True, timeout=10)
                logger.info(f"pkill result: {result.returncode}")
            except subprocess.TimeoutExpired:
                logger.warning("pkill command timed out")
            except Exception as e:
                logger.warning(f"pkill failed: {e}")
            
            # Wait a moment for processes to die and VRAM to clear
            logger.info("Waiting for processes to terminate...")
            await asyncio.sleep(5)
            
            # Additional VRAM clearing for NVIDIA GPUs
            try:
                result = subprocess.run(['nvidia-smi', '--gpu-reset'], 
                                      capture_output=True, text=True, timeout=15)
                logger.info(f"GPU reset attempted, result: {result.returncode}")
                await asyncio.sleep(3)
            except (subprocess.TimeoutExpired, FileNotFoundError):
                logger.info("nvidia-smi not available or timed out, continuing...")
            except Exception as e:
                logger.warning(f"GPU reset failed: {e}")
            
            # Try to automatically restart the server
            logger.info("Attempting to automatically restart server...")
            
            # Use the dedicated restart script
            import os
            restart_script = "/home/k/Desktop/Demonology/restart-llama.sh"
            
            if os.path.exists(restart_script) and os.access(restart_script, os.X_OK):
                try:
                    logger.info("Executing server restart script...")
                    result = subprocess.run([restart_script], 
                                          capture_output=True, text=True, timeout=30)
                    logger.info(f"Restart script completed with code: {result.returncode}")
                    if result.stdout:
                        logger.info(f"Script output: {result.stdout.strip()}")
                    await asyncio.sleep(10)  # Give server time to start
                except subprocess.TimeoutExpired:
                    logger.warning("Restart script timed out, but server may still be starting...")
                except Exception as e:
                    logger.error(f"Failed to run restart script: {e}")
            else:
                logger.info("Restart script not found. Please manually start your server:")
                logger.info("Run: ./llama-server-stable.sh")
            
            logger.info("Waiting for server to come back online (checking every 3 seconds)...")
            
            # Wait up to 120 seconds for server to come back online (model loading takes time)
            for attempt in range(40):  # 40 * 3 = 120 seconds
                await asyncio.sleep(3)
                if await self.test_connection():
                    logger.info(f"Server reconnected after {(attempt + 1) * 3} seconds")
                    return True
                    
                if attempt % 5 == 0:  # Log every 15 seconds
                    logger.info(f"Still waiting for server... ({(attempt + 1) * 3}s elapsed)")
                    if attempt == 0:
                        logger.info("Check /tmp/llama-server.log for server startup status")
            
            logger.error("Server did not come back online within 120 seconds")
            logger.error("Please manually start your server and try again")
            return False
            
        except Exception as e:
            logger.error(f"Error during server restart: {e}")
            return False


class DemonologyAPIError(Exception):
    """Custom exception for Demonology API errors."""
    pass