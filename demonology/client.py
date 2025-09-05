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
        
        # Server health monitoring
        self.consecutive_failures = 0
        self.last_successful_request = time.time()
        self.server_restart_threshold = 3  # Restart after 3 consecutive failures
        self.health_check_interval = 30.0  # Check health every 30 seconds during issues
        self.is_in_recovery_mode = False
        
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
        repetition_penalty: float = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Build the request payload for the API."""
        # Apply anti-repetition settings if specified
        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        
        if repetition_penalty:
            temperature = min(1.0, temperature + 0.3)  # Increase randomness
        
        # Resource-aware generation during server stress
        if self.consecutive_failures > 0:
            # Reduce token generation to prevent server overload
            max_tokens = min(max_tokens, 512)
            logger.debug(f"Reducing max_tokens to {max_tokens} due to server stress")
        
        payload = {
            "model": kwargs.get("model", self.model),
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": kwargs.get("top_p", self.top_p),
            "stream": stream
        }
        
        # Add repetition penalty if provided
        if repetition_penalty:
            payload["frequency_penalty"] = repetition_penalty
            payload["presence_penalty"] = 0.6  # Discourage repeated topics
        
        # Add tools if provided - Re-enabled with grammar workaround
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
            # Add stop sequences to prevent infinite generation
            payload["stop"] = ["<|endoftext|>", "<|im_end|>", "</s>", "\n\n---"]
            # Lower max_tokens for tool calls to prevent grammar issues
            payload["max_tokens"] = min(max_tokens, 2048)
        
        # Add any additional parameters
        for key in ["frequency_penalty", "presence_penalty", "stop"]:
            if key in kwargs:
                payload[key] = kwargs[key]
        
        return payload
    
    async def _exponential_backoff_delay(self, attempt: int) -> None:
        """Apply exponential backoff with jitter for retry attempts."""
        delay = self.retry_delay * (2 ** attempt) + random.uniform(0, 1)
        await asyncio.sleep(delay)
    
    def _detect_similar_patterns(self, buffer: List[str], threshold: float) -> bool:
        """Detect if content buffer contains similar patterns (not just exact matches)."""
        if len(buffer) < 2:
            return False
        
        # Simple similarity check based on character overlap
        for i in range(len(buffer)):
            for j in range(i + 1, len(buffer)):
                if self._calculate_similarity(buffer[i], buffer[j]) > threshold:
                    return True
        return False
    
    def _calculate_similarity(self, str1: str, str2: str) -> float:
        """Calculate similarity ratio between two strings."""
        if not str1 or not str2:
            return 0.0
        
        # Simple character-based similarity
        set1, set2 = set(str1.lower()), set(str2.lower())
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union if union > 0 else 0.0
    
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
        # Output validation for autonomous coding reliability
        if not content or len(content) > 50000:  # Prevent processing extremely long content
            return content, None
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
                original_function = function_name
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
                                # Determine operation based on original function name
                                path_val = param_value.strip()
                                if original_function in ["list_files", "directory_list", "_explorer", "explorer"]:
                                    arguments["operation"] = "list"
                                    arguments["path"] = path_val or "."
                                elif original_function in ["_reader", "file"]:
                                    arguments["operation"] = "read"
                                    arguments["path"] = path_val
                                else:
                                    # Fallback: Check if it's likely a file or directory
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
                
                # Handle file_operations without parameters - set default operation
                if function_name == "file_operations" and "operation" not in arguments:
                    if original_function in ["list_files", "directory_list", "_explorer", "explorer"]:
                        arguments["operation"] = "list"
                        arguments["path"] = "."
                    elif original_function in ["_reader", "file"]:
                        arguments["operation"] = "read"
                        arguments["path"] = "."
                
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
        # Enhanced repetition detection for autonomous coding reliability
        repetition_buffer = []
        repetition_threshold = 5  # Increased back to 5 - be less aggressive
        max_chunk_length = 1000   # Increased back to 1000 - allow longer responses
        pattern_buffer = []       # Track patterns, not just identical chunks
        similarity_threshold = 0.8  # Detect similar (not just identical) content
        min_content_length = 5    # Minimum content length to check for repetition
        url = f"{self.base_url}/chat/completions"
        payload = self._build_request_payload(messages, stream=True, tools=tools, **kwargs)
        
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            # Check if we should attempt server recovery before trying
            if attempt == 0 and self._should_attempt_server_restart():
                recovery_success = await self._handle_connection_recovery()
                if not recovery_success:
                    # If recovery failed, still try the request but log the issue
                    logger.warning("Server recovery failed, attempting request anyway")
            
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
                                            
                                            # Repetition detection for autonomous coding reliability
                                            content = preprocessed_delta.get("content", "")
                                            if content:
                                                # Check for excessively long chunks (potential repetition)
                                                if len(content) > max_chunk_length:
                                                    logger.error(f"Detected extremely long chunk ({len(content)} chars), likely repetitive generation")
                                                    break
                                                
                                                # Only track chunks with meaningful content for repetition detection
                                                if len(content.strip()) >= min_content_length:
                                                    # Track recent content for repetition detection
                                                    repetition_buffer.append(content.strip())
                                                    if len(repetition_buffer) > repetition_threshold:
                                                        repetition_buffer.pop(0)
                                                    
                                                    # Enhanced repetition detection (only for meaningful content)
                                                    if len(repetition_buffer) >= repetition_threshold:
                                                        # Check for exact repetition
                                                        if len(set(repetition_buffer)) == 1:
                                                            logger.error(f"CRITICAL: Exact repetition loop detected - '{repetition_buffer[0][:50]}...'")
                                                            break
                                                        
                                                        # Check for pattern repetition (similar content)
                                                        if self._detect_similar_patterns(repetition_buffer, similarity_threshold):
                                                            logger.error(f"CRITICAL: Pattern repetition loop detected in last {repetition_threshold} chunks")
                                                            break
                                            
                                            yield preprocessed_delta
                                            
                            except json.JSONDecodeError as e:
                                logger.warning(f"Failed to parse streaming response: {e}")
                                logger.debug(f"Raw line was: {line}")
                                continue
                        
                        # Check for heartbeat timeout
                        if time.time() - last_data_time > heartbeat_timeout:
                            raise DemonologyAPIError("Connection heartbeat timeout - server may be stalled")
                
                # If we reach here, the request was successful
                self._mark_request_success()
                return
            
            except (httpx.HTTPStatusError, httpx.TimeoutException, httpx.RequestError, DemonologyAPIError) as e:
                last_exception = e
                self._mark_request_failure()
                
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
                self._mark_request_failure()
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
            self._mark_request_success()
            return True
        except Exception as e:
            logger.debug(f"Connection test failed: {e}")
            self._mark_request_failure()
            return False
    
    def _mark_request_success(self):
        """Mark a successful request and reset failure tracking."""
        self.consecutive_failures = 0
        self.last_successful_request = time.time()
        if self.is_in_recovery_mode:
            logger.info("Server recovered successfully, exiting recovery mode")
            self.is_in_recovery_mode = False
    
    def _mark_request_failure(self):
        """Mark a failed request and update failure tracking."""
        self.consecutive_failures += 1
        logger.warning(f"Request failed ({self.consecutive_failures} consecutive failures)")
    
    def _should_attempt_server_restart(self) -> bool:
        """Determine if we should attempt to restart the server."""
        return (
            self.consecutive_failures >= self.server_restart_threshold and
            not self.is_in_recovery_mode and
            (time.time() - self.last_successful_request) > 60  # At least 1 minute since last success
        )
    
    async def _handle_connection_recovery(self) -> bool:
        """Handle server recovery when connection issues detected."""
        if not self._should_attempt_server_restart():
            return False
            
        logger.error(f"Server appears unresponsive after {self.consecutive_failures} failures")
        logger.info("Attempting automatic server recovery...")
        
        self.is_in_recovery_mode = True
        recovery_success = await self.restart_server_and_reconnect()
        
        if recovery_success:
            self._mark_request_success()
            return True
        else:
            logger.error("Server recovery failed")
            return False
    
    async def restart_server_and_reconnect(self) -> bool:
        """
        Kill the llama server process, clear VRAM, and wait for restart.
        Returns True if server becomes available, False otherwise.
        """
        import subprocess
        import asyncio
        import os
        
        logger.info("🔄 Attempting to restart llama server due to connection failures...")
        
        try:
            # Step 1: Aggressive process termination
            logger.info("🔪 Killing existing llama-server processes...")
            kill_commands = [
                ['pkill', '-f', 'llama-server'],
                ['pkill', '-f', 'llama.cpp'],
                ['pkill', '-9', '-f', 'llama-server'],  # Force kill if needed
            ]
            
            for cmd in kill_commands:
                try:
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
                    logger.debug(f"Command {' '.join(cmd)} result: {result.returncode}")
                except subprocess.TimeoutExpired:
                    logger.warning(f"Command {' '.join(cmd)} timed out")
                except Exception as e:
                    logger.debug(f"Command {' '.join(cmd)} failed: {e}")
                await asyncio.sleep(2)  # Short delay between kills
            
            # Step 2: Wait for processes to terminate and VRAM to clear
            logger.info("⏳ Waiting for processes to terminate and VRAM to clear...")
            await asyncio.sleep(8)
            
            # Step 3: GPU memory cleanup (if available)
            try:
                # Try to clear GPU memory more aggressively
                gpu_commands = [
                    ['nvidia-smi', '--gpu-reset-ecc=0'],  # Reset ECC if supported
                    ['nvidia-smi', '--reset-gpu-reset'],   # Reset GPU
                ]
                
                for cmd in gpu_commands:
                    try:
                        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
                        logger.debug(f"GPU command {' '.join(cmd)} result: {result.returncode}")
                        await asyncio.sleep(2)
                    except (subprocess.TimeoutExpired, FileNotFoundError):
                        logger.debug(f"GPU command {' '.join(cmd)} not available or timed out")
                    except Exception as e:
                        logger.debug(f"GPU command {' '.join(cmd)} failed: {e}")
                        
            except Exception as e:
                logger.debug(f"GPU cleanup failed: {e}")
            
            # Step 4: Find and execute restart script
            logger.info("🚀 Attempting to automatically restart server...")
            
            # Try multiple possible restart script locations
            restart_scripts = [
                "/home/k/Desktop/Demonology/restart-llama.sh",
                "/home/k/Desktop/Demonology/llama-server-stable.sh",
                "./restart-llama.sh",
                "./llama-server-stable.sh"
            ]
            
            restart_executed = False
            for script_path in restart_scripts:
                if os.path.exists(script_path) and os.access(script_path, os.X_OK):
                    try:
                        logger.info(f"📝 Executing server restart script: {script_path}")
                        result = subprocess.run([script_path], 
                                              capture_output=True, text=True, timeout=45)
                        logger.info(f"Restart script completed with code: {result.returncode}")
                        if result.stdout:
                            logger.debug(f"Script output: {result.stdout.strip()[:200]}")
                        if result.stderr:
                            logger.debug(f"Script errors: {result.stderr.strip()[:200]}")
                        restart_executed = True
                        await asyncio.sleep(15)  # Give server time to start
                        break
                    except subprocess.TimeoutExpired:
                        logger.warning("Restart script timed out, but server may still be starting...")
                        restart_executed = True
                        break
                    except Exception as e:
                        logger.warning(f"Failed to run restart script {script_path}: {e}")
                        continue
            
            if not restart_executed:
                logger.warning("❌ No restart script found. Please manually start your server:")
                logger.warning("Recommended: Run './llama-server-stable.sh' or './restart-llama.sh'")
                # Still wait for manual startup
                await asyncio.sleep(5)
            
            # Step 5: Wait for server to become available
            logger.info("⏱️  Waiting for server to come back online (checking every 4 seconds)...")
            
            # Increased wait time to 3 minutes for model loading
            max_wait_time = 180  # 3 minutes
            check_interval = 4   # Check every 4 seconds
            max_attempts = max_wait_time // check_interval
            
            for attempt in range(max_attempts):
                await asyncio.sleep(check_interval)
                
                # Use a simpler connection test during recovery
                try:
                    async with httpx.AsyncClient(timeout=8.0) as client:
                        response = await client.get(f"{self.base_url.replace('/v1', '')}/health")
                        if response.status_code in [200, 404]:  # 404 means server is up but no /health endpoint
                            logger.info(f"✅ Server health check passed after {(attempt + 1) * check_interval} seconds")
                            # Additional verification with models endpoint
                            try:
                                response = await client.get(f"{self.base_url}/models")
                                if response.status_code == 200:
                                    logger.info("✅ Server models endpoint responding")
                                    return True
                            except:
                                pass
                            return True
                except Exception as e:
                    logger.debug(f"Health check failed: {e}")
                
                # Periodic progress updates
                if attempt % 5 == 0 and attempt > 0:  # Every 20 seconds after first check
                    elapsed = (attempt + 1) * check_interval
                    logger.info(f"⏳ Still waiting for server... ({elapsed}s elapsed)")
                    if attempt == 5:  # After 20 seconds, suggest checking logs
                        logger.info("💡 Check server logs: tail -f /tmp/llama-server.log")
            
            logger.error("❌ Server did not come back online within 3 minutes")
            logger.error("🔧 Manual intervention required - please start your server and try again")
            return False
            
        except Exception as e:
            logger.error(f"💥 Error during server restart: {e}")
            return False


class DemonologyAPIError(Exception):
    """Custom exception for Demonology API errors."""
    pass