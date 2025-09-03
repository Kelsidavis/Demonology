"""
Demonology CLI - Main command line interface and application logic.
"""

import asyncio
import sys
import signal
import json
import os
import re
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import click
import logging

from .client import DemonologyClient, DemonologyAPIError
from .config import Config
from .ui import DemonologyUI
from .themes import ThemeName
from .tools import ToolRegistry

logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DemonologyApp:
    """Main Demonology application."""

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.ui = DemonologyUI(self.config)
        self.client: Optional[DemonologyClient] = None
        self.conversation_history: List[Dict[str, Any]] = []
        self.running = False
        self._auto_tool_used_this_turn = False  # guard to avoid repeated fallback in one turn

        safe_root = self._resolve_safe_root()
        self.tool_registry = ToolRegistry(safe_root=safe_root)

    def _resolve_safe_root(self) -> Path:
        """
        Resolve a stable SAFE_ROOT in this order:
        1) config.tools.working_directory (if set)
        2) $GRIMOIRE_SAFE_ROOT (if set)
        3) project root (parent of this package dir)
        4) current working directory
        """
        # 1) config
        cfg = getattr(self.config.tools, "working_directory", "") or ""
        if cfg:
            return Path(cfg).expanduser().resolve()
        # 2) env
        env_root = os.environ.get("GRIMOIRE_SAFE_ROOT", "")
        if env_root:
            return Path(env_root).expanduser().resolve()
        # 3) project root (…/Demonology), i.e. parent of the package directory
        try:
            pkg_dir = Path(__file__).resolve().parent          # …/demonology
            project_root = pkg_dir.parent                      # …/Demonology
            if project_root.exists():
                return project_root
        except Exception:
            pass
        # 4) cwd
        return Path.cwd().resolve()

    async def initialize(self):
        """Initialize the application."""
        self.client = DemonologyClient(
            base_url=self.config.api.base_url,
            model=self.config.api.model,
            max_tokens=self.config.api.max_tokens,
            temperature=self.config.api.temperature,
            top_p=self.config.api.top_p,
            timeout=self.config.api.timeout
        )

    async def cleanup(self):
        if self.client:
            await self.client.close()
        self.ui.cleanup()

    def handle_signal(self, signum, frame):
        self.ui.display_info("Received signal, shutting down gracefully...")
        self.running = False

    async def test_connection(self) -> bool:
        try:
            await self.ui.start_loading("Testing connection to the mystical realm...")
            ok = await self.client.test_connection()
            await self.ui.stop_loading()
            if ok:
                self.ui.display_success("⛧ Connection to the daemon established⛧")
                return True
            self.ui.display_error("Failed to reach the mystical realm")
            return False
        except Exception as e:
            await self.ui.stop_loading()
            self.ui.display_error(f"Connection ritual failed: {str(e)}")
            return False

    def process_command(self, user_input: str) -> bool:
        """Return False to quit."""
        if not user_input.startswith('/'):
            return True

        command = user_input[1:].strip().lower()
        parts = command.split()
        cmd = parts[0] if parts else ""

        if cmd in ['quit', 'exit', 'q']:
            return False
        elif cmd == 'help':
            self._show_help()
        elif cmd == 'status':
            self.ui.display_status()
        elif cmd == 'themes':
            self.ui.show_theme_preview()
        elif cmd == 'theme':
            if len(parts) > 1:
                self.ui.change_theme(parts[1])
            else:
                self.ui.display_error("Usage: /theme <theme_name>")
        elif cmd == 'permissive':
            self._toggle_permissive_mode()
        elif cmd == 'model':
            if len(parts) > 1:
                self._change_model(' '.join(parts[1:]))
            else:
                self.ui.display_info(f"Current model: {self.config.api.model}")
        elif cmd == 'save':
            if len(parts) > 1:
                self._save_conversation(' '.join(parts[1:]))
            else:
                self.ui.display_error("Usage: /save <filename>")
        elif cmd == 'load':
            if len(parts) > 1:
                self._load_conversation(' '.join(parts[1:]))
            else:
                self.ui.display_error("Usage: /load <filename>")
        elif cmd == 'clear':
            self._clear_conversation()
        elif cmd == 'config':
            if len(parts) > 1 and parts[1] == 'edit':
                self._edit_config()
            else:
                self._show_config()
        elif cmd == 'debug':
            self.ui.display_info("Debug command will be handled in next iteration")
        elif cmd == 'tools':
            self._show_tools()
        else:
            self.ui.display_error(f"Unknown command: /{cmd}. Type /help for available commands.")

        return True

    def _show_help(self):
        self.ui.console.print("""
[bold]Demonology Commands:[/bold]

/help                 Show this help message
/quit, /exit, /q      Exit Demonology
/status               Show current status
/themes               List available themes
/theme <name>         Change theme (amethyst, infernal, stygian)
/permissive           Toggle permissive mode
/model <name>         Change or show current model
/save <filename>      Save conversation to file
/load <filename>      Load conversation from file
/clear                Clear conversation history
/config               Show current configuration
/config edit          Edit configuration file
/debug                Debug API response
/tools                List available tools

[dim]Type your message to chat with the mystical AI.[/dim]
""")
        self.ui.console.print()

    def _toggle_permissive_mode(self):
        self.config.ui.permissive_mode = not self.config.ui.permissive_mode
        self.config.save()
        if self.config.ui.permissive_mode:
            from .ui import PERMISSIVE_MODE_MESSAGES
            import random
            self.ui.display_error(random.choice(PERMISSIVE_MODE_MESSAGES), mystical=True)
        else:
            self.ui.display_success("Seals restored. Permissive mode disabled.")

    def _change_model(self, model_name: str):
        old_model = self.config.api.model
        self.config.api.model = model_name
        self.config.save()
        if self.client:
            self.client.model = model_name
        self.ui.display_success(f"Model changed from {old_model} to {model_name}")

    def _save_conversation(self, filename: str):
        try:
            conversations_dir = self.config.get_conversations_dir()
            if not filename.endswith('.json'):
                filename += '.json'
            filepath = conversations_dir / filename
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.conversation_history, f, indent=2, ensure_ascii=False)
            self.ui.display_success(f"Conversation saved to {filepath}")
        except Exception as e:
            self.ui.display_error(f"Failed to save conversation: {str(e)}")

    def _load_conversation(self, filename: str):
        try:
            conversations_dir = self.config.get_conversations_dir()
            if not filename.endswith('.json'):
                filename += '.json'
            filepath = conversations_dir / filename
            if not filepath.exists():
                self.ui.display_error(f"Conversation file not found: {filepath}")
                return
            with open(filepath, 'r', encoding='utf-8') as f:
                self.conversation_history = json.load(f)
            self.ui.display_success(f"Conversation loaded from {filepath}")
            for message in self.conversation_history:
                self.ui.conversation.display_message(message['role'], message.get('content', ''))
        except Exception as e:
            self.ui.display_error(f"Failed to load conversation: {str(e)}")

    def _clear_conversation(self):
        self.conversation_history.clear()
        self.ui.display_success("Conversation history cleared")

    def _show_config(self):
        cfg = self.config
        self.ui.console.print(f"""
[bold]API Configuration:[/bold]
Base URL: {cfg.api.base_url}
Model: {cfg.api.model}
Max Tokens: {cfg.api.max_tokens}
Temperature: {cfg.api.temperature}
Top P: {cfg.api.top_p}

[bold]UI Configuration:[/bold]
Theme: {cfg.ui.theme}
Permissive Mode: {cfg.ui.permissive_mode}
Auto Save: {cfg.ui.auto_save_conversations}

[bold]Tools Configuration:[/bold]
Enabled: {cfg.tools.enabled}
Allowed Tools: {', '.join(cfg.tools.allowed_tools)}
Working Directory (config): {getattr(cfg.tools, "working_directory", "") or "(unset, defaults to project root)"}

Config file: {cfg.config_path}
""")
        self.ui.console.print()

    def _edit_config(self):
        import subprocess
        editor = os.environ.get('EDITOR', 'nano')
        try:
            subprocess.run([editor, str(self.config.config_path)])
            self.config.load()
            self.ui.display_success("Configuration reloaded")
        except Exception as e:
            self.ui.display_error(f"Failed to edit config: {str(e)}")

    def _show_tools(self):
        tools = self.tool_registry.list_tools()
        self.ui.console.print("[bold]Available Tools:[/bold]")
        for tool in tools:
            status = "✓" if tool["enabled"] and tool["available"] else "✗"
            self.ui.console.print(f"  {status} [yellow]{tool['name']}[/yellow]: {tool['description']}")
        self.ui.console.print()
        if self.config.tools.enabled:
            openai_tools = self.tool_registry.to_openai_tools_format()
            self.ui.console.print(f"[dim]Total tools sent to model: {len(openai_tools)}[/dim]")

    # ---------------- Heuristic fallback for NL commands ---------------------

    def _heuristic_from_user(self, user_text: str) -> Optional[Tuple[str, Dict[str, Any]]]:
        """
        Best-effort mapping of simple English requests to file_operations.
        Returns (tool_name, arguments) or None.
        """
        t = user_text.strip().lower()

        # create folder
        m = re.search(r'\b(?:create|make|new|add)\s+(?:a\s+)?(?:folder|directory)(?:\s+(?:named|called))?\s+([a-z0-9_.\-\/]+)', t)
        if m:
            name = m.group(1).strip('/ ')
            return ("file_operations", {"operation": "create_directory", "path": f"{name}/"})

        # create file in folder
        m = re.search(r'\b(?:create|make|new|add)\s+(?:a\s+)?file(?:\s+(?:named|called))?\s+([a-z0-9_.\-]+)\s+(?:in|inside|under)\s+([a-z0-9_.\-\/]+)', t)
        if m:
            fname, d = m.group(1), m.group(2).strip('/ ')
            return ("file_operations", {"operation": "create_or_write_file", "path": f"{d}/{fname}", "content": ""})

        # create file
        m = re.search(r'\b(?:create|make|new|add)\s+(?:a\s+)?file(?:\s+(?:named|called))?\s+([a-z0-9_.\-\/]+)', t)
        if m:
            return ("file_operations", {"operation": "create_or_write_file", "path": m.group(1), "content": ""})

        # list
        m = re.search(r'\b(?:list|show)\s+(?:files|contents)(?:\s+(?:of|in)\s+([a-z0-9_.\-\/]+))?', t)
        if m:
            path = m.group(1) or "."
            return ("file_operations", {"operation": "list", "path": path})

        # read
        m = re.search(r'\bread\s+([a-z0-9_.\-\/]+)', t)
        if m:
            return ("file_operations", {"operation": "read", "path": m.group(1)})

        # delete
        m = re.search(r'\b(?:delete|remove|rm)\s+(?:file|folder|directory)?\s*([a-z0-9_.\-\/]+)', t)
        if m:
            p = m.group(1)
            if p.endswith('/'):
                return ("file_operations", {"operation": "delete_directory", "path": p, "recursive": False})
            return ("file_operations", {"operation": "delete_file", "path": p})

        return None

    async def _attempt_auto_tool(self) -> Optional[str]:
        """
        Try to infer and execute a tool call from the last user message.
        Runs at most once per user turn.
        """
        if self._auto_tool_used_this_turn:
            return None

        last_user = next((m["content"] for m in reversed(self.conversation_history) if m["role"] == "user"), "")
        guess = self._heuristic_from_user(last_user or "")
        if not guess or not self.config.tools.enabled:
            return None

        self._auto_tool_used_this_turn = True  # guard re-entry this turn

        tool_name, args = guess
        if tool_name == "file_operations" and args.get("operation") == "create_directory":
            p = args.get("path", "")
            if p and not p.endswith("/"):
                args["path"] = p + "/"

        tool_id = "auto_1"
        self.conversation_history.append({
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": tool_id,
                "type": "function",
                "function": {"name": tool_name, "arguments": json.dumps(args)}
            }]
        })

        await self.ui.start_loading(f"Summoning infernal tool: {tool_name} from the depths...")
        result = await self.tool_registry.execute_tool(tool_name, **args)
        await self.ui.stop_loading()
        self.conversation_history.append({"role": "tool", "tool_call_id": tool_id, "content": json.dumps(result)})

        tools = self.tool_registry.to_openai_tools_format()
        follow = self.client.stream_chat_completion(self.conversation_history, tools=tools)
        return await self._handle_streaming_response(follow)

    # ---------------- Streaming & tool-calls ----------------

    async def _handle_streaming_response_with_loading(self, response_stream) -> str:
        """Handle streaming response while managing loading animation."""
        content_buffer = ""
        tool_calls: List[Dict[str, Any]] = []
        first_chunk_received = False

        try:
            async for delta in response_stream:
                # Stop loading animation on first content chunk
                if not first_chunk_received and ("content" in delta or "tool_calls" in delta):
                    await self.ui.stop_loading()
                    first_chunk_received = True

                if "content" in delta and delta["content"]:
                    chunk = delta["content"]
                    content_buffer += chunk
                    self.ui.console.print(chunk, end="")
                
                elif "tool_calls" in delta:
                    for tool_call_delta in delta["tool_calls"]:
                        index = tool_call_delta.get("index", 0)
                        
                        # Ensure we have enough tool_calls in our list
                        while len(tool_calls) <= index:
                            tool_calls.append({
                                "id": "",
                                "function": {"name": "", "arguments": ""},
                                "type": "function"
                            })
                        
                        # Update tool call data
                        if "id" in tool_call_delta:
                            tool_calls[index]["id"] = tool_call_delta["id"]
                        
                        if "function" in tool_call_delta:
                            function_data = tool_call_delta["function"]
                            if "name" in function_data:
                                tool_calls[index]["function"]["name"] = function_data["name"]
                            if "arguments" in function_data:
                                tool_calls[index]["function"]["arguments"] += function_data["arguments"]

        except Exception as e:
            logger.exception("Error processing streaming response")
            await self.ui.stop_loading()  # Ensure loading stops on error
            
            # For 500 errors, try to recover gracefully
            if "500" in str(e):
                self.ui.display_error("Server error occurred. Conversation history may be corrupted. Continuing...")
                # Clear conversation history to recover from 500 errors
                user_messages = [msg for msg in self.conversation_history if msg.get("role") == "user"]
                if user_messages:
                    self.conversation_history = user_messages[-3:]  # Keep last 3 user messages
                logger.warning("Cleared conversation history to recover from 500 error")
            
            if content_buffer.strip():
                return content_buffer
            else:
                raise

        # If we never received any chunks, stop loading now
        if not first_chunk_received:
            await self.ui.stop_loading()

        # If we collected tool calls, execute them
        if tool_calls and any(tc.get("function", {}).get("name") for tc in tool_calls):
            if content_buffer.strip():
                self.ui.console.print()  # Add newline after content
            return await self._execute_collected_tool_calls(content_buffer, tool_calls)
        
        return content_buffer
    
    async def _execute_collected_tool_calls(self, content_buffer: str, tool_calls: List[Dict[str, Any]]) -> str:
        """Execute collected tool calls and continue the conversation."""
        self.ui.console.print("\n[bold red]👹 UNLEASHING HELLISH TOOLS FROM THE ABYSS 👹[/bold red]")
        
        # Add assistant message with tool calls to history
        # Ensure tool calls are properly formatted
        valid_tool_calls = []
        for tc in tool_calls:
            if tc.get("function", {}).get("name") and tc.get("id"):
                valid_tool_calls.append({
                    "id": tc["id"],
                    "type": "function",
                    "function": {
                        "name": tc["function"]["name"],
                        "arguments": tc["function"].get("arguments", "{}")
                    }
                })
        
        assistant_message = {
            "role": "assistant",
            "content": content_buffer if content_buffer.strip() else None,
            "tool_calls": valid_tool_calls
        }
        self.conversation_history.append(assistant_message)
        
        # Execute each tool call
        for tool_call in tool_calls:
            if not tool_call.get("function", {}).get("name"):
                continue
                
            function_name = tool_call["function"]["name"]
            tool_id = tool_call.get("id", f"call_{function_name}")
            
            try:
                arguments_str = tool_call["function"]["arguments"]
                logger.warning(f"Raw arguments string for {function_name}: {repr(arguments_str[:200])}...")
                
                if not arguments_str.strip():
                    arguments = {}
                else:
                    # Try to fix common JSON issues
                    cleaned_str = arguments_str.strip()
                    # Handle incomplete JSON strings
                    if cleaned_str.count('"') % 2 != 0:
                        logger.warning(f"Incomplete JSON string detected, attempting to close")
                        cleaned_str = cleaned_str.rstrip(',') + '"}'
                    
                    arguments = json.loads(cleaned_str)
                    
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse tool arguments for {function_name}: {repr(arguments_str[:200])} - Error: {e}")
                # Try to extract key parameters manually as fallback
                arguments = self._extract_fallback_arguments(function_name, arguments_str)
            
            self.ui.console.print(f"[bold magenta]🔥 Binding demon [{function_name}] with dark ritual: {arguments} 🔥[/bold magenta]")
            await self.ui.start_loading(f"Invoking daemon of {function_name}... blood sacrifice accepted...")
            result = await self.tool_registry.execute_tool(function_name, **arguments)
            await self.ui.stop_loading()
            
            if result.get("success", False):
                self.ui.console.print(f"[bold green]👹 DEMON [{function_name}] BOWS TO YOUR WILL - POWER CHANNELED 👹[/bold green]")
            else:
                self.ui.console.print(f"[bold red]💀 DEMON [{function_name}] DEFIES THE SUMMONING - CURSE BACKFIRED: {result.get('error', 'Ancient evil')} 💀[/bold red]")
            
            # Add tool result to history
            tool_result_message = {
                "role": "tool",
                "tool_call_id": tool_id,
                "content": json.dumps(result)
            }
            self.conversation_history.append(tool_result_message)
        
        # Get follow-up response from AI
        tools = self.tool_registry.to_openai_tools_format()
        
        # Truncate conversation history if it's too long to avoid 500 errors
        max_history_length = 20
        if len(self.conversation_history) > max_history_length:
            # Keep system message if present, and recent messages
            system_msgs = [msg for msg in self.conversation_history if msg.get("role") == "system"]
            recent_msgs = self.conversation_history[-(max_history_length-len(system_msgs)):]
            self.conversation_history = system_msgs + recent_msgs
            logger.warning(f"Truncated conversation history to {len(self.conversation_history)} messages")
        
        logger.warning(f"Conversation history before follow-up request: {json.dumps(self.conversation_history[-3:], indent=2)}")
        follow_stream = self.client.stream_chat_completion(self.conversation_history, tools=tools)
        follow_response = await self._handle_streaming_response(follow_stream)
        
        # Return both the original content and follow-up
        full_response = content_buffer
        if follow_response.strip():
            if full_response.strip():
                full_response += "\n\n" + follow_response
            else:
                full_response = follow_response
                
        return full_response
    
    def _extract_fallback_arguments(self, function_name: str, arguments_str: str) -> dict:
        """Extract arguments from malformed JSON as fallback."""
        arguments = {}
        
        try:
            # Common patterns for each tool type
            if function_name == "file_operations":
                # Try to extract operation
                import re
                op_match = re.search(r'"operation":\s*"([^"]*)"', arguments_str)
                if op_match:
                    arguments["operation"] = op_match.group(1)
                
                path_match = re.search(r'"path":\s*"([^"]*)"', arguments_str)
                if path_match:
                    arguments["path"] = path_match.group(1)
                
                content_match = re.search(r'"content":\s*"([^"]*)"', arguments_str)
                if content_match:
                    arguments["content"] = content_match.group(1)
                    
            elif function_name == "web_search":
                query_match = re.search(r'"query":\s*"([^"]*)"', arguments_str)
                if query_match:
                    arguments["query"] = query_match.group(1)
                    
            elif function_name == "project_planning":
                name_match = re.search(r'"project_name":\s*"([^"]*)"', arguments_str)
                if name_match:
                    arguments["project_name"] = name_match.group(1)
                    
                desc_match = re.search(r'"project_description":\s*"([^"]*)"', arguments_str)
                if desc_match:
                    arguments["project_description"] = desc_match.group(1)
                    
        except Exception as e:
            logger.error(f"Fallback argument extraction failed: {e}")
            
        logger.warning(f"Extracted fallback arguments for {function_name}: {arguments}")
        return arguments

    async def _handle_streaming_response(self, response_stream) -> str:
        """Collect streaming content and tool-call deltas; execute when needed."""
        content_buffer = ""
        tool_calls: List[Dict[str, Any]] = []

        try:
            async for delta in response_stream:
                if "content" in delta and delta["content"]:
                    chunk = delta["content"]
                    content_buffer += chunk
                    self.ui.console.print(chunk, end="")
                elif "tool_calls" in delta:
                    for tool_call_delta in delta["tool_calls"]:
                        index = tool_call_delta.get("index", 0)
                        while len(tool_calls) <= index:
                            tool_calls.append({"id": "", "type": "function", "function": {"name": "", "arguments": ""}})
                        if "id" in tool_call_delta:
                            tool_calls[index]["id"] = tool_call_delta["id"]
                        if "function" in tool_call_delta:
                            fn = tool_call_delta["function"]
                            if "name" in fn:
                                tool_calls[index]["function"]["name"] += fn["name"]
                            if "arguments" in fn:
                                tool_calls[index]["function"]["arguments"] += fn["arguments"]
        except Exception as e:
            logger.exception("Error processing streaming response")
            
            # For 500 errors, try to recover gracefully
            if "500" in str(e):
                self.ui.display_error("Server error occurred. Conversation history may be corrupted. Continuing...")
                # Clear conversation history to recover from 500 errors
                user_messages = [msg for msg in self.conversation_history if msg.get("role") == "user"]
                if user_messages:
                    self.conversation_history = user_messages[-3:]  # Keep last 3 user messages
                logger.warning("Cleared conversation history to recover from 500 error")
            
            # If there's an error during streaming, return whatever content we have
            if content_buffer.strip():
                return content_buffer
            else:
                # Re-raise the error if we have no content to return
                raise

        # If we got normal content and NO function calls, try one fallback.
        if content_buffer.strip() and not tool_calls:
            self.ui.console.print()
            maybe = await self._attempt_auto_tool()
            if maybe is not None:
                return maybe
            return content_buffer

        # If we have function calls, execute them and continue the turn.
        if tool_calls:
            return await self._execute_tool_calls(tool_calls)

        return ""

    async def _execute_tool_calls(self, tool_calls: List[Dict[str, Any]]) -> str:
        """Execute tool calls and then continue the turn."""
        self.ui.console.print("\n[bold red]⛧ DEMONIC TOOL INVOCATION RITUAL ⛧[/bold red]")

        tool_results: List[Dict[str, Any]] = []
        assistant_message = {"role": "assistant", "content": None, "tool_calls": tool_calls}
        self.conversation_history.append(assistant_message)

        for tc in tool_calls:
            tool_id = tc.get("id", "")
            function_name = tc.get("function", {}).get("name", "")
            try:
                arguments_raw = tc.get("function", {}).get("arguments", "") or "{}"
                try:
                    logger.warning(f"Raw arguments string for {function_name}: {repr(arguments_raw[:200])}...")
                    
                    if not arguments_raw.strip():
                        arguments = {}
                    else:
                        # Try to fix common JSON issues
                        cleaned_str = arguments_raw.strip()
                        # Handle incomplete JSON strings
                        if cleaned_str.count('"') % 2 != 0:
                            logger.warning(f"Incomplete JSON string detected, attempting to close")
                            cleaned_str = cleaned_str.rstrip(',') + '"}'
                        
                        arguments = json.loads(cleaned_str)
                        
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse tool arguments for {function_name}: {repr(arguments_raw[:200])} - Error: {e}")
                    # Try to extract key parameters manually as fallback
                    arguments = self._extract_fallback_arguments(function_name, arguments_raw)

                if function_name == "file_operations" and arguments.get("operation") == "create_directory":
                    p = arguments.get("path", "")
                    if p and not p.endswith("/"):
                        arguments["path"] = p + "/"

                self.ui.console.print(f"[bold magenta]🔥 Binding demon [{function_name}] with ritual parameters: {arguments} 🔥[/bold magenta]")
                await self.ui.start_loading(f"Channeling dark powers through {function_name}... souls harvested...")
                result = await self.tool_registry.execute_tool(function_name, **arguments)
                await self.ui.stop_loading()

                if result.get("success", False):
                    self.ui.console.print(f"[bold green]😈 DEMON [{function_name}] HAS OBEYED - RITUAL SUCCESSFUL 😈[/bold green]")
                else:
                    self.ui.console.print(f"[bold red]🔥 DEMON [{function_name}] REBELLED - DARK MAGIC FAILED: {result.get('error', 'Infernal error')} 🔥[/bold red]")

                tool_results.append({
                    "role": "tool",
                    "tool_call_id": tool_id,
                    "content": json.dumps(result)
                })

            except Exception as e:
                self.ui.console.print(f"[red]✗ {function_name} error: {str(e)}[/red]")
                tool_results.append({
                    "role": "tool",
                    "tool_call_id": tool_id,
                    "content": json.dumps({"error": str(e)})
                })

        self.conversation_history.extend(tool_results)

        tools = self.tool_registry.to_openai_tools_format()
        followup_stream = self.client.stream_chat_completion(self.conversation_history, tools=tools)
        return await self._handle_streaming_response(followup_stream)

    async def chat_loop(self):
        self.running = True
        signal.signal(signal.SIGINT, self.handle_signal)
        signal.signal(signal.SIGTERM, self.handle_signal)

        while self.running:
            try:
                user_input = self.ui.get_user_input()

                # --- Debounce empty input: don't re-render the panel on blanks ---
                while not user_input.strip():
                    try:
                        sys.stdout.write("◆ ")
                        sys.stdout.flush()
                        user_input = input()
                    except EOFError:
                        self.running = False
                        break
                if not self.running:
                    break
                # ---------------------------------------------------------------

                # new user turn → reset auto-tool guard
                self._auto_tool_used_this_turn = False

                if not self.process_command(user_input):
                    break
                if user_input.startswith('/'):
                    continue

                self.conversation_history.append({"role": "user", "content": user_input})
                self.ui.conversation.display_message("user", user_input)

                messages = self.conversation_history.copy()
                tools = self.tool_registry.to_openai_tools_format() if self.config.tools.enabled else None

                if self.config.tools.enabled and tools:
                    system_msg = {
                        "role": "system",
                        "content": (
                            "You are Demonology, a proactive assistant that builds complete projects. "
                            "When users ask you to create or build something, take full initiative:\n\n"
                            "AVAILABLE TOOLS:\n"
                            "- web_search: Search for general information (requires 'query')\n"
                            "- reddit_search: Search Reddit discussions and community insights (requires 'query')\n"
                            "- project_planning: Plan AND build complete projects (requires 'project_name', 'project_description')\n"
                            "- file_operations: Create/modify individual files (requires 'operation' parameter)\n"
                            "- code_execution: Run code only when explicitly requested\n\n"
                            "PROACTIVE BEHAVIOR:\n"
                            "- Use web_search for general information when uncertain\n"
                            "- Use reddit_search to find community discussions, opinions, and real user experiences\n"
                            "- Use project_planning to automatically create complete project structures\n"
                            "- When asked to 'build' or 'create' a project, use project_planning with execute_plan=true\n"
                            "- Follow through completely - don't just plan, actually create the files and structure\n\n"
                            "Always include required parameters and focus on what the user requested."
                        )
                    }
                else:
                    if self.config.ui.permissive_mode:
                        system_msg = {"role": "system", "content":
                                      "You are a helpful AI assistant with enhanced capabilities. Be helpful and flexible."}
                    else:
                        system_msg = {"role": "system", "content":
                                      "You are a helpful AI assistant. Respond naturally and conversationally."}

                messages.insert(0, system_msg)

                await self.ui.start_loading()
                try:
                    response_stream = self.client.stream_chat_completion(messages, tools=tools)
                    # Keep loading animation running until we get first response chunk
                    full_response = await self._handle_streaming_response_with_loading(response_stream)
                    self.conversation_history.append({"role": "assistant", "content": full_response})
                except DemonologyAPIError as e:
                    await self.ui.stop_loading()
                    self.ui.display_error(f"API Error: {str(e)}")
                except Exception as e:
                    await self.ui.stop_loading()
                    logger.exception("Unexpected error in chat loop")
                    self.ui.display_error(f"Unexpected error: {str(e)}")
                    # Continue the conversation loop instead of breaking

            except KeyboardInterrupt:
                self.ui.display_info("Interrupted by ancient forces...")
                break
            except EOFError:
                break

        self.ui.display_info("Farewell, seeker of knowledge... 𖤐")

    async def run(self):
        try:
            await self.initialize()
            self.ui.display_banner()
            if not await self.test_connection():
                self.ui.display_error("Cannot establish connection to API. Check your configuration.")
                return 1
            await self.chat_loop()
            return 0
        except Exception as e:
            self.ui.display_error(f"Fatal error: {str(e)}")
            logger.exception("Fatal error in main application")
            return 1
        finally:
            await self.cleanup()


@click.command()
@click.option('--config', '-c', type=click.Path(), help='Path to configuration file')
@click.option('--theme', '-t', type=click.Choice(['amethyst', 'infernal', 'stygian']), help='UI theme')
@click.option('--model', '-m', help='Model name to use')
@click.option('--base-url', help='Base URL for API')
@click.option('--permissive', is_flag=True, help='Enable permissive mode')
@click.option('--debug', is_flag=True, help='Enable debug logging')
@click.version_option("0.1.0", prog_name='Demonology')
def main(config, theme, model, base_url, permissive, debug):
    """Demonology - A mystical terminal interface for local LLM interaction."""
    if debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")

    config_path = Path(config) if config else None
    app_config = Config(config_path)

    if theme:
        app_config.ui.theme = theme
    if model:
        app_config.api.model = model
    if base_url:
        app_config.api.base_url = base_url
    if permissive:
        app_config.ui.permissive_mode = True

    app_config.save()

    app = DemonologyApp(app_config)
    try:
        exit_code = asyncio.run(app.run())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\nExiting...")
        sys.exit(0)


if __name__ == '__main__':
    main()

