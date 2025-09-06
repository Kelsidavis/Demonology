"""
Demonology CLI - Main command line interface and application logic.
"""

import asyncio
import sys
import signal
import json
import os
import re
import time
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import platform
from collections import deque
try:
    import readline  # optional on Windows
except Exception:
    readline = None
import click
import logging
from logging.handlers import RotatingFileHandler

from .client import DemonologyClient, DemonologyAPIError
from .config import Config
from .ui import DemonologyUI
from .themes import ThemeName
from .tools import ToolRegistry, create_default_registry

# ---- injected safety helpers ----
def _json_loads_tolerant(s: str):
    """
    Tolerant JSON parser:
    - first try json.loads
    - then try minor repairs (strip trailing commas, fix unbalanced braces/brackets, swap single to double quotes if safe)
    - final attempt: wrap in {} if it looks like key:value pairs without braces
    Returns (obj, error) where obj is dict or None.
    """
    import json, re
    def _try(txt):
        try:
            return json.loads(txt), None
        except Exception as e:
            return None, str(e)

    obj, err = _try(s)
    if obj is not None:
        return obj, None

    t = s.strip()

    # If looks like key:value pairs without outer braces, add them
    if not t.startswith("{") and ":" in t and not any(t.startswith(ch) for ch in ("[", "\"")):
        t2 = "{" + t + "}"
        obj, err2 = _try(t2)
        if obj is not None:
            return obj, None

    # Replace single quotes with double quotes cautiously (not inside already valid JSON)
    t2 = re.sub(r"(?<!\\)'", '"', t)
    obj, err2 = _try(t2)
    if obj is not None:
        return obj, None

    # Remove trailing commas before } or ]
    t3 = re.sub(r",\s*([}\]])", r"\1", t2)
    obj, err3 = _try(t3)
    if obj is not None:
        return obj, None

    # Balance braces/brackets if off by one
    def _balance_braces(u):
        open_b = u.count("{"); close_b = u.count("}")
        if open_b > close_b:
            u += "}" * (open_b - close_b)
        open_a = u.count("["); close_a = u.count("]")
        if open_a > close_a:
            u += "]" * (open_a - close_a)
        return u
    t4 = _balance_braces(t3)
    obj, err4 = _try(t4)
    if obj is not None:
        return obj, None

    # As a last resort, extract a JSON object substring
    m = re.search(r"\{.*\}", t4, re.S)
    if m:
        obj, err5 = _try(m.group(0))
        if obj is not None:
            return obj, None

    return None, err or err2 or err3 or err4 or "unable to parse json"

def _exp_backoff(attempt: int, base: float = 0.5, cap: float = 6.0) -> float:
    """Exponential backoff seconds for attempt N (0-based)."""
    import math
    return min(cap, base * (2 ** max(0, attempt)))

def _choose_editor() -> str:
    """Pick a sensible editor with fallbacks."""
    env = os.environ.get("EDITOR")
    if env:
        return env
    for cand in ("nano", "vim", "vi"):
        import shutil
        if shutil.which(cand):
            return cand
    return "vi"
# ---- end injected helpers ----


# Enhanced logging setup for autonomous coding workflows
def setup_enhanced_logging():
    """Setup comprehensive logging with file rotation and error tracking."""
    log_dir = Path.home() / ".demonology" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Main log file with rotation
    main_log = log_dir / "demonology.log"
    main_handler = RotatingFileHandler(
        main_log, maxBytes=10*1024*1024, backupCount=5  # 10MB files, keep 5
    )
    main_handler.setLevel(logging.INFO)
    
    # Error-only log for critical issues
    error_log = log_dir / "errors.log"
    error_handler = RotatingFileHandler(
        error_log, maxBytes=5*1024*1024, backupCount=3  # 5MB files, keep 3
    )
    error_handler.setLevel(logging.ERROR)
    
    # Console handler for immediate feedback
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)
    
    # Detailed formatter for log files
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
    )
    
    # Simple formatter for console
    console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    main_handler.setFormatter(file_formatter)
    error_handler.setFormatter(file_formatter)
    console_handler.setFormatter(console_formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(main_handler)
    root_logger.addHandler(error_handler)
    root_logger.addHandler(console_handler)
    
    return str(main_log), str(error_log)

# Setup enhanced logging
MAIN_LOG_PATH, ERROR_LOG_PATH = setup_enhanced_logging()
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
        self._tool_usage_stats = {}  # Track tool usage statistics
        self._session_start_time = time.time()  # Track session duration
        self._error_count = 0  # Track errors for autonomous debugging
        self._repetition_count = 0  # Track repetition loops
        self._last_repetition_time = 0  # Track when last repetition occurred

        
        # Input history
        self._history_max = int(os.environ.get("DEMONOLOGY_HISTORY_MAX", "1000"))
        self._history = deque(maxlen=self._history_max)
        self._history_file = (self.config.get_conversations_dir().parent / "history.txt")
        self._readline_enabled = False
        try:
            if readline and sys.stdin.isatty():
                self._readline_enabled = True
                try:
                    readline.set_history_length(self._history_max)
                except Exception:
                    pass
                if self._history_file.exists():
                    try:
                        readline.read_history_file(str(self._history_file))
                        # also seed deque for /history popover
                        for idx in range(min(readline.get_current_history_length() or 0, self._history_max)):
                            try:
                                self._history.append(readline.get_history_item(idx+1))
                            except Exception:
                                break
                    except Exception:
                        pass
        except Exception:
            # Non-fatal; continue without readline
            pass
        self.tool_registry = create_default_registry()
        
        # Log startup info
        logger.info(f"Demonology CLI started - Log files: {MAIN_LOG_PATH} (main), {ERROR_LOG_PATH} (errors)")

    def _log_structured_error(self, error_type: str, error_message: str, context: Dict[str, Any] = None):
        """Log structured error information for autonomous debugging."""
        self._error_count += 1
        error_data = {
            "error_id": self._error_count,
            "error_type": error_type,
            "error_message": error_message,
            "timestamp": time.time(),
            "session_duration": time.time() - self._session_start_time,
            "conversation_length": len(self.conversation_history),
            "context": context or {}
        }
        
        logger.error(f"STRUCTURED_ERROR: {json.dumps(error_data, indent=2)}")
        
        # Also log a human-readable version
        context_str = f" | Context: {context}" if context else ""
        logger.error(f"Error #{self._error_count} ({error_type}): {error_message}{context_str}")

    def _handle_repetition_recovery(self, is_critical: bool = False):
        """Enhanced repetition loop recovery with context storms handling."""
        current_time = time.time()
        self._repetition_count += 1
        
        time_since_last = current_time - self._last_repetition_time
        self._last_repetition_time = current_time
        
        # Critical repetition (from client-level detection) or frequent repetitions
        is_repetition_storm = (
            is_critical or 
            (time_since_last < 60 and self._repetition_count > 2) or  # 2+ loops in 1 minute
            (time_since_last < 300 and self._repetition_count > 5)    # 5+ loops in 5 minutes  
        )
        
        if is_repetition_storm:
            storm_level = "CRITICAL" if is_critical else "HIGH"
            logger.error(f"{storm_level} repetition storm detected ({self._repetition_count} total loops)")
            self.ui.display_warning(f"⚠️  {storm_level} repetition detected - applying recovery strategies...")
            
            # Strategy 1: Aggressive context trimming based on storm severity
            if is_critical or self._repetition_count > 5:
                # Nuclear option - keep only last 3 exchanges
                logger.warning("Applying nuclear context reset - keeping only last 3 exchanges")
                self.ui.display_warning("🔥 Nuclear context reset - clearing most conversation history")
                self._nuclear_context_reset()
            elif len(self.conversation_history) > 8:
                # Aggressive trimming - keep only last 8 messages 
                logger.warning("Applying aggressive context trimming")
                self.ui.display_warning("🔄 Aggressive context trimming to break patterns")
                self._trim_conversation_history(8)
            
            # Strategy 2: Add pattern-breaking system message
            pattern_breaker = self._generate_pattern_breaker_message()
            self.conversation_history.append(pattern_breaker)
            
            # Strategy 3: Request server health check and possible restart
            if is_critical and hasattr(self, 'client') and self.client:
                logger.warning("Critical repetition may indicate server stress - marking for health check")
                # Mark client for potential server restart on next request
                self.client.consecutive_failures = max(self.client.consecutive_failures, 2)
            
            # Strategy 4: Reset repetition tracking after recovery
            if self._repetition_count > 10:  # Reset after too many attempts
                logger.info("Resetting repetition counter after recovery attempt")
                self._repetition_count = 0
            # Inject exponential backoff to avoid tight retry loops
            try:
                time.sleep(_exp_backoff(self._repetition_count))
            except Exception:
                pass
                
            return True
            
        elif time_since_last < 120 and self._repetition_count > 1:  # Minor repetition
            logger.info(f"Minor repetition detected ({self._repetition_count} loops), applying light recovery")
            self.ui.display_info("🔄 Minor repetition detected - nudging conversation forward")
            
            # Light strategy: Just add a gentle nudge
            nudge_msg = {
                "role": "system", 
                "content": "Note: Please try a different approach or provide more specific details."
            }
            self.conversation_history.append(nudge_msg)
            return True
        
        return False
    
    def _nuclear_context_reset(self):
        """Nuclear option: Keep only the most recent essential exchanges."""
        if len(self.conversation_history) <= 6:
            return  # Already minimal
            
        # Find the last user message and preserve context around it
        last_user_idx = -1
        for i in range(len(self.conversation_history) - 1, -1, -1):
            if self.conversation_history[i].get('role') == 'user':
                last_user_idx = i
                break
        
        if last_user_idx >= 0:
            # Keep system message (if exists) + last user message + any assistant response
            preserved = []
            
            # Preserve initial system message if it exists
            if (self.conversation_history and 
                self.conversation_history[0].get('role') == 'system' and
                'pattern-breaker' not in self.conversation_history[0].get('content', '').lower()):
                preserved.append(self.conversation_history[0])
            
            # Keep last 2-3 exchanges maximum
            start_idx = max(0, last_user_idx - 2)  # At most 2 messages before last user message
            preserved.extend(self.conversation_history[start_idx:])
            
            removed_count = len(self.conversation_history) - len(preserved)
            self.conversation_history = preserved
            
            logger.info(f"Nuclear context reset: removed {removed_count} messages, kept {len(preserved)}")
        else:
            # Fallback: keep only last 3 messages
            removed_count = len(self.conversation_history) - 3
            self.conversation_history = self.conversation_history[-3:]
            logger.info(f"Nuclear context reset (fallback): removed {removed_count} messages")
    
    def _generate_pattern_breaker_message(self) -> Dict[str, str]:
        """Generate a pattern-breaking system message based on context."""
        breakers = [
            "SYSTEM INTERVENTION: Previous responses were repetitive. Please take a completely different approach. If you're stuck, ask for clarification or break down the problem into smaller steps.",
            "PATTERN BREAK: The conversation has entered a loop. Please step back, reassess the situation, and try a fresh perspective or ask what specific aspect needs attention.",
            "RECOVERY MODE: Repetitive generation detected. Please provide a concise, direct response that moves the conversation forward in a new direction.",
            "CONTEXT RESET: Previous approach wasn't working. Please try a different method entirely or ask for more specific requirements.",
        ]
        
        # Choose based on repetition count for variety
        breaker_idx = (self._repetition_count - 1) % len(breakers)
        
        return {
            "role": "system",
            "content": f"{breakers[breaker_idx]} Remember: emit only new code, no repeats."
        }
    
    def _generate_context_summary(self, messages: List[Dict[str, str]]) -> str:
        """Generate a concise summary of conversation context."""
        if not messages:
            return "No prior context"
        
        # Extract key topics and actions
        topics = set()
        tools_used = set()
        
        for msg in messages:
            content = str(msg.get('content', '')).lower()
            
            # Extract topics
            if 'file' in content or 'read' in content:
                topics.add('file operations')
            if 'code' in content or 'python' in content or 'script' in content:
                topics.add('coding')
            if 'analyze' in content or 'debug' in content:
                topics.add('analysis')
            if 'create' in content or 'generate' in content:
                topics.add('creation')
                
            # Extract tool usage
            if 'file_operations' in content:
                tools_used.add('files')
            if 'code_execution' in content:
                tools_used.add('execution')
            if 'web_search' in content:
                tools_used.add('web')
        
        summary_parts = []
        if topics:
            summary_parts.append(f"Topics: {', '.join(list(topics)[:3])}")
        if tools_used:
            summary_parts.append(f"Tools: {', '.join(list(tools_used)[:3])}")
        
        return '; '.join(summary_parts) if summary_parts else "General conversation"

    

    async def initialize(self):
        """Initialize the application."""
        self.client = DemonologyClient(
            base_url=self.config.api.base_url,
            model=self.config.api.model,
            max_tokens=self.config.api.max_tokens,
            temperature=self.config.api.temperature,
            top_p=self.config.api.top_p,
            timeout=self.config.api.timeout,
            max_retries=getattr(self.config.api, 'max_retries', 3),
            retry_delay=getattr(self.config.api, 'retry_delay', 2.0)
        )

    async def cleanup(self):
        if self.client:
            await self.client.close()
        self.ui.cleanup()
        # Persist readline history
        try:
            if self._readline_enabled and readline:
                readline.set_history_length(self._history_max)
                readline.write_history_file(str(self._history_file))
        except Exception:
            pass

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
            self._log_structured_error("connection_failed", str(e), {
                "base_url": self.config.api.base_url,
                "model": self.config.api.model
            })
            self.ui.display_error(f"Connection ritual failed: {str(e)}")
            return False
    
    async def _manual_server_restart(self):
        """Handle manual server restart command."""
        try:
            await self.ui.start_loading("Manually restarting server...")
            success = await self.client.restart_server_and_reconnect()
            await self.ui.stop_loading()
            if success:
                self.ui.display_success("✅ Server manually restarted successfully!")
            else:
                self.ui.display_error("❌ Manual server restart failed.")
        except Exception as e:
            await self.ui.stop_loading()
            self.ui.display_error(f"❌ Error during manual restart: {e}")
    
    async def ensure_server_connection(self) -> bool:
        """
        Ensure server connection is available, attempting restart if needed.
        Returns True if connection is established, False otherwise.
        """
        # First try normal connection
        if await self.test_connection():
            return True
        
        # If connection failed, try to restart server
        self.ui.display_error("🔄 Server connection failed. Attempting automatic restart...")
        await self.ui.start_loading("Restarting server and clearing VRAM...")
        
        try:
            success = await self.client.restart_server_and_reconnect()
            await self.ui.stop_loading()
            
            if success:
                self.ui.display_success("✅ Server restarted successfully! Connection restored.")
                return True
            else:
                self.ui.display_error("❌ Server restart failed. Please start your llama server manually.")
                return False
                
        except Exception as e:
            await self.ui.stop_loading()
            self.ui.display_error(f"❌ Error during server restart: {e}")
            return False

    
    def _history_add(self, line: str) -> None:
        """Record a user input (excluding commands). Persist to readline if available."""
        if not line or line.strip().startswith('/'):
            return
        self._history.append(line)
        if self._readline_enabled and readline:
            try:
                readline.add_history(line)
                # Opportunistically persist
                try:
                    readline.write_history_file(str(self._history_file))
                except Exception:
                    pass
            except Exception:
                pass

    def _history_tail(self, n: int = 20) -> list:
        """Return last n history entries as a list (most recent last)."""
        if n <= 0:
            n = 1
        return list(self._history)[-n:]
    
    def process_command(self, user_input: str) -> bool:
        """Return False to quit."""
        if not user_input.startswith('/'):
            return True

        command = user_input[1:].strip().lower()
        parts = command.split()
        cmd = parts[0] if parts else ""

        # Enhanced command aliases for better UX
        if cmd in ['quit', 'exit', 'q', 'bye', 'goodbye']:
            return False
        elif cmd in ['help', 'h', '?']:
            self._show_help()
        elif cmd in ['status', 'stat', 's']:
            self.ui.display_status()
        elif cmd in ['themes', 'theme-list']:
            self.ui.show_theme_preview()
        elif cmd in ['history', 'his', 'hi']:
            # Usage: /history [N]
            count = 20
            if len(parts) > 1 and parts[1].isdigit():
                try:
                    count = max(1, min(int(parts[1]), self._history_max))
                except Exception:
                    count = 20
            items = self._history_tail(count)
            if hasattr(self.ui, "show_history_popover"):
                self.ui.show_history_popover(items, title=f"Last {len(items)} inputs")
            else:
                self.ui.console.print("\\n[bold]History (newest last):[/bold]")
                for i, line in enumerate(items, 1):
                    self.ui.console.print(f"{i:>3}. {line}")
        elif cmd in ['theme', 't']:
            if len(parts) > 1:
                self.ui.change_theme(parts[1])
            else:
                self.ui.display_error("Usage: /theme <theme_name>")
        elif cmd in ['permissive', 'perm', 'p']:
            self._toggle_permissive_mode()
        elif cmd in ['model', 'm']:
            if len(parts) > 1:
                self._change_model(' '.join(parts[1:]))
            else:
                self.ui.display_info(f"Current model: {self.config.api.model}")
        elif cmd in ['save', 'sv']:
            if len(parts) > 1:
                self._save_conversation(' '.join(parts[1:]))
            else:
                self.ui.display_error("Usage: /save <filename>")
        elif cmd in ['load', 'ld']:
            if len(parts) > 1:
                self._load_conversation(' '.join(parts[1:]))
            else:
                self.ui.display_error("Usage: /load <filename>")
        elif cmd in ['clear', 'cls', 'clean']:
            self._clear_conversation()
        elif cmd in ['config', 'cfg', 'conf']:
            if len(parts) > 1 and parts[1] == 'edit':
                self._edit_config()
            else:
                self._show_config()
        elif cmd in ['debug', 'dbg']:
            self.ui.display_info("Debug command will be handled in next iteration")
        elif cmd in ['restart', 'rs', 'reset']:
            asyncio.create_task(self._manual_server_restart())
        elif cmd in ['tools', 'tl']:
            self._show_tools()
        elif cmd in ['diagnose', 'diag', 'dx']:
            self._handle_diagnose_command(user_input)
        elif cmd in ['logs', 'log']:
            self._show_logs()
        elif cmd in ['context', 'ctx', 'c']:
            self._show_context_stats()
        elif cmd in ['trim', 'tr']:
            parts = user_input.split()
            if len(parts) > 1 and parts[1] == 'smart':
                # Force smart trimming
                removed = self._smart_trim_context()
                if removed == 0:
                    self.ui.display_info("Context is already optimally sized.")
            else:
                keep_count = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 10
                self._trim_conversation_history(keep_count)
        elif cmd in ['optimize', 'opt', 'o']:
            # Optimize context using smart trimming
            stats = self._get_context_stats()
            if stats['context_usage_percent'] < 50:
                self.ui.display_info(f"Context usage is only {stats['context_usage_percent']:.1f}% - no optimization needed.")
            else:
                removed = self._smart_trim_context()
                if removed > 0:
                    new_stats = self._get_context_stats()
                    self.ui.display_success(f"✅ Context optimized: {stats['context_usage_percent']:.1f}% → {new_stats['context_usage_percent']:.1f}%")
                else:
                    self.ui.display_info("Context is already optimally sized.")
        elif cmd in ['autocontinue', 'auto', 'ac']:
            if len(parts) > 1:
                if parts[1] == 'on':
                    self.config.ui.auto_continue_enabled = True
                    self.config.save()
                    self.ui.display_success(f"✅ Auto-continue enabled (timeout: {self.config.ui.auto_continue_timeout}s)")
                elif parts[1] == 'off':
                    self.config.ui.auto_continue_enabled = False
                    self.config.save()
                    self.ui.display_success("✅ Auto-continue disabled")
                elif parts[1].replace('.', '').isdigit():
                    timeout = float(parts[1])
                    if 10 <= timeout <= 300:  # Between 10 seconds and 5 minutes
                        self.config.ui.auto_continue_timeout = timeout
                        self.config.save()
                        self.ui.display_success(f"✅ Auto-continue timeout set to {timeout}s")
                    else:
                        self.ui.display_error("Timeout must be between 10 and 300 seconds")
                else:
                    self.ui.display_error("Usage: /autocontinue <on|off|timeout_seconds>")
            else:
                status = "enabled" if self.config.ui.auto_continue_enabled else "disabled"
                self.ui.display_info(f"Auto-continue: {status} (timeout: {self.config.ui.auto_continue_timeout}s)")
        else:
            self.ui.display_error(f"Unknown command: /{cmd}. Type /help for available commands.")

        return True

    def _show_help(self):
        """Display help information for available commands."""
        self.ui.console.print(f"""
[bold red]⛧ DEMONOLOGY COMMAND GRIMOIRE ⛧[/bold red]

[bold]Basic Commands:[/bold]
/help, /h, /?         Show this grimoire of dark commands
/quit, /exit, /q      Banish the daemon and return to the void
/status, /s           Show current summoning status
/themes               Preview available visual themes
/theme <name>         Change the visual theme
/permissive, /p       Toggle permissive mode (dangerous magic)

[bold]Model & Connection:[/bold]
/model [name]         View or change the AI model
/config               Show current configuration
/config edit          Edit configuration file
/debug, /dbg          Debug API response
/restart, /rs         Restart llama server and clear VRAM
/tools, /tl           List available tools
/diagnose, /diag, /dx <binary> [--dll-dir <path>]  Diagnose reverse engineering setup
/logs, /log           Show log file locations for error review

[bold]Context Management:[/bold]
/context, /ctx, /c    Show context usage statistics
/trim, /tr [number]   Keep only recent messages (default: 10)
/trim smart           Intelligent context optimization
/optimize, /opt, /o   Auto-optimize context usage

[bold]Auto-Continue:[/bold]
/autocontinue, /ac    Show auto-continue status
/auto on              Enable auto-continue (resumes work after timeout)
/auto off             Disable auto-continue
/auto <seconds>       Set timeout (10-300 seconds, default: 60)

[bold]Command History:[/bold]
/history              Show recent command history
/last                 Recall and resend last command
/h<number>            Recall specific command (e.g., /h1, /h2)

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
    
    def _get_context_stats(self) -> Dict[str, Any]:
        """Get current context usage statistics."""
        total_chars = 0
        total_messages = len(self.conversation_history)
        
        for message in self.conversation_history:
            content = message.get('content', '')
            if isinstance(content, str):
                total_chars += len(content)
            elif isinstance(content, list):
                # Handle tool calls and complex content
                total_chars += len(str(content))
        
        # More accurate token estimate for coding content (3.5 chars per token average)
        estimated_tokens = int(total_chars / 3.5)
        
        # Use configured context length
        context_limit = getattr(self.config.api, 'context_length', 26768)
        context_buffer = getattr(self.config.api, 'context_buffer', 2048)
        available_context = context_limit - context_buffer
        
        return {
            "total_messages": total_messages,
            "total_characters": total_chars,
            "estimated_tokens": estimated_tokens,
            "context_limit": context_limit,
            "available_context": available_context,
            "context_usage_percent": min(100, (estimated_tokens / available_context) * 100),
            "needs_trimming": estimated_tokens > (available_context * getattr(self.config.api, 'auto_trim_threshold', 0.85))
        }
    
    def _trim_conversation_history(self, keep_recent: int = 10):
        """Trim conversation history to keep only recent messages."""
        if len(self.conversation_history) > keep_recent:
            removed_count = len(self.conversation_history) - keep_recent
            self.conversation_history = self.conversation_history[-keep_recent:]
            self.ui.display_warning(f"Trimmed {removed_count} old messages to manage context size")
            return removed_count
        return 0
    
    def _smart_trim_context(self) -> int:
        """Intelligently trim context while preserving important messages."""
        if not getattr(self.config.api, 'smart_trimming', True):
            # Fall back to basic trimming
            return self._trim_conversation_history(10)
        
        stats = self._get_context_stats()
        if not stats['needs_trimming']:
            return 0
        
        # Preserve system messages and recent interactions
        system_messages = []
        regular_messages = []
        
        for msg in self.conversation_history:
            if msg.get('role') == 'system':
                system_messages.append(msg)
            else:
                regular_messages.append(msg)
        
        # Calculate how many regular messages to keep (keep 20% headroom)
        target_context = int(stats['available_context'] * 0.8)  # Aim for 80% usage max
        current_tokens = stats['estimated_tokens']
        tokens_to_remove = current_tokens - target_context
        
        if tokens_to_remove <= 0:
            return 0
        
        # Remove messages from oldest first, but preserve recent context
        messages_to_keep = []
        tokens_kept = 0
        
        # Always keep last 6 messages for immediate context
        recent_messages = regular_messages[-6:] if len(regular_messages) >= 6 else regular_messages
        for msg in recent_messages:
            content_length = len(str(msg.get('content', '')))
            tokens_kept += int(content_length / 3.5)
            messages_to_keep.append(msg)
        
        # Add rolling summary if significant content was removed
        if len(regular_messages) > 6:
            removed_messages = regular_messages[:-6]
            if len(removed_messages) > 2:
                summary = self._generate_context_summary(removed_messages[-10:])  # Summary of last 10 removed
                summary_msg = {
                    "role": "system",
                    "content": f"[CONTEXT SUMMARY] Previous conversation: {summary}"
                }
                messages_to_keep.insert(0, summary_msg)
        
        # Add older messages if we have token budget
        remaining_tokens = target_context - tokens_kept
        for msg in reversed(regular_messages[:-6]):  # Work backwards from older messages
            content_length = len(str(msg.get('content', '')))
            msg_tokens = int(content_length / 3.5)
            
            if msg_tokens <= remaining_tokens:
                messages_to_keep.insert(-6 if len(messages_to_keep) >= 6 else 0, msg)
                remaining_tokens -= msg_tokens
            else:
                break
        
        # Rebuild conversation history
        original_count = len(self.conversation_history)
        self.conversation_history = system_messages + messages_to_keep
        removed_count = original_count - len(self.conversation_history)
        
        if removed_count > 0:
            self.ui.display_warning(f"🧠 Smart-trimmed {removed_count} messages to manage context ({stats['context_usage_percent']:.1f}% → ~70%)")
        
        return removed_count
    
    def _auto_manage_context(self):
        """Automatically manage context size and show warnings."""
        stats = self._get_context_stats()
        
        # Show periodic warnings
        if stats['context_usage_percent'] >= 90:
            self.ui.display_error("🚨 Context 90%+ full! Auto-trimming to prevent overflow...")
            self._smart_trim_context()
        elif stats['context_usage_percent'] >= 75:
            # Only show warning every few messages to avoid spam
            if len(self.conversation_history) % 5 == 0:  # Every 5th message
                self.ui.display_warning(f"⚠️  Context {stats['context_usage_percent']:.0f}% full. Use `/trim` or `/clear` if needed.")
        
        # Trigger automatic smart trimming if enabled and threshold exceeded
        if getattr(self.config.api, 'smart_trimming', True) and stats['needs_trimming']:
            removed = self._smart_trim_context()
            if removed > 0:
                # Refresh stats after trimming
                stats = self._get_context_stats()
    
    def _show_context_stats(self):
        """Display current context usage statistics."""
        stats = self._get_context_stats()
        
        # Color-code usage based on percentage
        if stats['context_usage_percent'] >= 90:
            usage_color = "red"
            warning = "\n⚠️  [red]CRITICAL: Context nearly full! Auto-trim will occur soon.[/red]"
        elif stats['context_usage_percent'] >= 75:
            usage_color = "yellow"
            warning = "\n⚠️  [yellow]WARNING: Context getting full. Consider trimming.[/yellow]"
        else:
            usage_color = "green"
            warning = ""
        
        self.ui.console.print(f"""
[bold]Context Usage Statistics:[/bold]

[dim]Messages:[/dim] {stats['total_messages']}
[dim]Characters:[/dim] {stats['total_characters']:,}
[dim]Estimated Tokens:[/dim] {stats['estimated_tokens']:,}
[dim]Context Limit:[/dim] {stats['context_limit']:,} (Available: {stats['available_context']:,})
[dim]Context Usage:[/dim] [{usage_color}]{stats['context_usage_percent']:.1f}%[/{usage_color}]
[dim]Smart Trimming:[/dim] {'Enabled' if getattr(self.config.api, 'smart_trimming', True) else 'Disabled'}{warning}

[yellow]Commands:[/yellow]
• `/trim [number]` - Keep only N recent messages
• `/trim smart` - Intelligent context optimization
• `/optimize` - Auto-optimize context usage
• `/clear` - Clear entire conversation history
• `/context` - Show these statistics
""")

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

    def _handle_diagnose_command(self, user_input: str):
        """Handle the diagnose command for troubleshooting reverse engineering setup."""
        from .diagnostic import diagnose_binary, print_diagnostic_report
        
        # Parse command line arguments
        parts = user_input.split()
        if len(parts) < 2:
            self.ui.display_error("Usage: /diagnose <binary_path> [--dll-dir <path>] [--verbose]")
            self.ui.display_info("Examples:")
            self.ui.display_info("  /diagnose REBEXE.EXE --dll-dir .dlls")
            self.ui.display_info("  /diagnose program.exe --verbose")
            return
        
        binary_path = parts[1]
        dll_dir = None
        verbose = False
        
        # Simple argument parsing
        i = 2
        while i < len(parts):
            if parts[i] == "--dll-dir" and i + 1 < len(parts):
                dll_dir = parts[i + 1]
                i += 2
            elif parts[i] == "--verbose" or parts[i] == "-v":
                verbose = True
                i += 1
            else:
                i += 1
        
        try:
            self.ui.display_info("🔍 Running diagnostic analysis...")
            results = diagnose_binary(binary_path, dll_dir, verbose)
            print_diagnostic_report(results, verbose)
            
            # Suggest next steps based on results
            status = results.get("overall_status", "unknown")
            if status == "ready":
                self.ui.display_success("\n🎉 Your setup looks good! You can proceed with reverse engineering analysis.")
            elif status == "minor_issues":
                self.ui.display_info("\n⚡ Your setup should work but could be improved. Check the suggestions above.")
            else:
                self.ui.display_error("\n⚠️  Your setup has issues that may prevent successful analysis. Please address the problems listed above.")
                
        except Exception as e:
            self.ui.display_error(f"Diagnostic failed: {e}")
            logger.error(f"Diagnostic command error: {e}", exc_info=True)
    
    def _show_logs(self):
        """Display log file locations for debugging."""
        self.ui.display_info(f"📋 Log Files for Error Review:")
        self.ui.display_info(f"  • Main log: {MAIN_LOG_PATH}")
        self.ui.display_info(f"  • Errors only: {ERROR_LOG_PATH}")
        self.ui.display_info(f"  • Session errors: {self._error_count}")
        
        # Show recent error if any
        if self._error_count > 0:
            self.ui.display_info(f"\n💡 Tip: Use 'tail -f {ERROR_LOG_PATH}' to monitor errors in real-time")

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
        
        # Don't auto-trigger tools for simple greetings or conversations
        simple_responses = ['hello', 'hi', 'hey', 'thanks', 'thank you', 'ok', 'okay', 'yes', 'no']
        if (last_user or "").strip().lower() in simple_responses:
            return None
        
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
            # Check if this is a repetition-related error
            if "repetition loop" in str(e).lower() or "repetitive generation" in str(e).lower():
                is_critical = "critical" in str(e).lower() or "exactly" in str(e).lower()
                recovery_applied = self._handle_repetition_recovery(is_critical=is_critical)
                if recovery_applied:
                    if is_critical:
                        self.ui.display_warning("⚠️  CRITICAL repetition detected - applied nuclear recovery")
                    else:
                        self.ui.display_info("🔄 Applied repetition recovery - retrying with modified context")
                    await self.ui.stop_loading()
                    return  # Let the caller retry
            
            self._log_structured_error("streaming_response_failed", str(e), {
                "error_type": type(e).__name__,
                "conversation_length": len(self.conversation_history)
            })
            logger.exception("Error processing streaming response")
            await self.ui.stop_loading()  # Ensure loading stops on error
            
            # For 500 errors, try to recover gracefully
            if "500" in str(e) or isinstance(e, DemonologyAPIError):
                self.ui.display_error("𐕣 Server error occurred. Conversation history may be corrupted. Continuing... 𐕣")
                # More graceful conversation history recovery
                try:
                    # Keep system messages and recent user/assistant pairs
                    system_msgs = [msg for msg in self.conversation_history if msg.get("role") == "system"]
                    user_msgs = [msg for msg in self.conversation_history if msg.get("role") == "user"]
                    assistant_msgs = [msg for msg in self.conversation_history if msg.get("role") == "assistant"]
                    
                    # Keep last 2 user messages and their responses
                    if user_msgs:
                        recent_users = user_msgs[-2:]
                        # Try to maintain conversation pairs
                        self.conversation_history = system_msgs + recent_users
                        # Conversation history recovered
                except Exception as recovery_error:
                    logger.error(f"Failed to recover conversation history: {recovery_error}")
                    self.conversation_history = []
            
            # Return partial content if available, otherwise display error but don't crash
            if content_buffer.strip():
                return content_buffer
            else:
                self.ui.display_error(f"𐕣 API Error: {str(e)} 𐕣")
                return ""

        # If we never received any chunks, stop loading now
        if not first_chunk_received:
            await self.ui.stop_loading()

        # If we collected tool calls, execute them
        if tool_calls and any(tc.get("function", {}).get("name") for tc in tool_calls):
            if content_buffer.strip():
                self.ui.console.print()  # Add newline after content
            return await self._execute_collected_tool_calls(content_buffer, tool_calls)
        
        # Add newline after response for proper spacing
        if content_buffer.strip():
            self.ui.console.print()
        
        return content_buffer
    
    async def _execute_collected_tool_calls(self, content_buffer: str, tool_calls: List[Dict[str, Any]]) -> str:
        """Execute collected tool calls and continue the conversation."""
        self.ui.console.print("\\n[bold red]👹 UNLEASHING HELLISH TOOLS FROM THE ABYSS 👹[/bold red]")
        
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
                # Debug logging removed for cleaner output
                
                if not arguments_str.strip():
                    arguments = {}
                else:
                    # Try to fix common JSON issues
                    cleaned_str = arguments_str.strip()
                    # Handle incomplete JSON strings
                    if cleaned_str.count('"') % 2 != 0:
                        # Auto-fixing incomplete JSON
                        cleaned_str = cleaned_str.rstrip(',') + '"}'
                    
                    arguments = json.loads(cleaned_str)
                    
            except json.JSONDecodeError as e:
                logger.debug(f"JSON parse failed for {function_name}, using fallback")
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
        max_history_length = 6  # Greatly reduced to prevent server overload
        if len(self.conversation_history) > max_history_length:
            try:
                # Keep system message if present, and recent messages
                system_msgs = [msg for msg in self.conversation_history if msg.get("role") == "system"]
                user_msgs = [msg for msg in self.conversation_history if msg.get("role") == "user"]
                assistant_msgs = [msg for msg in self.conversation_history if msg.get("role") == "assistant"]
                tool_msgs = [msg for msg in self.conversation_history if msg.get("role") == "tool"]
                
                # Keep recent conversation pairs (user + assistant + tool results)
                recent_length = max_history_length - len(system_msgs)
                if recent_length > 0:
                    # Take the most recent messages, trying to keep complete conversation turns
                    recent_msgs = self.conversation_history[-recent_length:]
                    self.conversation_history = system_msgs + recent_msgs
                else:
                    # If too many system messages, just keep them
                    self.conversation_history = system_msgs
                
                # Conversation history truncated for performance
                
                # Also check for excessive tool calling loops
                recent_tool_calls = sum(1 for msg in self.conversation_history[-10:] 
                                      if msg.get("role") == "tool")
                if recent_tool_calls > 6:  # More than 6 tool calls in last 10 messages
                    # Clear some tool results to break the loop
                    self.conversation_history = [msg for msg in self.conversation_history 
                                               if not (msg.get("role") == "tool" and 
                                                      "demo_project" in str(msg.get("content", "")))][:8]
                    logger.warning("Truncated conversation history to break tool calling loop")
                    
            except Exception as truncate_error:
                logger.error(f"Failed to truncate conversation history: {truncate_error}")
                # Fallback: keep only recent messages
                self.conversation_history = self.conversation_history[-5:]
        
        # Debug logging removed for cleaner output
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
            import re
            # Common patterns for each tool type
            if function_name == "file_operations":
                # Try to extract operation
                op_match = re.search(r'"operation":\s*"([^"]*)"', arguments_str)
                if op_match:
                    arguments["operation"] = op_match.group(1)
                else:
                    # Default operation when none specified
                    arguments["operation"] = "list"
                
                path_match = re.search(r'"path":\s*"([^"]*)"', arguments_str)
                if path_match:
                    arguments["path"] = path_match.group(1)
                else:
                    # Default path when none specified
                    arguments["path"] = "."
                
                content_match = re.search(r'"content":\s*"([^"]*)"', arguments_str)
                if content_match:
                    arguments["content"] = content_match.group(1)
                
                # Fix common operation name issues
                if "operation" in arguments:
                    op = arguments["operation"]
                    if op in ["list_directory", "list_files", "dir", "ls"]:
                        arguments["operation"] = "list"
                    elif op in ["read_file", "cat", "view"]:
                        arguments["operation"] = "read"
                    elif op in ["write_file", "create", "save"]:
                        arguments["operation"] = "create_or_write_file"
                    
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
                    
            elif function_name == "code_execution":
                # Extract language parameter
                lang_match = re.search(r'"language":\s*"([^"]*)"', arguments_str)
                if lang_match:
                    arguments["language"] = lang_match.group(1)
                else:
                    arguments["language"] = "bash"  # Default to bash
                
                # Extract code parameter - handle multiline code
                code_match = re.search(r'"code":\s*"((?:[^"\\]|\\.)*)"', arguments_str, re.DOTALL)
                if code_match:
                    # Unescape JSON string
                    code = code_match.group(1).replace('\\"', '"').replace('\\n', '\n').replace('\\\\', '\\')
                    arguments["code"] = code
                else:
                    # Try without quotes as fallback
                    code_match = re.search(r'"code":\s*([^,}]+)', arguments_str)
                    if code_match:
                        arguments["code"] = code_match.group(1).strip()
                
                # Extract optional parameters
                workdir_match = re.search(r'"workdir":\s*"([^"]*)"', arguments_str)
                if workdir_match:
                    arguments["workdir"] = workdir_match.group(1)
                
                timeout_match = re.search(r'"timeout":\s*(\d+)', arguments_str)
                if timeout_match:
                    arguments["timeout"] = int(timeout_match.group(1))
                    
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
            if "500" in str(e) or isinstance(e, DemonologyAPIError):
                self.ui.display_error("𐕣 Server error occurred. Conversation history may be corrupted. Continuing... 𐕣")
                # More graceful conversation history recovery
                try:
                    # Keep system messages and recent user/assistant pairs
                    system_msgs = [msg for msg in self.conversation_history if msg.get("role") == "system"]
                    user_msgs = [msg for msg in self.conversation_history if msg.get("role") == "user"]
                    
                    # Keep last 2 user messages
                    if user_msgs:
                        recent_users = user_msgs[-2:]
                        self.conversation_history = system_msgs + recent_users
                        # Conversation history recovered
                except Exception as recovery_error:
                    logger.error(f"Failed to recover conversation history: {recovery_error}")
                    self.conversation_history = []
            
            # Return partial content if available, otherwise return empty string
            if content_buffer.strip():
                return content_buffer
            else:
                # Don't re-raise errors - return empty response to continue conversation
                self.ui.display_error(f"𐕣 API Error: {str(e)} 𐕣")
                return ""

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
        self.ui.console.print("\\n[bold red]👹 UNLEASHING HELLISH TOOLS FROM THE ABYSS 👹[/bold red]")

        tool_results: List[Dict[str, Any]] = []
        assistant_message = {"role": "assistant", "content": None, "tool_calls": tool_calls}
        self.conversation_history.append(assistant_message)

        for tc in tool_calls:
            tool_id = tc.get("id", "")
            function_name = tc.get("function", {}).get("name", "")
            arguments = {}  # Initialize arguments to prevent NameError
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
                            # Auto-fixing incomplete JSON
                            cleaned_str = cleaned_str.rstrip(',') + '"}'
                        
                        arguments = json.loads(cleaned_str)
                        
                except json.JSONDecodeError as e:
                    logger.debug(f"JSON parse failed for {function_name}, using fallback")
                    # Try to extract key parameters manually as fallback
                    arguments = self._extract_fallback_arguments(function_name, arguments_raw)

                if function_name == "file_operations" and arguments.get("operation") == "create_directory":
                    p = arguments.get("path", "")
                    if p and not p.endswith("/"):
                        arguments["path"] = p + "/"

                self.ui.console.print(f"[bold magenta]🔥 Binding demon [{function_name}] with ritual parameters: {arguments} 🔥[/bold magenta]")
                await self.ui.start_loading(f"Invoking daemon of {function_name}... blood sacrifice accepted...")
                result = await self.tool_registry.execute_tool(function_name, **arguments)
                await self.ui.stop_loading()

                if result.get("success", False):
                    self.ui.console.print(f"[bold green]👹 DEMON [{function_name}] BOWS TO YOUR WILL - POWER CHANNELED 👹[/bold green]")
                else:
                    error_msg = result.get('error', 'Ancient evil')
                    self.ui.console.print(f"[bold red]💀 DEMON [{function_name}] DEFIES THE SUMMONING - CURSE BACKFIRED: {error_msg} 💀[/bold red]")
                    # Log failed tool results for review
                    self._log_structured_error("tool_result_failed", error_msg, {
                        "tool_name": function_name,
                        "tool_arguments": arguments,
                        "call_id": tool_id,
                        "result": result
                    })

                tool_results.append({
                    "role": "tool",
                    "tool_call_id": tool_id,
                    "content": json.dumps(result)
                })

            except Exception as e:
                self._log_structured_error("tool_execution_failed", str(e), {
                    "tool_name": function_name,
                    "tool_arguments": arguments,
                    "call_id": tool_id
                })
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

    def _generate_auto_continue_prompt(self) -> str:
        """Generate an intelligent auto-continue prompt based on conversation context."""
        if not self.conversation_history:
            return "Please continue with our conversation."
            
        # Get recent context (last few messages)
        recent_messages = self.conversation_history[-3:]
        
        # Analyze what the user was working on
        work_indicators = {
            'project': ['build', 'create', 'project', 'implement', 'develop', 'code'],
            'search': ['search', 'find', 'look', 'research', 'information'],
            'analysis': ['analyze', 'examine', 'review', 'check', 'investigate'],
            'file_work': ['file', 'edit', 'write', 'modify', 'update'],
            'problem_solving': ['fix', 'debug', 'solve', 'issue', 'error', 'problem']
        }
        
        context_type = 'general'
        last_user_message = ""
        
        # Find the last user message
        for msg in reversed(recent_messages):
            if msg.get('role') == 'user':
                last_user_message = msg.get('content', '').lower()
                break
        
        # Determine context type
        for work_type, keywords in work_indicators.items():
            if any(keyword in last_user_message for keyword in keywords):
                context_type = work_type
                break
        
        # Generate context-appropriate continuation prompts
        prompts = {
            'project': "Please continue working on the project. What's the next step?",
            'search': "Please continue with the search or provide more information on this topic.",
            'analysis': "Please continue your analysis or provide additional insights.",
            'file_work': "Please continue with the file operations or show me what to do next.",
            'problem_solving': "Please continue troubleshooting or suggest the next solution step.",
            'general': "Please continue where we left off."
        }
        
        return prompts.get(context_type, prompts['general'])
    
    async def _prompt_auto_continue_setup(self):
        """Prompt user to configure auto-continue on startup."""
        current_status = "enabled" if self.config.ui.auto_continue_enabled else "disabled"
        timeout = self.config.ui.auto_continue_timeout
        
        # Create a nice startup prompt
        self.ui.console.print()
        self.ui.console.print("[bold yellow]⚙️  Auto-Continue Setup[/bold yellow]")
        self.ui.console.print(f"[dim]Auto-continue is currently [bold]{current_status}[/bold] (timeout: {timeout}s)[/dim]")
        self.ui.console.print("[dim]Auto-continue automatically resumes work after timeout periods[/dim]")
        self.ui.console.print()
        
        # Simple prompt with clear options
        self.ui.console.print("[yellow]Quick setup:[/yellow]")
        self.ui.console.print("  [green]y[/green]/[green]yes[/green] - Enable auto-continue")  
        self.ui.console.print("  [red]n[/red]/[red]no[/red] - Disable auto-continue")
        self.ui.console.print("  [cyan]<number>[/cyan] - Enable with custom timeout (seconds)")
        self.ui.console.print("  [dim]<enter>[/dim] - Keep current setting")
        
        try:
            response = await asyncio.wait_for(
                asyncio.to_thread(input, "\nAuto-continue setting: "), 
                timeout=15.0
            )
            response = response.strip().lower()
            
            if response in ['y', 'yes']:
                self.config.ui.auto_continue_enabled = True
                self.ui.display_success(f"✅ Auto-continue enabled (timeout: {self.config.ui.auto_continue_timeout}s)")
            elif response in ['n', 'no']:
                self.config.ui.auto_continue_enabled = False
                self.ui.display_success("✅ Auto-continue disabled")
            elif response.isdigit():
                timeout_val = int(response)
                if 10 <= timeout_val <= 300:  # 10 seconds to 5 minutes
                    self.config.ui.auto_continue_enabled = True
                    self.config.ui.auto_continue_timeout = float(timeout_val)
                    self.ui.display_success(f"✅ Auto-continue enabled with {timeout_val}s timeout")
                else:
                    self.ui.display_warning("Timeout must be between 10-300 seconds. Keeping current setting.")
            elif response == "":
                self.ui.display_info("✅ Keeping current auto-continue setting")
            else:
                self.ui.display_warning("Invalid response. Keeping current setting.")
                
        except asyncio.TimeoutError:
            # Auto-timeout after 15 seconds
            self.ui.display_info("⏰ Setup timed out. Keeping current auto-continue setting.")
        except KeyboardInterrupt:
            self.ui.display_info("⏹️  Setup cancelled. Keeping current setting.")
        except EOFError:
            # Non-interactive mode (no stdin available)
            self.ui.display_info("⚙️  Non-interactive mode detected. Keeping current auto-continue setting.")
        
        self.ui.console.print()
    
    async def _get_input_with_timeout(self, timeout: float) -> Optional[str]:
        """Get user input with timeout. Returns None if timeout occurs."""
        try:
            # Create a task for getting input
            def get_input():
                try:
                    return input()
                except EOFError:
                    return None
                    
            # Run input in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            input_task = loop.run_in_executor(None, get_input)
            
            # Wait with timeout
            result = await asyncio.wait_for(input_task, timeout=timeout)
            return result
            
        except asyncio.TimeoutError:
            return None
        except Exception:
            return None

    async def chat_loop(self):
        self.running = True
        # guarded for cross-platform

        try:

            signal.signal(signal.SIGINT, self.handle_signal)

        except (AttributeError, ValueError):

            pass
        # guarded for cross-platform

        try:

            signal.signal(signal.SIGTERM, self.handle_signal)

        except (AttributeError, ValueError):

            pass

        while self.running:
            try:
                # Get initial user input
                user_input = self.ui.get_user_input()

                # Handle timeout-based auto-continue
                auto_continued = False
                
                # --- Input handling with timeout and auto-continue ---
                while not user_input.strip():
                    try:
                        sys.stdout.write("◆ ")
                        sys.stdout.flush()
                        
                        # Use timeout-based input if auto-continue is enabled
                        if self.config.ui.auto_continue_enabled and self.conversation_history:
                            timeout_input = await self._get_input_with_timeout(self.config.ui.auto_continue_timeout)
                            
                            if timeout_input is None:
                                # Timeout occurred - auto-continue
                                auto_continue_prompt = self._generate_auto_continue_prompt()
                                self.ui.console.print(f"\n[dim yellow]⏰ Auto-continuing after {self.config.ui.auto_continue_timeout}s timeout...[/dim yellow]")
                                self.ui.console.print(f"[dim]Auto-prompt: {auto_continue_prompt}[/dim]\n")
                                user_input = auto_continue_prompt
                                auto_continued = True
                                break
                            else:
                                user_input = timeout_input or ""
                        else:
                            # Standard input without timeout
                            user_input = input()
                            
                    except EOFError:
                        self.running = False
                        break
                        
                if not self.running:
                    break
                # ---------------------------------------------------------------

                # new user turn → reset auto-tool guard
                self._auto_tool_used_this_turn = False

                # Process commands unless this is an auto-continue
                if not auto_continued:
                    if not self.process_command(user_input):
                        break
                    if user_input.startswith('/'):
                        continue

# Context management now handled by _auto_manage_context()
                
                self.conversation_history.append({"role": "user", "content": user_input})
                try:
                    self._history_add(user_input)
                except Exception:
                    pass
                # Note: User message display is handled by the input system

                # Automatically manage context before sending request
                self._auto_manage_context()

                messages = self.conversation_history.copy()
                tools = self.tool_registry.to_openai_tools_format() if self.config.tools.enabled else None

                if self.config.tools.enabled and tools:
                    system_msg = {
                        "role": "system", 
                        "content": (
                            "You are Demonology AI with 17 system tools. Use tools when users need files, code, or tasks done. "
                            "Tool names: file_operations, codebase_analysis, code_execution, project_planning, "
                            "image_analysis, image_generation, disassembler, hex_editor, pattern_search, debugger, "
                            "ghidra_analysis, waveform_generator, synthesizer, audio_analysis, midi_tool, "
                            "web_search, reddit_search. No synonyms - use exact names."
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
                    # Add spacing after response before next input
                    self.ui.console.print()
                    
                    self.conversation_history.append({"role": "assistant", "content": full_response})
                except DemonologyAPIError as e:
                    await self.ui.stop_loading()
                    
                    # If it's a server error, try to restart and reconnect
                    if "500" in str(e) or "connection" in str(e).lower():
                        self.ui.display_error("𐕣 Server error detected. Attempting automatic recovery... 𐕣")
                        
                        # Try to restart server and reconnect
                        if await self.ensure_server_connection():
                            self.ui.display_success("🔄 Connection restored! Retrying your request...")
                            # Retry the request once
                            try:
                                response_stream = self.client.stream_chat_completion(messages, tools=tools)
                                full_response = await self._handle_streaming_response_with_loading(response_stream)
                                self.ui.console.print()
                                self.conversation_history.append({"role": "assistant", "content": full_response})
                                continue  # Successfully recovered and processed
                            except Exception as retry_e:
                                self.ui.display_error(f"𐕣 Retry failed: {str(retry_e)} 𐕣")
                        else:
                            self.ui.display_error("𐕣 Server recovery failed. Please check your server manually. 𐕣")
                    else:
                        # Non-server errors, display normally
                        self.ui.display_error(f"𐕣 API Error: {str(e)} 𐕣")
                    # Continue conversation instead of breaking
                except Exception as e:
                    await self.ui.stop_loading()
                    logger.exception("Unexpected error in chat loop")
                    # More consistent error formatting
                    self.ui.display_error(f"𐕣 Unexpected error: {str(e)} 𐕣")
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
            if not await self.ensure_server_connection():
                self.ui.display_error("Cannot establish connection to API. Check your configuration and ensure your server is running.")
                return 1
            
            # Prompt for auto-continue setting on startup
            await self._prompt_auto_continue_setup()
            
            # Layout mode disabled due to stability issues
            # self.ui.start_layout_mode()
            
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
