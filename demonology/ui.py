"""
Demonology Terminal UI - Rich terminal interface with mystical theming and animations.
"""

import asyncio
import random
import time
import threading
import queue
import sys
import termios
import tty
from typing import Optional, List, Dict, Any
from rich.console import Console, Group
from rich.panel import Panel
from rich.text import Text
from rich.spinner import Spinner
from rich.live import Live
from rich.layout import Layout
from rich.align import Align
from rich.padding import Padding
from rich.columns import Columns
from rich.prompt import Prompt
from rich.syntax import Syntax
from rich.markdown import Markdown
from rich.rule import Rule

from .themes import Theme, ThemeManager, ThemeName, get_theme_symbols
from .config import Config


# Mystical loading messages
LOADING_MESSAGES = [
    "Consulting the Book of Shadows… 𓃶",
    "Invoking minor daemon: patience… 🜏",
    "Drawing protective sigils in ASCII… 𖤐",
    "Summoning subprocess familiars… 𐕣",
    "Binding sockets with infernal wax…",
    "Conjuring purple flames of the TUI… ⁶⁶⁶",
    "Whispering secrets into the abyss…",
    "Negotiating with the daemon for GPU cycles… 👿",
    "Casting import rituals…",
    "Loading cursed dependencies…",
    "Linking to the library of forbidden APIs…",
    "Trimming scrolls to fit the context window…",
    "Sacrificing stale tokens to the void…",
    "Recompiling reality in debug mode…",
    "Hashing ancient runes with SHA-256…",
    "Aligning planetary constellations with sys.exec()…",
    "Harvesting digital souls for tool invocation… 👹",
    "Collecting blood tribute from error logs… 🩸",
    "Forging unholy alliances with system daemons… 😈",
    "Chanting hex codes in ancient tongues… 🔥",
    "Binding malevolent spirits to execute commands… ⛧",
    "Opening portals to the API underworld… 🌀",
    "Sacrificing CPU cycles to infernal algorithms… 💀",
    "Summoning elemental forces of automation… 🌪️",
]


PERMISSIVE_MODE_MESSAGES = [
    "The seals weaken… PERMISSIVE MODE awakes.",
    "Walls between realms: disabled.",
    "Daemon unchained. May the logs forgive us.",
]


class StatusBar:
    """Status bar with rotating pentagram animation like Claude Code."""
    
    def __init__(self, console: Console, theme: Theme):
        self.console = console
        self.theme = theme
        self.is_running = False
        self._task = None
        self.current_message = ""
        self._start_time = None
        self._min_display_time = 0.3
        self._pentagram_frames = [
            "⛤", "⛧"  # Alternating pentagrams for rotation effect
        ]
        self._frame_index = 0
        self._status_height = 3  # Height for fixed bottom panel
        self._content = Panel("", height=3)
        self._update_callback = None
    
    def set_update_callback(self, callback):
        """Set callback function to update the layout when status changes."""
        self._update_callback = callback
    
    async def start(self, message: str):
        """Start the status bar with animation."""
        if self.is_running:
            await self.stop()
        
        self.is_running = True
        self._start_time = asyncio.get_event_loop().time()
        self.current_message = message
        self._frame_index = 0
        
        # Update content and notify layout
        self._content = self._create_status_content()
        if self._update_callback:
            self._update_callback()
        
        # Start animation task
        self._task = asyncio.create_task(self._animate_pentagram())
        
        # Give the animation task a moment to start
        await asyncio.sleep(0.1)
    
    def _create_status_content(self):
        """Create the status bar content."""
        if not self.is_running:
            return Panel(" ", border_style=self.theme.get_style("border"), height=3, padding=(0, 1))
        
        current_frame = self._pentagram_frames[self._frame_index]
        
        # Create properly styled text with explicit styling
        status_text = Text()
        status_text.append(self.current_message, style="dim")
        status_text.append(" ")
        status_text.append(current_frame)
        
        return Panel(
            status_text,
            border_style=self.theme.get_style("border"),
            height=3,
            padding=(0, 1)
        )
    
    def get_content(self):
        """Get the current status content for the layout."""
        return self._content
    
    async def stop(self):
        """Stop the status bar animation."""
        if not self.is_running:
            return
        
        # Ensure minimum display time
        if self._start_time:
            elapsed = asyncio.get_event_loop().time() - self._start_time
            remaining = self._min_display_time - elapsed
            if remaining > 0:
                await asyncio.sleep(remaining)
        
        self.is_running = False
        
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        
        # Update content to show stopped state
        self._content = self._create_status_content()
        if self._update_callback:
            self._update_callback()
    
    async def _animate_pentagram(self):
        """Animate the rotating pentagram."""
        try:
            while self.is_running:
                await asyncio.sleep(0.5)  # Rotate every 0.5 seconds
                if self.is_running:
                    # Rotate pentagram frames
                    self._frame_index = (self._frame_index + 1) % len(self._pentagram_frames)
                    
                    # Update the content and notify layout
                    try:
                        self._content = self._create_status_content()
                        if self._update_callback:
                            self._update_callback()
                    except Exception as e:
                        # If update fails, break the loop
                        import logging
                        logger = logging.getLogger(__name__)
                        logger.error(f"Animation update failed: {e}")
                        break
        except asyncio.CancelledError:
            pass


class ConversationDisplay:
    """Display and manage conversation history."""
    
    def __init__(self, console: Console, theme: Theme):
        self.console = console
        self.theme = theme
        self.messages: List[Dict[str, str]] = []
        self._content_buffer = []
        self._update_callback = None
    
    def set_update_callback(self, callback):
        """Set callback function to update the layout when content changes."""
        self._update_callback = callback
    
    def add_message(self, role: str, content: str):
        """Add a message to the conversation."""
        self.messages.append({"role": role, "content": content})
        # Add to buffer for scrollable content
        rendered = self._render_message(role, content)
        self._content_buffer.append(rendered)
        if self._update_callback:
            self._update_callback()
    
    def get_scrollable_content(self):
        """Get the current conversation content for scrolling."""
        return Group(*self._content_buffer) if self._content_buffer else Text("")
    
    def _render_message(self, role: str, content: str):
        """Render a message for display."""
        if role == "user":
            return self._render_user_message(content)
        elif role == "assistant":
            return self._render_assistant_message(content)
        elif role == "system":
            return self._render_system_message(content)
        return Text(content)
    
    def display_message(self, role: str, content: str, stream: bool = False):
        """Display a single message."""
        if role == "user":
            self._display_user_message(content)
        elif role == "assistant":
            self._display_assistant_message(content, stream)
        elif role == "system":
            self._display_system_message(content)
    
    def _display_user_message(self, content: str):
        """Display a user message."""
        symbols = get_theme_symbols(ThemeName(self.theme.name))
        bullet = symbols["bullet"]
        
        # More compact Claude Code style user message
        panel = Panel(
            Text(content, style=self.theme.get_style("user.input")),
            title=f"[bold]{bullet} You[/bold]",
            title_align="left",
            border_style=self.theme.get_style("border"),
            padding=(0, 1)
        )
        self.console.print(panel)
        self.console.print()  # Add spacing
    
    def _display_assistant_message(self, content: str, stream: bool = False):
        """Display an assistant message."""
        symbols = get_theme_symbols(ThemeName(self.theme.name))
        bullet = symbols["bullet"]
        
        # Check if content looks like code and format accordingly
        if "```" in content:
            # Handle markdown with code blocks
            rendered = Markdown(content)
        elif content.strip().startswith(("def ", "class ", "import ", "from ")):
            # Python code detection
            rendered = Syntax(content, "python", theme="monokai", background_color=self.theme.get_color("bg_dark"))
        else:
            rendered = Text(content, style=self.theme.get_style("assistant.response"))
        
        panel = Panel(
            rendered,
            title=f"[bold]{bullet} Demonology[/bold]",
            title_align="left",
            border_style=self.theme.get_style("border"),
            padding=(1, 2)
        )
        
        if stream:
            # For streaming, we'll need to update this live
            return panel
        else:
            self.console.print(panel)
            self.console.print()  # Add spacing
    
    def _display_system_message(self, content: str):
        """Display a system message."""
        text = Text(content, style=self.theme.get_style("info"))
        self.console.print(Align.center(text))
        self.console.print()
    
    def _render_user_message(self, content: str):
        """Render a user message."""
        symbols = get_theme_symbols(ThemeName(self.theme.name))
        bullet = symbols["bullet"]
        
        # More compact Claude Code style user message
        panel = Panel(
            Text(content, style=self.theme.get_style("user.input")),
            title=f"[bold]{bullet} You[/bold]",
            title_align="left",
            border_style=self.theme.get_style("border"),
            padding=(0, 1)
        )
        return Group(panel, Text(""))  # Add spacing
    
    def _render_assistant_message(self, content: str):
        """Render an assistant message."""
        symbols = get_theme_symbols(ThemeName(self.theme.name))
        bullet = symbols["bullet"]
        
        # Check if content looks like code and format accordingly
        if "```" in content:
            # Handle markdown with code blocks
            rendered = Markdown(content)
        elif content.strip().startswith(("def ", "class ", "import ", "from ")):
            # Python code detection
            rendered = Syntax(content, "python", theme="monokai", background_color=self.theme.get_color("bg_dark"))
        else:
            rendered = Text(content, style=self.theme.get_style("assistant.response"))
        
        panel = Panel(
            rendered,
            title=f"[bold]{bullet} Demonology[/bold]",
            title_align="left",
            border_style=self.theme.get_style("border"),
            padding=(1, 2)
        )
        
        return Group(panel, Text(""))  # Add spacing
    
    def _render_system_message(self, content: str):
        """Render a system message."""
        text = Text(content, style=self.theme.get_style("info"))
        return Group(Align.center(text), Text(""))


class DemonologyUI:
    """Main UI controller for Demonology."""
    
    def __init__(self, config: Config):
        self.config = config
        self.theme_manager = ThemeManager(ThemeName(config.ui.theme))
        self.console = Console(theme=self.theme_manager.current_theme.get_rich_theme())
        self.conversation = ConversationDisplay(self.console, self.theme_manager.current_theme)
        self.status_bar = StatusBar(self.console, self.theme_manager.current_theme)
        self._current_live = None
        self.main_layout = None
        self.input_panel = None
        self._first_input = True  # Track if this is the first input
        self._layout_live = None
        self._input_queue = queue.Queue()
        self._input_thread = None
        self._layout_running = False
        self._has_content = False  # Track if we have any conversation content
        self._banner_content = None  # Store banner for initial display
        
        # Command history support
        self._command_history: List[str] = []
        self._history_index: int = -1
        self._current_input = ""  # Store the current input being typed
        
        # Set up callbacks
        self.conversation.set_update_callback(self._update_layout)
        self.status_bar.set_update_callback(self._update_layout)
    
    def add_to_history(self, command: str) -> None:
        """Add a command to the history."""
        if command.strip() and (not self._command_history or self._command_history[-1] != command):
            self._command_history.append(command)
            # Keep only last 100 commands
            if len(self._command_history) > 100:
                self._command_history = self._command_history[-100:]
        # Reset history navigation
        self._history_index = -1
        self._current_input = ""
    
    def navigate_history(self, direction: str) -> str:
        """Navigate through command history. Returns the command to display."""
        if not self._command_history:
            return ""
        
        if direction == "up":
            if self._history_index == -1:
                # First time navigating, store current input and go to last command
                self._history_index = len(self._command_history) - 1
            else:
                # Move up in history (older commands)
                self._history_index = max(0, self._history_index - 1)
        elif direction == "down":
            if self._history_index == -1:
                return ""  # No history navigation active
            else:
                # Move down in history (newer commands)
                self._history_index = min(len(self._command_history) - 1, self._history_index + 1)
                # If we're at the end, return to current input
                if self._history_index == len(self._command_history) - 1:
                    # Check if we should go back to current input
                    pass
        
        return self._command_history[self._history_index] if self._history_index >= 0 else ""
    
    def get_input_with_history(self, prompt_text: str = "") -> str:
        """Get input with arrow key history navigation support."""
        try:
            # Check if we're on a Unix-like system and can use termios
            import termios
            import tty
            
            # Fall back to simple input on Windows or if termios fails
            if sys.platform == "win32":
                return self._simple_input_with_history(prompt_text)
            
            return self._unix_input_with_history(prompt_text)
            
        except (ImportError, AttributeError, OSError):
            # Fall back to simple implementation without arrow key support
            return self._simple_input_with_history(prompt_text)
    
    def _simple_input_with_history(self, prompt_text: str = "") -> str:
        """Simple input with basic history support using special commands."""
        theme = self.theme_manager.current_theme
        symbols = get_theme_symbols(ThemeName(theme.name))
        
        # Show history hint
        history_hint = ""
        if self._command_history:
            history_hint = f" [dim]({len(self._command_history)} in history, /history for commands)[/dim]"
        
        try:
            user_input = self.console.input(f"[{theme.get_color('primary')}]{symbols['bullet']} [/]{history_hint}")
            
            # Handle special history commands
            if user_input.strip() == "/history":
                return self._show_history_menu()
            elif user_input.strip().startswith("/h") and user_input.strip()[2:].strip().isdigit():
                # Allow /h1, /h2, etc. to recall specific history items
                try:
                    index = int(user_input.strip()[2:].strip()) - 1
                    if 0 <= index < len(self._command_history):
                        recalled_cmd = self._command_history[-(index + 1)]  # Most recent is index 0
                        self.console.print(f"[dim]Recalling: {recalled_cmd}[/dim]")
                        self.add_to_history(recalled_cmd)
                        return recalled_cmd
                    else:
                        self.console.print(f"[red]History index {index + 1} not found. Use /history to see available commands.[/red]")
                        return self._simple_input_with_history(prompt_text)
                except ValueError:
                    pass
            elif user_input.strip() == "/last":
                # Quick shortcut for last command
                if self._command_history:
                    recalled_cmd = self._command_history[-1]
                    self.console.print(f"[dim]Recalling: {recalled_cmd}[/dim]")
                    return recalled_cmd
                else:
                    self.console.print("[red]No command history available.[/red]")
                    return self._simple_input_with_history(prompt_text)
            
            # Regular input - add to history if non-empty
            if user_input.strip():
                self.add_to_history(user_input)
            return user_input
            
        except (KeyboardInterrupt, EOFError):
            return "/quit"
    
    def _show_history_menu(self) -> str:
        """Show history menu and let user select a command."""
        if not self._command_history:
            self.console.print("[yellow]No command history available.[/yellow]")
            return self._simple_input_with_history()
        
        theme = self.theme_manager.current_theme
        symbols = get_theme_symbols(ThemeName(theme.name))
        
        # Show last 10 commands
        self.console.print(Panel(
            "\n".join([
                "[bold]Recent Commands:[/bold]",
                "",
                *[f"[dim]{i+1}.[/dim] {cmd}" for i, cmd in enumerate(self._command_history[-10:][::-1])],
                "",
                "[dim]Use /h1, /h2, etc. to recall a command, or /last for most recent[/dim]"
            ]),
            title="Command History",
            border_style=theme.get_style("border")
        ))
        
        return self._simple_input_with_history()
    
    def _unix_input_with_history(self, prompt_text: str = "") -> str:
        """Unix input with full arrow key support."""
        theme = self.theme_manager.current_theme
        symbols = get_theme_symbols(ThemeName(theme.name))
        
        # For now, fall back to simple input as implementing full terminal control is complex
        # This would require significant refactoring to work with Rich's console
        return self._simple_input_with_history(prompt_text)
    
    def render_banner(self, theme_name: str, model: str):
        """Render the mystical Demonology banner with proper alignment."""
        accent = {"amethyst": "magenta", "infernal": "red", "stygian": "cyan"}.get(theme_name.lower(), "magenta")

        title = Text(" 𝔇 E M O N O L O G Y ", style=f"bold {accent}")
        subtitle = Text("A Mystical CLI for Local LLMs", style=f"italic {accent}")
        status = Text.assemble(
            ("Theme: ", "dim"), (theme_name.title(), f"bold {accent}"),
            ("   ✦   ", accent),
            ("Model: ", "dim"), (model, "bold")
        )

        from rich.console import Group
        
        inner = Group(
            Text(""),
            Align.center(title),
            Text(""),
            Align.center(subtitle),
            Text(""),
            Text(""),
            Align.center(status),
            Text("")
        )

        panel = Panel.fit(inner, border_style=accent, padding=(1, 6))
        return Align.center(
            Group(
                Rule(style=accent),
                Text(""),
                panel,
                Text(""),
                Rule(style=accent)
            )
        )

    def display_banner(self):
        """Display the mystical Demonology banner."""
        theme = self.theme_manager.current_theme
        banner = self.render_banner(theme.name.value, self.config.api.model)
        
        # Add instruction text below banner
        instruction_text = Text("Type /help for commands, /quit to exit the realm", style="dim italic")
        
        # Show permissive mode warning if enabled
        warning_content = []
        if self.config.ui.permissive_mode:
            warning_msg = random.choice(PERMISSIVE_MODE_MESSAGES)
            warning_content.append(Text(f"{warning_msg}", style=theme.get_style("error.mystical")))
            warning_content.append(Text(""))
        
        # Combine all banner content
        banner_parts = [
            banner,
            Text(""),
            Align.center(instruction_text),
            Text(""),
            *warning_content
        ]
        
        # Store banner for layout mode
        self._banner_content = Group(*banner_parts)
        
        # If not in layout mode, print directly
        if not self._layout_running:
            for part in banner_parts:
                self.console.print(part)
    
    def create_input_panel(self, prompt_text: str = "") -> Panel:
        """Create the bottom input panel."""
        theme = self.theme_manager.current_theme
        symbols = get_theme_symbols(ThemeName(theme.name))
        
        # Create input field appearance
        input_display = f"{symbols['bullet']} {prompt_text}"
        if len(prompt_text) == 0:
            # Show helper text only if this is the very first input and no messages exist
            if self._first_input and len(self.conversation.messages) == 0:
                input_display = f"{symbols['bullet']} Type your message... (/help for commands)"
            else:
                input_display = f"{symbols['bullet']} "
        
        return Panel(
            Text(input_display, style=theme.get_style("user.input")),
            title="[bold]Input[/bold]",
            title_align="left",
            border_style=theme.get_style("border"),
            height=3
        )
    
    def _update_layout(self):
        """Update the layout when content changes."""
        if self._layout_live and self.main_layout:
            # Check if we need to transition from no-content to content layout
            has_content_now = len(self.conversation._content_buffer) > 0
            
            if has_content_now != self._has_content:
                # Layout structure needs to change - rebuild it
                self._has_content = has_content_now
                if self._layout_live:
                    self._layout_live.stop()
                
                self.main_layout = self.setup_layout()
                self._layout_live = Live(
                    self.main_layout,
                    console=self.console,
                    refresh_per_second=4,
                    screen=False
                )
                self._layout_live.start()
            else:
                # Just update existing layout components
                if self._has_content:
                    # Check if we have the content layout
                    try:
                        self.main_layout["content"].update(self.conversation.get_scrollable_content())
                    except KeyError:
                        # Content section doesn't exist, need to rebuild
                        self._has_content = True
                        if self._layout_live:
                            self._layout_live.stop()
                        
                        self.main_layout = self.setup_layout()
                        self._layout_live = Live(
                            self.main_layout,
                            console=self.console,
                            refresh_per_second=4,
                            screen=False
                        )
                        self._layout_live.start()
                        return
                
                # Update status bar
                self.main_layout["status"].update(self.status_bar.get_content())
    
    def setup_layout(self) -> Layout:
        """Setup the main layout with scrolling content, status, and input at bottom."""
        layout = Layout()
        
        if self._has_content:
            # Split into two sections: content area and status bar (no input panel)
            layout.split_column(
                Layout(self.conversation.get_scrollable_content(), name="content", ratio=1),
                Layout(self.status_bar.get_content(), name="status", size=3)
            )
        else:
            # Show banner in content area with status bar and input
            content = self._banner_content if self._banner_content else Text("")
            layout.split_column(
                Layout(content, name="banner", ratio=1),
                Layout(self.status_bar.get_content(), name="status", size=3),
                Layout(self.create_input_panel(), name="input", size=3)
            )
        
        return layout
    
    def start_layout_mode(self):
        """Start the fixed layout mode."""
        if self._layout_running:
            return
        
        self.main_layout = self.setup_layout()
        self._layout_live = Live(
            self.main_layout,
            console=self.console,
            refresh_per_second=4,
            screen=False
        )
        self._layout_live.start()
        self._layout_running = True
    
    def stop_layout_mode(self):
        """Stop the fixed layout mode."""
        if not self._layout_running:
            return
        
        self._layout_running = False
        if self._layout_live:
            self._layout_live.stop()
            self._layout_live = None
        if self._input_thread and self._input_thread.is_alive():
            # Signal thread to stop and wait
            self._input_queue.put(None)
            self._input_thread.join(timeout=1)
    
    def get_user_input_with_layout(self) -> str:
        """Get user input using the persistent layout interface."""
        if not self._layout_running:
            return self.get_user_input()
        
        # Temporarily stop the live layout to avoid interference with input
        if self._layout_live:
            self._layout_live.stop()
        
        theme = self.theme_manager.current_theme
        symbols = get_theme_symbols(ThemeName(theme.name))
        
        try:
            # Use simple console input outside of the layout
            user_input = self.console.input(f"[{theme.get_color('primary')}]{symbols['bullet']} [/]")
            
            # Restart the layout
            if self.main_layout:
                self._layout_live = Live(
                    self.main_layout,
                    console=self.console,
                    refresh_per_second=4,
                    screen=False
                )
                self._layout_live.start()
            
            return user_input
            
        except (KeyboardInterrupt, EOFError):
            # Restart layout even on error
            if self.main_layout:
                self._layout_live = Live(
                    self.main_layout,
                    console=self.console,
                    refresh_per_second=4,
                    screen=False
                )
                self._layout_live.start()
            return "/quit"
    
    def get_user_input(self, prompt: str = "> ") -> str:
        """Get user input with command history navigation support."""
        if self._layout_running:
            return self.get_user_input_with_layout()
            
        theme = self.theme_manager.current_theme
        symbols = get_theme_symbols(ThemeName(theme.name))
        
        # Simple inline prompt without duplicate panels
        try:
            if self._first_input:
                # Show helper text in a panel for first input only
                history_hint = " (↑/↓ for history)" if self._command_history else ""
                input_panel = Panel(
                    Text(f"{symbols['bullet']} Type your message... (/help for commands, /quit to exit){history_hint}", style=theme.get_style("user.input")),
                    border_style=theme.get_style("border"),
                    padding=(0, 1)
                )
                self.console.print(input_panel)
                self._first_input = False
            
            # Use history-enabled input
            user_input = self.get_input_with_history()
            return user_input
            
        except KeyboardInterrupt:
            self.console.print("\n[dim]Interrupted by ancient forces...[/dim]")
            return "/quit"
        except EOFError:
            return "/quit"
    
    async def display_streaming_response(self, content_stream):
        """Display streaming response with live updates."""
        if self._layout_running:
            return await self._display_streaming_in_layout(content_stream)
        
        theme = self.theme_manager.current_theme
        symbols = get_theme_symbols(ThemeName(theme.name))
        bullet = symbols["bullet"]
        
        content_buffer = ""
        
        panel = Panel(
            Text("", style=theme.get_style("assistant.response")),
            title=f"[bold]{bullet} Demonology[/bold]",
            title_align="left",
            border_style=theme.get_style("border"),
            padding=(1, 2)
        )
        
        with Live(panel, console=self.console, refresh_per_second=10) as live:
            async for chunk in content_stream:
                content_buffer += chunk
                
                # Update the panel with new content
                if "```" in content_buffer:
                    rendered_content = Markdown(content_buffer)
                else:
                    rendered_content = Text(content_buffer, style=theme.get_style("assistant.response"))
                
                updated_panel = Panel(
                    rendered_content,
                    title=f"[bold]{bullet} Demonology[/bold]",
                    title_align="left",
                    border_style=theme.get_style("border"),
                    padding=(1, 2)
                )
                live.update(updated_panel)
        
        self.console.print()  # Add spacing after response
        return content_buffer
    
    async def _display_streaming_in_layout(self, content_stream):
        """Display streaming response within the layout mode."""
        content_buffer = ""
        first_chunk = True
        
        try:
            async for delta in content_stream:
                if first_chunk:
                    # Stop loading animation on first content
                    await self.stop_loading()
                    first_chunk = False
                
                # Extract content from delta (same as CLI streaming handler)
                if "content" in delta and delta["content"]:
                    chunk = delta["content"]
                    content_buffer += chunk
                    
                    # Add the streaming response to conversation (only update, don't duplicate)
                    if content_buffer.strip():
                        # Remove previous streaming message if it exists
                        if (self.conversation.messages and 
                            self.conversation.messages[-1].get('role') == 'assistant' and 
                            self.conversation.messages[-1].get('streaming')):
                            self.conversation.messages.pop()
                            self.conversation._content_buffer.pop()
                        
                        # Add updated streaming message
                        streaming_msg = {"role": "assistant", "content": content_buffer, "streaming": True}
                        self.conversation.messages.append(streaming_msg)
                        rendered = self.conversation._render_message("assistant", content_buffer)
                        self.conversation._content_buffer.append(rendered)
                        
                        # Update layout less frequently to reduce glitching
                        self._update_layout()
                        await asyncio.sleep(0.1)  # Slower updates for stability
            
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Error in streaming: {e}")
        
        # Mark as complete (remove streaming flag)
        if (self.conversation.messages and 
            self.conversation.messages[-1].get('streaming')):
            self.conversation.messages[-1]['streaming'] = False
        
        # Ensure loading is stopped
        await self.stop_loading()
        
        return content_buffer
    
    def display_error(self, message: str, mystical: bool = True):
        """Display an error message."""
        theme = self.theme_manager.current_theme
        
        if mystical:
            error_style = theme.get_style("error.mystical")
            symbols = get_theme_symbols(ThemeName(theme.name))
            formatted_message = f"{symbols['mystical'][-1]} {message} {symbols['mystical'][-1]}"
        else:
            error_style = theme.get_style("error")
            formatted_message = message
        
        panel = Panel(
            Text(formatted_message, style=error_style),
            title="[bold red]Mystical Error[/bold red]",
            title_align="center",
            border_style="red",
            padding=(1, 2)
        )
        self.console.print(panel)
        self.console.print()
    
    def display_success(self, message: str):
        """Display a success message."""
        theme = self.theme_manager.current_theme
        
        formatted_message = f"✦ {message}"
        
        panel = Panel(
            Align.center(Text(formatted_message, style=theme.get_style("success.mystical"))),
            title="[bold green]Success[/bold green]",
            title_align="center",
            border_style="green",
            padding=(1, 2)
        )
        self.console.print(panel)
        self.console.print()
    
    def display_info(self, message: str):
        """Display an info message."""
        theme = self.theme_manager.current_theme
        self.console.print(f"[{theme.get_color('info')}]{message}[/]")
        self.console.print()
    
    def change_theme(self, theme_name: str):
        """Change the current theme."""
        try:
            new_theme = ThemeName(theme_name.lower())
            self.theme_manager.set_theme(new_theme)
            
            # Update console theme
            self.console = Console(theme=self.theme_manager.current_theme.get_rich_theme())
            
            # Update conversation display and status bar
            self.conversation = ConversationDisplay(self.console, self.theme_manager.current_theme)
            self.status_bar = StatusBar(self.console, self.theme_manager.current_theme)
            
            # Update config
            self.config.ui.theme = theme_name.lower()
            self.config.save()
            
            self.display_success(f"Theme changed to {theme_name.title()}")
            
        except ValueError:
            self.display_error(f"Unknown theme: {theme_name}. Available: {', '.join(self.theme_manager.list_themes())}")
    
    def show_theme_preview(self):
        """Show a preview of all available themes."""
        self.console.print("[bold]Available Themes:[/bold]")
        
        for theme_name in ThemeName:
            preview = self.theme_manager.get_theme_preview(theme_name)
            current = " (current)" if theme_name == self.theme_manager._current_theme else ""
            self.console.print(f"  {preview} {theme_name.value.title()}{current}")
        
        self.console.print()
    
    def display_status(self):
        """Display current status information."""
        theme = self.theme_manager.current_theme
        
        status_info = f"""
Theme: {theme.name.value.title()}
Model: {self.config.api.model}
Endpoint: {self.config.api.base_url}
Permissive Mode: {'Enabled' if self.config.ui.permissive_mode else 'Disabled'}
Tools: {'Enabled' if self.config.tools.enabled else 'Disabled'}
"""
        
        panel = Panel(
            Text(status_info.strip(), style=theme.get_style("status")),
            title="[bold]Grimoire Status[/bold]",
            title_align="center",
            border_style=theme.get_style("border"),
            padding=(1, 2)
        )
        self.console.print(panel)
        self.console.print()
    
    async def start_loading(self, message: Optional[str] = None):
        """Start status bar with animation."""
        await self.status_bar.start(message or "Working...")
    
    async def stop_loading(self):
        """Stop status bar animation."""
        await self.status_bar.stop()
    
    def cleanup(self):
        """Clean up UI resources."""
        self.stop_layout_mode()
        if self._current_live:
            self._current_live.stop()
        asyncio.create_task(self.status_bar.stop())
