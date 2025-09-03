"""
Demonology Terminal UI - Rich terminal interface with mystical theming and animations.
"""

import asyncio
import random
import time
from typing import Optional, List, Dict, Any
from rich.console import Console
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


class LoadingAnimation:
    """Mystical loading animation with rotating messages."""
    
    def __init__(self, console: Console, theme: Theme):
        self.console = console
        self.theme = theme
        self.is_running = False
        self._task = None
        self._live = None
        self.current_message = ""
        self._start_time = None
        self._min_display_time = 1.5  # Minimum seconds to display loading
    
    async def start(self, initial_message: Optional[str] = None):
        """Start the loading animation."""
        if self.is_running:
            return
        
        self.is_running = True
        self._start_time = asyncio.get_event_loop().time()
        self.current_message = initial_message or random.choice(LOADING_MESSAGES)
        
        # Create spinner based on theme
        theme_symbols = get_theme_symbols(ThemeName(self.theme.name))
        spinner_chars = theme_symbols.get("loading", ["⚬", "⚮", "⚯"])
        
        spinner = Spinner(
            "dots",
            text=Text(self.current_message, style=self.theme.get_style("loading")),
            style=self.theme.get_style("primary")
        )
        
        self._live = Live(
            Panel(
                Align.center(spinner),
                border_style=self.theme.get_style("border"),
                title="[bold]Demonology[/bold]",
                title_align="center"
            ),
            console=self.console,
            refresh_per_second=4,
            transient=True
        )
        
        self._live.start()
        self._task = asyncio.create_task(self._rotate_messages())
    
    async def stop(self):
        """Stop the loading animation."""
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
        
        if self._live:
            self._live.stop()
    
    async def _rotate_messages(self):
        """Rotate through loading messages."""
        try:
            while self.is_running:
                await asyncio.sleep(2.0)  # Change message every 2 seconds
                if self.is_running:
                    self.current_message = random.choice(LOADING_MESSAGES)
                    if self._live:
                        # Update the spinner text
                        theme_symbols = get_theme_symbols(ThemeName(self.theme.name))
                        spinner = Spinner(
                            "dots",
                            text=Text(self.current_message, style=self.theme.get_style("loading")),
                            style=self.theme.get_style("primary")
                        )
                        self._live.update(
                            Panel(
                                Align.center(spinner),
                                border_style=self.theme.get_style("border"),
                                title="[bold]Demonology[/bold]",
                                title_align="center"
                            )
                        )
        except asyncio.CancelledError:
            pass


class ConversationDisplay:
    """Display and manage conversation history."""
    
    def __init__(self, console: Console, theme: Theme):
        self.console = console
        self.theme = theme
        self.messages: List[Dict[str, str]] = []
    
    def add_message(self, role: str, content: str):
        """Add a message to the conversation."""
        self.messages.append({"role": role, "content": content})
    
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
        
        panel = Panel(
            Text(content, style=self.theme.get_style("user.input")),
            title=f"[bold]{bullet} You[/bold]",
            title_align="left",
            border_style=self.theme.get_style("border"),
            padding=(1, 2)
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


class DemonologyUI:
    """Main UI controller for Demonology."""
    
    def __init__(self, config: Config):
        self.config = config
        self.theme_manager = ThemeManager(ThemeName(config.ui.theme))
        self.console = Console(theme=self.theme_manager.current_theme.get_rich_theme())
        self.conversation = ConversationDisplay(self.console, self.theme_manager.current_theme)
        self.loading_animation = LoadingAnimation(self.console, self.theme_manager.current_theme)
        self._current_live = None
        self.main_layout = None
        self.input_panel = None
    
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
        self.console.print(banner)
        
        # Add instruction text below banner
        instruction_text = Text("Type /help for commands, /quit to exit the realm", style="dim italic")
        self.console.print(Align.center(instruction_text))
        self.console.print()
        
        # Show permissive mode warning if enabled
        if self.config.ui.permissive_mode:
            warning_msg = random.choice(PERMISSIVE_MODE_MESSAGES)
            self.console.print(
                f"\n[bold red]{warning_msg}[/bold red]",
                style=theme.get_style("error.mystical")
            )
        
        self.console.print()
    
    def create_input_panel(self, prompt_text: str = "") -> Panel:
        """Create the bottom input panel."""
        theme = self.theme_manager.current_theme
        symbols = get_theme_symbols(ThemeName(theme.name))
        
        # Create input field appearance
        input_display = f"{symbols['bullet']} {prompt_text}"
        if len(prompt_text) == 0:
            input_display = f"{symbols['bullet']} Type your message... (/help for commands)"
        
        return Panel(
            Text(input_display, style=theme.get_style("user.input")),
            title="[bold]Input[/bold]",
            title_align="left",
            border_style=theme.get_style("border"),
            height=3
        )
    
    def setup_layout(self) -> Layout:
        """Setup the main layout with fixed input at bottom."""
        layout = Layout()
        
        # Split into main content area and input area
        layout.split_column(
            Layout(name="main", ratio=1),
            Layout(self.create_input_panel(), name="input", size=3)
        )
        
        return layout
    
    def get_user_input_with_layout(self) -> str:
        """Get user input using the persistent layout interface."""
        import threading
        import queue
        import sys
        
        # For now, fall back to the original input method
        # This is a complex feature that would require significant restructuring
        # to implement properly with rich's input handling
        return self.get_user_input()
    
    def get_user_input(self, prompt: str = "> ") -> str:
        """Get user input with mystical styling and boxed input area."""
        theme = self.theme_manager.current_theme
        symbols = get_theme_symbols(ThemeName(theme.name))
        
        # Add some spacing before the input box
        self.console.print()
        
        # Create input prompt inside a panel
        input_prompt = f"{symbols['bullet']} Type your message... (/help for commands, /quit to exit)"
        input_panel = Panel(
            Text(input_prompt, style=theme.get_style("user.input")),
            title="[bold]Input[/bold]",
            title_align="left",
            border_style=theme.get_style("border"),
            padding=(0, 1)
        )
        self.console.print(input_panel)
        
        # Get input with styled prompt
        styled_prompt = f"[{theme.get_color('primary')}]{symbols['bullet']} [/]"
        
        try:
            user_input = self.console.input(styled_prompt)
            return user_input
        except KeyboardInterrupt:
            self.console.print("\n[dim]Interrupted by ancient forces...[/dim]")
            return "/quit"
        except EOFError:
            return "/quit"
    
    async def display_streaming_response(self, content_stream):
        """Display streaming response with live updates."""
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
            
            # Update conversation display and loading animation
            self.conversation = ConversationDisplay(self.console, self.theme_manager.current_theme)
            self.loading_animation = LoadingAnimation(self.console, self.theme_manager.current_theme)
            
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
        """Start loading animation."""
        await self.loading_animation.start(message)
    
    async def stop_loading(self):
        """Stop loading animation."""
        await self.loading_animation.stop()
    
    def cleanup(self):
        """Clean up UI resources."""
        if self._current_live:
            self._current_live.stop()
        asyncio.create_task(self.loading_animation.stop())
