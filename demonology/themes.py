"""
Demonology Theming System - Mystical color schemes and visual styling.
"""

from enum import Enum
from typing import Dict, Any
from rich.style import Style
from rich.theme import Theme as RichTheme
from rich.color import Color


class ThemeName(str, Enum):
    """Available theme names."""
    AMETHYST = "amethyst"
    INFERNAL = "infernal"
    STYGIAN = "stygian"


class Theme:
    """Theme configuration for Demonology interface."""
    
    def __init__(self, name: ThemeName):
        self.name = name
        self._colors = self._get_color_scheme()
        self._styles = self._build_styles()
        self._rich_theme = self._create_rich_theme()
    
    def _get_color_scheme(self) -> Dict[str, str]:
        """Get color scheme based on theme name."""
        schemes = {
            ThemeName.AMETHYST: {
                "primary": "#9966CC",      # Deep purple
                "secondary": "#6A4C93",    # Dark purple
                "accent": "#E6E6FA",       # Lavender
                "bg_dark": "#1A0D26",      # Very dark purple
                "bg_light": "#2D1B3D",     # Dark purple
                "text_primary": "#E6E6FA", # Lavender
                "text_secondary": "#B39AC7", # Light purple
                "success": "#90EE90",      # Light green
                "warning": "#FFD700",      # Gold
                "error": "#FF6B6B",        # Light red
                "info": "#87CEEB",         # Sky blue
                "border": "#9966CC",       # Deep purple
            },
            ThemeName.INFERNAL: {
                "primary": "#CC3333",      # Deep red
                "secondary": "#FF6600",    # Orange-red
                "accent": "#FFCC99",       # Light orange
                "bg_dark": "#1A0A0A",      # Very dark red
                "bg_light": "#330A0A",     # Dark red
                "text_primary": "#FFCC99", # Light orange
                "text_secondary": "#FF9966", # Orange
                "success": "#FF8C00",      # Dark orange
                "warning": "#FFD700",      # Gold
                "error": "#FF0000",        # Bright red
                "info": "#FF6347",         # Tomato
                "border": "#CC3333",       # Deep red
            },
            ThemeName.STYGIAN: {
                "primary": "#008B8B",      # Dark cyan
                "secondary": "#20B2AA",    # Light sea green
                "accent": "#AFEEEE",       # Pale turquoise
                "bg_dark": "#0A1A1A",      # Very dark teal
                "bg_light": "#0F2A2A",     # Dark teal
                "text_primary": "#AFEEEE", # Pale turquoise
                "text_secondary": "#7FFFD4", # Aquamarine
                "success": "#00FF7F",      # Spring green
                "warning": "#FFD700",      # Gold
                "error": "#FF6B6B",        # Light red
                "info": "#00CED1",         # Dark turquoise
                "border": "#008B8B",       # Dark cyan
            }
        }
        return schemes[self.name]
    
    def _build_styles(self) -> Dict[str, Style]:
        """Build Rich styles from color scheme."""
        return {
            "primary": Style(color=self._colors["primary"], bold=True),
            "secondary": Style(color=self._colors["secondary"]),
            "accent": Style(color=self._colors["accent"]),
            "text.primary": Style(color=self._colors["text_primary"]),
            "text.secondary": Style(color=self._colors["text_secondary"]),
            "success": Style(color=self._colors["success"], bold=True),
            "warning": Style(color=self._colors["warning"], bold=True),
            "error": Style(color=self._colors["error"], bold=True),
            "info": Style(color=self._colors["info"]),
            "border": Style(color=self._colors["border"]),
            "bg.dark": Style(bgcolor=self._colors["bg_dark"]),
            "bg.light": Style(bgcolor=self._colors["bg_light"]),
            
            # Specialized styles
            "user.input": Style(color=self._colors["accent"], bold=True),
            "assistant.response": Style(color=self._colors["text_primary"]),
            "loading": Style(color=self._colors["secondary"], italic=True),
            "error.mystical": Style(color=self._colors["error"], bold=True, italic=True),
            "success.mystical": Style(color=self._colors["success"], bold=True),
            "title": Style(color=self._colors["primary"], bold=True, underline=True),
            "status": Style(color=self._colors["info"], dim=True),
            "code": Style(color=self._colors["accent"], bgcolor=self._colors["bg_dark"]),
        }
    
    def _create_rich_theme(self) -> RichTheme:
        """Create a Rich theme from the styles."""
        theme_dict = {name: style for name, style in self._styles.items()}
        return RichTheme(theme_dict)
    
    def get_color(self, name: str) -> str:
        """Get a color value by name."""
        return self._colors.get(name, self._colors["text_primary"])
    
    def get_style(self, name: str) -> Style:
        """Get a style by name."""
        return self._styles.get(name, self._styles["text.primary"])
    
    def get_rich_theme(self) -> RichTheme:
        """Get the Rich theme object."""
        return self._rich_theme
    
    @property
    def colors(self) -> Dict[str, str]:
        """Get all colors in the theme."""
        return self._colors.copy()
    
    @property
    def styles(self) -> Dict[str, Style]:
        """Get all styles in the theme."""
        return self._styles.copy()


class ThemeManager:
    """Manages theme switching and customization."""
    
    def __init__(self, default_theme: ThemeName = ThemeName.AMETHYST):
        self._themes = {name: Theme(name) for name in ThemeName}
        self._current_theme = default_theme
    
    @property
    def current_theme(self) -> Theme:
        """Get the current active theme."""
        return self._themes[self._current_theme]
    
    def set_theme(self, theme_name: ThemeName) -> None:
        """Set the current theme."""
        if theme_name in self._themes:
            self._current_theme = theme_name
        else:
            raise ValueError(f"Unknown theme: {theme_name}")
    
    def get_theme(self, theme_name: ThemeName) -> Theme:
        """Get a specific theme."""
        if theme_name in self._themes:
            return self._themes[theme_name]
        else:
            raise ValueError(f"Unknown theme: {theme_name}")
    
    def list_themes(self) -> list[str]:
        """List all available theme names."""
        return [theme.value for theme in ThemeName]
    
    def get_theme_preview(self, theme_name: ThemeName) -> str:
        """Get a preview string for a theme."""
        theme = self.get_theme(theme_name)
        preview_colors = ["primary", "secondary", "accent", "text_primary"]
        preview_text = f"[{theme.name.value}] "
        
        for i, color_name in enumerate(preview_colors):
            color = theme.get_color(color_name)
            preview_text += f"[{color}]●[/] "
        
        return preview_text.strip()


# Mystical symbols and decorations for different themes
THEME_SYMBOLS = {
    ThemeName.AMETHYST: {
        "loading": ["⚬", "⚮", "⚯"],
        "bullet": "◆",
        "separator": "┃",
        "corner": "┌",
        "mystical": ["𓃶", "🜏", "𖤐", "𐕣"],
    },
    ThemeName.INFERNAL: {
        "loading": ["🔥", "⚡", "💀"],
        "bullet": "▲",
        "separator": "║",
        "corner": "╔",
        "mystical": ["👿", "⁶⁶⁶", "🜄", "𖤍"],
    },
    ThemeName.STYGIAN: {
        "loading": ["〰", "≈", "∼"],
        "bullet": "◇",
        "separator": "│",
        "corner": "┌",
        "mystical": ["⊕", "⊗", "⊙", "○"],
    }
}


def get_theme_symbols(theme_name: ThemeName) -> Dict[str, Any]:
    """Get symbols associated with a theme."""
    return THEME_SYMBOLS.get(theme_name, THEME_SYMBOLS[ThemeName.AMETHYST])